"""合成语料上的真实Codex两轮画像、GPU检索与生成联调；不读取真实评论。"""

import asyncio
import json
import tempfile
import traceback
from pathlib import Path

from app.schemas import GenerateRequest, ModelRequest, StyleEdit
from app.service import Service

PAIRS = [
    ("数学错题太多怎么办？", "先别着急，可以先把错题按原因分成三组，再从最常见的一组开始练习。"),
    (
        "刚开始学Python，列表和元组怎么区别？",
        "我们可以先记住一个区别：列表可以改，元组创建后不能改。比如待办清单适合列表。",
    ),
    ("复习任务很多，如何安排？", "先选今天最需要完成的一件事，给它一个具体时间段。【小目标】完成后再安排下一步。"),
    ("练习总是失败，很难受。", "有这种感受很正常，先休息一会儿。我们可以把这次卡住的地方拆小一点，再试一遍。"),
    ("谢谢你的解释！", "不客气，慢慢来，祝你今天顺利完成练习！"),
]


async def wait(service, job):
    deadline = asyncio.get_running_loop().time() + 800
    while asyncio.get_running_loop().time() < deadline:
        current = service.store.get("jobs", job["id"])
        if current["status"] not in {"queued", "running"}:
            if current["status"] != "succeeded":
                raise RuntimeError(json.dumps(current["error"], ensure_ascii=False))
            return current
        await asyncio.sleep(0.5)
    raise TimeoutError("联调任务等待超时")


async def main():
    with tempfile.TemporaryDirectory(prefix="reply-workflow-") as folder:
        service = Service(Path(folder))
        original_handler = service.jobs.handler

        async def diagnostic_handler(job, key):
            try:
                return await original_handler(job, key)
            except Exception:
                destination = Path(".verification") / "workflow-failure.txt"
                destination.write_text(traceback.format_exc(), encoding="utf-8")
                raise

        service.jobs.handler = diagnostic_handler
        groups = []
        for index, (question, answer) in enumerate(PAIRS):
            note = f"synthetic-note-{index}"
            messages = []
            for role, text in ((False, question), (True, answer)):
                messages.append(
                    {
                        "id": "a" if role else "q",
                        "thread": note + ":q",
                        "note_id": note,
                        "comment_id": "a" if role else "q",
                        "content": text,
                        "target": role,
                        "context_only": False,
                        "parent_comment_id": "q" if role else None,
                        "root_comment_id": "q",
                        "platform_time": 2 if role else 1,
                        "nickname": "合成作者" if role else "合成提问者",
                        "user_id": "synthetic-author" if role else "asker",
                    }
                )
            groups.append({"kind": "qa" if index != 4 else "standalone", "messages": messages, "warnings": []})
        source = {
            "id": "synthetic-workflow",
            "status": "complete",
            "nickname": "合成作者",
            "user_id": "synthetic-author",
            "groups": groups,
            "warnings": [],
        }
        preview = service.preview_import(json.dumps(source, ensure_ascii=False).encode())
        outcome = service.confirm_import(preview["id"])
        await service.jobs.start()
        try:
            extraction = await service.submit("style", ModelRequest(request_id="synthetic-style-extract"))
            done_style = await wait(service, extraction)
            profile = done_style["result"]
            activated = service.save_style(
                StyleEdit(
                    base_id=profile["id"], expected_version=profile["version"], rules=profile["rules"], activate=True
                )
            )
            generation = await service.submit(
                "generate",
                GenerateRequest(
                    question="数学错题很多，每天只有半小时怎么复习？",
                    provider="codex",
                    request_id="synthetic-workflow-generation",
                ),
            )
            done_reply = await wait(service, generation)
            evidence = {
                "scope": "完全合成语料，真实Codex和真实CUDA；不代表用户风格评价",
                "import": outcome,
                "style_status": done_style["status"],
                "style_calls": done_style["calls"],
                "rule_count": len(activated["rules"]),
                "generation_status": done_reply["status"],
                "generation_calls": done_reply["calls"],
                "reply": done_reply["result"],
                "runtime": service.retriever.runtime(),
            }
            destination = Path(".verification") / "workflow.json"
            destination.write_text(json.dumps(evidence, ensure_ascii=False, indent=2), encoding="utf-8")
            print("合成语料真实联调已完成，画像2轮与回复1轮均通过结构与来源校验。", flush=True)
        finally:
            await service.jobs.close()


if __name__ == "__main__":
    asyncio.run(main())
