"""生成本地留出集，或在用户主动执行时运行三方案对比。"""

from __future__ import annotations

import argparse
import json
import time

import httpx

from app.config import DATA
from app.service import Service
from app.utils import digest, dumps, now, uid

EXTRA = [
    ("条件变化", "初一数学基础很薄弱，是否适合立即大量刷高三模拟题？"),
    ("条件变化", "只有每天二十分钟复习时间，怎么选择先复习的内容？"),
    ("条件变化", "英语单词基本认识，但长难句读不懂，怎样调整练习？"),
    ("条件变化", "已经能理解课堂内容，但考试时间总是不够，怎么练习？"),
    ("新主题", "第一次给阳台的薄荷换盆，怎样安排操作步骤？"),
    ("新主题", "想开始练习手机摄影，怎么安排一个星期的练习？"),
    ("新主题", "怎么向朋友解释为什么要做文件备份？"),
    ("新主题", "阅读一个陌生领域的入门书，怎样做笔记更有效？"),
    ("情绪回应", "我很努力，但最近几次考试都没进步，心里很难受。"),
    ("澄清追问", "我学不会，应该怎么办？"),
    ("感谢祝福", "谢谢你的建议，我今天终于完成了第一步！"),
    ("短互动", "收到，我先去试试看。"),
]


def source_keys(example):
    return {f"{m.get('note_id')}:{m.get('comment_id')}" for m in example.get("original_messages", [])}


def prepare():
    service = Service(DATA)
    if any(j["status"] in {"queued", "running"} for j in service.store.all("jobs")):
        raise SystemExit("请先等待或取消全部任务")
    examples = service.examples()
    candidates = sorted(
        (e for e in examples if e["status"] == "ready" and e["question_text"].strip()),
        key=lambda e: digest(e["source_thread_key"]),
    )
    chosen, threads, refs = [], set(), set()
    for example in candidates:
        keys = source_keys(example)
        if example["source_thread_key"] in threads or keys & refs:
            continue
        chosen.append(example)
        threads.add(example["source_thread_key"])
        refs |= keys
        if len(chosen) == 8:
            break
    if len(chosen) < 8:
        raise SystemExit("至少需要8个互不重叠的已整理问答会话，先整理更多资料")
    # 完整来源会话的所有重叠与近重复材料一起隔离，避免从前文带回答案。
    held = set()
    held_answers = {e["answer_text"].strip() for e in chosen}
    while True:
        previous = len(held)
        for e in examples:
            if (
                e["id"] in held
                or e["source_thread_key"] in threads
                or source_keys(e) & refs
                or e["answer_text"].strip() in held_answers
            ):
                held.add(e["id"])
                threads.add(e["source_thread_key"])
                refs |= source_keys(e)
        if previous == len(held):
            break
    backup = service.backup()
    with service.store.transaction() as db:
        for e in examples:
            if e["id"] in held:
                e.update(evaluation_holdout=True, content_enabled=False, style_enabled=False)
                service.store.put("examples", e, db)
        for p in service.store.all("style_profiles", db):
            if set(p["sample_ids"]) & held:
                p["active"] = False
                service.store.put("style_profiles", p, db)
    questions = [
        {
            "id": f"history-{i + 1}",
            "category": "留出会话",
            "question": e["question_text"],
            "background": e["context_text"],
            "held_source_id": e["id"],
        }
        for i, e in enumerate(chosen)
    ]
    questions += [
        {"id": f"new-{i + 1}", "category": kind, "question": text, "background": ""}
        for i, (kind, text) in enumerate(EXTRA)
    ]
    result = {
        "schema_version": 1,
        "id": uid(),
        "prepared_at": now(),
        "questions": questions,
        "held_example_ids": sorted(held),
        "backup": backup,
        "library_hash": digest(service.examples()),
        "human_ratings": "等待用户评价",
    }
    folder = DATA / "evaluation"
    folder.mkdir(exist_ok=True)
    path = folder / "study.json"
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print("已建立20题留出集，相关画像已停用；请用剩余示例重新提取并启用画像。")
    print(path)


def run(provider, base_url):
    path = DATA / "evaluation" / "study.json"
    study = json.loads(path.read_text(encoding="utf-8"))
    service = Service(DATA)
    if digest(service.examples()) != study["library_hash"]:
        raise SystemExit("资料已变化，请重新固定评测集，不采用变化前后的混合结果")
    if not service.styles()["active"]:
        raise SystemExit("请先从非留出材料提取并启用画像")
    results = []
    headers = {"X-Requested-With": "reply-assistant"}
    with httpx.Client(base_url=base_url, headers=headers, timeout=30) as client:
        for question in study["questions"]:
            for mode in ("base", "style", "full"):
                body = {
                    "question": question["question"],
                    "background": question["background"],
                    "length": "自然",
                    "provider": provider,
                    "mode": mode,
                    "request_id": digest({"study": study["id"], "case": question["id"], "mode": mode}),
                }
                response = client.post("/api/generations", json=body)
                response.raise_for_status()
                job = response.json()
                deadline = time.monotonic() + 1000
                while job["status"] in {"queued", "running"} and time.monotonic() < deadline:
                    time.sleep(1)
                    job = client.get("/api/jobs/" + job["id"]).json()
                results.append(
                    {
                        "case_id": question["id"],
                        "mode": mode,
                        "job_id": job["id"],
                        "status": job["status"],
                        "result": job.get("result"),
                        "error": job.get("error"),
                        "tone_score": None,
                        "applicability_score": None,
                        "adoptability_score": None,
                        "preferred": None,
                        "final_text": None,
                        "edit_seconds": None,
                    }
                )
                (path.parent / "results.json").write_text(
                    dumps({"study": study["id"], "results": results}), encoding="utf-8"
                )
                print(f"{question['id']} / {mode}：{job['status']}", flush=True)
                if job["status"] != "succeeded":
                    raise SystemExit("已保存部分结果，请修复失败原因后主动重试")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="留出会话并运行由用户评分的20题三方案评测")
    parser.add_argument("action", choices=["prepare", "run"])
    parser.add_argument("--provider", choices=["codex", "deepseek"], default="codex")
    parser.add_argument("--url", default="http://127.0.0.1:8081")
    args = parser.parse_args()
    prepare() if args.action == "prepare" else run(args.provider, args.url)
