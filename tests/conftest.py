import json
import time

import pytest
from fastapi.testclient import TestClient

from app.main import create_app
from app.providers.base import Provider

HEADERS = {"X-Requested-With": "reply-assistant"}


def sample_source(answer="先整理错题，再按薄弱点练习。", kind="qa", collection="collection-1"):
    def msg(comment_id, content, target, parent=None, time_value=1):
        return {
            "id": comment_id,
            "thread": "note-1:q1",
            "note_id": "note-1",
            "comment_id": comment_id,
            "content": content,
            "target": target,
            "context_only": False,
            "parent_comment_id": parent,
            "root_comment_id": "q1",
            "parent": parent,
            "user_id": "author-1" if target else "other-1",
            "nickname": "合成作者" if target else "合成提问者",
            "platform_time": time_value,
        }

    return json.dumps(
        {
            "id": collection,
            "status": "partial",
            "nickname": "合成作者",
            "user_id": "author-1",
            "groups": [
                {
                    "kind": kind,
                    "warnings": [],
                    "messages": [
                        msg("q1", "数学错题很多，该如何复习？", False),
                        msg("a1", answer, True, "q1", 2),
                    ],
                }
            ],
            "warnings": [],
        },
        ensure_ascii=False,
    ).encode()


class FakeRetriever:
    settings = {}

    def runtime(self):
        return {"available": False, "vector_ready": False, "device": None, "dtype": "float16"}

    def rebuild(self, examples):
        return {"ok": False, "runtime": self.runtime(), "warnings": ["合成GPU不可用"]}

    def search(self, question, background, examples, scene="auto", length="自然"):
        eligible = [e for e in examples if e["status"] == "ready"]
        return {
            "content": [e for e in eligible if e["content_enabled"]][:3],
            "style": [e for e in eligible if e["style_enabled"]][:3],
            "warnings": [],
        }


class FakeProvider(Provider):
    name = "codex"
    model = "gpt-6-luna"

    def __init__(self):
        self.calls = []
        self.fail_expression = False
        self.fake_reference = False

    async def request(self, system, prompt, schema):
        payload = json.loads(prompt)
        self.calls.append((system, payload, schema.__name__))
        if schema.__name__ == "Reply":
            refs = []
            if self.fake_reference:
                refs = [{"example_id": "unknown", "purpose": "content", "note": "伪造"}]
            elif payload.get("references"):
                first = payload["references"][0]
                refs = [{"example_id": first["id"], "purpose": first["purposes"][0], "note": "复习方法参考"}]
            return {
                "reply_text": "先别着急，可以先把错题按原因分组，再安排练习。",
                "reference_usage": refs,
                "supplement_summary": "补充一般建议。",
                "clarification_needed": False,
                "review_notes": [],
            }, None
        if schema.__name__ in {"Strategy", "Expression"}:
            sample = payload["samples"][0]
            observation = {
                "description": "先回应，再提供步骤",
                "applicable_scenes": ["方法解释"],
                "evidence_ids": [sample["id"]],
                "evidence": [{"example_id": sample["id"], "quote": sample["answer_text"]}],
            }
            if schema.__name__ == "Strategy":
                return {
                    "observations": [observation],
                    "content_candidates": [],
                    "contradictions": [],
                    "boundaries": ["不复用个人经历"],
                    "gaps": ["感谢场景不足"],
                }, None
            if self.fail_expression:
                raise ValueError("合成第二轮失败")
            return {
                "rules": [
                    {
                        "rule_id": "rule1",
                        "dimension": "回复结构",
                        "description": "按步骤解释方法。",
                        "applicable_scenes": ["方法解释"],
                        "evidence_ids": [sample["id"]],
                        "counterexample_ids": [],
                        "strength": "场景倾向",
                        "origin": "observed",
                        "enabled": True,
                    }
                ],
                "dimensions": [observation],
                "cross_checks": ["已核对原文"],
                "gaps": [],
            }, None
        if schema.__name__ == "Organization":
            messages = payload["messages"]
            return {
                "units": [
                    {
                        "question_ids": [m["id"] for m in messages if not m["target"]],
                        "context_ids": [],
                        "answer_ids": [m["id"] for m in messages if m["target"]],
                    }
                ]
            }, None
        raise AssertionError(schema.__name__)


@pytest.fixture
def client(tmp_path):
    provider = FakeProvider()
    app = create_app(tmp_path / "data", lambda settings, key: provider, FakeRetriever())
    with TestClient(app, headers=HEADERS) as test:
        test.provider = provider
        test.service = app.state.service
        yield test


def import_sample(client, **kwargs):
    preview = client.post("/api/imports/preview", files={"file": ("sample.json", sample_source(**kwargs))})
    assert preview.status_code == 200, preview.text
    data = preview.json()
    response = client.post(f"/api/imports/{data['id']}/confirm", json={})
    assert response.status_code == 200, response.text
    return data, response.json()


def finished(client, job):
    for _ in range(200):
        current = client.get("/api/jobs/" + job["id"]).json()
        if current["status"] not in {"running", "queued"}:
            return current
        time.sleep(0.01)
    raise AssertionError("合成任务没有完成")
