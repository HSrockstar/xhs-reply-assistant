import copy

import pytest
from conftest import finished, import_sample

from app import style
from app.utils import AppError


def test_large_corpus_keeps_whole_units_and_does_not_drop_after_first_budget():
    records = [
        {
            "id": str(index),
            "status": "ready",
            "style_enabled": True,
            "source_thread_key": str(index),
            "version": 1,
            "question_text": "问题",
            "context_text": "",
            "answer_text": "回答" * 5000,
        }
        for index in range(4)
    ]
    samples = style.corpus(records)
    batches = style.batches(samples)
    assert len(samples) == 4 and len(batches) == 4
    assert [item for batch in batches for item in batch] == samples
    assert all(len(batch[0]["answer_text"]) == 10000 for batch in batches)
    samples[0]["answer_text"] = "长" * 18001
    with pytest.raises(AppError, match="完整回复单元"):
        style.batches(samples)


def test_multi_batch_second_round_failure_resumes_without_repeating_first_round(client):
    import_sample(client)
    original = client.service.examples()[0]
    for index in range(2):
        example = copy.deepcopy(original)
        example.update(
            id=f"batch-example-{index}", source_thread_key=f"batch-thread-{index}", answer_text="回答" * 5000
        )
        client.service.store.put("examples", example)
    client.provider.fail_expression = True
    job = client.post("/api/style/extract", json={"request_id": "multi-batch-extract"}).json()
    failed = finished(client, job)
    count = len(job["input"]["batches"])
    assert count > 1
    assert failed["status"] == "failed"
    assert len(failed["checkpoint"]["strategy_batches"]) == count
    assert len(failed["calls"]) == count
    client.provider.fail_expression = False
    resumed = client.post(f"/api/style/jobs/{job['id']}/resume", json={"request_id": "multi-batch-resume"}).json()
    result = finished(client, resumed)
    assert result["status"] == "succeeded", result.get("error")
    assert len(result["calls"]) == count
    assert result["result"]["batch_count"] == count
    assert len(result["result"]["sample_ids"]) == 3
    assert len(result["result"]["rules"]) <= 8


def test_deleted_rule_stays_deleted_after_explicit_refresh(client):
    import_sample(client)
    first = finished(client, client.post("/api/style/extract", json={"request_id": "deleted-first"}).json())
    profile = first["result"]
    saved = client.post(
        "/api/style/versions",
        json={"base_id": profile["id"], "expected_version": profile["version"], "rules": [], "activate": True},
    ).json()
    assert saved["deleted_rules"][0]["rule_id"] == "rule1"
    refreshed = finished(client, client.post("/api/style/extract", json={"request_id": "deleted-refresh"}).json())
    assert refreshed["status"] == "succeeded"
    assert refreshed["result"]["rules"] == []
