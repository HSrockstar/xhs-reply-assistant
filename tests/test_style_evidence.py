import json

import pytest
from conftest import finished, import_sample

from app.schemas import Expression, ExpressionSelection, Strategy, StrategySelection, output_schema, selection_schema
from app.style import attach_answer_evidence, validate_observations
from app.utils import AppError


def id_only_provider(client, unknown=False):
    original = client.provider.request

    async def request(system, prompt, schema):
        # 模型只选择依据 ID；模拟上下文中同时存在另一段目标作者回答。
        payload = json.loads(prompt)
        stored_schema = Strategy if "Strategy" in schema.__name__ else Expression
        raw, usage = await original(system, prompt, stored_schema)
        definitions = output_schema(schema).get("$defs", {}).values()
        assert not any("quote" in node.get("properties", {}) for node in definitions)
        fields = ("observations", "content_candidates", "contradictions", "dimensions")
        for field in fields:
            for observation in raw.get(field, []):
                observation.pop("evidence")
                if unknown:
                    observation["evidence_ids"] = ["not-in-this-batch"]
        assert payload["samples"][0]["context_text"]
        return raw, usage

    client.provider.request = request


def prepare_context(client):
    import_sample(client)
    example = client.service.examples()[0]
    client.service.store.update("examples", example["id"], {"context_text": "上下文中的旧回答不能作为当前回答依据。"})
    return example


def test_two_pass_evidence_is_loaded_from_answer_not_copied_by_model(client):
    example = prepare_context(client)
    id_only_provider(client)
    job = client.post("/api/style/extract", json={"request_id": "source-evidence-extract"}).json()
    done = finished(client, job)
    assert done["status"] == "succeeded", done.get("error")
    assert len(done["calls"]) == 2
    profile = done["result"]
    for observation in profile["strategy"]["observations"] + profile["expression"]["dimensions"]:
        assert observation["evidence"] == [{"example_id": example["id"], "quote": example["answer_text"]}]


def test_unknown_model_evidence_id_does_not_create_profile(client):
    prepare_context(client)
    id_only_provider(client, unknown=True)
    done = finished(client, client.post("/api/style/extract", json={"request_id": "unknown-evidence-extract"}).json())
    assert done["status"] == "failed"
    assert done["error"]["code"] == "invalid_output"
    assert client.service.store.all("style_profiles") == []


def test_context_quote_remains_invalid_for_stored_evidence():
    sample = {"id": "example-1", "answer_text": "当前回答。", "context_text": "上下文中的旧回答。"}
    observation = {
        "evidence_ids": [sample["id"]],
        "evidence": [{"example_id": sample["id"], "quote": sample["context_text"]}],
    }
    with pytest.raises(AppError, match="风格摘录与目标作者原文不一致"):
        validate_observations([observation], [sample])


@pytest.mark.parametrize("schema", [StrategySelection, ExpressionSelection])
def test_output_schema_limits_all_evidence_to_current_batch(schema):
    ids = ["example-1", "example-2"]
    constrained = output_schema(selection_schema(schema, ids))
    for definition in constrained["$defs"].values():
        properties = definition.get("properties", {})
        for field in ("evidence_ids", "counterexample_ids"):
            if field in properties:
                assert properties[field]["items"]["enum"] == ids


def test_backend_also_rejects_cross_batch_id():
    data = {"dimensions": [{"evidence_ids": ["other-batch"]}]}
    with pytest.raises(AppError, match="本批未提供"):
        attach_answer_evidence(data, [{"id": "this-batch", "answer_text": "回答。"}], ("dimensions",))
