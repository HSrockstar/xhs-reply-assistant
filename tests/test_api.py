import io
import json
import zipfile

from conftest import finished, import_sample, sample_source


def test_import_preview_confirmation_duplicate_and_conflict(client):
    preview, outcome = import_sample(client)
    assert preview["stats"]["target_comments"] == 1
    assert outcome["imported"] == 1
    repeated = client.post(f"/api/imports/{preview['id']}/confirm", json={}).json()
    assert repeated["duplicate"]
    _, crossed = import_sample(client, collection="collection-2")
    assert crossed["duplicates"] == 1
    _, conflict = import_sample(client, answer="新的正文版本。", collection="collection-3")
    assert conflict["conflicts"] == 1
    items = client.get("/api/examples").json()["items"]
    assert len(items) == 2
    assert sum(e["style_enabled"] for e in items) == 1
    assert next(e for e in items if e["status"] == "ready")["answer_text"].startswith("先整理")


def test_original_text_preserved_and_optimistic_version(client):
    import_sample(client)
    example = client.get("/api/examples").json()["items"][0]
    changed = client.patch(
        "/api/examples/" + example["id"],
        json={"expected_version": 1, "answer_text": "人工编辑稿。", "style_enabled": False},
    )
    assert changed.status_code == 200
    assert changed.json()["original_messages"] == example["original_messages"]
    assert changed.json()["version"] == 2
    assert (
        client.patch(
            "/api/examples/" + example["id"], json={"expected_version": 1, "answer_text": "迟到稿。"}
        ).status_code
        == 409
    )


def test_source_snapshot_hash_checked(client):
    preview = client.post("/api/imports/preview", files={"file": ("data.json", sample_source())}).json()
    path = client.service.data / "imports" / preview["filename"]
    path.write_bytes(b"{}")
    assert client.post(f"/api/imports/{preview['id']}/confirm", json={}).status_code == 409
    assert not client.get("/api/examples").json()["items"]


def test_two_pass_style_snapshot_activation_and_resume(client):
    import_sample(client)
    client.provider.fail_expression = True
    first = client.post("/api/style/extract", json={"request_id": "style-request-1"}).json()
    failed = finished(client, first)
    assert failed["status"] == "failed"
    assert failed["checkpoint"]["strategy"]
    assert client.get("/api/style").json()["active"] is None
    client.provider.fail_expression = False
    resumed = client.post(f"/api/style/jobs/{first['id']}/resume", json={"request_id": "style-request-2"}).json()
    done = finished(client, resumed)
    assert done["status"] == "succeeded", done
    calls = client.provider.calls
    assert [c[2] for c in calls] == ["StrategySelection", "ExpressionSelection", "ExpressionSelection"]
    assert calls[0][1]["samples"] == calls[2][1]["samples"]
    assert calls[2][1]["strategy"] == failed["checkpoint"]["strategy"]
    profile = done["result"]
    activated = client.post(
        "/api/style/versions",
        json={
            "base_id": profile["id"],
            "expected_version": profile["version"],
            "rules": profile["rules"],
            "activate": True,
        },
    )
    assert activated.status_code == 200, activated.text
    assert client.get("/api/style").json()["active"]["id"] == activated.json()["id"]
    exported = client.get(f"/api/style/versions/{profile['id']}/export")
    with zipfile.ZipFile(io.BytesIO(exported.content)) as z:
        assert "analysis/reply-strategy.md" in z.namelist()
        assert "prompt.txt" in z.namelist()


def test_stopped_style_evidence_not_silently_used(client):
    import_sample(client)
    job = client.post("/api/style/extract", json={"request_id": "style-request-1"}).json()
    profile = finished(client, job)["result"]
    client.post(
        "/api/style/versions",
        json={
            "base_id": profile["id"],
            "expected_version": profile["version"],
            "rules": profile["rules"],
            "activate": True,
        },
    )
    example = client.get("/api/examples").json()["items"][0]
    client.patch(f"/api/examples/{example['id']}", json={"expected_version": 1, "style_enabled": False})
    assert client.get("/api/style").json()["stale"]
    assert not client.service.effective_profile()["rules"][0]["enabled"]


def test_generate_dedupe_reference_mapping_feedback_and_adoption(client):
    import_sample(client)
    body = {"request_id": "generate-request-1", "question": "数学怎么复习？"}
    job = client.post("/api/generations", json=body).json()
    again = client.post("/api/generations", json=body).json()
    assert again["id"] == job["id"]
    completed = finished(client, job)
    assert completed["status"] == "succeeded", completed
    assert (
        completed["result"]["reference_usage"][0]["example_id"] == completed["result"]["references"]["content"][0]["id"]
    )
    assert client.post("/api/generations", json={**body, "question": "新问题"}).status_code == 409
    feedback = client.post(
        "/api/feedback",
        json={"job_id": job["id"], "final_text": "我认可的最终稿。", "request_id": "feedback-request-1"},
    ).json()
    assert len(client.get("/api/examples").json()["items"]) == 1
    adopted = client.post(f"/api/feedback/{feedback['id']}/adopt", json={}).json()
    repeated = client.post(f"/api/feedback/{feedback['id']}/adopt", json={}).json()
    assert adopted["id"] == repeated["id"]
    assert adopted["origin"] == "user_approved"
    assert len(client.get("/api/examples").json()["items"]) == 2
    assert "我认可的最终稿" in client.get("/api/exports/finals").text


def test_fake_reference_rejected(client):
    client.provider.fake_reference = True
    job = client.post("/api/generations", json={"question": "新的问题？", "request_id": "generate-invalid"}).json()
    done = finished(client, job)
    assert done["status"] == "failed"
    assert done["error"]["code"] == "invalid_reference"
    assert done["result"] is None


def test_model_organization_requires_manual_confirmation(client):
    import_sample(client, kind="unresolved")
    example = client.get("/api/examples").json()["items"][0]
    assert not example["style_enabled"]
    job = client.post(f"/api/examples/{example['id']}/organize", json={"request_id": "organize-request-1"}).json()
    done = finished(client, job)
    assert done["status"] == "succeeded", done
    assert done["result"]["confirmation_required"]
    assert not client.get(f"/api/examples/{example['id']}").json()["style_enabled"]
    response = client.post(
        f"/api/examples/{example['id']}/resolve", json={"expected_version": 1, "units": done["result"]["units"]}
    )
    assert response.status_code == 200, response.text
    assert response.json()["items"][0]["style_enabled"]


def test_backup_restore_consistent_and_zip_slip_rejected(client):
    import_sample(client)
    saved = client.post("/api/backups", json={}).json()
    raw = client.get(saved["url"]).content
    example = client.get("/api/examples").json()["items"][0]
    client.patch(f"/api/examples/{example['id']}", json={"expected_version": 1, "answer_text": "后来编辑稿"})
    restored = client.post("/api/restores", files={"file": ("backup.zip", raw)})
    assert restored.status_code == 200, restored.text
    assert restored.json()["backup"]["url"]
    assert client.get(f"/api/examples/{example['id']}").json()["answer_text"] == example["answer_text"]
    payload = io.BytesIO()
    with zipfile.ZipFile(payload, "w") as z:
        z.writestr("../../evil", "danger")
        z.writestr("manifest.json", json.dumps({"schema_version": 1}))
    assert client.post("/api/restores", files={"file": ("bad.zip", payload.getvalue())}).status_code == 400


def test_same_origin_and_secret_redaction(client):
    blocked = client.post("/api/settings", headers={"Origin": "https://evil.example"}, json={})
    assert blocked.status_code == 403
    secret = "synthetic-secret-not-a-real-key"
    invalid = client.post("/api/generations", json={"api_key": secret, "question": "", "request_id": "invalid-req"})
    assert secret not in invalid.text
    assert invalid.status_code == 422
    job = client.post("/api/providers/deepseek/test", json={"api_key": secret, "request_id": "provider-request"}).json()
    finished(client, job)
    assert secret not in client.get("/api/jobs").text
    assert secret.encode() not in client.service.store.path.read_bytes()


def test_defaults_and_no_implicit_provider_switch(client):
    settings = client.get("/api/settings").json()
    assert settings["provider"] == "codex"
    assert settings["embedding_device"] == "cuda:0"
    assert client.patch("/api/settings", json={"provider": "deepseek"}).status_code == 200
    assert client.patch("/api/settings", json={"restore_defaults": True}).json()["provider"] == "codex"
    assert client.patch("/api/settings", json={"embedding_device": "cpu"}).status_code == 400
    failed = finished(client, client.post("/api/retrieval/rebuild", json={}).json())
    assert failed["status"] == "failed"
    assert failed["error"]["code"] == "gpu_unavailable"
