import io
import json
import os
import sqlite3
import zipfile
from pathlib import Path

from conftest import import_sample, sample_source


def test_restore_without_rowid_rejected_before_live_replacement(client, tmp_path):
    import_sample(client)
    before = client.service.store.path.read_bytes()
    archive = client.post("/api/backups", json={}).json()
    raw = client.get(archive["url"]).content
    staged = tmp_path / "bad.db"
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        staged.write_bytes(z.read("reply.sqlite3"))
    connection = sqlite3.connect(staged)
    connection.execute("DROP TABLE jobs")
    connection.execute("CREATE TABLE jobs(id TEXT PRIMARY KEY,body TEXT NOT NULL) WITHOUT ROWID")
    connection.commit()
    connection.close()
    malicious = io.BytesIO()
    with zipfile.ZipFile(io.BytesIO(raw)) as source, zipfile.ZipFile(malicious, "w") as output:
        for name in source.namelist():
            output.writestr(name, staged.read_bytes() if name == "reply.sqlite3" else source.read(name))
    response = client.post("/api/restores", files={"file": ("bad.zip", malicious.getvalue())})
    assert response.status_code == 400
    assert client.service.store.path.read_bytes() == before
    assert client.get("/api/jobs").status_code == 200


def test_source_without_ui_message_id_can_be_manually_resolved_without_changing_raw(client):
    source = json.loads(sample_source(kind="unresolved"))
    for message in source["groups"][0]["messages"]:
        del message["id"]
    preview = client.post("/api/imports/preview", files={"file": ("sample.json", json.dumps(source).encode())}).json()
    client.post(f"/api/imports/{preview['id']}/confirm", json={})
    original = client.get("/api/examples").json()["items"][0]
    response = client.post(
        f"/api/examples/{original['id']}/resolve",
        json={
            "expected_version": 1,
            "units": [{"question_ids": ["message-0"], "context_ids": [], "answer_ids": ["message-1"]}],
        },
    )
    assert response.status_code == 200, response.text
    assert response.json()["items"][0]["original_messages"] == original["original_messages"]


def test_resolve_does_not_overwrite_edited_derived_answer(client):
    import_sample(client, kind="unresolved")
    original = client.get("/api/examples").json()["items"][0]
    payload = {"expected_version": 1, "units": [{"question_ids": ["q1"], "context_ids": [], "answer_ids": ["a1"]}]}
    derived = client.post(f"/api/examples/{original['id']}/resolve", json=payload).json()["items"][0]
    edited = client.patch(
        f"/api/examples/{derived['id']}",
        json={
            "expected_version": derived["version"],
            "answer_text": "人工稿必须保留",
            "content_enabled": False,
            "style_enabled": False,
        },
    ).json()
    old = client.get(f"/api/examples/{original['id']}").json()
    response = client.post(
        f"/api/examples/{original['id']}/resolve", json={**payload, "expected_version": old["version"]}
    )
    assert response.status_code == 409
    assert client.get(f"/api/examples/{derived['id']}").json() == edited


def test_restore_failure_after_replace_automatically_rolls_back(client, monkeypatch):
    import_sample(client)
    original = client.get("/api/examples").json()["items"][0]
    archive = client.post("/api/backups", json={}).json()
    raw = client.get(archive["url"]).content
    edited = client.patch(
        f"/api/examples/{original['id']}", json={"expected_version": original["version"], "answer_text": "故障前人工稿"}
    ).json()

    def fail_after_replace(*args, **kwargs):
        raise RuntimeError("合成恢复阶段故障")

    monkeypatch.setattr("app.service.Retriever", fail_after_replace)
    response = client.post("/api/restores", files={"file": ("backup.zip", raw)})
    assert response.status_code == 500
    assert response.json()["code"] == "restore_failed"
    assert client.get(f"/api/examples/{original['id']}").json() == edited


def test_restore_works_when_system_temp_cannot_be_atomically_moved_to_data(client, monkeypatch):
    import_sample(client)
    original = client.get("/api/examples").json()["items"][0]
    archive = client.post("/api/backups", json={}).json()
    raw = client.get(archive["url"]).content
    client.patch(f"/api/examples/{original['id']}", json={"expected_version": 1, "answer_text": "备份后的修改"})
    native_replace = os.replace

    def same_data_volume(source, destination):
        if not Path(source).resolve().is_relative_to(client.service.data.resolve()):
            raise OSError(17, "合成Windows跨卷原子移动失败")
        return native_replace(source, destination)

    monkeypatch.setattr("app.service.os.replace", same_data_volume)
    response = client.post("/api/restores", files={"file": ("backup.zip", raw)})
    assert response.status_code == 200, response.text
    assert client.get(f"/api/examples/{original['id']}").json()["answer_text"] == original["answer_text"]
