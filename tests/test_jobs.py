import asyncio

from app.db import Store
from app.jobs import Jobs


async def test_queue_running_cancellation_and_no_late_result(tmp_path):
    store = Store(tmp_path / "jobs.db")
    entered = asyncio.Event()
    second_entered = asyncio.Event()
    calls = []

    async def handler(job, key):
        calls.append(job["id"])
        if job["input"]["case"] == 1:
            entered.set()
            try:
                await asyncio.sleep(30)
            except asyncio.CancelledError:
                return {"late": "must-not-be-stored"}
        second_entered.set()
        return {"done": True}

    jobs = Jobs(store, handler)
    await jobs.start()
    first = await jobs.submit("test", {"case": 1}, "first-request", "synthetic-key")
    await asyncio.wait_for(entered.wait(), 2)
    queued = await jobs.submit("test", {"case": 2}, "second-request")
    await jobs.cancel(queued["id"])
    await jobs.cancel(first["id"])
    await asyncio.wait_for(jobs.queue.join(), 2)
    assert len(calls) == 1
    assert not second_entered.is_set()
    assert store.get("jobs", first["id"])["result"] is None
    assert store.get("jobs", first["id"])["status"] == "cancelled"
    assert not jobs.keys
    await jobs.close()


async def test_restart_marks_running_and_queued_interrupted(tmp_path):
    store = Store(tmp_path / "jobs.db")
    store.put("jobs", {"id": "first", "status": "running"})
    store.put("jobs", {"id": "second", "status": "queued"})

    async def handler(job, key):
        raise AssertionError("重启不得自动模型调用")

    jobs = Jobs(store, handler)
    await jobs.start()
    assert all(j["status"] == "interrupted" for j in store.all("jobs"))
    assert jobs.queue.empty()
    await jobs.close()
