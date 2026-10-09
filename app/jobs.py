from __future__ import annotations

import asyncio

from app.utils import AppError, digest, now, uid


class Jobs:
    def __init__(self, store, handler):
        self.store, self.handler = store, handler
        self.queue = asyncio.Queue()
        self.keys = {}
        self.running = None
        self.worker = None
        self.submission_lock = asyncio.Lock()

    async def start(self):
        for job in self.store.all("jobs"):
            if job["status"] in {"queued", "running"}:
                self.store.update(
                    "jobs",
                    job["id"],
                    {
                        "status": "interrupted",
                        "stage": "重启中断，请主动重试",
                        "error": {"code": "interrupted", "message": "服务曾中断，未自动调用模型", "retryable": True},
                    },
                )
        self.worker = asyncio.create_task(self._work())

    async def close(self):
        if self.worker:
            self.worker.cancel()
            try:
                await self.worker
            except asyncio.CancelledError:
                pass
        self.keys.clear()

    async def submit(self, kind: str, snapshot: dict, request_id: str, api_key="", checkpoint=None):
        async with self.submission_lock:
            request_hash = digest({"kind": kind, "input": snapshot})
            for old in self.store.all("jobs"):
                if old["request_id"] == request_id:
                    if old["request_hash"] != request_hash:
                        raise AppError("请求标识已用于不同任务", "request_conflict", 409)
                    return old
            if self.queue.qsize() >= 50:
                raise AppError("排队任务过多，请等待已有任务结束", "queue_full", 409)
            job = {
                "id": uid(),
                "kind": kind,
                "request_id": request_id,
                "request_hash": request_hash,
                "input": snapshot,
                "status": "queued",
                "stage": "等待处理",
                "result": None,
                "error": None,
                "created_at": now(),
                "checkpoint": checkpoint or {},
                "calls": [],
            }
            self.store.put("jobs", job)
            if api_key:
                self.keys[job["id"]] = api_key
            self.queue.put_nowait(job["id"])
            return job

    def stage(self, job_id, text, checkpoint=None):
        job = self.store.get("jobs", job_id)
        if job["status"] != "running":
            raise asyncio.CancelledError()
        changes = {"stage": text}
        if checkpoint is not None:
            changes["checkpoint"] = checkpoint
        return self.store.update("jobs", job_id, changes)

    def record_call(self, job_id, result):
        job = self.store.get("jobs", job_id)
        job["calls"].append({k: result[k] for k in ("provider", "model", "duration_ms", "usage")})
        self.store.put("jobs", job)

    async def cancel(self, job_id):
        job = self.store.get("jobs", job_id)
        if job["status"] in {"queued", "running"}:
            self.store.update("jobs", job_id, {"status": "cancelled", "stage": "已取消", "finished_at": now()})
            self.keys.pop(job_id, None)
            if self.running and self.running[0] == job_id:
                task = self.running[1]
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        return self.store.get("jobs", job_id)

    async def _work(self):
        while True:
            job_id = await self.queue.get()
            try:
                job = self.store.get("jobs", job_id)
                if job["status"] != "queued":
                    continue
                self.store.update("jobs", job_id, {"status": "running", "stage": "正在处理", "started_at": now()})
                task = asyncio.create_task(self.handler(self.store.get("jobs", job_id), self.keys.get(job_id, "")))
                self.running = (job_id, task)
                try:
                    result = await task
                    if self.store.get("jobs", job_id)["status"] == "running":
                        self.store.update(
                            "jobs",
                            job_id,
                            {"status": "succeeded", "result": result, "stage": "已完成", "finished_at": now()},
                        )
                except asyncio.CancelledError:
                    # 外层关闭继续传播；单任务取消后继续处理其他排队项。
                    if asyncio.current_task().cancelling():
                        self.store.update("jobs", job_id, {"status": "interrupted", "stage": "服务停止"})
                        raise
                except Exception as exc:
                    if self.store.get("jobs", job_id)["status"] == "running":
                        error = (
                            {"code": exc.code, "message": exc.message, "retryable": exc.retryable}
                            if isinstance(exc, AppError)
                            else {
                                "code": "task_failed",
                                "message": "任务失败，已保留输入和检查点",
                                "retryable": True,
                                "error_type": type(exc).__name__,
                            }
                        )
                        self.store.update(
                            "jobs",
                            job_id,
                            {"status": "failed", "error": error, "stage": "处理失败", "finished_at": now()},
                        )
            finally:
                self.keys.pop(job_id, None)
                self.running = None
                self.queue.task_done()
