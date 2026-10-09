from __future__ import annotations

import asyncio
import io
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated
from urllib.parse import urlsplit

from fastapi import Body, FastAPI, File, Query, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import ValidationError
from starlette.middleware.trustedhost import TrustedHostMiddleware

from app.config import DATA, ROOT
from app.schemas import ExamplePatch, FeedbackRequest, GenerateRequest, ModelRequest, ResolveRequest, StyleEdit
from app.service import Service
from app.utils import AppError, uid


def create_app(data: Path = DATA, provider_factory=None, retriever=None):
    @asynccontextmanager
    async def lifespan(app):
        app.state.service = Service(data, provider_factory, retriever)
        await app.state.service.jobs.start()
        yield
        await app.state.service.jobs.close()

    app = FastAPI(title="个人回复助手", lifespan=lifespan)
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=["localhost", "127.0.0.1", "testserver"])

    @app.middleware("http")
    async def local_boundary(request: Request, call_next):
        if request.method in {"POST", "PATCH", "PUT", "DELETE"}:
            if request.headers.get("x-requested-with") != "reply-assistant":
                return JSONResponse(
                    {"code": "same_origin_required", "message": "写请求须由本地页面发起", "retryable": False},
                    status_code=403,
                )
            origin = request.headers.get("origin")
            expected = urlsplit(str(request.base_url))
            supplied = urlsplit(origin) if origin else None
            if supplied and (supplied.scheme != expected.scheme or supplied.netloc != expected.netloc):
                return JSONResponse(
                    {"code": "origin_rejected", "message": "拒绝非同源写请求", "retryable": False}, status_code=403
                )
            size = request.headers.get("content-length", "")
            cap = 101 * 1024 * 1024 if request.url.path == "/api/restores" else 11 * 1024 * 1024
            if not size.isdigit() or int(size) > cap:
                return JSONResponse(
                    {"code": "upload_too_large", "message": "请求缺少长度或体积超限", "retryable": False},
                    status_code=413,
                )
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "same-origin"
        response.headers["Cache-Control"] = "no-store"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; "
            "connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'"
        )
        return response

    @app.exception_handler(AppError)
    async def app_error(_, exc):
        return JSONResponse(
            {"code": exc.code, "message": exc.message, "retryable": exc.retryable}, status_code=exc.status
        )

    @app.exception_handler(RequestValidationError)
    async def invalid_input(_, exc):
        return JSONResponse(
            {"code": "invalid_input", "message": "请求字段、长度或类型不正确", "retryable": False}, status_code=422
        )

    @app.exception_handler(ValidationError)
    async def internal_validation(_, exc):
        return JSONResponse(
            {"code": "invalid_input", "message": "请求字段、长度或类型不正确", "retryable": False}, status_code=422
        )

    def service(request):
        return request.app.state.service

    app.mount("/static", StaticFiles(directory=ROOT / "app" / "static"), name="static")

    @app.get("/", response_class=HTMLResponse)
    async def index():
        return (ROOT / "app" / "templates" / "index.html").read_text(encoding="utf-8")

    @app.get("/api/settings")
    async def settings(request: Request):
        return service(request).settings()

    @app.patch("/api/settings")
    async def patch_settings(request: Request, body: Annotated[dict, Body()]):
        return service(request).save_settings(body)

    @app.get("/api/providers")
    async def providers(request: Request):
        return await service(request).providers()

    @app.post("/api/providers/{name}/test")
    async def provider_test(name: str, body: ModelRequest, request: Request):
        if name not in {"codex", "deepseek"}:
            raise AppError("不支持的模型入口")
        body.provider = name
        return await service(request).submit("test", body)

    @app.post("/api/imports/preview")
    async def preview(request: Request, file: Annotated[UploadFile, File()]):
        raw = await file.read(10 * 1024 * 1024 + 1)
        try:
            return service(request).preview_import(raw)
        except ValueError as exc:
            if isinstance(exc, AppError):
                raise
            raise AppError(str(exc)) from None
        finally:
            await file.close()

    @app.get("/api/imports")
    async def imports(request: Request):
        return {"items": service(request).store.all("imports")}

    @app.post("/api/imports/{import_id}/confirm")
    async def confirm(import_id: str, request: Request):
        return service(request).confirm_import(import_id)

    @app.get("/api/examples")
    async def examples(
        request: Request,
        status: str = "",
        offset: int = Query(default=0, ge=0),
        limit: int = Query(default=50, ge=1, le=200),
    ):
        items = service(request).examples()
        if status:
            items = [e for e in items if e["status"] == status]
        return {"items": items[offset : offset + limit], "total": len(items)}

    @app.get("/api/examples/{example_id}")
    async def example(example_id: str, request: Request):
        return service(request).store.get("examples", example_id)

    @app.patch("/api/examples/{example_id}")
    async def patch_example(example_id: str, body: ExamplePatch, request: Request):
        return service(request).edit_example(example_id, body)

    @app.post("/api/examples/{example_id}/organize")
    async def organize(example_id: str, body: ModelRequest, request: Request):
        s = service(request)
        example = s.store.get("examples", example_id)
        return await s.submit("organize", body, {"example": example})

    @app.post("/api/examples/{example_id}/resolve")
    async def resolve(example_id: str, body: ResolveRequest, request: Request):
        return service(request).resolve(example_id, body)

    @app.get("/api/style")
    async def styles(request: Request):
        return service(request).styles()

    @app.post("/api/style/extract")
    async def extract(body: ModelRequest, request: Request):
        return await service(request).submit("style", body)

    @app.post("/api/style/jobs/{job_id}/resume")
    async def resume(job_id: str, body: ModelRequest, request: Request):
        return await service(request).resume_style(job_id, body)

    @app.post("/api/style/versions")
    async def save_style(body: StyleEdit, request: Request):
        return service(request).save_style(body)

    @app.get("/api/style/versions/{profile_id}/export")
    async def style_export(profile_id: str, request: Request):
        raw = service(request).style_export(profile_id)
        return StreamingResponse(
            io.BytesIO(raw),
            media_type="application/zip",
            headers={"Content-Disposition": f'attachment; filename="style-{profile_id}.zip"'},
        )

    @app.get("/api/retrieval/runtime")
    async def runtime(request: Request):
        return await asyncio.to_thread(service(request).retriever.runtime)

    @app.post("/api/retrieval/rebuild")
    async def rebuild(request: Request, body: Annotated[dict | None, Body()] = None):
        body = body or {}
        return await service(request).submit("index", ModelRequest(request_id=body.get("request_id") or uid()))

    @app.post("/api/retrieval/preview")
    async def retrieve(request: Request, body: Annotated[dict, Body()]):
        req = GenerateRequest(request_id=uid(), **body)
        s = service(request)
        return await asyncio.to_thread(
            s.retriever.search, req.question, req.background, s.examples(), req.scene, req.length
        )

    @app.post("/api/generations")
    async def generate(body: GenerateRequest, request: Request):
        return await service(request).submit("generate", body)

    @app.get("/api/jobs")
    async def jobs(request: Request, limit: int = Query(default=30, ge=1, le=100)):
        return {"items": list(reversed(service(request).store.all("jobs")))[0:limit]}

    @app.get("/api/jobs/{job_id}")
    async def job(job_id: str, request: Request):
        return service(request).store.get("jobs", job_id)

    @app.post("/api/jobs/{job_id}/cancel")
    async def cancel(job_id: str, request: Request):
        return await service(request).jobs.cancel(job_id)

    @app.post("/api/feedback")
    async def feedback(body: FeedbackRequest, request: Request):
        return service(request).feedback(body)

    @app.get("/api/feedback")
    async def feedback_list(request: Request):
        return {"items": list(reversed(service(request).store.all("feedback")))}

    @app.post("/api/feedback/{feedback_id}/adopt")
    async def adopt(feedback_id: str, request: Request):
        return service(request).adopt(feedback_id)

    @app.get("/api/exports/finals")
    async def finals(request: Request):
        text = "# 已保存的最终稿\n\n"
        for f in service(request).store.all("feedback"):
            j = service(request).store.get("jobs", f["job_id"])
            text += f"## {f['created_at']}\n\n问题：{j['input']['question']}\n\n{f['final_text']}\n\n"
        return StreamingResponse(
            io.BytesIO(text.encode("utf-8")),
            media_type="text/markdown; charset=utf-8",
            headers={"Content-Disposition": 'attachment; filename="finals.md"'},
        )

    @app.post("/api/backups")
    async def backup(request: Request):
        return await asyncio.to_thread(service(request).backup)

    @app.get("/api/backups/{backup_id}/download")
    async def download_backup(backup_id: str, request: Request):
        if not all(c in "0123456789abcdef-" for c in backup_id) or len(backup_id) != 36:
            raise AppError("备份不存在", "not_found", 404)
        path = service(request).data / "exports" / (backup_id + ".zip")
        if not path.is_file():
            raise AppError("备份不存在", "not_found", 404)
        return FileResponse(path, filename="reply-backup-" + backup_id + ".zip")

    @app.post("/api/restores")
    async def restore(request: Request, file: Annotated[UploadFile, File()]):
        raw = await file.read(100 * 1024 * 1024 + 1)
        try:
            return service(request).restore(raw)
        finally:
            await file.close()

    return app


app = create_app()
