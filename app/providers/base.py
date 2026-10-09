from __future__ import annotations

import time

from pydantic import BaseModel, ValidationError

from app.utils import AppError


class ProviderError(AppError):
    def __init__(self, message, code="provider_failed", retryable=True):
        super().__init__(message, code, 503, retryable)


class Provider:
    async def complete(self, system: str, prompt: str, schema: type[BaseModel]) -> dict:
        start = time.perf_counter()
        raw, usage = await self.request(system, prompt, schema)
        try:
            data = schema.model_validate(raw).model_dump()
        except ValidationError:
            raise ProviderError("模型输出未通过结构校验，请主动重试", "invalid_output") from None
        return {
            "provider": self.name,
            "model": self.model,
            "data": data,
            "usage": usage,
            "duration_ms": round((time.perf_counter() - start) * 1000),
        }
