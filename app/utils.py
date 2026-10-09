from __future__ import annotations

import hashlib
import json
import uuid
from datetime import UTC, datetime


def now() -> str:
    return datetime.now(UTC).isoformat()


def uid() -> str:
    return str(uuid.uuid4())


def dumps(value) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def digest(value) -> str:
    return hashlib.sha256(dumps(value).encode("utf-8")).hexdigest()


class AppError(ValueError):
    def __init__(self, message: str, code="invalid_input", status=400, retryable=False):
        self.message, self.code, self.status, self.retryable = message, code, status, retryable
        super().__init__(message)
