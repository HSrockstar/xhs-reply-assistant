from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.getenv("REPLY_DATA_DIR", str(ROOT / "data"))).resolve()
DEFAULTS = {
    "provider": "codex",
    "codex_model": "gpt-6-luna",
    "codex_reasoning_effort": "high",
    "codex_timeout_seconds": 240,
    "deepseek_model": "deepseek-flash",
    "deepseek_timeout_seconds": 180,
    "embedding_model": "BAAI/bge-small-zh-v1.5",
    "embedding_revision": "7999e1d3359715c523056ef9478215996d62a620",
    "embedding_device": "cuda:0",
    "embedding_dtype": "float16",
    "embedding_batch_size": 32,
    "local_model_path": str(ROOT / "data" / "models" / "bge-small-zh-v1.5"),
    "confirmed_background": "",
    "user_preferences": "",
    "automatic_cpu_fallback": False,
    "automatic_provider_fallback": False,
}
