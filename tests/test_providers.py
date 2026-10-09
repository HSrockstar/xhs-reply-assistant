import asyncio
import sys

import httpx
import pytest

from app.config import DEFAULTS
from app.providers import codex, deepseek
from app.providers.base import ProviderError
from app.schemas import Reply

VALID = {
    "reply_text": "已收到问题。",
    "reference_usage": [],
    "supplement_summary": "",
    "clarification_needed": False,
    "review_notes": [],
}


@pytest.mark.parametrize(
    "status,finish,content,expected",
    [
        (401, "stop", VALID, "credentials_invalid"),
        (429, "stop", VALID, "rate_limit"),
        (200, "length", VALID, "truncated"),
        (200, "stop", "invalid-json", "invalid_output"),
    ],
)
async def test_deepseek_error_no_secret_echo(monkeypatch, status, finish, content, expected):
    import json

    key = "synthetic-credential"

    def handler(request):
        assert request.headers["authorization"] == "Bearer " + key
        return httpx.Response(
            status,
            json={
                "choices": [
                    {
                        "finish_reason": finish,
                        "message": {"content": content if isinstance(content, str) else json.dumps(content)},
                    }
                ]
            },
        )

    native = httpx.AsyncClient
    monkeypatch.setattr(
        deepseek.httpx, "AsyncClient", lambda **kwargs: native(**kwargs, transport=httpx.MockTransport(handler))
    )
    with pytest.raises(ProviderError) as raised:
        await deepseek.DeepSeek(DEFAULTS, key).complete("system", "prompt", Reply)
    assert raised.value.code == expected
    assert key not in str(raised.value)


async def test_deepseek_missing_credentials(monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    with pytest.raises(ProviderError, match="密钥") as error:
        await deepseek.DeepSeek(DEFAULTS).complete("system", "prompt", Reply)
    assert error.value.code == "credentials_missing"


async def test_codex_unicode_stdin_and_isolated_flags(tmp_path, monkeypatch):
    script = tmp_path / "中文 path.py"
    script.write_text(
        """import json,sys,pathlib
args=sys.argv
assert '--ignore-user-config' in args and '--ephemeral' in args and '--ignore-rules' in args
assert args[args.index('--model')+1]=='gpt-6-luna'
assert 'read-only' in args
body=sys.stdin.buffer.read().decode('utf-8')
assert '中文 $() & ` 特殊字符' in body
output=pathlib.Path(args[args.index('--output-last-message')+1])
output.write_text(json.dumps({'reply_text':'收到。','reference_usage':[],'supplement_summary':'',
 'clarification_needed':False,'review_notes':[]},ensure_ascii=False),encoding='utf-8')
""",
        encoding="utf-8",
    )
    monkeypatch.setattr(codex, "command", lambda: [sys.executable, str(script)])
    result = await codex.Codex(DEFAULTS).complete("system", "中文 $() & ` 特殊字符", Reply)
    assert result["data"]["reply_text"] == "收到。"
    assert result["usage"] is None


@pytest.mark.parametrize("action", ["timeout", "cancel"])
async def test_codex_task_process_reclaimed(tmp_path, monkeypatch, action):
    script = tmp_path / "wait.py"
    script.write_text("import time\ntime.sleep(30)\n", encoding="utf-8")
    monkeypatch.setattr(codex, "command", lambda: [sys.executable, str(script)])
    settings = {**DEFAULTS, "codex_timeout_seconds": 0.1 if action == "timeout" else 10}
    task = asyncio.create_task(codex.Codex(settings).complete("system", "question", Reply))
    if action == "cancel":
        await asyncio.sleep(0.1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    else:
        with pytest.raises(ProviderError) as error:
            await task
        assert error.value.code == "timeout"
