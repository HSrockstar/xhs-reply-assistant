from __future__ import annotations

import asyncio
import json
import os
import shutil
import signal
import tempfile
from pathlib import Path

from app.providers.base import Provider, ProviderError
from app.schemas import output_schema

HIDDEN = 0x08000000 if os.name == "nt" else 0
REQUIRED = ("--ignore-user-config", "--ignore-rules", "--ephemeral", "--output-schema", "--output-last-message")


def command() -> list[str] | None:
    raw = os.getenv("REPLY_CODEX_EXECUTABLE") or shutil.which("codex")
    if not raw:
        return None
    path = Path(raw)
    if path.suffix.lower() in {".cmd", ".ps1"}:
        package = path.parent / "node_modules" / "@openai" / "codex"
        exe = sorted(package.glob("node_modules/@openai/codex-win32-*/vendor/*/bin/codex.exe"))
        if exe:
            return [str(exe[0])]
        node, entry = shutil.which("node"), package / "bin" / "codex.js"
        return [node, str(entry)] if node and entry.exists() else None
    return [str(path)] if path.is_file() else None


def environment() -> dict:
    return {k: v for k, v in os.environ.items() if k not in {"OPENAI_API_KEY", "DEEPSEEK_API_KEY", "CODEX_API_KEY"}}


async def stop_process(process):
    if process.returncode is not None:
        return
    if os.name == "nt":
        cleanup = await asyncio.create_subprocess_exec(
            "taskkill",
            "/PID",
            str(process.pid),
            "/T",
            "/F",
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
            creationflags=HIDDEN,
        )
        await cleanup.wait()
    else:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    await process.wait()


async def probe() -> dict:
    cmd = command()
    result = {
        "installed": bool(cmd),
        "logged_in": False,
        "supported": False,
        "version": "",
        "message": "未找到 Codex CLI，请安装并登录",
    }
    if not cmd:
        return result

    async def run(args):
        p = await asyncio.create_subprocess_exec(
            *cmd,
            *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=environment(),
            creationflags=HIDDEN,
            start_new_session=os.name != "nt",
        )
        try:
            out, err = await asyncio.wait_for(p.communicate(), 20)
            return p.returncode, (out + err).decode("utf-8", errors="replace")
        except (TimeoutError, asyncio.CancelledError):
            await stop_process(p)
            raise

    try:
        code, version = await run(["--version"])
        result["version"] = version.strip() if code == 0 else "未知"
        code, help_text = await run(["exec", "--help"])
        result["supported"] = code == 0 and all(flag in help_text for flag in REQUIRED)
        code, _ = await run(["login", "status"])
        result["logged_in"] = code == 0
        result["message"] = (
            "已安装并登录；实际生成需连通性测试"
            if result["supported"] and result["logged_in"]
            else "CLI参数不受支持或尚未登录，请检查安装与 codex login"
        )
    except (OSError, TimeoutError):
        result["message"] = "Codex 状态检测失败或超时"
    return result


class Codex(Provider):
    name = "codex"

    def __init__(self, settings):
        self.model = settings["codex_model"]
        self.effort = settings["codex_reasoning_effort"]
        self.timeout = settings["codex_timeout_seconds"]

    async def request(self, system, prompt, schema):
        cmd = command()
        if not cmd:
            raise ProviderError("未找到 Codex CLI，请安装并完成 codex login", "not_installed", False)
        with tempfile.TemporaryDirectory(prefix="reply-assistant-") as folder:
            root = Path(folder)
            schema_path, output = root / "schema.json", root / "result.json"
            schema_path.write_text(json.dumps(output_schema(schema)), encoding="utf-8")
            args = [
                *cmd,
                "exec",
                "--ignore-user-config",
                "--ignore-rules",
                "--ephemeral",
                "--skip-git-repo-check",
                "--sandbox",
                "read-only",
                "--model",
                self.model,
                "-c",
                f'model_reasoning_effort="{self.effort}"',
                "-c",
                "project_doc_max_bytes=0",
                "-c",
                'web_search="disabled"',
                "--cd",
                folder,
                "--output-schema",
                str(schema_path),
                "--output-last-message",
                str(output),
            ]
            for feature in (
                "shell_tool",
                "apps",
                "plugins",
                "browser_use",
                "computer_use",
                "memories",
                "multi_agent",
                "image_generation",
                "view_image",
                "skill_search",
            ):
                args.extend(["--disable", feature])
            args.append("-")
            try:
                process = await asyncio.create_subprocess_exec(
                    *args,
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.DEVNULL,
                    stderr=asyncio.subprocess.DEVNULL,
                    cwd=folder,
                    env=environment(),
                    creationflags=HIDDEN,
                    start_new_session=os.name != "nt",
                )
            except OSError:
                raise ProviderError("无法启动 Codex CLI", "not_installed", False) from None
            try:
                await asyncio.wait_for(process.communicate((system + "\n\n" + prompt).encode("utf-8")), self.timeout)
            except asyncio.CancelledError:
                await stop_process(process)
                raise
            except TimeoutError:
                await stop_process(process)
                raise ProviderError("Codex 调用超时，已回收本任务进程", "timeout") from None
            if process.returncode != 0 or not output.exists():
                raise ProviderError("Codex 执行失败，请检查登录、指定模型可用性与额度", "codex_failed")
            try:
                return json.loads(output.read_text(encoding="utf-8")), None
            except (ValueError, OSError):
                raise ProviderError("Codex 未返回有效 JSON", "invalid_output") from None
