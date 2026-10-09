"""仅使用合成问题验证默认Codex；不会读取或发送资料库。"""

import asyncio
import json
from pathlib import Path

from app.config import DEFAULTS
from app.generation import SYSTEM
from app.providers.codex import Codex, probe
from app.schemas import Reply


async def main():
    diagnostic = await probe()
    print(json.dumps(diagnostic, ensure_ascii=False))
    result = await Codex(DEFAULTS).complete(
        SYSTEM,
        json.dumps(
            {
                "question": "用一句中文说明已收到问题。",
                "references": [],
                "instructions": "不引用来源，reference_usage为空数组。",
            },
            ensure_ascii=False,
        ),
        Reply,
    )
    folder = Path(".verification")
    folder.mkdir(exist_ok=True)
    (folder / "codex-smoke.json").write_text(
        json.dumps({"diagnostic": diagnostic, **result}, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("Codex 合成问题已返回有效结构", result["model"], result["duration_ms"])


if __name__ == "__main__":
    asyncio.run(main())
