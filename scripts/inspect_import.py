"""只解析导入文件，报告数量；不入库、不调用模型、不输出评论正文。"""

import argparse
import json
from pathlib import Path

from app.importers.xhs_qa import parse_upload

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("--output", type=Path, default=Path(".verification/import-report.json"))
    args = parser.parse_args()
    parsed = parse_upload(args.source.read_bytes())
    report = {
        "scope": "本机解析，未入库或发送模型",
        "file_hash": parsed["file_hash"],
        "unit_count": len(parsed["units"]),
        "stats": parsed["stats"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
