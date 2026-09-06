#!/usr/bin/env python3
"""Deterministically repair non-semantic metadata in core-gold JSONL files."""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any


def normalize(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).casefold()
    text = re.sub(r"[^\w]+", " ", text, flags=re.UNICODE)
    return " ".join(text.split())


def answer_type(answer: str) -> str:
    if re.fullmatch(r"\d+(?:[.,]\d+)?", answer.strip()):
        return "\u6570\u503c"
    if re.search(r"\b(?:1[0-9]{3}|20[0-9]{2})\b", answer):
        return "\u65e5\u671f\u6216\u5e74\u4efd"
    if len(normalize(answer).split()) <= 6:
        return "\u77ed\u5b9e\u4f53"
    return "\u77ed\u6587\u672c"


def repair(path: Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    changes = 0
    counts: Counter[str] = Counter()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            canonical = str((row.get("答案") or [""])[0])
            expected = answer_type(canonical)
            metadata = row.setdefault("元数据", {})
            if metadata.get("答案类型") != expected:
                metadata["答案类型"] = expected
                changes += 1
            counts[expected] += 1
            rows.append(row)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    temporary.replace(path)
    return {
        "path": str(path),
        "rows": len(rows),
        "changed_rows": changes,
        "answer_type_counts": dict(counts),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--files", type=Path, nargs="+", required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    missing = [str(path) for path in args.files if not path.exists()]
    if missing:
        raise FileNotFoundError(missing)
    report = {
        "stage": "core-gold metadata finalization",
        "repairs": [repair(path) for path in args.files],
        "model_outputs_modified": False,
        "semantic_content_modified": False,
        "recomputed_field": "元数据.答案类型",
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
