#!/usr/bin/env python3
"""Remove deterministic answer leakage from supporting-only core candidates."""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    temporary.replace(path)


def normalize(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).casefold()
    text = re.sub(r"(?<=\d),(?=\d)", "", text)
    text = text.replace("'", "").replace("’", "")
    text = "".join(
        " " if unicodedata.category(character).startswith(("P", "S")) else character
        for character in text
    )
    return " ".join(token for token in text.split() if token not in {"a", "an", "the"})


def phrase_in_text(phrase: str, text: str) -> bool:
    needle = normalize(phrase)
    return bool(needle) and f" {needle} " in f" {normalize(text)} "


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rejected", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    rows = load_jsonl(args.input)
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for row in rows:
        query = str(row.get("问题") or "")
        aliases = [str(value) for value in row.get("答案", []) if str(value).strip()]
        leaking = [alias for alias in aliases if phrase_in_text(alias, query)]
        if leaking:
            rejected.append({
                "sample_id": row.get("样本ID"),
                "dataset": row.get("来源"),
                "reason": "ANSWER_ALIAS_IN_QUERY",
                "leaking_aliases": leaking,
                "query": query,
            })
        else:
            accepted.append(row)
    write_jsonl(args.output, accepted)
    write_jsonl(args.rejected, rejected)
    report = {
        "stage": "core-gold deterministic candidate sanitation",
        "input_rows": len(rows),
        "accepted_rows": len(accepted),
        "rejected_rows": len(rejected),
        "rejection_counts": dict(Counter(row["reason"] for row in rejected)),
        "policy": "Any normalized canonical answer or alias occurring in query is rejected before model calls.",
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
