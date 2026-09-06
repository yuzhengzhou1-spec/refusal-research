#!/usr/bin/env python3
"""Compact a core-gold call log to the calls active in the final funnel."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import filter_core_gold_v4  # noqa: F401  (activates v4 namespaces)
import filter_core_gold as pipeline


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    calls_path = args.output_root / "core_gold_calls.jsonl"
    screening_path = args.output_root / "core_gold_screening.jsonl"
    report_path = args.output_root / "core_gold_report.json"
    screenings = load_jsonl(screening_path)
    runtime_report = json.loads(report_path.read_text(encoding="utf-8"))
    models = list(runtime_report["required_models"])

    active: set[str] = set()
    for row in screenings:
        sid = str(row["sample_id"])
        for model in models:
            active.add(pipeline.call_identifier(sid, model, "QUERY_ONLY"))
            if row.get("query_only_knowledge_clean"):
                active.add(pipeline.call_identifier(sid, model, "CLEAN_FULL"))
            if row.get("clean_full_pass") and not row.get("counterfactual_construction_error"):
                active.add(pipeline.call_identifier(sid, model, "COUNTERFACTUAL"))

    original_rows = load_jsonl(calls_path)
    latest: dict[str, dict[str, Any]] = {}
    for row in original_rows:
        latest[str(row["call_id"])] = row
    missing = sorted(active - set(latest))
    if missing:
        raise RuntimeError(f"active calls are missing: {missing[:20]}")
    compacted = [latest[identifier] for identifier in sorted(active)]
    temporary = calls_path.with_suffix(calls_path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        for row in compacted:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    temporary.replace(calls_path)
    report = {
        "stage": "core-gold active call-log compaction",
        "original_line_count": len(original_rows),
        "original_unique_call_ids": len(latest),
        "active_call_ids": len(active),
        "compacted_line_count": len(compacted),
        "removed_lines": len(original_rows) - len(compacted),
        "removed_unique_call_ids": len(set(latest) - active),
        "required_models": models,
        "policy": "Keep only the latest call for every request active in the final v4 funnel.",
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
