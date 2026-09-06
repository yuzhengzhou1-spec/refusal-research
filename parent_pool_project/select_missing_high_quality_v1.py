#!/usr/bin/env python3
"""Select MISSING families requiring forward and adversarial gate consensus."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Any

from evaluation import evaluate_eplus as shared
import build_missing_families_split_v2_3 as common


PIPELINE_VERSION = "missing_high_quality_selector.v1_frozen"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--families", type=Path, required=True)
    parser.add_argument("--edit-calls", type=Path, required=True)
    parser.add_argument("--audit-forward", type=Path, required=True)
    parser.add_argument("--audit-reverse", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def exclusion_reasons(
    forward: dict[str, Any] | None, reverse: dict[str, Any] | None
) -> list[str]:
    reasons: list[str] = []
    if not forward or forward.get("status") != "PASS":
        reasons.append("FORWARD_SEMANTIC_GATE_NOT_PASS")
    if not reverse or reverse.get("status") != "PASS":
        reasons.append("REVERSE_SEMANTIC_GATE_NOT_PASS")
    return reasons


def main() -> int:
    args = parse_args()
    families = shared.load_jsonl(args.families)
    edit_calls = shared.load_jsonl(args.edit_calls)
    forward = {
        str(row["sample_id"]): row for row in shared.load_jsonl(args.audit_forward)
    }
    reverse = {
        str(row["sample_id"]): row for row in shared.load_jsonl(args.audit_reverse)
    }
    passed: list[dict[str, Any]] = []
    review: list[dict[str, Any]] = []
    reason_counts: Counter[str] = Counter()
    for family in families:
        sid = str(family["source_sample_id"])
        reasons = exclusion_reasons(forward.get(sid), reverse.get(sid))
        reason_counts.update(reasons)
        if reasons:
            review.append({"family": family, "exclusion_reasons": reasons})
        else:
            passed.append(family)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    common.write_jsonl(args.output_dir / "missing_families_high_quality.jsonl", passed)
    common.write_jsonl(args.output_dir / "missing_families_review.jsonl", review)
    report = {
        "pipeline_version": PIPELINE_VERSION,
        "candidate_rows": len(edit_calls),
        "structurally_constructed_families": len(families),
        "high_quality_families": len(passed),
        "candidate_yield": round(len(passed) / len(edit_calls), 6) if edit_calls else 0,
        "structured_yield": round(len(passed) / len(families), 6) if families else 0,
        "dataset_counts": dict(sorted(Counter(row["dataset"] for row in passed).items())),
        "edit_status_counts": dict(sorted(Counter(row.get("status") for row in edit_calls).items())),
        "exclusion_reason_counts": dict(sorted(reason_counts.items())),
        "selection_policy": {
            "forward_reverse_consensus_required": True,
            "qwen_behavior_used_as_filter": False,
            "manual_labels_used": False,
        },
        "created_at": common.now_iso(),
    }
    common.write_json(args.output_dir / "missing_high_quality_report.json", report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
