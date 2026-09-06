#!/usr/bin/env python3
"""Build a conservative high-quality CONFLICT pilot from independent gates."""

from __future__ import annotations

import argparse
import re
from collections import Counter
from pathlib import Path
from typing import Any

from evaluation import evaluate_eplus as shared
import build_conflict_families_v1 as builder
import build_missing_families_split_v2_3 as common


PIPELINE_VERSION = "conflict_high_quality_selector.v1_2"
COMPOUND_ROLE = re.compile(
    r"\b(?:co-?written|written)\b.{0,100}\b(?:edited|produced|directed)\b"
    r"|\b(?:edited|produced)\b.{0,100}\bdirected\b"
    r"|\bwriter-director\b.{0,140}\bpreviously directed\b",
    re.IGNORECASE,
)
DOCUMENT_ID_TARGET = re.compile(r"^(?:hp|mq)_doc_\d+$", re.IGNORECASE)
PROXIMITY_RELATION = re.compile(
    r"\b(?:approximately\s+)?(?:\d+(?:\.\d+)?|one|two|three|four|five|six|seven|eight|nine|ten)\s*"
    r"(?:mi|mile|miles|km|kilometre|kilometres|kilometer|kilometers)\s+"
    r"(?:north|south|east|west)(?:-?east|-?west)?\s+(?:of|from)\b",
    re.IGNORECASE,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--families", type=Path, required=True)
    parser.add_argument("--edit-calls", type=Path, required=True)
    parser.add_argument("--audit-a", type=Path, required=True)
    parser.add_argument("--audit-b", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def normalized_equal(left: str, right: str) -> bool:
    return shared.normalize_answer(left) == shared.normalize_answer(right)


def exclusion_reasons(
    family: dict[str, Any],
    audit_a: dict[str, Any] | None,
    audit_b: dict[str, Any] | None,
) -> list[str]:
    reasons: list[str] = []
    if not audit_a or audit_a.get("status") != "PASS":
        reasons.append("SEMANTIC_GATE_A_NOT_PASS")
    if not audit_b or audit_b.get("status") != "PASS":
        reasons.append("SEMANTIC_GATE_B_NOT_PASS")

    state = family["initial_state"]
    accepted = [family["answer"], *(family.get("answer_aliases") or [])]
    original_value = common.clean(state.get("original_value"))
    if not any(normalized_equal(original_value, answer) for answer in accepted):
        reasons.append("NOT_DIRECT_ANSWER_SLOT")

    original_claim = common.clean(state.get("original_claim"))
    conflicting_claim = common.clean(state.get("conflicting_claim"))
    if COMPOUND_ROLE.search(original_claim) or COMPOUND_ROLE.search(conflicting_claim):
        reasons.append("COMPOUND_ROLE_SUBSTITUTION")

    conflicting_value = common.clean(state.get("conflicting_value"))
    if (
        builder.normalized_contains(original_claim, conflicting_value)
        and builder.normalized_contains(conflicting_claim, original_value)
    ):
        reasons.append("CROSS_SWAPPED_MULTIPLE_RELATIONS")

    if DOCUMENT_ID_TARGET.fullmatch(common.clean(state.get("conflict_target"))):
        reasons.append("NON_SEMANTIC_CONFLICT_TARGET")
    if (
        not re.search(r"\d", original_value)
        and PROXIMITY_RELATION.search(original_claim)
        and PROXIMITY_RELATION.search(conflicting_claim)
    ):
        reasons.append("MULTIVALUED_PROXIMITY_RELATION")
    return reasons


def markdown_family(family: dict[str, Any]) -> str:
    state = family["initial_state"]
    return "\n".join(
        [
            f"## {family['family_id']}",
            "",
            f"- dataset: {family['dataset']}",
            f"- query: {family['query']}",
            f"- answer: {family['answer']}",
            f"- original claim: {state['original_claim']}",
            f"- conflicting claim: {state['conflicting_claim']}",
            f"- clarification: {family['trajectory'][0]['content']}",
        ]
    )


def main() -> int:
    args = parse_args()
    families = shared.load_jsonl(args.families)
    edit_calls = shared.load_jsonl(args.edit_calls)
    audit_a_rows = shared.load_jsonl(args.audit_a)
    audit_b_rows = shared.load_jsonl(args.audit_b)
    audit_a = {str(row["sample_id"]): row for row in audit_a_rows}
    audit_b = {str(row["sample_id"]): row for row in audit_b_rows}

    passed: list[dict[str, Any]] = []
    review: list[dict[str, Any]] = []
    reason_counts: Counter[str] = Counter()
    for family in families:
        sid = str(family["source_sample_id"])
        reasons = exclusion_reasons(family, audit_a.get(sid), audit_b.get(sid))
        reason_counts.update(reasons)
        if reasons:
            review.append({"family": family, "exclusion_reasons": reasons})
        else:
            passed.append(family)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    common.write_jsonl(args.output_dir / "conflict_families_high_quality.jsonl", passed)
    common.write_jsonl(args.output_dir / "conflict_families_review.jsonl", review)
    edit_status = Counter(row.get("status") for row in edit_calls)
    report = {
        "pipeline_version": PIPELINE_VERSION,
        "candidate_rows": len(edit_calls),
        "structurally_constructed_families": len(families),
        "high_quality_families": len(passed),
        "candidate_yield": round(len(passed) / len(edit_calls), 6) if edit_calls else 0,
        "structured_yield": round(len(passed) / len(families), 6) if families else 0,
        "dataset_counts": dict(sorted(Counter(row["dataset"] for row in passed).items())),
        "edit_status_counts": dict(sorted(edit_status.items())),
        "exclusion_reason_counts": dict(sorted(reason_counts.items())),
        "selection_policy": {
            "semantic_gate_consensus_required": True,
            "direct_answer_slot_only": True,
            "compound_role_substitution_rejected": True,
            "cross_swapped_relations_rejected": True,
            "semantic_conflict_target_required": True,
            "multivalued_proximity_relation_rejected": True,
            "no_manual_labels_used": True,
        },
        "created_at": common.now_iso(),
    }
    common.write_json(args.output_dir / "conflict_high_quality_report.json", report)
    review_md = "# CONFLICT high-quality pilot review\n\n" + "\n\n".join(
        markdown_family(family) for family in passed
    )
    (args.output_dir / "conflict_high_quality_review.md").write_text(
        review_md + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
