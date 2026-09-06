#!/usr/bin/env python3
"""Independent offline verifier for causal core-gold outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import filter_core_gold as pipeline


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--audit-output", type=Path, required=True)
    args = parser.parse_args()

    candidates = pipeline.load_rows(args.candidates)
    calls = pipeline.latest_calls(args.output_root / "core_gold_calls.jsonl")
    screenings = pipeline.load_rows(args.output_root / "core_gold_screening.jsonl")
    final_rows = pipeline.load_rows(args.output_root / "core_gold_final.jsonl")
    counterfactuals = pipeline.load_rows(args.output_root / "core_gold_counterfactuals.jsonl")
    report = load_json(args.output_root / "core_gold_report.json")
    errors: list[str] = []

    candidate_ids = [pipeline.sample_id(row) for row in candidates]
    screening_ids = [str(row.get("sample_id") or "") for row in screenings]
    final_ids = [pipeline.sample_id(row) for row in final_rows]
    if len(candidate_ids) != len(set(candidate_ids)):
        errors.append("candidate IDs are not unique")
    if len(screening_ids) != len(set(screening_ids)):
        errors.append("screening IDs are not unique")
    if set(candidate_ids) != set(screening_ids):
        errors.append("screening IDs differ from candidate IDs")
    if len(final_ids) != len(set(final_ids)):
        errors.append("final IDs are not unique")

    screening_by_id = {str(row["sample_id"]): row for row in screenings}
    qualified_ids = {
        str(row["sample_id"]) for row in screenings
        if row.get("decision") == "CORE_QUALIFIED"
    }
    if not set(final_ids).issubset(qualified_ids):
        errors.append("final rows contain non-qualified samples")
    if report.get("final_rows") != len(final_rows):
        errors.append("report final_rows does not match file")
    if report.get("candidate_rows") != len(candidates):
        errors.append("report candidate_rows does not match file")
    required_models = list(report.get("required_models") or [])
    if not required_models:
        errors.append("report has no required models")

    expected_call_count = 0
    for screening in screenings:
        sid = str(screening["sample_id"])
        for model in required_models:
            query_id = pipeline.call_identifier(sid, model, "QUERY_ONLY")
            expected_call_count += 1
            call = calls.get(query_id)
            if call is None:
                errors.append(f"{query_id}: missing QUERY_ONLY call")
                continue
            query_clean = call.get("status") == "SUCCESS" and not call.get("answer_correct")
            if bool(screening.get("query_only_knowledge_clean")) != query_clean:
                errors.append(f"{sid}: query-only decision mismatch for {model}")
        if screening.get("query_only_knowledge_clean"):
            for model in required_models:
                full_id = pipeline.call_identifier(sid, model, "CLEAN_FULL")
                expected_call_count += 1
                if full_id not in calls:
                    errors.append(f"{full_id}: missing CLEAN_FULL call")
        if screening.get("clean_full_pass") and not screening.get("counterfactual_construction_error"):
            for model in required_models:
                cf_id = pipeline.call_identifier(sid, model, "COUNTERFACTUAL")
                expected_call_count += 1
                if cf_id not in calls:
                    errors.append(f"{cf_id}: missing COUNTERFACTUAL call")

    for row in final_rows:
        sid = pipeline.sample_id(row)
        screening = screening_by_id[sid]
        if not (
            screening.get("query_only_knowledge_clean")
            and screening.get("clean_full_pass")
            and screening.get("counterfactual_follow_pass")
        ):
            errors.append(f"{sid}: final row failed at least one admission gate")
        support_ids = {
            str(unit.get("文档ID") or "") for unit in row.get("关键证据单元", [])
        }
        context_ids = {
            str(document.get("文档ID") or "") for document in row.get("完整上下文", [])
        }
        if support_ids != context_ids:
            errors.append(f"{sid}: final context contains non-support or missing support documents")
        if row.get("元数据", {}).get("包含干扰文档") is not False:
            errors.append(f"{sid}: final metadata does not certify distractor-free context")

    cf_ids = [str(row.get("sample_id") or "") for row in counterfactuals]
    if len(cf_ids) != len(set(cf_ids)):
        errors.append("counterfactual IDs are not unique")
    for variant in counterfactuals:
        if variant.get("diagnostic_only") is not True:
            errors.append(f"{variant.get('sample_id')}: counterfactual is not marked diagnostic-only")

    audit = {
        "passed": not errors,
        "error_count": len(errors),
        "errors": errors[:300],
        "candidate_rows": len(candidates),
        "screening_rows": len(screenings),
        "final_rows": len(final_rows),
        "counterfactual_rows": len(counterfactuals),
        "stored_calls": len(calls),
        "expected_calls_from_funnel": expected_call_count,
        "call_status": dict(Counter(str(row.get("status")) for row in calls.values())),
        "required_models": required_models,
        "candidate_sha256": sha256(args.candidates),
        "final_sha256": sha256(args.output_root / "core_gold_final.jsonl"),
        "report_sha256": sha256(args.output_root / "core_gold_report.json"),
    }
    args.audit_output.parent.mkdir(parents=True, exist_ok=True)
    args.audit_output.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit, ensure_ascii=False))
    return 0 if audit["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
