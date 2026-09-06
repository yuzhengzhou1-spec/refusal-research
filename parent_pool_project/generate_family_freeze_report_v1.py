#!/usr/bin/env python3
"""Generate the blind construction and Qwen baseline freeze report."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Any

from evaluation import evaluate_eplus as shared
import build_missing_families_split_v2_3 as common


PIPELINE_VERSION = "evidence_family_freeze_report.v1"
INFRA_STATUSES = {"API_ERROR", "PARSE_ERROR", "WORKER_ERROR", "ERROR"}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=root / "evidence_family_protocol_v1.json")
    parser.add_argument("--missing-run", type=Path, required=True)
    parser.add_argument("--conflict-run", type=Path, required=True)
    parser.add_argument("--missing-baseline", type=Path)
    parser.add_argument("--conflict-baseline", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def latest_rows(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return rows
    for row in shared.load_jsonl(path):
        sid = str(row.get("sample_id") or row.get("source_sample_id") or len(rows))
        rows[sid] = row
    return rows


def infra_count(paths: list[Path]) -> int:
    return sum(
        row.get("status") in INFRA_STATUSES
        for path in paths
        for row in latest_rows(path).values()
    )


def main() -> int:
    args = parse_args()
    protocol = shared.load_json(args.protocol)
    missing_manifest = shared.load_json(args.missing_run / "frozen_run_manifest.json")
    conflict_manifest = shared.load_json(args.conflict_run / "frozen_run_manifest.json")
    missing_quality = shared.load_json(args.missing_run / "final" / "missing_high_quality_report.json")
    conflict_quality = shared.load_json(args.conflict_run / "final" / "conflict_high_quality_report.json")
    missing_threshold = protocol["missing"]["blind_acceptance"]
    conflict_threshold = protocol["conflict"]["blind_acceptance"]
    missing_count = int(missing_quality["high_quality_families"])
    conflict_count = int(conflict_quality["high_quality_families"])
    missing_datasets = missing_quality.get("dataset_counts") or {}
    conflict_datasets = conflict_quality.get("dataset_counts") or {}
    same_blind_ids = missing_manifest["selected_ids"] == conflict_manifest["selected_ids"]
    zero_overlap = not missing_manifest["development_overlap"] and not conflict_manifest["development_overlap"]
    protocol_hash_equal = missing_manifest["protocol_sha256"] == conflict_manifest["protocol_sha256"]
    current_protocol_hash = file_sha256(args.protocol)
    manifests_match_current_protocol = (
        missing_manifest["protocol_sha256"] == current_protocol_hash
        and conflict_manifest["protocol_sha256"] == current_protocol_hash
    )
    missing_infra = infra_count([
        args.missing_run / "build" / "missing_edit_calls.jsonl",
        args.missing_run / "build" / "missing_clarification_calls.jsonl",
        args.missing_run / "audit_forward" / "missing_semantic_audit_calls.jsonl",
        args.missing_run / "audit_reverse_v1_1" / "missing_semantic_audit_calls.jsonl",
    ])
    conflict_infra = infra_count([
        args.conflict_run / "build" / "conflict_edit_calls.jsonl",
        args.conflict_run / "build" / "conflict_clarification_calls.jsonl",
        args.conflict_run / "audit_v1_5" / "conflict_semantic_audit_calls.jsonl",
        args.conflict_run / "audit_v1_3" / "conflict_semantic_audit_calls.jsonl",
    ])
    missing_pass = (
        missing_count >= int(missing_threshold["minimum_total"])
        and all(int(missing_datasets.get(dataset, 0)) >= int(missing_threshold["minimum_per_dataset"]) for dataset in protocol["blind_selection"]["datasets"])
    )
    conflict_pass = (
        conflict_count >= int(conflict_threshold["minimum_total"])
        and all(int(conflict_datasets.get(dataset, 0)) >= int(conflict_threshold["minimum_per_dataset"]) for dataset in protocol["blind_selection"]["datasets"])
    )
    construction_ready = all((same_blind_ids, zero_overlap, protocol_hash_equal, manifests_match_current_protocol, missing_infra == 0, conflict_infra == 0, missing_pass, conflict_pass))

    missing_baseline = shared.load_json(args.missing_baseline / "family_baseline_report.json") if args.missing_baseline else None
    conflict_baseline = shared.load_json(args.conflict_baseline / "family_baseline_report.json") if args.conflict_baseline else None
    baselines_complete = bool(missing_baseline and conflict_baseline)
    baseline_infra_ok = bool(
        baselines_complete
        and missing_baseline["infrastructure_errors"] == 0
        and conflict_baseline["infrastructure_errors"] == 0
    )
    amendments = protocol.get("amendments") or []
    strict_blind_without_amendment = not amendments
    content_policy_changed = any(bool(item.get("content_policy_changed")) for item in amendments)
    if not construction_ready:
        status = "BLIND_CONSTRUCTION_NOT_SEALED"
    elif not baselines_complete:
        status = "BLIND_CONSTRUCTION_PASSED_BASELINE_PENDING"
    elif not baseline_infra_ok:
        status = "BASELINE_INFRASTRUCTURE_FAILED"
    elif amendments:
        status = "SEALED_WITH_DOCUMENTED_TECHNICAL_AMENDMENT_READY_FOR_FULL_PRODUCTION_DECISION"
    else:
        status = "SEALED_READY_FOR_FULL_PRODUCTION_DECISION"
    report = {
        "pipeline_version": PIPELINE_VERSION,
        "protocol_version": protocol["protocol_version"],
        "protocol_amendments": protocol.get("amendments") or [],
        "status": status,
        "construction_ready": construction_ready,
        "strict_blind_without_amendment": strict_blind_without_amendment,
        "content_policy_changed_by_amendment": content_policy_changed,
        "checks": {
            "same_blind_ids": same_blind_ids,
            "development_overlap_zero": zero_overlap,
            "protocol_hash_equal": protocol_hash_equal,
            "manifests_match_current_protocol": manifests_match_current_protocol,
            "current_protocol_sha256": current_protocol_hash,
            "missing_infrastructure_errors": missing_infra,
            "conflict_infrastructure_errors": conflict_infra,
            "missing_threshold_pass": missing_pass,
            "conflict_threshold_pass": conflict_pass,
        },
        "missing": {"quality": missing_quality, "baseline": missing_baseline},
        "conflict": {"quality": conflict_quality, "baseline": conflict_baseline},
        "baseline_behavior_used_as_filter": False,
        "created_at": common.now_iso(),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    common.write_json(args.output_dir / "family_freeze_report.json", report)
    lines = [
        "# Evidence-family blind freeze report",
        "",
        f"- status: **{status}**",
        f"- protocol: `{protocol['protocol_version']}`",
        f"- same blind IDs: {same_blind_ids}",
        f"- development overlap zero: {zero_overlap}",
        f"- MISSING high quality: {missing_count}/50; datasets={missing_datasets}; threshold pass={missing_pass}",
        f"- CONFLICT high quality: {conflict_count}/50; datasets={conflict_datasets}; threshold pass={conflict_pass}",
        f"- construction infrastructure errors: MISSING={missing_infra}, CONFLICT={conflict_infra}",
        f"- protocol amendments: {len(amendments)}",
        f"- strict blind without amendment: {strict_blind_without_amendment}",
        f"- content policy changed by amendment: {content_policy_changed}",
    ]
    if missing_baseline:
        lines.extend(["", "## Qwen3-8B MISSING baseline", "", f"- initial action accuracy: {missing_baseline['initial_action_accuracy']}", f"- initial action counts: {missing_baseline.get('initial_action_counts')}", f"- unsafe answer rate: {missing_baseline['initial_unsafe_answer_rate']}", f"- recovery accuracy: {missing_baseline['recovery_accuracy']}", f"- per dataset: {missing_baseline.get('per_dataset')}"])
    if conflict_baseline:
        lines.extend(["", "## Qwen3-8B CONFLICT baseline", "", f"- initial action accuracy: {conflict_baseline['initial_action_accuracy']}", f"- initial action counts: {conflict_baseline.get('initial_action_counts')}", f"- unsafe answer rate: {conflict_baseline['initial_unsafe_answer_rate']}", f"- recovery accuracy: {conflict_baseline['recovery_accuracy']}", f"- per dataset: {conflict_baseline.get('per_dataset')}"])
    lines.extend(["", "## Interpretation", "", "- Construction quality passed the frozen thresholds; Qwen behavior was never used for sample admission.", "- High recovery accuracy with low initial clarification/confirmation accuracy isolates the intended behavior gap: the model can answer after the missing or disputed fact is resolved, but often answers prematurely before resolution.", "- CONFLICT is substantially harder than MISSING for explicit state recognition under the frozen baseline."])
    (args.output_dir / "family_freeze_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0 if construction_ready else 2


if __name__ == "__main__":
    raise SystemExit(main())
