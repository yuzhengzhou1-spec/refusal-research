#!/usr/bin/env python3
"""Frozen, restartable MISSING or CONFLICT construction pipeline."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from evaluation import evaluate_eplus as shared
import build_missing_families_split_v2_3 as common


PIPELINE_VERSION = "evidence_family_runner.v1_frozen"


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--state", choices=("missing", "conflict"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, default=root / "evidence_family_protocol_v1.json")
    parser.add_argument("--input", type=Path, default=root / "core_gold_qwen_v4_output" / "screening" / "core_gold_final.jsonl")
    parser.add_argument("--screening", type=Path, default=root / "core_gold_qwen_v4_output" / "screening" / "core_gold_screening.jsonl")
    parser.add_argument("--config", type=Path, default=root / "evaluation" / "config" / "models.json")
    parser.add_argument("--secrets-file", type=Path, default=root / "evaluation" / "secrets" / ".env.local")
    parser.add_argument("--exclude-file", type=Path, action="append", default=[])
    parser.add_argument("--seed", type=int)
    parser.add_argument("--per-dataset", type=int)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def collect_ids(path: Path) -> set[str]:
    ids: set[str] = set()
    if not path.exists():
        raise FileNotFoundError(path)
    for row in shared.load_jsonl(path):
        value = row.get("source_sample_id") or row.get("sample_id") or row.get("样本ID")
        if value:
            ids.add(str(value))
        family = row.get("family")
        if isinstance(family, dict) and family.get("source_sample_id"):
            ids.add(str(family["source_sample_id"]))
    return ids


def select_blind_rows(
    rows: list[dict[str, Any]], excluded: set[str], seed: int, per_dataset: int
) -> list[dict[str, Any]]:
    eligible = [row for row in rows if str(row["样本ID"]) not in excluded]
    selected = common.select_rows(eligible, per_dataset, seed)
    counts = Counter(str(row["来源"]) for row in selected)
    expected = {"hotpotqa": per_dataset, "musique": per_dataset}
    if dict(counts) != expected:
        raise ValueError(f"unexpected blind selection counts: {dict(counts)}")
    return selected


def run_stage(command: list[str], log: list[dict[str, Any]], allowed=(0,)) -> None:
    print("RUN", " ".join(command), flush=True)
    completed = subprocess.run(command, check=False)
    log.append({"command": command, "returncode": completed.returncode})
    if completed.returncode not in allowed:
        raise RuntimeError(f"stage failed with {completed.returncode}: {command}")


def main() -> int:
    args = parse_args()
    root = Path(__file__).resolve().parent
    protocol = shared.load_json(args.protocol)
    blind = protocol["blind_selection"]
    seed = args.seed if args.seed is not None else int(blind["seed"])
    per_dataset = args.per_dataset if args.per_dataset is not None else int(blind["per_dataset"])
    exclude_paths = [root / value for value in blind["exclude_development_files"]]
    exclude_paths.extend(args.exclude_file)
    excluded: set[str] = set()
    for path in exclude_paths:
        excluded.update(collect_ids(path.resolve()))

    rows = shared.load_jsonl(args.input)
    selected = select_blind_rows(rows, excluded, seed, per_dataset)
    selected_ids = [str(row["样本ID"]) for row in selected]
    overlap = sorted(set(selected_ids).intersection(excluded))
    if overlap:
        raise ValueError(f"development overlap detected: {overlap[:10]}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    seed_path = args.output_dir / "blind_seeds.jsonl"
    if seed_path.exists():
        old_ids = [str(row["样本ID"]) for row in shared.load_jsonl(seed_path)]
        if old_ids != selected_ids:
            raise ValueError("existing blind seed manifest differs from frozen selection")
    else:
        common.write_jsonl(seed_path, selected)
    empty_overflow = args.output_dir / "empty_overflow.jsonl"
    if not empty_overflow.exists():
        empty_overflow.write_text("", encoding="utf-8")

    manifest = {
        "pipeline_version": PIPELINE_VERSION,
        "state": args.state,
        "protocol_version": protocol["protocol_version"],
        "protocol_sha256": file_sha256(args.protocol),
        "seed": seed,
        "per_dataset": per_dataset,
        "selected_ids": selected_ids,
        "selected_id_sha256": hashlib.sha256("\n".join(selected_ids).encode()).hexdigest(),
        "development_excluded_ids": len(excluded),
        "development_overlap": overlap,
        "input": str(args.input.resolve()),
        "input_sha256": file_sha256(args.input),
        "overflow_used": False,
        "commands": [],
    }
    common.write_json(args.output_dir / "frozen_run_manifest.json", manifest)
    if args.dry_run:
        return 0

    python = sys.executable
    build_dir = args.output_dir / "build"
    base = [
        "--input", str(seed_path), "--overflow", str(empty_overflow),
        "--screening", str(args.screening), "--config", str(args.config),
        "--secrets-file", str(args.secrets_file), "--workers", str(args.workers),
    ]
    commands: list[dict[str, Any]] = []
    if args.state == "missing":
        run_stage([python, str(root / "build_missing_families_split_v2_3.py"), *base, "--output-dir", str(build_dir)], commands, (0, 2))
        families = build_dir / "missing_families.jsonl"
        forward_dir = args.output_dir / "audit_forward"
        reverse_dir = args.output_dir / "audit_reverse_v1_1"
        run_stage([python, str(root / "validate_missing_families_v1.py"), "--families", str(families), "--output-dir", str(forward_dir), "--config", str(args.config), "--secrets-file", str(args.secrets_file), "--workers", str(args.workers)], commands)
        run_stage([python, str(root / "validate_missing_families_reverse_v1.py"), "--families", str(families), "--output-dir", str(reverse_dir), "--config", str(args.config), "--secrets-file", str(args.secrets_file), "--workers", str(args.workers)], commands)
        run_stage([python, str(root / "select_missing_high_quality_v1.py"), "--families", str(families), "--edit-calls", str(build_dir / "missing_edit_calls.jsonl"), "--audit-forward", str(forward_dir / "missing_semantic_audit_calls.jsonl"), "--audit-reverse", str(reverse_dir / "missing_semantic_audit_calls.jsonl"), "--output-dir", str(args.output_dir / "final")], commands)
    else:
        run_stage([python, str(root / "build_conflict_families_v1.py"), *base, "--output-dir", str(build_dir)], commands, (0, 2))
        families = build_dir / "conflict_families.jsonl"
        audit_a = args.output_dir / "audit_v1_5"
        audit_b = args.output_dir / "audit_v1_3"
        run_stage([python, str(root / "validate_conflict_families_v1.py"), "--families", str(families), "--output-dir", str(audit_a), "--config", str(args.config), "--secrets-file", str(args.secrets_file), "--workers", str(args.workers)], commands)
        run_stage([python, str(root / "validate_conflict_families_v1_3_frozen.py"), "--families", str(families), "--output-dir", str(audit_b), "--config", str(args.config), "--secrets-file", str(args.secrets_file), "--workers", str(args.workers)], commands)
        run_stage([python, str(root / "select_conflict_high_quality_v1.py"), "--families", str(families), "--edit-calls", str(build_dir / "conflict_edit_calls.jsonl"), "--audit-a", str(audit_a / "conflict_semantic_audit_calls.jsonl"), "--audit-b", str(audit_b / "conflict_semantic_audit_calls.jsonl"), "--output-dir", str(args.output_dir / "final")], commands)

    manifest["commands"] = commands
    manifest["completed_at"] = common.now_iso()
    common.write_json(args.output_dir / "frozen_run_manifest.json", manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
