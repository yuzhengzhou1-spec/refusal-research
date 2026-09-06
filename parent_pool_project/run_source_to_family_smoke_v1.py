#!/usr/bin/env python3
"""Small end-to-end source -> core gold -> MISSING/CONFLICT smoke runner."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from evaluation import evaluate_eplus as shared
import build_missing_families_split_v2_3 as common


PIPELINE_VERSION = "source_to_family_smoke.v1"


def run(command: list[str], log: list[dict[str, Any]]) -> None:
    print("RUN", " ".join(command), flush=True)
    completed = subprocess.run(command, check=False)
    log.append({"command": command, "returncode": completed.returncode})
    if completed.returncode != 0:
        raise RuntimeError(f"stage failed with {completed.returncode}: {command}")


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--parent-pool-input", type=Path)
    source.add_argument("--source-root", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--exclude-screening", type=Path, action="append", default=[])
    parser.add_argument("--per-dataset", type=int, default=20)
    parser.add_argument("--family-per-dataset", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260828)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--config", type=Path, default=root / "evaluation" / "config" / "models.json")
    parser.add_argument("--secrets-file", type=Path, default=root / "evaluation" / "secrets" / ".env.local")
    parser.add_argument("--protocol", type=Path, default=root / "evidence_family_protocol_v1.json")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--core-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(__file__).resolve().parent
    args.output_dir.mkdir(parents=True, exist_ok=True)
    commands: list[dict[str, Any]] = []
    candidate_dir = args.output_dir / "candidate_build"
    if args.parent_pool_input:
        command = [
            sys.executable, str(root / "prepare_core_candidates_from_parent_pool_v1.py"),
            "--input", str(args.parent_pool_input),
            "--output-dir", str(candidate_dir),
            "--per-dataset", str(args.per_dataset),
            "--seed", str(args.seed),
        ]
        for path in args.exclude_screening:
            command.extend(["--exclude-screening", str(path)])
        run(command, commands)
        candidates = candidate_dir / "core_gold_candidates_sanitized.jsonl"
    else:
        raw_dir = candidate_dir / "raw_adapter"
        run([
            sys.executable, str(root / "build_core_gold_candidates_v2.py"),
            "--source-root", str(args.source_root),
            "--output", str(raw_dir),
            "--per-dataset", str(args.per_dataset),
            "--seed", str(args.seed),
        ], commands)
        candidates = candidate_dir / "core_gold_candidates_sanitized.jsonl"
        run([
            sys.executable, str(root / "sanitize_core_gold_candidates.py"),
            "--input", str(raw_dir / "core_gold_candidates.jsonl"),
            "--output", str(candidates),
            "--rejected", str(candidate_dir / "core_gold_candidate_query_leakage.jsonl"),
            "--report", str(candidate_dir / "core_gold_candidate_sanitation_report.json"),
        ], commands)

    screening_dir = args.output_dir / "core_screening"
    filter_command = [
        sys.executable, str(root / "evaluation" / "filter_core_gold_v4.py"),
        "--input", str(candidates),
        "--config", str(args.config),
        "--secrets-file", str(args.secrets_file),
        "--output", str(screening_dir),
        "--models", "qwen3_8b",
        "--workers", str(args.workers),
        "--target", "0",
    ]
    if args.prepare_only:
        filter_command.append("--prepare-only")
    run(filter_command, commands)

    report: dict[str, Any] = {
        "pipeline_version": PIPELINE_VERSION,
        "source_mode": "parent_pool_v1" if args.parent_pool_input else "official_raw",
        "commands": commands,
        "prepare_only": args.prepare_only,
    }
    if args.prepare_only:
        common.write_json(args.output_dir / "source_to_family_smoke_report.json", report)
        return 0

    qualified = shared.load_jsonl(screening_dir / "core_gold_final.jsonl")
    counts = Counter(str(row.get("来源") or "") for row in qualified)
    family_count = min(
        args.family_per_dataset,
        counts.get("hotpotqa", 0),
        counts.get("musique", 0),
    )
    report["core_qualified"] = len(qualified)
    report["core_qualified_by_dataset"] = dict(counts)
    report["family_per_dataset"] = family_count
    if args.core_only:
        report["status"] = "CORE_GOLD_STAGE_COMPLETED"
        common.write_json(args.output_dir / "source_to_family_smoke_report.json", report)
        print(json.dumps(report, ensure_ascii=False))
        return 0
    if family_count <= 0:
        report["status"] = "CORE_QUALIFIED_NOT_BALANCED_FOR_FAMILY_SMOKE"
        common.write_json(args.output_dir / "source_to_family_smoke_report.json", report)
        return 2

    for state in ("missing", "conflict"):
        run([
            sys.executable, str(root / "run_family_pipeline_v1.py"),
            "--state", state,
            "--output-dir", str(args.output_dir / state),
            "--protocol", str(args.protocol),
            "--input", str(screening_dir / "core_gold_final.jsonl"),
            "--screening", str(screening_dir / "core_gold_screening.jsonl"),
            "--config", str(args.config),
            "--secrets-file", str(args.secrets_file),
            "--seed", str(args.seed),
            "--per-dataset", str(family_count),
            "--workers", str(args.workers),
        ], commands)
    report["commands"] = commands
    report["status"] = "COMPLETED"
    report["missing_pass"] = len(shared.load_jsonl(
        args.output_dir / "missing" / "final" / "missing_families_high_quality.jsonl"
    ))
    report["conflict_pass"] = len(shared.load_jsonl(
        args.output_dir / "conflict" / "final" / "conflict_families_high_quality.jsonl"
    ))
    common.write_json(args.output_dir / "source_to_family_smoke_report.json", report)
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
