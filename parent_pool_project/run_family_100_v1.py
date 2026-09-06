#!/usr/bin/env python3
"""Project-default one-command entry for the current 100-row family targets."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state", choices=("missing", "conflict"), required=True)
    parser.add_argument("--target", type=int, default=100)
    parser.add_argument("--batch-per-dataset", type=int, default=40)
    parser.add_argument("--max-waves", type=int, default=4)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(__file__).resolve().parent
    command = [
        sys.executable, str(root / "run_family_target_v1.py"),
        "--state", args.state,
        "--target", str(args.target),
        "--output-dir", str(root / "family_100_v1" / args.state),
        "--seed-pool", str(root / "core_gold_qwen_v4_output" / "screening" / "core_gold_final.jsonl"),
        "--seed-pool", str(root / "core_gold_qwen_v4_output" / "screening" / "core_gold_overflow.jsonl"),
        "--screening", str(root / "core_gold_qwen_v4_output" / "screening" / "core_gold_screening.jsonl"),
        "--batch-per-dataset", str(args.batch_per_dataset),
        "--max-waves", str(args.max_waves),
        "--workers", str(args.workers),
    ]
    if args.state == "missing":
        command.extend([
            "--accepted", str(root / "missing_family_split_v2_5_2_grounded_same50_output" / "semantic_audit_v1_2" / "missing_families_semantic_pass.jsonl"),
            "--accepted", str(root / "family_freeze_blind_v1" / "missing" / "final" / "missing_families_high_quality.jsonl"),
            "--exclude-file", str(root / "family_freeze_blind_v1" / "missing" / "blind_seeds.jsonl"),
        ])
    else:
        command.extend([
            "--accepted", str(root / "conflict_family_v1_2_smoke50_output" / "high_quality_v1_2" / "conflict_families_high_quality.jsonl"),
            "--accepted", str(root / "family_freeze_blind_v1" / "conflict" / "final" / "conflict_families_high_quality.jsonl"),
            "--exclude-file", str(root / "family_freeze_blind_v1" / "conflict" / "blind_seeds.jsonl"),
        ])
        expanded_gold = root / "source_to_family_smoke_v1" / "core_screening" / "core_gold_final.jsonl"
        expanded_screening = root / "source_to_family_smoke_v1" / "core_screening" / "core_gold_screening.jsonl"
        if expanded_gold.exists() and expanded_screening.exists():
            command.extend([
                "--seed-pool", str(expanded_gold),
                "--screening", str(expanded_screening),
            ])
    if args.dry_run:
        command.append("--dry-run")
    print("RUN", " ".join(command), flush=True)
    return subprocess.run(command, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
