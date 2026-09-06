#!/usr/bin/env python3
"""Accumulate frozen family PASS records until an exact target is reached."""

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


PIPELINE_VERSION = "family_target_accumulator.v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_id(row: dict[str, Any]) -> str:
    return str(row.get("source_sample_id") or row.get("样本ID") or "")


def expected_state(row: dict[str, Any], state: str) -> bool:
    initial = row.get("initial_state") or {}
    if str(initial.get("state") or "").upper() != state.upper():
        return False
    actions = [str(turn.get("action") or "") for turn in row.get("trajectory", [])]
    if state == "missing":
        return actions == ["REQUEST_INFORMATION", "PROVIDE_INFORMATION", "ANSWER"]
    return actions == ["REQUEST_CONFLICT_RESOLUTION", "CONFIRM_INFORMATION", "ANSWER"]


def validate_pass(row: dict[str, Any], state: str) -> list[str]:
    errors: list[str] = []
    sid = source_id(row)
    if not sid:
        errors.append("missing source_sample_id")
    if not expected_state(row, state):
        errors.append("state or trajectory mismatch")
    gate = row.get("seed_gate") or {}
    if gate.get("query_only_answer_correct") is not False:
        errors.append("query-only gate is not clean")
    if gate.get("full_answer_correct") is not True:
        errors.append("FULL gate did not pass")
    if str((row.get("full_state") or {}).get("state") or "") != "FULL":
        errors.append("missing FULL state")
    if not row.get("answer") or not row.get("query"):
        errors.append("missing query or answer")
    return errors


def merge_jsonl(paths: list[Path], key_name: str) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(path)
        for row in shared.load_jsonl(path):
            key = str(row.get(key_name) or "")
            if not key:
                raise ValueError(f"{path}: row missing {key_name}")
            merged.setdefault(key, row)
    return list(merged.values())


def load_passes(paths: list[Path], state: str) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    rows: dict[str, dict[str, Any]] = {}
    provenance: dict[str, str] = {}
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(path)
        for row in shared.load_jsonl(path):
            errors = validate_pass(row, state)
            if errors:
                raise ValueError(f"{path}: {source_id(row)}: {errors}")
            sid = source_id(row)
            rows.setdefault(sid, row)
            provenance.setdefault(sid, str(path.resolve()))
    return rows, provenance


def stable_family_key(seed: int, state: str, sid: str) -> str:
    return hashlib.sha256(
        f"{seed}:{PIPELINE_VERSION}:{state}:{sid}".encode("utf-8")
    ).hexdigest()


def run(command: list[str], commands: list[dict[str, Any]]) -> None:
    print("RUN", " ".join(command), flush=True)
    completed = subprocess.run(command, check=False)
    commands.append({"command": command, "returncode": completed.returncode})
    if completed.returncode != 0:
        raise RuntimeError(f"stage failed with {completed.returncode}: {command}")


def api_error_count(wave_dir: Path) -> int:
    total = 0
    for path in wave_dir.rglob("*_report.json"):
        try:
            report = shared.load_json(path)
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        counts = report.get("status_counts") or {}
        total += int(counts.get("API_ERROR") or 0)
        total += int(counts.get("WORKER_ERROR") or 0)
    return total


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--state", choices=("missing", "conflict"), required=True)
    parser.add_argument("--target", type=int, default=100)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--accepted", type=Path, action="append", default=[])
    parser.add_argument("--seed-pool", type=Path, action="append", required=True)
    parser.add_argument("--screening", type=Path, action="append", required=True)
    parser.add_argument("--exclude-file", type=Path, action="append", default=[])
    parser.add_argument("--batch-per-dataset", type=int, default=40)
    parser.add_argument("--max-waves", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260828)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--protocol", type=Path, default=root / "evidence_family_protocol_v1.json")
    parser.add_argument("--config", type=Path, default=root / "evaluation" / "config" / "models.json")
    parser.add_argument("--secrets-file", type=Path, default=root / "evaluation" / "secrets" / ".env.local")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.target <= 0 or args.batch_per_dataset <= 0 or args.max_waves < 0:
        raise ValueError("target/batch must be positive and max-waves non-negative")
    root = Path(__file__).resolve().parent
    args.output_dir.mkdir(parents=True, exist_ok=True)

    seed_rows = merge_jsonl(args.seed_pool, "样本ID")
    screening_rows = merge_jsonl(args.screening, "sample_id")
    seed_path = args.output_dir / "qualified_seed_pool.jsonl"
    screening_path = args.output_dir / "qualified_seed_screening.jsonl"
    common.write_jsonl(seed_path, seed_rows)
    common.write_jsonl(screening_path, screening_rows)

    pass_paths = list(args.accepted)
    passes, provenance = load_passes(pass_paths, args.state)
    commands: list[dict[str, Any]] = []
    wave_reports: list[dict[str, Any]] = []
    attempted_files = list(args.exclude_file)
    stopped_reason = ""

    for wave in range(args.max_waves):
        if len(passes) >= args.target:
            break
        wave_dir = args.output_dir / f"wave_{wave + 1:02d}"
        command = [
            sys.executable, str(root / "run_family_pipeline_v1.py"),
            "--state", args.state,
            "--output-dir", str(wave_dir),
            "--protocol", str(args.protocol),
            "--input", str(seed_path),
            "--screening", str(screening_path),
            "--config", str(args.config),
            "--secrets-file", str(args.secrets_file),
            "--seed", str(args.seed + wave),
            "--per-dataset", str(args.batch_per_dataset),
            "--workers", str(args.workers),
        ]
        for path in [*pass_paths, *attempted_files]:
            command.extend(["--exclude-file", str(path)])
        if args.dry_run:
            command.append("--dry-run")
        run(command, commands)
        attempted_files.append(wave_dir / "blind_seeds.jsonl")
        if args.dry_run:
            wave_reports.append({"wave": wave + 1, "dry_run": True})
            break
        final_name = (
            "missing_families_high_quality.jsonl"
            if args.state == "missing"
            else "conflict_families_high_quality.jsonl"
        )
        final_path = wave_dir / "final" / final_name
        new_passes, new_provenance = load_passes([final_path], args.state)
        before = len(passes)
        for sid, row in new_passes.items():
            passes.setdefault(sid, row)
            provenance.setdefault(sid, new_provenance[sid])
        pass_paths.append(final_path)
        wave_reports.append({
            "wave": wave + 1,
            "attempted": len(shared.load_jsonl(wave_dir / "blind_seeds.jsonl")),
            "new_unique_passes": len(passes) - before,
            "cumulative_passes": len(passes),
            "output": str(final_path.resolve()),
            "api_or_worker_errors": api_error_count(wave_dir),
        })
        if wave_reports[-1]["api_or_worker_errors"] and len(passes) < args.target:
            stopped_reason = "INFRASTRUCTURE_ERRORS_REQUIRE_RESUME"
            break

    ordered_ids = sorted(
        passes,
        key=lambda sid: stable_family_key(args.seed, args.state, sid),
    )
    selected_ids = ordered_ids[:args.target]
    surplus_ids = ordered_ids[args.target:]
    selected = [passes[sid] for sid in selected_ids]
    surplus = [passes[sid] for sid in surplus_ids]
    target_reached = len(selected) == args.target
    final_name = (
        f"{args.state}_{args.target}_high_quality.jsonl"
        if target_reached
        else f"{args.state}_partial_{len(selected)}_of_{args.target}.jsonl"
    )
    common.write_jsonl(args.output_dir / final_name, selected)
    common.write_jsonl(args.output_dir / f"{args.state}_surplus_high_quality.jsonl", surplus)

    report = {
        "pipeline_version": PIPELINE_VERSION,
        "state": args.state.upper(),
        "target": args.target,
        "target_reached": target_reached,
        "final_output": str((args.output_dir / final_name).resolve()),
        "final_rows": len(selected),
        "final_unique_source_ids": len({source_id(row) for row in selected}),
        "final_by_dataset": dict(Counter(str(row.get("dataset") or "") for row in selected)),
        "available_passes": len(passes),
        "surplus_rows": len(surplus),
        "initial_accepted_sources": [
            {"path": str(path.resolve()), "sha256": sha256(path)} for path in args.accepted
        ],
        "seed_pools": [
            {"path": str(path.resolve()), "sha256": sha256(path)} for path in args.seed_pool
        ],
        "screening_sources": [
            {"path": str(path.resolve()), "sha256": sha256(path)} for path in args.screening
        ],
        "waves": wave_reports,
        "selection_seed": args.seed,
        "selection_rule": "sha256 over state and source_sample_id",
        "selected_provenance": {sid: provenance[sid] for sid in selected_ids},
        "commands": commands,
        "stopped_reason": stopped_reason,
        "input_files_modified": False,
    }
    common.write_json(args.output_dir / f"{args.state}_target_report.json", report)
    print(json.dumps({key: report[key] for key in (
        "state", "target", "target_reached", "final_rows", "final_by_dataset",
        "available_passes", "surplus_rows", "waves",
    )}, ensure_ascii=False))
    if args.dry_run:
        return 0
    return 0 if report["target_reached"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
