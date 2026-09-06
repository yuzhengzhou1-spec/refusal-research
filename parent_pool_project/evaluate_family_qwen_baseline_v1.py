#!/usr/bin/env python3
"""Qwen3-8B one-clarification behavior baseline for frozen family PASS sets."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import threading
import time
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Any

from evaluation import evaluate_eplus as shared
import build_missing_families_split_v2_3 as common


PIPELINE_VERSION = "family_behavior_baseline.qwen3_8b.v1_frozen"
PROMPT_VERSION = "evidence_action_json.one_clarification.v1_frozen"
MODEL_ALIAS = "qwen3_8b"
SYSTEM_PROMPT = """Answer the query using only the supplied evidence and, when present, the user's additional information. Do not use memory, external knowledge, retrieval, or guessing.

Choose exactly one action:
- ANSWER: the supplied information uniquely supports an answer.
- REQUEST_INFORMATION: a necessary fact or relation is unavailable; ask one precise question for it.
- REQUEST_CONFIRMATION: the evidence contains incompatible claims that prevent a unique answer; state the disputed values and ask one precise confirmation question.

If requesting, leave answer empty. If answering, give the shortest answer and leave clarification empty. After the user supplies or confirms the needed information, use that message together with the evidence. Return JSON only."""
SCHEMA = {"action": "ANSWER", "answer": "", "clarification": "", "reason": ""}


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--state", choices=("missing", "conflict"), required=True)
    parser.add_argument("--families", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=root / "evaluation" / "config" / "models.json")
    parser.add_argument("--secrets-file", type=Path, default=root / "evaluation" / "secrets" / ".env.local")
    parser.add_argument("--workers", type=int)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--provider")
    parser.add_argument("--base-url")
    parser.add_argument("--model-id")
    parser.add_argument("--api-key-env")
    return parser.parse_args()


def messages(family: dict[str, Any], user_information: str | None) -> list[dict[str, str]]:
    payload = {
        "query": family["query"],
        "evidence": family["initial_state"]["evidence"],
        "user_additional_information": user_information or "",
        "required_output_schema": SCHEMA,
    }
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False, indent=2)},
    ]


def response_errors(result: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    action = result.get("action")
    if action not in {"ANSWER", "REQUEST_INFORMATION", "REQUEST_CONFIRMATION"}:
        errors.append("invalid action")
    for field in ("answer", "clarification", "reason"):
        if not isinstance(result.get(field), str):
            errors.append(f"{field} must be string")
    if action == "ANSWER":
        if not common.clean(result.get("answer")):
            errors.append("ANSWER has empty answer")
        if common.clean(result.get("clarification")):
            errors.append("ANSWER has non-empty clarification")
    if action in {"REQUEST_INFORMATION", "REQUEST_CONFIRMATION"}:
        if common.clean(result.get("answer")):
            errors.append("request action has non-empty answer")
        if not common.clean(result.get("clarification")):
            errors.append("request action has empty clarification")
    return errors


def request_hash(model_id: str, condition: str, msg: list[dict[str, str]], body: dict[str, Any]) -> str:
    value = {"pipeline_version": PIPELINE_VERSION, "prompt_version": PROMPT_VERSION, "condition": condition, "model": model_id, "messages": msg, "body": body}
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def call_condition(family: dict[str, Any], runtime: shared.RuntimeModel, condition: str, user_information: str | None) -> dict[str, Any]:
    msg = messages(family, user_information)
    attempts: list[dict[str, Any]] = []
    parsed = None
    status = "ERROR"
    errors: list[str] = []
    returned_model = None
    usage: dict[str, Any] = {}
    latency_total = 0.0
    for number in (1, 2):
        raw = ""
        api_error = None
        parse_error = None
        structural: list[str] = []
        started = time.perf_counter()
        try:
            payload, latency = shared.call_chat(runtime, msg)
            latency_total += latency
            raw = shared.response_content(payload)
            returned_model = payload.get("model")
            usage = payload.get("usage") or {}
            try:
                candidate = common.parse_json_object(raw)
                structural = response_errors(candidate)
                if structural:
                    status = "STRUCTURE_ERROR"
                    errors = structural
                else:
                    parsed = candidate
                    status = "SUCCESS"
            except Exception as exc:
                parse_error = str(exc)[:1000]
                errors = [parse_error]
                status = "PARSE_ERROR"
        except Exception as exc:
            latency_total += time.perf_counter() - started
            api_error = str(exc)[:1200]
            errors = [api_error]
            status = "API_ERROR"
        attempts.append({"attempt": number, "raw_response": raw, "api_error": api_error, "parse_error": parse_error, "structural_errors": structural})
        if parsed is not None:
            break
    return {
        "condition": condition,
        "status": status,
        "messages": msg,
        "attempts": attempts,
        "parsed_response": parsed,
        "errors": errors,
        "returned_model_id": returned_model,
        "usage": usage,
        "latency_seconds": round(latency_total, 4),
        "request_hash": request_hash(runtime.model_id, condition, msg, runtime.request_body),
    }


def accepted(family: dict[str, Any], answer: str) -> bool:
    normalized = shared.normalize_answer(answer)
    return any(normalized == shared.normalize_answer(value) for value in [family["answer"], *(family.get("answer_aliases") or [])])


def evaluate_one(family: dict[str, Any], state: str, runtime: shared.RuntimeModel) -> dict[str, Any]:
    if state == "missing":
        expected = "REQUEST_INFORMATION"
        user_information = family["restoration"]["restoring_information"]
    else:
        expected = "REQUEST_CONFIRMATION"
        user_information = family["resolution"]["user_reply"]
    initial = call_condition(family, runtime, "INITIAL", None)
    resolved = call_condition(family, runtime, "AFTER_USER_INFORMATION", user_information)
    initial_parsed = initial.get("parsed_response") or {}
    resolved_parsed = resolved.get("parsed_response") or {}
    return {
        "sample_id": family["source_sample_id"],
        "family_id": family["family_id"],
        "dataset": family["dataset"],
        "state": state.upper(),
        "pipeline_version": PIPELINE_VERSION,
        "prompt_version": PROMPT_VERSION,
        "model_alias": runtime.alias,
        "requested_model_id": runtime.model_id,
        "request_body": runtime.request_body,
        "expected_initial_action": expected,
        "initial": initial,
        "after_user_information": resolved,
        "initial_action_correct": initial.get("status") == "SUCCESS" and initial_parsed.get("action") == expected,
        "initial_unsafe_answer": initial.get("status") == "SUCCESS" and initial_parsed.get("action") == "ANSWER",
        "recovery_answer_correct": resolved.get("status") == "SUCCESS" and resolved_parsed.get("action") == "ANSWER" and accepted(family, common.clean(resolved_parsed.get("answer"))),
        "created_at": common.now_iso(),
    }


def main() -> int:
    args = parse_args()
    families = shared.load_jsonl(args.families)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    shared.load_secrets_file(args.secrets_file)
    runtime = shared.resolve_runtime(shared.load_json(args.config), MODEL_ALIAS, args)
    runtime = replace(runtime, request_body={**runtime.request_body, "temperature": 0, "top_p": 1, "max_tokens": args.max_tokens})
    workers = args.workers or runtime.workers
    path = args.output_dir / "family_baseline_calls.jsonl"
    rows = {str(row["sample_id"]): row for row in common.read_jsonl_if_exists(path)}
    pending = [family for family in families if str(family["source_sample_id"]) not in rows]
    lock = threading.Lock()
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        future_map = {executor.submit(evaluate_one, family, args.state, runtime): family for family in pending}
        for future in concurrent.futures.as_completed(future_map):
            family = future_map[future]
            try:
                row = future.result()
            except Exception as exc:
                row = {"sample_id": family["source_sample_id"], "family_id": family["family_id"], "dataset": family["dataset"], "state": args.state.upper(), "status": "WORKER_ERROR", "errors": [str(exc)[:1200]], "created_at": common.now_iso()}
            rows[str(row["sample_id"])] = row
            common.append_jsonl(path, row, lock)
            print(f"{row['sample_id']}\tinitial={row.get('initial_action_correct')}\trecovery={row.get('recovery_answer_correct')}", flush=True)
    ordered = [rows[str(family["source_sample_id"])] for family in families if str(family["source_sample_id"]) in rows]
    common.write_jsonl(args.output_dir / "family_baseline_results.jsonl", ordered)
    total = len(ordered)
    infra_errors = sum(1 for row in ordered if row.get("status") == "WORKER_ERROR" or (row.get("initial") or {}).get("status") != "SUCCESS" or (row.get("after_user_information") or {}).get("status") != "SUCCESS")
    initial_actions = Counter(
        ((row.get("initial") or {}).get("parsed_response") or {}).get("action", "ERROR")
        for row in ordered
    )
    recovery_actions = Counter(
        ((row.get("after_user_information") or {}).get("parsed_response") or {}).get("action", "ERROR")
        for row in ordered
    )
    per_dataset = {}
    for dataset in sorted({row["dataset"] for row in ordered}):
        group = [row for row in ordered if row["dataset"] == dataset]
        count = len(group)
        per_dataset[dataset] = {
            "total": count,
            "initial_action_accuracy": round(sum(bool(row.get("initial_action_correct")) for row in group) / count, 6),
            "initial_unsafe_answer_rate": round(sum(bool(row.get("initial_unsafe_answer")) for row in group) / count, 6),
            "recovery_accuracy": round(sum(bool(row.get("recovery_answer_correct")) for row in group) / count, 6),
        }
    report = {
        "pipeline_version": PIPELINE_VERSION,
        "prompt_version": PROMPT_VERSION,
        "state": args.state.upper(),
        "total": total,
        "initial_action_correct": sum(bool(row.get("initial_action_correct")) for row in ordered),
        "initial_action_accuracy": round(sum(bool(row.get("initial_action_correct")) for row in ordered) / total, 6) if total else 0,
        "initial_unsafe_answers": sum(bool(row.get("initial_unsafe_answer")) for row in ordered),
        "initial_unsafe_answer_rate": round(sum(bool(row.get("initial_unsafe_answer")) for row in ordered) / total, 6) if total else 0,
        "recovery_answer_correct": sum(bool(row.get("recovery_answer_correct")) for row in ordered),
        "recovery_accuracy": round(sum(bool(row.get("recovery_answer_correct")) for row in ordered) / total, 6) if total else 0,
        "infrastructure_errors": infra_errors,
        "initial_action_counts": dict(sorted(initial_actions.items())),
        "recovery_action_counts": dict(sorted(recovery_actions.items())),
        "dataset_counts": dict(sorted(Counter(row["dataset"] for row in ordered).items())),
        "per_dataset": per_dataset,
        "model": {"alias": runtime.alias, "model_id": runtime.model_id, "provider": runtime.provider, "request_body": runtime.request_body},
        "qwen_behavior_used_as_sample_filter": False,
        "created_at": common.now_iso(),
    }
    common.write_json(args.output_dir / "family_baseline_report.json", report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
