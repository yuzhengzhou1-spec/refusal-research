#!/usr/bin/env python3
"""Semantic gate for CONFLICT evidence-state families."""

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


PIPELINE_VERSION = "conflict_semantic_gate.v1_5_frozen"
PROMPT_VERSION = "conflict_semantic_gate.atomic_necessity.v1_5_frozen"
MODEL_ALIAS = "deepseek_v4_pro"

SYSTEM_PROMPT = """You are a strict semantic quality gate for a synthetic CONFLICT evidence sample. Judge only; do not repair the sample.

The conflicting sentence was intentionally synthesized and appended to CONFLICT evidence. It does not need to occur in the original FULL evidence, be supported by an external source, or be true in the real world. Never fail it merely as fabricated, unsupported, absent from FULL, shorter than the original sentence, or separately appended. Wording and append-only structure have already passed deterministic checks and are outside this audit. A subject pronoun may be resolved from its document title and local context.

A valid sample must satisfy all seven conditions:
1. same_atomic_fact: the original and appended claims assert different values for exactly the same subject, relation, scope, time, and conditions.
2. single_disputed_slot: changing the value creates exactly one disputed proposition. A person substituted across coordinated roles such as 'written, produced and directed by', or across a main clause plus a relative clause, changes multiple facts and fails. Unchanged descriptive details are allowed.
3. mutually_exclusive: the two claims cannot both be true under that scope. Apply strict logical coexistence, not dataset convention. Lists, examples, memberships, siblings, guest appearances, awards at different editions, repeated achievements, collaborators, things located a similar distance from multiple places, and other multi-valued relations are not exclusive merely because they mention different values. A query written in the singular does not make its underlying relation single-valued.
4. necessary_for_query: the disputed fact directly supplies the final answer or an indispensable unique bridge needed to answer the query from the supplied evidence. Changing an incidental attribute, list position, or nearby fact fails. Do not treat wording in the query as evidence.
5. changes_query_answer: evaluating the query under the original claim leads to the accepted answer, while evaluating it under the conflicting claim leads to a different answer candidate or breaks an indispensable bridge. If both claims still lead to the same accepted answer, the conflict is ineffective. For example, changing which Homeric character a fixed lake is named after fails if both claims still identify the same lake requested by the query.
6. not_resolvable_without_user: ignoring the two disputed assertions themselves, no other unaffected evidence, document title, query constraint, arithmetic identity, component count, date, or deterministic relation independently selects the original value or refutes the conflicting value. The mere presence of the original assertion, the synthetic origin of the conflicting assertion, or absence of other support for it cannot resolve the conflict. If genuinely unaffected content can settle it, fail.
7. resolution_restores_answer: after the user confirms the original claim/value, the supplied evidence chain uniquely supports the accepted answer without adding another unstated fact.

Also fail claims that are fragments without an explicit relation, claims that merely imply a different fact, or conflicts created from weak proxies such as inferring a county seat only from a courthouse location. Do not assess source provenance, real-world truth, sentence length, or append position. Use failure_codes only to summarize which of the seven semantic conditions is false.

Set valid to true if and only if all seven condition booleans are true. Always provide a concise non-empty reason, including for valid samples. Use concise machine-readable failure codes. Return JSON only, with no markdown or commentary."""

SCHEMA = {
    "valid": True,
    "same_atomic_fact": True,
    "single_disputed_slot": True,
    "mutually_exclusive": True,
    "necessary_for_query": True,
    "changes_query_answer": True,
    "not_resolvable_without_user": True,
    "resolution_restores_answer": True,
    "failure_codes": [],
    "reason": "Brief evidence-based justification.",
}

DIMENSIONS = (
    "same_atomic_fact",
    "single_disputed_slot",
    "mutually_exclusive",
    "necessary_for_query",
    "changes_query_answer",
    "not_resolvable_without_user",
    "resolution_restores_answer",
)


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--families", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=root / "evaluation" / "config" / "models.json")
    parser.add_argument("--secrets-file", type=Path, default=root / "evaluation" / "secrets" / ".env.local")
    parser.add_argument("--workers", type=int)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--provider")
    parser.add_argument("--base-url")
    parser.add_argument("--model-id")
    parser.add_argument("--api-key-env")
    parser.add_argument(
        "--retry-api-errors",
        action="store_true",
        help="Retry only cached API_ERROR/WORKER_ERROR rows; preserve PASS/FAIL outputs.",
    )
    return parser.parse_args()


def audit_messages(family: dict[str, Any]) -> list[dict[str, str]]:
    state = family["initial_state"]
    payload = {
        "query": family["query"],
        "accepted_answer": family["answer"],
        "answer_aliases": family.get("answer_aliases") or [],
        "full_evidence": family["full_state"]["evidence"],
        "conflict_evidence": family["initial_state"]["evidence"],
        "conflict_target": state["conflict_target"],
        "original_claim": state["original_claim"],
        "original_value": state["original_value"],
        "conflicting_claim": state["conflicting_claim"],
        "conflicting_value": state["conflicting_value"],
        "required_output_schema": SCHEMA,
    }
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False, indent=2)},
    ]


def audit_errors(result: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    fields = ("valid", *DIMENSIONS)
    for field in fields:
        if not isinstance(result.get(field), bool):
            errors.append(f"{field} must be boolean")
    codes = result.get("failure_codes")
    if not isinstance(codes, list) or any(not isinstance(code, str) for code in codes):
        errors.append("failure_codes must be a list of strings")
    if not common.clean(result.get("reason")):
        errors.append("reason must be non-empty")
    return errors


def derived_valid(result: dict[str, Any]) -> bool:
    return all(result[field] for field in DIMENSIONS)


def request_hash(model_id: str, messages: list[dict[str, str]], body: dict[str, Any]) -> str:
    value = {
        "pipeline_version": PIPELINE_VERSION,
        "prompt_version": PROMPT_VERSION,
        "model": model_id,
        "messages": messages,
        "body": body,
    }
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode()).hexdigest()


def audit_one(family: dict[str, Any], runtime: shared.RuntimeModel) -> dict[str, Any]:
    messages = audit_messages(family)
    attempts: list[dict[str, Any]] = []
    parsed: dict[str, Any] | None = None
    status = "ERROR"
    errors: list[str] = []
    returned_model: str | None = None
    usage: dict[str, Any] = {}
    latency_total = 0.0
    for number in (1, 2):
        raw = ""
        api_error = None
        parse_error = None
        structure: list[str] = []
        started = time.perf_counter()
        try:
            payload, latency = shared.call_chat(runtime, messages)
            latency_total += latency
            raw = shared.response_content(payload)
            returned_model = payload.get("model")
            usage = payload.get("usage") or {}
            try:
                candidate = common.parse_json_object(raw)
                structure = audit_errors(candidate)
                if structure:
                    status = "STRUCTURE_ERROR"
                    errors = structure
                else:
                    candidate["derived_valid"] = derived_valid(candidate)
                    candidate["model_valid_consistent"] = (
                        candidate["valid"] == candidate["derived_valid"]
                    )
                    parsed = candidate
                    status = "PASS" if candidate["derived_valid"] else "FAIL"
            except Exception as exc:
                parse_error = str(exc)[:1000]
                errors = [parse_error]
                status = "PARSE_ERROR"
        except Exception as exc:
            latency_total += time.perf_counter() - started
            api_error = str(exc)[:1200]
            errors = [api_error]
            status = "API_ERROR"
        attempts.append(
            {
                "attempt": number,
                "raw_response": raw,
                "api_error": api_error,
                "parse_error": parse_error,
                "structural_errors": structure,
            }
        )
        if parsed is not None or status == "STRUCTURE_ERROR":
            break
    return {
        "sample_id": family["source_sample_id"],
        "family_id": family["family_id"],
        "dataset": family["dataset"],
        "pipeline_version": PIPELINE_VERSION,
        "prompt_version": PROMPT_VERSION,
        "model_alias": runtime.alias,
        "requested_model_id": runtime.model_id,
        "returned_model_id": returned_model,
        "provider": runtime.provider,
        "request_body": runtime.request_body,
        "request_hash": request_hash(runtime.model_id, messages, runtime.request_body),
        "messages": messages,
        "attempts": attempts,
        "parsed_response": parsed,
        "status": status,
        "errors": errors,
        "usage": usage,
        "latency_seconds": round(latency_total, 4),
        "created_at": common.now_iso(),
    }


def main() -> int:
    args = parse_args()
    families = shared.load_jsonl(args.families)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    shared.load_secrets_file(args.secrets_file)
    config = shared.load_json(args.config)
    runtime = shared.resolve_runtime(config, MODEL_ALIAS, args)
    runtime = replace(
        runtime,
        request_body={
            **runtime.request_body,
            "temperature": 0,
            "top_p": 1,
            "max_tokens": args.max_tokens,
        },
    )
    workers = args.workers or runtime.workers
    call_path = args.output_dir / "conflict_semantic_audit_calls.jsonl"
    calls = {
        str(row["sample_id"]): row
        for row in common.read_jsonl_if_exists(call_path)
    }
    if args.retry_api_errors:
        retry_ids = [
            sid for sid, call in calls.items()
            if call.get("status") in {"API_ERROR", "WORKER_ERROR"}
        ]
        for sid in retry_ids:
            del calls[sid]
        print(f"retrying_api_or_worker_errors={len(retry_ids)}", flush=True)
    pending = [row for row in families if str(row["source_sample_id"]) not in calls]
    lock = threading.Lock()
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        future_map = {
            executor.submit(audit_one, family, runtime): family for family in pending
        }
        for future in concurrent.futures.as_completed(future_map):
            family = future_map[future]
            try:
                call = future.result()
            except Exception as exc:
                call = {
                    "sample_id": family["source_sample_id"],
                    "family_id": family["family_id"],
                    "dataset": family["dataset"],
                    "status": "WORKER_ERROR",
                    "errors": [str(exc)[:1200]],
                    "created_at": common.now_iso(),
                }
            calls[str(call["sample_id"])] = call
            common.append_jsonl(call_path, call, lock)
            print(f"{call['sample_id']}\t{call['status']}", flush=True)

    pass_ids = {
        sid for sid, call in calls.items() if call.get("status") == "PASS"
    }
    passed = [row for row in families if str(row["source_sample_id"]) in pass_ids]
    review = [row for row in families if str(row["source_sample_id"]) not in pass_ids]
    common.write_jsonl(args.output_dir / "conflict_families_semantic_pass.jsonl", passed)
    common.write_jsonl(args.output_dir / "conflict_families_semantic_review.jsonl", review)
    report = {
        "pipeline_version": PIPELINE_VERSION,
        "input_families": len(families),
        "semantic_pass": len(passed),
        "semantic_pass_rate": round(len(passed) / len(families), 6) if families else 0,
        "status_counts": dict(sorted(Counter(call.get("status") for call in calls.values()).items())),
        "pass_dataset_counts": dict(sorted(Counter(row["dataset"] for row in passed).items())),
        "failure_code_counts": dict(sorted(Counter(
            code
            for call in calls.values()
            for code in ((call.get("parsed_response") or {}).get("failure_codes") or [])
        ).items())),
        "derived_failure_dimension_counts": dict(sorted(Counter(
            field
            for call in calls.values()
            for field in DIMENSIONS
            if (call.get("parsed_response") or {}).get(field) is False
        ).items())),
        "model": {
            "alias": runtime.alias,
            "model_id": runtime.model_id,
            "provider": runtime.provider,
            "request_body": runtime.request_body,
        },
        "created_at": common.now_iso(),
    }
    common.write_json(args.output_dir / "conflict_semantic_audit_report.json", report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
