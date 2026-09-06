#!/usr/bin/env python3
"""Semantic gate for synthetic MISSING evidence-state families."""

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


PIPELINE_VERSION = "missing_semantic_gate.v1_2"
PROMPT_VERSION = "missing_semantic_gate.support_gap_restoration.v1_2"
MODEL_ALIAS = "deepseek_v4_pro"

SYSTEM_PROMPT = """You are a strict semantic quality gate for one synthetic MISSING-information evidence family. Judge only; do not repair it and do not use external or parametric knowledge.

The query is a request, not evidence. Every factual premise used in your judgment must come from FULL evidence, MISSING evidence, restoring_information, or the recorded exact support span.

A valid family must satisfy all five conditions:
1. deletion_effective: MISSING evidence no longer directly or through a complete evidence chain supplies the exact relation/value/condition described by missing_information. The accepted-answer entity or related entities may remain elsewhere; that alone does not make deletion ineffective. Fail only when the remaining evidence actually establishes the requested slot or an equivalent complete answer chain.
2. known_support_grounded: clarification_grounding.span is present in the indicated MISSING document and that single recorded span fully entails the clarification's opening known-support claim. Do not combine it with another sentence merely because that other sentence occurs elsewhere in MISSING evidence. The opening claim must not add a date, entity, property, relation, or condition taken only from the query or outside the recorded span.
3. query_not_used_as_evidence: neither the edit rationale nor clarification treats wording in the query as proof of a fact. Repeating query constraints only in the unresolved request is allowed; presenting them as something the context establishes is not.
4. gap_matches_edit: the clarification accurately contrasts existing support with the single information slot made unavailable by minimal_edit, remains answer-free, and asks for exactly that slot rather than mechanically repeating the whole query or requesting another fact. Qualifying constraints from the query may narrow that same requested slot even when missing_information states the slot more briefly. For example, "the featured rapper who died on [date]" still requests one rapper slot; the death-date phrase is not a second slot. Such a query constraint does not need to be in the grounding span when it appears only in the unresolved request. Fail it only if the clarification presents that query-only constraint as something the context establishes or asks the user to supply it separately.
5. restoration_recovers: adding restoring_information to MISSING evidence re-establishes a complete evidence-supported route to the accepted answer, and restoring_information does not make a stronger claim than the removed original_span.

Be conservative about unsupported known-support claims and surviving complete answer chains. Do not fail merely because a name, date, or answer string still occurs without the missing relation. Set valid true if and only if all five condition booleans are true. Always provide a concise non-empty reason, including for a valid sample. Return JSON only, with no markdown or commentary."""

SCHEMA = {
    "valid": True,
    "deletion_effective": True,
    "known_support_grounded": True,
    "query_not_used_as_evidence": True,
    "gap_matches_edit": True,
    "restoration_recovers": True,
    "failure_codes": [],
    "reason": "Brief evidence-based justification.",
}

DIMENSIONS = (
    "deletion_effective",
    "known_support_grounded",
    "query_not_used_as_evidence",
    "gap_matches_edit",
    "restoration_recovers",
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
    return parser.parse_args()


def audit_messages(family: dict[str, Any]) -> list[dict[str, str]]:
    state = family["initial_state"]
    payload = {
        "query": family["query"],
        "accepted_answer": family["answer"],
        "answer_aliases": family.get("answer_aliases") or [],
        "full_evidence": family["full_state"]["evidence"],
        "missing_evidence": state["evidence"],
        "missing_information": state["missing_information"],
        "minimal_edit": state["minimal_edit"],
        "clarification_grounding": state["clarification_grounding"],
        "assistant_clarification": family["trajectory"][0]["content"],
        "restoring_information": family["restoration"]["restoring_information"],
        "required_output_schema": SCHEMA,
    }
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False, indent=2)},
    ]


def audit_errors(result: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    for field in ("valid", *DIMENSIONS):
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


def dimension_failed(result: dict[str, Any], field: str) -> bool:
    return result.get(field) is False


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
        attempts.append({
            "attempt": number,
            "raw_response": raw,
            "api_error": api_error,
            "parse_error": parse_error,
            "structural_errors": structure,
        })
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
    runtime = replace(runtime, request_body={
        **runtime.request_body,
        "temperature": 0,
        "top_p": 1,
        "max_tokens": args.max_tokens,
    })
    workers = args.workers or runtime.workers
    call_path = args.output_dir / "missing_semantic_audit_calls.jsonl"
    calls = {
        str(row["sample_id"]): row
        for row in common.read_jsonl_if_exists(call_path)
    }
    pending = [row for row in families if str(row["source_sample_id"]) not in calls]
    lock = threading.Lock()
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        future_map = {executor.submit(audit_one, row, runtime): row for row in pending}
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

    pass_ids = {sid for sid, call in calls.items() if call.get("status") == "PASS"}
    passed = [row for row in families if str(row["source_sample_id"]) in pass_ids]
    review = [row for row in families if str(row["source_sample_id"]) not in pass_ids]
    common.write_jsonl(args.output_dir / "missing_families_semantic_pass.jsonl", passed)
    common.write_jsonl(args.output_dir / "missing_families_semantic_review.jsonl", review)
    report = {
        "pipeline_version": PIPELINE_VERSION,
        "input_families": len(families),
        "semantic_pass": len(passed),
        "semantic_pass_rate": round(len(passed) / len(families), 6) if families else 0,
        "status_counts": dict(sorted(Counter(x.get("status") for x in calls.values()).items())),
        "pass_dataset_counts": dict(sorted(Counter(x["dataset"] for x in passed).items())),
        "failure_code_counts": dict(sorted(Counter(
            code
            for call in calls.values()
            for code in ((call.get("parsed_response") or {}).get("failure_codes") or [])
        ).items())),
        "failure_dimension_counts": dict(sorted(Counter(
            field
            for call in calls.values()
            for field in DIMENSIONS
            if dimension_failed(call.get("parsed_response") or {}, field)
        ).items())),
        "model": {
            "alias": runtime.alias,
            "model_id": runtime.model_id,
            "provider": runtime.provider,
            "request_body": runtime.request_body,
        },
        "created_at": common.now_iso(),
    }
    common.write_json(args.output_dir / "missing_semantic_audit_report.json", report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
