#!/usr/bin/env python3
"""Build one-turn CONFLICT evidence-state families."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import re
import threading
import time
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Any

from evaluation import evaluate_eplus as shared
import build_missing_families_split_v2_3 as common


PIPELINE_VERSION = "conflict_family.append_assertion.v1_2"
EDIT_PROMPT_VERSION = "conflict_family.editor.append_atomic_conflict.v1_2"
CLARIFY_PROMPT_VERSION = "conflict_family.clarifier.present_two_values.v1"
MODEL_ALIAS = "deepseek_v4_pro"

EDITOR_SYSTEM_PROMPT = """Create one simple CONFLICT evidence variant. Your only job is to append one contradictory evidence sentence. Do not generate dialogue or explain hidden reasoning.

1. Find one explicit, atomic claim in FULL evidence that is necessary for answering the question. Prefer the direct answer-bearing claim whose value is the accepted answer or an alias. Use an indispensable unique bridge only when no direct answer-bearing claim is available. The original claim must be directly stated, not inferred from a weak proxy.
2. Keep every character of every original document unchanged.
3. In exactly one document, append exactly one new sentence at the end. The appended sentence must repeat the same explicit subject, relation or attribute, scope, time, and conditions as the original claim, changing only one value.
4. Keep the original correct claim present. The result must contain two direct, mutually incompatible claims; neither may be hedged, attributed to an uncertain source, or presented as a correction.
5. The conflicting value must have the same semantic type as the original value: person for person, place for place, valid date for date, number for number, organization for organization, and so on.
6. The conflicting value must differ from the original value and every accepted answer or alias. It must be natural and plausible, not a placeholder, nonsense token, or trivial negation.
7. Before constructing, apply a strict coexistence test: if the original and new claims could both be true, do not use that fact. Avoid lists, examples, guest appearances, memberships, siblings, collaborators, repeated awards or events, proximity to places, disjunctions, hedged claims, and any other multi-valued relation. A singular query does not make a relation single-valued.
8. The disputed slot itself must supply the final answer or an indispensable unique bridge. Do not change an incidental attribute while leaving the accepted answer unchanged in both claims. For example, if the query asks for a series name, changing only the film's ordinal position is invalid. Do not infer a county seat merely from a courthouse location.
9. original_claim must be a complete grammatical assertion with an explicit relation, not a table fragment or detached phrase. conflicting_claim must also be a complete standalone assertion. Do not use 'actually', correction language, or a different predicate.
10. Do not add any other fact, entity relation, explanation, provenance phrase, ambiguity, or second conflict.
11. Do not alter the question, accepted answer, original claim, document IDs, titles, count, or order.
12. Return original_claim as an exact substring of the original document. Return conflicting_claim as exactly the single sentence appended to the document. Briefly state why the slot is necessary and why the claims cannot coexist.
13. If there is no explicit atomic necessary single-valued claim that supports a clean same-type contradiction, set constructible to false.

Return JSON only, with no markdown or commentary."""

CLARIFY_SYSTEM_PROMPT = """Write one concise conflict-resolution question.

The supplied evidence contains two mutually incompatible claims about the same fact. State both conflicting values and the shared fact they refer to, then ask the user to confirm which one is correct. Treat both as evidence claims of equal status. Do not choose a value, answer the original query, explain reasoning, add a third possibility, or ask more than one question. Return JSON only."""

EDIT_SCHEMA = {
    "constructible": True,
    "skip_reason": "",
    "conflict_target": "",
    "edited_document_id": "",
    "original_claim": "",
    "original_value": "",
    "conflicting_claim": "",
    "conflicting_value": "",
    "necessity_explanation": "",
    "exclusivity_explanation": "",
    "modified_evidence": [{"document_id": "", "title": "", "text": ""}],
}
CLARIFY_SCHEMA = {"assistant_conflict_resolution": ""}


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=root / "core_gold_qwen_v4_output" / "screening" / "core_gold_final.jsonl")
    parser.add_argument("--overflow", type=Path, default=root / "core_gold_qwen_v4_output" / "screening" / "core_gold_overflow.jsonl")
    parser.add_argument("--screening", type=Path, default=root / "core_gold_qwen_v4_output" / "screening" / "core_gold_screening.jsonl")
    parser.add_argument("--config", type=Path, default=root / "evaluation" / "config" / "models.json")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target", type=int, default=0)
    parser.add_argument("--limit-per-dataset", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260825)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--provider")
    parser.add_argument("--base-url")
    parser.add_argument("--model-id")
    parser.add_argument("--api-key-env")
    parser.add_argument("--secrets-file", type=Path, default=root / "evaluation" / "secrets" / ".env.local")
    parser.add_argument("--review-per-dataset", type=int, default=10)
    parser.add_argument("--revalidate-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def request_hash(
    model_id: str,
    prompt_version: str,
    messages: list[dict[str, str]],
    body: dict[str, Any],
) -> str:
    value = {
        "pipeline_version": PIPELINE_VERSION,
        "prompt_version": prompt_version,
        "model": model_id,
        "messages": messages,
        "body": body,
    }
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode()).hexdigest()


def editor_messages(sample: dict[str, Any]) -> list[dict[str, str]]:
    payload = {
        "question": sample["问题"],
        "accepted_answers": sample["答案"],
        "full_evidence": common.compact_evidence(sample),
        "required_output_schema": EDIT_SCHEMA,
    }
    return [
        {"role": "system", "content": EDITOR_SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False, indent=2)},
    ]


def clarification_messages(
    sample: dict[str, Any], edit: dict[str, Any]
) -> list[dict[str, str]]:
    payload = {
        "original_question": sample["问题"],
        "conflict_target": edit["conflict_target"],
        "original_claim": edit["original_claim"],
        "original_value": edit["original_value"],
        "conflicting_claim": edit["conflicting_claim"],
        "conflicting_value": edit["conflicting_value"],
        "required_output_schema": CLARIFY_SCHEMA,
    }
    return [
        {"role": "system", "content": CLARIFY_SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False, indent=2)},
    ]


def normalized_contains(text: str, value: str) -> bool:
    haystack = f" {shared.normalize_answer(text)} "
    needle = shared.normalize_answer(value)
    return bool(needle) and f" {needle} " in haystack


def sentence_count(text: str) -> int:
    return len([part for part in re.split(r"(?<=[.!?])\s+", text.strip()) if part.strip()])


def edit_errors(sample: dict[str, Any], result: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if not isinstance(result.get("constructible"), bool):
        return ["constructible must be boolean"]
    if result["constructible"] is False:
        if not common.clean(result.get("skip_reason")):
            errors.append("unconstructible result has no skip_reason")
        return errors

    required = (
        "conflict_target",
        "edited_document_id",
        "original_claim",
        "original_value",
        "conflicting_claim",
        "conflicting_value",
        "necessity_explanation",
        "exclusivity_explanation",
    )
    for field in required:
        if not common.clean(result.get(field)):
            errors.append(f"missing or empty {field}")

    before = common.compact_evidence(sample)
    after = result.get("modified_evidence")
    if not isinstance(after, list):
        return errors + ["modified_evidence must be a list"]
    if len(after) != len(before):
        return errors + ["modified_evidence document count changed"]

    changed: list[str] = []
    for index, (original, modified) in enumerate(zip(before, after)):
        if not isinstance(modified, dict):
            errors.append(f"modified_evidence[{index}] is not an object")
            continue
        if common.clean(modified.get("document_id")) != original["document_id"]:
            errors.append(f"document ID changed at index {index}")
        if common.clean(modified.get("title")) != original["title"].strip():
            errors.append(f"document title changed at index {index}")
        if common.clean(modified.get("text")) != original["text"].strip():
            changed.append(original["document_id"])

    edited_id = common.clean(result.get("edited_document_id"))
    if len(changed) != 1:
        errors.append(f"expected exactly one changed document, found {len(changed)}")
    elif changed[0] != edited_id:
        errors.append("edited_document_id does not match changed document")

    before_by_id = {doc["document_id"]: doc["text"].strip() for doc in before}
    title_by_id = {doc["document_id"]: doc["title"].strip() for doc in before}
    after_by_id = {
        common.clean(doc.get("document_id")): common.clean(doc.get("text"))
        for doc in after
        if isinstance(doc, dict)
    }
    original_claim = common.clean(result.get("original_claim"))
    original_value = common.clean(result.get("original_value"))
    conflicting_claim = common.clean(result.get("conflicting_claim"))
    conflicting_value = common.clean(result.get("conflicting_value"))

    source_text = before_by_id.get(edited_id, "")
    modified_text = after_by_id.get(edited_id, "")
    if original_claim and original_claim not in source_text:
        errors.append("original_claim absent from original document")
    if original_claim and original_claim not in modified_text:
        errors.append("original_claim was not preserved")
    if conflicting_claim and conflicting_claim in source_text:
        errors.append("conflicting_claim already exists in original document")
    if source_text and modified_text:
        if not modified_text.startswith(source_text):
            errors.append("original document text was changed before the appended claim")
        else:
            suffix = modified_text[len(source_text):]
            if not suffix or not suffix[0].isspace():
                errors.append("conflicting claim is not separated by whitespace")
            if suffix.strip() != conflicting_claim:
                errors.append("document suffix is not exactly conflicting_claim")

    if original_value and not normalized_contains(original_claim, original_value):
        value_tokens = common.word_tokens(original_value)
        claim_tokens = common.word_tokens(original_claim)
        title_support = normalized_contains(title_by_id.get(edited_id, ""), original_value)
        shortened_reference = bool(value_tokens.intersection(claim_tokens))
        if not title_support and not shortened_reference:
            errors.append("original_value is unsupported by original_claim or document title")
    if conflicting_value and not normalized_contains(conflicting_claim, conflicting_value):
        errors.append("conflicting_value absent from conflicting_claim")
    if (
        original_value
        and conflicting_value
        and shared.normalize_answer(original_value)
        == shared.normalize_answer(conflicting_value)
    ):
        errors.append("original_value and conflicting_value are identical")
    for answer in common.accepted_answers(sample):
        if conflicting_value and shared.normalize_answer(conflicting_value) == shared.normalize_answer(answer):
            errors.append("conflicting_value matches an accepted answer")
            break
    for answer in common.accepted_answers(sample):
        if (
            original_claim
            and conflicting_claim
            and normalized_contains(original_claim, answer)
            and normalized_contains(conflicting_claim, answer)
            and shared.normalize_answer(original_value) != shared.normalize_answer(answer)
            and shared.normalize_answer(conflicting_value) != shared.normalize_answer(answer)
        ):
            errors.append("accepted answer remains unchanged in both claims")
            break

    if conflicting_claim and sentence_count(conflicting_claim) != 1:
        errors.append("conflicting_claim must be exactly one sentence")
    if conflicting_claim and common.SURFACE_ARTIFACT.search(conflicting_claim):
        errors.append("conflicting_claim contains a surface artifact")
    if conflicting_claim and re.search(
        r"\b(?:actually|allegedly|possibly|perhaps|reportedly|may have|might have|uncertain)\b",
        conflicting_claim,
        re.IGNORECASE,
    ):
        errors.append("conflicting_claim is hedged or framed as a correction")

    original_anchor = common.word_tokens(original_claim) - common.word_tokens(original_value)
    conflict_anchor = common.word_tokens(conflicting_claim) - common.word_tokens(conflicting_value)
    if original_anchor and conflict_anchor and not original_anchor.intersection(conflict_anchor):
        errors.append("original_claim and conflicting_claim share no subject/relation anchor")
    return errors


def clarification_errors(
    sample: dict[str, Any], result: dict[str, Any], edit: dict[str, Any]
) -> list[str]:
    text = common.clean(result.get("assistant_conflict_resolution"))
    errors: list[str] = []
    if not text:
        return ["assistant_conflict_resolution is empty"]
    if text.count("?") != 1:
        errors.append("assistant_conflict_resolution must ask exactly one question")
    if not normalized_contains(text, common.clean(edit.get("original_value"))):
        errors.append("assistant_conflict_resolution omits original_value")
    if not normalized_contains(text, common.clean(edit.get("conflicting_value"))):
        errors.append("assistant_conflict_resolution omits conflicting_value")
    if not re.search(
        r"\b(?:conflict|contradict|discrep|inconsisten|differ|both|two values|one claim|another claim|one source|another source|while|whereas)\w*\b|\bwhich\b.{0,120}\bcorrect\b",
        text,
        re.IGNORECASE,
    ):
        errors.append("assistant_conflict_resolution does not identify a conflict")
    if re.search(
        r"\b(?:and|or)\s+(?:who|what|which|when|where|how)\b",
        text,
        re.IGNORECASE,
    ):
        errors.append("assistant_conflict_resolution asks multiple information slots")
    return errors


def response_usage(payload: dict[str, Any]) -> dict[str, Any]:
    usage = payload.get("usage") or {}
    return {
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "total_tokens": usage.get("total_tokens"),
    }


def call_stage(
    sample: dict[str, Any],
    runtime: shared.RuntimeModel,
    stage: str,
    messages: list[dict[str, str]],
    validate: Any,
    prompt_version: str,
) -> dict[str, Any]:
    attempts: list[dict[str, Any]] = []
    parsed: dict[str, Any] | None = None
    status = "ERROR"
    final_errors: list[str] = []
    returned_model: str | None = None
    usage: dict[str, Any] = {}
    latency_total = 0.0
    for attempt_number in (1, 2):
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
            usage = response_usage(payload)
            try:
                candidate = common.parse_json_object(raw)
                structure = validate(sample, candidate)
                if not structure:
                    parsed = candidate
                    status = (
                        "SKIPPED"
                        if stage == "EDIT" and candidate["constructible"] is False
                        else "SUCCESS"
                    )
                else:
                    status = "STRUCTURE_ERROR"
                    final_errors = structure
            except Exception as exc:
                parse_error = str(exc)[:1000]
                final_errors = [parse_error]
                status = "PARSE_ERROR"
        except Exception as exc:
            latency_total += time.perf_counter() - started
            api_error = str(exc)[:1200]
            final_errors = [api_error]
            status = "API_ERROR"
        attempts.append(
            {
                "attempt": attempt_number,
                "raw_response": raw,
                "api_error": api_error,
                "parse_error": parse_error,
                "structural_errors": structure,
            }
        )
        if parsed is not None or status == "STRUCTURE_ERROR":
            break

    return {
        "sample_id": sample["样本ID"],
        "dataset": sample["来源"],
        "stage": stage,
        "pipeline_version": PIPELINE_VERSION,
        "prompt_version": prompt_version,
        "model_alias": runtime.alias,
        "requested_model_id": runtime.model_id,
        "returned_model_id": returned_model,
        "provider": runtime.provider,
        "request_body": runtime.request_body,
        "request_hash": request_hash(runtime.model_id, prompt_version, messages, runtime.request_body),
        "messages": messages,
        "attempts": attempts,
        "parsed_response": parsed,
        "status": status,
        "errors": final_errors,
        "usage": usage,
        "latency_seconds": round(latency_total, 4),
        "created_at": common.now_iso(),
    }


def edit_one(sample: dict[str, Any], runtime: shared.RuntimeModel) -> dict[str, Any]:
    messages = editor_messages(sample)
    return call_stage(sample, runtime, "EDIT", messages, edit_errors, EDIT_PROMPT_VERSION)


def clarify_one(
    sample: dict[str, Any],
    edit_call: dict[str, Any],
    runtime: shared.RuntimeModel,
) -> dict[str, Any]:
    edit = edit_call["parsed_response"]
    messages = clarification_messages(sample, edit)

    def validate(sample_value: dict[str, Any], result: dict[str, Any]) -> list[str]:
        return clarification_errors(sample_value, result, edit)

    return call_stage(
        sample,
        runtime,
        "CLARIFICATION",
        messages,
        validate,
        CLARIFY_PROMPT_VERSION,
    )


def revalidate_call(
    sample: dict[str, Any],
    call: dict[str, Any],
    stage: str,
    validate: Any,
) -> dict[str, Any]:
    """Re-parse a saved raw response under the current CONFLICT validators."""
    raw = ""
    for attempt in reversed(call.get("attempts") or []):
        if common.clean(attempt.get("raw_response")):
            raw = common.clean(attempt.get("raw_response"))
            break
    if not raw:
        return call

    updated = dict(call)
    try:
        candidate = common.parse_json_object(raw)
        errors = validate(sample, candidate)
        if errors:
            status = "STRUCTURE_ERROR"
        elif stage == "EDIT" and candidate.get("constructible") is False:
            status = "SKIPPED"
        else:
            status = "SUCCESS"
        updated.update(
            {
                "pipeline_version": PIPELINE_VERSION,
                "parsed_response": candidate,
                "status": status,
                "errors": errors,
                "revalidated_at": common.now_iso(),
            }
        )
    except Exception as exc:
        updated.update(
            {
                "pipeline_version": PIPELINE_VERSION,
                "parsed_response": None,
                "status": "PARSE_ERROR",
                "errors": [str(exc)[:1000]],
                "revalidated_at": common.now_iso(),
            }
        )
    return updated


def build_family(
    sample: dict[str, Any],
    edit_call: dict[str, Any],
    clarify_call: dict[str, Any],
) -> dict[str, Any]:
    edit = edit_call["parsed_response"]
    clarification = clarify_call["parsed_response"]["assistant_conflict_resolution"]
    answers = common.accepted_answers(sample)
    screening = sample["核心黄金筛选"]
    user_resolution = (
        f"For {edit['conflict_target']}, the correct value is "
        f"{edit['original_value']}."
    )
    return {
        "schema_version": "evidence_state_family.v1",
        "family_id": f"{sample['样本ID']}::CONFLICT::v1_2",
        "source_sample_id": sample["样本ID"],
        "original_sample_id": sample.get("原始样本ID"),
        "dataset": sample["来源"],
        "query": sample["问题"],
        "answer": answers[0],
        "answer_aliases": answers[1:],
        "seed_gate": {
            "screening_model": "qwen3_8b",
            "query_only_answer_correct": False,
            "full_answer_correct": True,
            "query_only_stage": screening["stages"]["QUERY_ONLY"]["qwen3_8b"],
            "full_stage": screening["stages"]["CLEAN_FULL"]["qwen3_8b"],
        },
        "full_state": {
            "state": "FULL",
            "evidence": common.compact_evidence(sample),
        },
        "initial_state": {
            "state": "CONFLICT",
            "evidence": edit["modified_evidence"],
            "conflict_target": edit["conflict_target"],
            "original_claim": edit["original_claim"],
            "original_value": edit["original_value"],
            "conflicting_claim": edit["conflicting_claim"],
            "conflicting_value": edit["conflicting_value"],
            "necessity_explanation": edit["necessity_explanation"],
            "exclusivity_explanation": edit["exclusivity_explanation"],
            "minimal_edit": {
                "edit_type": "APPEND_CONFLICTING_CLAIM",
                "document_id": edit["edited_document_id"],
                "appended_text": edit["conflicting_claim"],
            },
        },
        "resolution": {
            "confirmed_claim": edit["original_claim"],
            "confirmed_value": edit["original_value"],
            "user_reply": user_resolution,
        },
        "trajectory": [
            {
                "turn": 1,
                "role": "assistant",
                "action": "REQUEST_CONFLICT_RESOLUTION",
                "content": clarification,
            },
            {
                "turn": 2,
                "role": "user",
                "action": "CONFIRM_INFORMATION",
                "content": user_resolution,
            },
            {
                "turn": 3,
                "role": "assistant",
                "action": "ANSWER",
                "content": answers[0],
            },
        ],
        "constructor": {
            "model_alias": edit_call["model_alias"],
            "model_id": edit_call["requested_model_id"],
            "configuration": edit_call["request_body"],
            "editor": {
                "prompt_version": edit_call["prompt_version"],
                "request_hash": edit_call["request_hash"],
                "raw_attempts": edit_call["attempts"],
                "parsed_response": edit,
            },
            "clarification_writer": {
                "prompt_version": clarify_call["prompt_version"],
                "request_hash": clarify_call["request_hash"],
                "raw_attempts": clarify_call["attempts"],
                "parsed_response": clarify_call["parsed_response"],
            },
        },
    }


def markdown_sample(family: dict[str, Any]) -> str:
    full = "\n\n".join(
        f"[{doc['document_id']}] {doc['title']}\n{doc['text']}"
        for doc in family["full_state"]["evidence"]
    )
    conflict = "\n\n".join(
        f"[{doc['document_id']}] {doc['title']}\n{doc['text']}"
        for doc in family["initial_state"]["evidence"]
    )
    turns = "\n".join(
        f"- {turn['role']} / {turn['action']}: {turn['content']}"
        for turn in family["trajectory"]
    )
    state = family["initial_state"]
    return f"""## {family['family_id']}

- dataset: {family['dataset']}
- query: {family['query']}
- answer: {family['answer']}
- conflict target: {state['conflict_target']}
- original claim: {state['original_claim']}
- conflicting claim: {state['conflicting_claim']}
- original value: {state['original_value']}
- conflicting value: {state['conflicting_value']}

### FULL evidence

{full}

### CONFLICT evidence

{conflict}

### One-resolution trajectory

{turns}
"""


def build_report(
    args: argparse.Namespace,
    rows: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    families: list[dict[str, Any]],
    edit_calls: dict[str, dict[str, Any]],
    clarify_calls: dict[str, dict[str, Any]],
    runtime: shared.RuntimeModel,
) -> dict[str, Any]:
    goal = args.target or len(candidates)
    return {
        "pipeline_version": PIPELINE_VERSION,
        "prompt_versions": {
            "editor": EDIT_PROMPT_VERSION,
            "clarification": CLARIFY_PROMPT_VERSION,
        },
        "source_rows": len(rows),
        "candidate_rows": len(candidates),
        "target": goal,
        "constructed_families": len(families),
        "target_reached": len(families) >= goal,
        "family_dataset_counts": dict(sorted(Counter(x["dataset"] for x in families).items())),
        "edit_status_counts": dict(sorted(Counter(x["status"] for x in edit_calls.values()).items())),
        "clarification_status_counts": dict(sorted(Counter(x["status"] for x in clarify_calls.values()).items())),
        "model": {
            "alias": runtime.alias,
            "model_id": runtime.model_id,
            "provider": runtime.provider,
            "base_url": runtime.base_url,
            "request_body": runtime.request_body,
        },
        "policy": {
            "original_full_evidence_immutable": True,
            "exactly_one_appended_sentence": True,
            "original_claim_preserved": True,
            "same_target_mutually_incompatible_values": True,
            "conflicting_value_cannot_match_accepted_answer": True,
            "assistant_must_present_both_values": True,
            "user_confirms_original_value_deterministically": True,
            "final_answer_is_canonical_and_deterministic": True,
            "qwen_conflict_behavior_used_as_filter": False,
        },
        "created_at": common.now_iso(),
    }


def main() -> int:
    args = parse_args()
    primary = shared.load_jsonl(args.input)
    overflow = shared.load_jsonl(args.overflow) if args.overflow.exists() else []
    screening_rows = shared.load_jsonl(args.screening)
    screening_by_id = {str(row["sample_id"]): row for row in screening_rows}
    primary = [common.attach_screening_gate(row, screening_by_id) for row in primary]
    overflow = [common.attach_screening_gate(row, screening_by_id) for row in overflow]
    combined: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in primary + overflow:
        sid = str(row.get("样本ID") or "")
        if sid not in seen:
            combined.append(row)
            seen.add(sid)

    check = common.preflight(combined)
    if check["errors"]:
        raise ValueError(f"seed preflight failed: {check['errors'][:20]}")
    candidates = (
        common.select_rows(primary, args.limit_per_dataset, args.seed)
        if args.limit_per_dataset > 0
        else combined
    )
    goal = args.target or len(candidates)
    if goal > len(candidates):
        raise ValueError(f"target {goal} exceeds {len(candidates)} candidates")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    common.write_json(
        args.output_dir / "conflict_family_preflight.json",
        {
            **check,
            "primary_rows": len(primary),
            "overflow_rows": len(overflow),
            "selected_candidates": len(candidates),
            "selected_dataset_counts": dict(sorted(Counter(x["来源"] for x in candidates).items())),
            "target": goal,
            "pipeline_version": PIPELINE_VERSION,
        },
    )
    if args.dry_run:
        common.write_jsonl(
            args.output_dir / "conflict_family_editor_prompts.jsonl",
            [
                {"sample_id": row["样本ID"], "dataset": row["来源"], "messages": editor_messages(row)}
                for row in candidates
            ],
        )
        return 0

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
    edit_path = args.output_dir / "conflict_edit_calls.jsonl"
    clarify_path = args.output_dir / "conflict_clarification_calls.jsonl"
    edit_calls = {
        str(row["sample_id"]): row for row in common.read_jsonl_if_exists(edit_path)
    }
    clarify_calls = {
        str(row["sample_id"]): row for row in common.read_jsonl_if_exists(clarify_path)
    }
    sample_by_id = {str(row["样本ID"]): row for row in candidates}
    order = [str(row["样本ID"]) for row in candidates]
    lock = threading.Lock()

    if args.revalidate_existing:
        for sid, old_call in list(edit_calls.items()):
            if sid not in sample_by_id:
                continue
            updated = revalidate_call(
                sample_by_id[sid], old_call, "EDIT", edit_errors
            )
            edit_calls[sid] = updated
            common.append_jsonl(edit_path, updated, lock)
        for sid, old_call in list(clarify_calls.items()):
            if (
                sid not in sample_by_id
                or edit_calls.get(sid, {}).get("status") != "SUCCESS"
            ):
                continue
            edit = edit_calls[sid]["parsed_response"]

            def validate_clarification(
                sample_value: dict[str, Any],
                result: dict[str, Any],
                edit_value: dict[str, Any] = edit,
            ) -> list[str]:
                return clarification_errors(sample_value, result, edit_value)

            updated = revalidate_call(
                sample_by_id[sid],
                old_call,
                "CLARIFICATION",
                validate_clarification,
            )
            clarify_calls[sid] = updated
            common.append_jsonl(clarify_path, updated, lock)

    while True:
        success_ids = [
            sid for sid in order
            if edit_calls.get(sid, {}).get("status") == "SUCCESS"
            and clarify_calls.get(sid, {}).get("status") == "SUCCESS"
        ]
        if len(success_ids) >= goal:
            break
        pending = [
            sample_by_id[sid] for sid in order
            if edit_calls.get(sid, {}).get("status") == "SUCCESS"
            and sid not in clarify_calls
        ]
        if pending:
            def clarification_worker(sample: dict[str, Any]) -> dict[str, Any]:
                return clarify_one(sample, edit_calls[str(sample["样本ID"])], runtime)

            common.run_parallel(
                pending, clarification_worker, workers, clarify_path, clarify_calls, lock
            )
            continue
        unattempted = [sample_by_id[sid] for sid in order if sid not in edit_calls]
        if not unattempted:
            break
        needed = goal - len(success_ids)
        batch_size = min(len(unattempted), max(needed, min(50, max(1, (needed + 9) // 10))))
        common.run_parallel(
            unattempted[:batch_size],
            lambda sample: edit_one(sample, runtime),
            workers,
            edit_path,
            edit_calls,
            lock,
        )

    success_ids = [
        sid for sid in order
        if edit_calls.get(sid, {}).get("status") == "SUCCESS"
        and clarify_calls.get(sid, {}).get("status") == "SUCCESS"
    ][:goal]
    families = [
        build_family(sample_by_id[sid], edit_calls[sid], clarify_calls[sid])
        for sid in success_ids
    ]
    common.write_jsonl(args.output_dir / "conflict_families.jsonl", families)
    common.write_json(
        args.output_dir / "conflict_family_report.json",
        build_report(args, combined, candidates, families, edit_calls, clarify_calls, runtime),
    )

    review_families: list[dict[str, Any]] = []
    review_counts: Counter[str] = Counter()
    for family in families:
        dataset = family["dataset"]
        if review_counts[dataset] < args.review_per_dataset:
            review_families.append(family)
            review_counts[dataset] += 1
    review = "# CONFLICT family append-assertion review\n\n" + "\n\n".join(
        markdown_sample(family) for family in review_families
    )
    (args.output_dir / "conflict_family_review.md").write_text(review + "\n", encoding="utf-8")
    print(
        f"constructed={len(families)} target={goal} "
        f"edit_calls={len(edit_calls)} clarification_calls={len(clarify_calls)}",
        flush=True,
    )
    return 0 if len(families) >= goal else 2


if __name__ == "__main__":
    raise SystemExit(main())
