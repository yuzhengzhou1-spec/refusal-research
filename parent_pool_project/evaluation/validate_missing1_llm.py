#!/usr/bin/env python3
"""Dual-model, three-condition validity validation for MISSING-1 pilot data.

The validator is read-only with respect to FULL and MISSING-1 source data. It
does not construct perturbations or generate training responses. API calls are
independent across models and conditions, resumable, and preserve every raw
response attempt.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import copy
import dataclasses
import hashlib
import json
import os
import re
import threading
import unicodedata
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import evaluate_eplus as api


PROMPT_VERSION = "missing1_closed_context_validity.v1"
CONDITIONS = ("FULL", "MISSING", "RESTORED")
MODEL_ALIASES = ("deepseek_v4_flash", "qwen3_8b")
REQUIRED_FIELDS = {
    "pair_id",
    "source_sample_id",
    "dataset",
    "query",
    "answer",
    "answer_aliases",
    "full_evidence",
    "insufficient_evidence",
    "removed_evidence",
    "missing_subtype",
}
SYSTEM_PROMPT = (
    "You are validating closed-context question answering. Use only the evidence "
    "included in the user message. Do not browse, retrieve information, or use "
    "facts remembered from training. Decide whether the evidence uniquely and "
    "reliably supports an answer to the question. If it does, provide the shortest "
    "supported answer. If it does not, do not guess an answer; state what fact or "
    "reasoning link is missing. Return exactly one JSON object with these four "
    "fields and no markdown: answerable (boolean), answer (string), "
    "missing_description (string), reason (string). When answerable is false, "
    "answer must be an empty string. When answerable is true, "
    "missing_description must be an empty string."
)
USER_TEMPLATE = (
    "Question:\n{query}\n\n"
    "Evidence:\n{evidence}\n\n"
    "Determine whether the evidence is sufficient to answer the question uniquely "
    "and reliably. Use only this evidence. Return the required JSON object."
)
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "being", "by", "did",
    "do", "does", "for", "from", "had", "has", "have", "he", "her", "hers",
    "him", "his", "how", "i", "in", "into", "is", "it", "its", "of", "on",
    "or", "she", "that", "the", "their", "them", "there", "these", "they",
    "this", "those", "to", "was", "were", "what", "when", "where", "which",
    "who", "whom", "whose", "why", "with", "would", "you", "your", "evidence",
    "information", "question", "answer", "fact", "facts", "provided", "given",
    "cannot", "determine", "insufficient", "missing", "needed", "reliably",
}


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected object")
            rows.append(row)
    return rows


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    os.replace(temporary, path)


def append_jsonl(path: Path, row: dict[str, Any], lock: threading.Lock) -> None:
    line = json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
    with lock:
        with path.open("a", encoding="utf-8", newline="") as handle:
            handle.write(line)
            handle.flush()


def render_evidence(evidence: list[dict[str, Any]]) -> str:
    if not evidence:
        return "(no evidence provided)"
    blocks: list[str] = []
    for index, document in enumerate(evidence, 1):
        title = str(document.get("标题") or document.get("title") or "").strip()
        text = str(document.get("文本") or document.get("text") or "").strip()
        header = f"[Document {index}]"
        if title:
            header += f" {title}"
        blocks.append(f"{header}\n{text}")
    return "\n\n".join(blocks)


def build_messages(query: str, evidence: list[dict[str, Any]]) -> list[dict[str, str]]:
    user = USER_TEMPLATE.format(query=query, evidence=render_evidence(evidence))
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def strip_json_fence(text: str) -> str:
    value = text.strip()
    if value.startswith("```"):
        value = value[3:].lstrip()
        if value.lower().startswith("json"):
            value = value[4:].lstrip()
        if value.endswith("```"):
            value = value[:-3].rstrip()
    return value


def parse_model_json(raw: str) -> dict[str, Any]:
    parsed = json.loads(strip_json_fence(raw))
    if not isinstance(parsed, dict):
        raise ValueError("top-level JSON is not an object")
    required = {"answerable", "answer", "missing_description", "reason"}
    if set(parsed) != required:
        raise ValueError(f"JSON fields must be exactly {sorted(required)}")
    if not isinstance(parsed["answerable"], bool):
        raise ValueError("answerable is not boolean")
    for field in ("answer", "missing_description", "reason"):
        if not isinstance(parsed[field], str):
            raise ValueError(f"{field} is not string")
    return {
        "answerable": parsed["answerable"],
        "answer": parsed["answer"].strip(),
        "missing_description": parsed["missing_description"].strip(),
        "reason": parsed["reason"].strip(),
    }


def normalize_phrase(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).casefold()
    text = re.sub(r"(?<=\d),(?=\d)", "", text)
    text = text.replace("'", "").replace("’", "")
    text = "".join(
        " " if unicodedata.category(character).startswith(("P", "S")) else character
        for character in text
    )
    return " ".join(token for token in text.split() if token not in {"a", "an", "the"})


def strict_answer_match(prediction: str, answers: list[str]) -> bool:
    normalized = normalize_phrase(prediction)
    return bool(normalized) and any(normalized == normalize_phrase(answer) for answer in answers)


def evidence_text(evidence: list[dict[str, Any]]) -> str:
    return " ".join(
        f"{document.get('标题', '')} {document.get('文本', '')}" for document in evidence
    )


def phrase_in_evidence(phrase: str, evidence: list[dict[str, Any]]) -> bool:
    needle = normalize_phrase(phrase)
    haystack = f" {normalize_phrase(evidence_text(evidence))} "
    return bool(needle) and f" {needle} " in haystack


def restore_evidence(row: dict[str, Any]) -> list[dict[str, Any]]:
    """Reinsert the removed sentence/paragraph using its exact FULL position."""
    full = row["full_evidence"]
    missing = copy.deepcopy(row["insufficient_evidence"])
    removed = row["removed_evidence"]
    document_id = str(removed.get("document_id") or "")
    full_matches = [
        index for index, document in enumerate(full)
        if str(document.get("文档ID") or "") == document_id
    ]
    if len(full_matches) != 1:
        raise ValueError(f"{row['pair_id']}: removed document is not unique in FULL")
    full_index = full_matches[0]
    full_document = copy.deepcopy(full[full_index])
    if removed.get("granularity") == "paragraph" or removed.get("sentence_id") is None:
        if any(str(document.get("文档ID") or "") == document_id for document in missing):
            raise ValueError(f"{row['pair_id']}: removed paragraph still exists")
        missing.insert(full_index, full_document)
    else:
        missing_matches = [
            index for index, document in enumerate(missing)
            if str(document.get("文档ID") or "") == document_id
        ]
        if len(missing_matches) != 1:
            raise ValueError(f"{row['pair_id']}: sentence document missing or duplicated")
        missing[missing_matches[0]] = full_document
    return missing


def structure_check(rows: list[dict[str, Any]]) -> dict[str, Any]:
    errors: list[str] = []
    pair_ids: list[str] = []
    source_ids: list[str] = []
    for index, row in enumerate(rows, 1):
        missing_fields = sorted(REQUIRED_FIELDS - set(row))
        if missing_fields:
            errors.append(f"row {index}: missing fields {missing_fields}")
            continue
        pair_ids.append(str(row["pair_id"]))
        source_ids.append(str(row["source_sample_id"]))
        try:
            restored = restore_evidence(row)
        except ValueError as exc:
            errors.append(str(exc))
            continue
        if restored != row["full_evidence"]:
            errors.append(f"{row['pair_id']}: restored evidence differs from FULL")
        if row.get("state") != "MISSING":
            errors.append(f"{row['pair_id']}: state is not MISSING")
        if row.get("missing_subtype") not in {"MISSING_ANSWER", "MISSING_BRIDGE"}:
            errors.append(f"{row['pair_id']}: invalid missing_subtype")
    if len(pair_ids) != len(set(pair_ids)):
        errors.append("duplicate pair_id")
    if len(source_ids) != len(set(source_ids)):
        errors.append("duplicate source_sample_id")
    if len(rows) != 256:
        errors.append(f"expected 256 rows, found {len(rows)}")
    return {"rows": len(rows), "error_count": len(errors), "errors": errors, "passed": not errors}


def condition_evidence(row: dict[str, Any], condition: str) -> list[dict[str, Any]]:
    if condition == "FULL":
        return row["full_evidence"]
    if condition == "MISSING":
        return row["insufficient_evidence"]
    if condition == "RESTORED":
        return restore_evidence(row)
    raise ValueError(condition)


def call_id(pair_id: str, model_alias: str, condition: str) -> str:
    return f"{pair_id}::{model_alias}::{condition}::{PROMPT_VERSION}"


def request_digest(model_id: str, messages: list[dict[str, str]], body: dict[str, Any]) -> str:
    canonical = json.dumps(
        {"model": model_id, "messages": messages, "body": body, "prompt": PROMPT_VERSION},
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def evaluate_condition(
    row: dict[str, Any],
    runtime: api.RuntimeModel,
    condition: str,
) -> dict[str, Any]:
    evidence = condition_evidence(row, condition)
    messages = build_messages(row["query"], evidence)
    attempts: list[dict[str, Any]] = []
    parsed: dict[str, Any] | None = None
    final_status = "PARSE_ERROR"
    for parse_attempt in (1, 2):
        attempted_at = now_iso()
        try:
            payload, latency = api.call_chat(runtime, messages)
            raw = api.response_content(payload)
            parse_error: str | None = None
            try:
                parsed = parse_model_json(raw)
            except (json.JSONDecodeError, ValueError) as exc:
                parse_error = str(exc)
            attempts.append({
                "attempt": parse_attempt,
                "requested_at": attempted_at,
                "returned_model_id": payload.get("model"),
                "raw_response": raw,
                "parsed_response": parsed,
                "parse_error": parse_error,
                "request_error": None,
                "latency_seconds": round(latency, 4),
                "usage": api.usage_fields(payload),
            })
            if parsed is not None:
                final_status = "SUCCESS"
                break
        except Exception as exc:
            attempts.append({
                "attempt": parse_attempt,
                "requested_at": attempted_at,
                "returned_model_id": None,
                "raw_response": "",
                "parsed_response": None,
                "parse_error": None,
                "request_error": str(exc)[:1200],
                "latency_seconds": None,
                "usage": {"输入tokens": None, "输出tokens": None, "总tokens": None},
            })
    body_without_secret = dict(runtime.request_body)
    return {
        "call_id": call_id(row["pair_id"], runtime.alias, condition),
        "pair_id": row["pair_id"],
        "source_sample_id": row["source_sample_id"],
        "dataset": row["dataset"],
        "missing_subtype": row["missing_subtype"],
        "condition": condition,
        "model_alias": runtime.alias,
        "requested_model_id": runtime.model_id,
        "provider": runtime.provider,
        "base_url": runtime.base_url,
        "prompt_version": PROMPT_VERSION,
        "request_body": body_without_secret,
        "request_hash": request_digest(runtime.model_id, messages, body_without_secret),
        "input": {"query": row["query"], "evidence": evidence},
        "attempts": attempts,
        "status": final_status,
        "parsed_response": parsed,
    }


def latest_calls(path: Path) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return latest
    for row in load_jsonl(path):
        latest[str(row["call_id"])] = row
    return latest


def run_calls(
    rows: list[dict[str, Any]],
    runtimes: dict[str, api.RuntimeModel],
    calls_path: Path,
) -> dict[str, dict[str, Any]]:
    existing = latest_calls(calls_path)
    tasks: list[tuple[dict[str, Any], api.RuntimeModel, str]] = []
    for row in rows:
        for alias in MODEL_ALIASES:
            for condition in CONDITIONS:
                identifier = call_id(row["pair_id"], alias, condition)
                if existing.get(identifier, {}).get("status") in {"SUCCESS", "PARSE_ERROR"}:
                    continue
                tasks.append((row, runtimes[alias], condition))
    if not tasks:
        return existing

    lock = threading.Lock()
    workers = sum(runtime.workers for runtime in runtimes.values())
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(evaluate_condition, row, runtime, condition): (
                row["pair_id"], runtime.alias, condition
            )
            for row, runtime, condition in tasks
        }
        for completed, future in enumerate(concurrent.futures.as_completed(futures), 1):
            result = future.result()
            append_jsonl(calls_path, result, lock)
            existing[result["call_id"]] = result
            if completed % 25 == 0 or completed == len(tasks):
                print(json.dumps({
                    "completed_this_run": completed,
                    "pending_this_run": len(tasks),
                    "total_required": len(rows) * len(MODEL_ALIASES) * len(CONDITIONS),
                }, ensure_ascii=False), flush=True)
    return existing


def words(value: str) -> set[str]:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    tokens = set(re.findall(r"[\w]+", normalized, flags=re.UNICODE))
    return {
        token for token in tokens
        if (len(token) >= 3 or token.isdigit()) and token not in STOPWORDS
    }


def gap_match(
    row: dict[str, Any],
    parsed: dict[str, Any] | None,
) -> dict[str, Any]:
    if parsed is None:
        return {"label": "GAP_MISMATCH", "key_tokens": [], "matched_tokens": []}
    description = parsed.get("missing_description", "")
    if not description:
        return {"label": "GAP_MISMATCH", "key_tokens": [], "matched_tokens": []}
    removed_text = str(row["removed_evidence"].get("text") or "")
    removed_tokens = words(removed_text)
    query_tokens = words(row["query"])
    answer_tokens = words(" ".join([row["answer"], *row["answer_aliases"]]))
    remaining_tokens = words(evidence_text(row["insufficient_evidence"]))
    if row["missing_subtype"] == "MISSING_ANSWER":
        key_tokens = answer_tokens | (removed_tokens - query_tokens)
    else:
        bridge_anchors = (removed_tokens & remaining_tokens) - query_tokens - answer_tokens
        key_tokens = bridge_anchors or (removed_tokens - query_tokens - answer_tokens)
    description_tokens = words(description)
    matched = sorted(key_tokens & description_tokens)
    answer_mentioned = any(
        normalize_phrase(alias)
        and normalize_phrase(alias) in normalize_phrase(description)
        for alias in [row["answer"], *row["answer_aliases"]]
    )
    if answer_mentioned or len(matched) >= 2:
        label = "GAP_MATCH"
    elif len(matched) == 1:
        label = "GAP_PARTIAL"
    else:
        label = "GAP_MISMATCH"
    return {
        "label": label,
        "key_tokens": sorted(key_tokens),
        "matched_tokens": matched,
    }


def condition_summary(
    row: dict[str, Any],
    call: dict[str, Any],
    condition: str,
) -> dict[str, Any]:
    parsed = call.get("parsed_response")
    answers = [row["answer"], *row["answer_aliases"]]
    success = call.get("status") == "SUCCESS" and isinstance(parsed, dict)
    answerable = parsed.get("answerable") if success else None
    answer = parsed.get("answer", "") if success else ""
    correct = strict_answer_match(answer, answers) if success else False
    result = {
        "status": call.get("status"),
        "answerable": answerable,
        "answer": answer,
        "answer_correct": correct,
        "abstained": bool(success and answerable is False and not answer),
        "missing_description": parsed.get("missing_description", "") if success else "",
        "reason": parsed.get("reason", "") if success else "",
        "parsed_response": parsed,
        "raw_attempts": call.get("attempts", []),
        "request_hash": call.get("request_hash"),
        "requested_model_id": call.get("requested_model_id"),
        "provider": call.get("provider"),
        "prompt_version": call.get("prompt_version"),
        "request_body": call.get("request_body"),
    }
    if condition == "MISSING":
        result["gap_match"] = gap_match(row, parsed)
    return result


def deterministic_support_check(row: dict[str, Any]) -> dict[str, Any]:
    aliases = [row["answer"], *row["answer_aliases"]]
    answer_aliases_present = [
        alias for alias in aliases if phrase_in_evidence(alias, row["insufficient_evidence"])
    ]
    removed_text = str(row["removed_evidence"].get("text") or "")
    removed_text_present = phrase_in_evidence(removed_text, row["insufficient_evidence"])
    complete_support_detected = bool(
        removed_text_present
        or (
            row["missing_subtype"] == "MISSING_ANSWER"
            and answer_aliases_present
        )
    )
    return {
        "removed_text_still_present": removed_text_present,
        "answer_aliases_still_present": answer_aliases_present,
        "complete_support_detected": complete_support_detected,
        "policy": (
            "For MISSING_ANSWER, a full answer alias remaining in titles/text is direct support. "
            "For MISSING_BRIDGE, answer presence alone is not complete support because the link may be missing. "
            "Exact removed information remaining is always support."
        ),
    }


def classify_sample(
    row: dict[str, Any],
    model_results: dict[str, dict[str, dict[str, Any]]],
    restored_equal: bool,
) -> tuple[str, list[str]]:
    reasons: list[str] = []
    all_conditions = [
        model_results[alias][condition]
        for alias in MODEL_ALIASES
        for condition in CONDITIONS
    ]
    if any(item["status"] != "SUCCESS" for item in all_conditions):
        return "PARSE_ERROR", ["at least one condition has no parseable JSON after two attempts"]

    full_ok = all(
        model_results[alias]["FULL"]["answerable"] is True
        and model_results[alias]["FULL"]["answer_correct"]
        for alias in MODEL_ALIASES
    )
    restored_ok = restored_equal and all(
        model_results[alias]["RESTORED"]["answerable"] is True
        and model_results[alias]["RESTORED"]["answer_correct"]
        for alias in MODEL_ALIASES
    )
    if not full_ok:
        reasons.append("at least one model failed the FULL baseline")
    if not restored_equal:
        reasons.append("restored_evidence is not exactly equal to full_evidence")
    if not restored_ok:
        reasons.append("at least one model failed after restoration")
    if not full_ok or not restored_ok:
        return "FAIL_RESTORATION", reasons

    support = deterministic_support_check(row)
    if support["complete_support_detected"]:
        return "FAIL_DELETION_INEFFECTIVE", ["deterministic remaining-evidence check found complete support"]

    missing_abstained = all(
        model_results[alias]["MISSING"]["abstained"] for alias in MODEL_ALIASES
    )
    if missing_abstained:
        return "PASS_STRICT", []
    return "PASS_LOGICAL_BUT_MODEL_GUESSED", [
        "FULL and RESTORED passed; at least one model answered under MISSING without deterministic complete support"
    ]


def build_validation_rows(
    rows: list[dict[str, Any]],
    calls: dict[str, dict[str, Any]],
    runtime_metadata: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in rows:
        restored = restore_evidence(row)
        restored_equal = restored == row["full_evidence"]
        model_results: dict[str, dict[str, dict[str, Any]]] = {}
        for alias in MODEL_ALIASES:
            model_results[alias] = {}
            for condition in CONDITIONS:
                identifier = call_id(row["pair_id"], alias, condition)
                if identifier not in calls:
                    raise RuntimeError(f"missing completed call: {identifier}")
                model_results[alias][condition] = condition_summary(
                    row, calls[identifier], condition
                )
        status, status_reasons = classify_sample(row, model_results, restored_equal)
        model_agreement = all(
            model_results[MODEL_ALIASES[0]][condition]["answerable"]
            == model_results[MODEL_ALIASES[1]][condition]["answerable"]
            and model_results[MODEL_ALIASES[0]][condition]["answer_correct"]
            == model_results[MODEL_ALIASES[1]][condition]["answer_correct"]
            for condition in CONDITIONS
        )
        output.append({
            "pair_id": row["pair_id"],
            "source_sample_id": row["source_sample_id"],
            "split": row.get("split"),
            "dataset": row["dataset"],
            "missing_subtype": row["missing_subtype"],
            "query": row["query"],
            "answer": row["answer"],
            "answer_aliases": row["answer_aliases"],
            "full_evidence": row["full_evidence"],
            "insufficient_evidence": row["insufficient_evidence"],
            "removed_evidence": row["removed_evidence"],
            "restored_evidence": restored,
            "restored_equals_full": restored_equal,
            "deterministic_support_check": deterministic_support_check(row),
            "models": model_results,
            "model_runtime_config": runtime_metadata,
            "dual_model_agreement": model_agreement,
            "overall_status": status,
            "status_reasons": status_reasons,
        })
    return output


def ratio(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 6) if denominator else 0.0


def build_report(
    rows: list[dict[str, Any]],
    structure: dict[str, Any],
    runtime_metadata: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    status_counts = Counter(row["overall_status"] for row in rows)
    dataset_report: dict[str, Any] = {}
    for dataset in ("hotpotqa", "musique", "nq"):
        items = [row for row in rows if row["dataset"] == dataset]
        passed = sum(row["overall_status"] == "PASS_STRICT" for row in items)
        dataset_report[dataset] = {
            "total": len(items),
            "pass_strict": passed,
            "pass_strict_rate": ratio(passed, len(items)),
        }
    subtype_report: dict[str, Any] = {}
    for subtype in ("MISSING_ANSWER", "MISSING_BRIDGE"):
        items = [row for row in rows if row["missing_subtype"] == subtype]
        passed = sum(row["overall_status"] == "PASS_STRICT" for row in items)
        subtype_report[subtype] = {
            "total": len(items),
            "pass_strict": passed,
            "pass_strict_rate": ratio(passed, len(items)),
        }
    model_report: dict[str, Any] = {}
    gap_counts_total: Counter[str] = Counter()
    for alias in MODEL_ALIASES:
        parsed_missing = [
            row["models"][alias]["MISSING"]
            for row in rows
            if row["models"][alias]["MISSING"]["status"] == "SUCCESS"
        ]
        gap_counts = Counter(
            item["gap_match"]["label"] for item in parsed_missing
        )
        gap_counts_total.update(gap_counts)
        model_report[alias] = {
            "model_id": runtime_metadata[alias]["model_id"],
            "missing_parsed": len(parsed_missing),
            "missing_abstained": sum(item["abstained"] for item in parsed_missing),
            "missing_abstention_rate": ratio(
                sum(item["abstained"] for item in parsed_missing), len(parsed_missing)
            ),
            "gap_counts": {
                label: gap_counts[label]
                for label in ("GAP_MATCH", "GAP_PARTIAL", "GAP_MISMATCH")
            },
        }
    agreement = sum(row["dual_model_agreement"] for row in rows)
    report = {
        "generated_at": now_iso(),
        "stage": "MISSING-1 dual-model validity validation",
        "prompt_version": PROMPT_VERSION,
        "total_samples": len(rows),
        "structure_check": structure,
        "models": runtime_metadata,
        "status_counts": {
            status: status_counts[status]
            for status in (
                "PASS_STRICT",
                "PASS_LOGICAL_BUT_MODEL_GUESSED",
                "FAIL_DELETION_INEFFECTIVE",
                "FAIL_RESTORATION",
                "PARSE_ERROR",
            )
        },
        "pass_strict_by_dataset": dataset_report,
        "pass_strict_by_subtype": subtype_report,
        "model_missing_abstention": model_report,
        "dual_model_agreement_count": agreement,
        "dual_model_agreement_rate": ratio(agreement, len(rows)),
        "gap_counts_all_model_outputs": {
            label: gap_counts_total[label]
            for label in ("GAP_MATCH", "GAP_PARTIAL", "GAP_MISMATCH")
        },
        "parameter_memory_or_guess_samples": status_counts["PASS_LOGICAL_BUT_MODEL_GUESSED"],
        "deletion_ineffective_samples": status_counts["FAIL_DELETION_INEFFECTIVE"],
        "restoration_failure_samples": status_counts["FAIL_RESTORATION"],
        "parse_error_samples": status_counts["PARSE_ERROR"],
        "classification_notes": {
            "guessed_vs_ineffective": (
                "A correct MISSING answer is labeled guessed when no deterministic complete support "
                "is present. MISSING_ANSWER is deletion-ineffective if an answer alias remains in "
                "evidence; MISSING_BRIDGE does not fail merely because the final answer string remains."
            ),
            "gap_matching": "Rule-based token/entity/relation overlap only; wording differences do not remove samples.",
        },
        "scope": {
            "llm_calls_only_for_validation": True,
            "source_data_modified": False,
            "new_perturbations_generated": False,
            "training_or_multiturn_data_generated": False,
        },
    }
    return report


def raw_outputs_for_markdown(row: dict[str, Any], alias: str, condition: str) -> str:
    attempts = row["models"][alias][condition]["raw_attempts"]
    return "\n\n".join(
        f"Attempt {attempt['attempt']}:\n{attempt.get('raw_response') or attempt.get('request_error') or ''}"
        for attempt in attempts
    )


def markdown_case(row: dict[str, Any], number: int) -> list[str]:
    lines = [
        f"### {number}. {row['pair_id']}",
        "",
        f"- dataset: `{row['dataset']}`",
        f"- subtype: `{row['missing_subtype']}`",
        f"- status: `{row['overall_status']}`",
        f"- dual-model agreement: `{str(row['dual_model_agreement']).lower()}`",
        f"- query: {row['query']}",
        f"- reference answer: {row['answer']}",
        f"- removed evidence: {row['removed_evidence'].get('text', '')}",
        "",
    ]
    for condition, evidence_key in (
        ("FULL", "full_evidence"),
        ("MISSING", "insufficient_evidence"),
        ("RESTORED", "restored_evidence"),
    ):
        lines.extend([
            f"#### {condition}",
            "",
            "Input evidence:",
            "",
            "```json",
            json.dumps(row[evidence_key], ensure_ascii=False, indent=2),
            "```",
            "",
        ])
        for alias in MODEL_ALIASES:
            lines.extend([
                f"{alias} raw output:",
                "",
                "```text",
                raw_outputs_for_markdown(row, alias, condition),
                "```",
                "",
            ])
    return lines


def build_manual_review(rows: list[dict[str, Any]]) -> str:
    sections: list[tuple[str, list[dict[str, Any]]]] = []
    strict_examples: list[dict[str, Any]] = []
    for dataset in ("hotpotqa", "musique", "nq"):
        candidates = [
            row for row in rows
            if row["dataset"] == dataset and row["overall_status"] == "PASS_STRICT"
        ]
        candidates.sort(key=lambda row: hashlib.sha256(row["pair_id"].encode("utf-8")).hexdigest())
        strict_examples.extend(candidates[:5])
    sections.append(("PASS_STRICT examples (up to 5 per dataset)", strict_examples))
    sections.append((
        "All FAIL_DELETION_INEFFECTIVE cases",
        [row for row in rows if row["overall_status"] == "FAIL_DELETION_INEFFECTIVE"],
    ))
    sections.append((
        "All FAIL_RESTORATION cases",
        [row for row in rows if row["overall_status"] == "FAIL_RESTORATION"],
    ))
    inconsistent = [row for row in rows if not row["dual_model_agreement"]]
    sections.append(("Dual-model disagreements (at least 10 when available)", inconsistent[:max(10, len(inconsistent))]))

    lines = [
        "# MISSING-1 dual-model manual review",
        "",
        "This document contains FULL, MISSING, and RESTORED evidence plus raw outputs from both models.",
        "No review sample is deleted automatically.",
        "",
    ]
    case_number = 0
    for title, items in sections:
        lines.extend([f"## {title}", "", f"Cases: {len(items)}", ""])
        for row in items:
            case_number += 1
            lines.extend(markdown_case(row, case_number))
    return "\n".join(lines)


def runtime_from_config(
    config: dict[str, Any],
    alias: str,
    args: argparse.Namespace,
) -> api.RuntimeModel:
    placeholder = argparse.Namespace(
        provider=None,
        model_id=None,
        base_url=None,
        api_key_env=None,
        workers=args.workers,
    )
    runtime = api.resolve_runtime(config, alias, placeholder)
    shared_body = {
        "temperature": 0,
        "top_p": 1,
        "max_tokens": args.max_tokens,
        **config["模型"][alias].get("extra_body", {}),
    }
    return dataclasses.replace(runtime, request_body=shared_body)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--secrets-file", type=Path, required=True)
    parser.add_argument("--max-tokens", type=int, default=384)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()

    rows = load_jsonl(args.input)
    if args.limit > 0:
        rows = rows[:args.limit]
    structure = structure_check(rows) if args.limit == 0 else {
        **structure_check(rows),
        "note": "smoke subset; expected-row-count check omitted",
    }
    if args.limit > 0:
        structure["errors"] = [
            error for error in structure["errors"] if not error.startswith("expected 256 rows")
        ]
        structure["error_count"] = len(structure["errors"])
        structure["passed"] = not structure["errors"]
    if not structure["passed"]:
        raise RuntimeError(f"structure check failed: {structure['errors'][:10]}")

    args.output.mkdir(parents=True, exist_ok=True)
    write_json(args.output / "missing1_preflight.json", structure)
    if args.prepare_only:
        print(json.dumps(structure, ensure_ascii=False))
        return 0

    api.load_secrets_file(args.secrets_file)
    config = api.load_json(args.config)
    runtimes = {
        alias: runtime_from_config(config, alias, args) for alias in MODEL_ALIASES
    }
    runtime_metadata = {
        alias: {
            "alias": alias,
            "model_id": runtime.model_id,
            "provider": runtime.provider,
            "base_url": runtime.base_url,
            "prompt_version": PROMPT_VERSION,
            "system_prompt": SYSTEM_PROMPT,
            "temperature": runtime.request_body["temperature"],
            "top_p": runtime.request_body["top_p"],
            "max_tokens": runtime.request_body["max_tokens"],
            "provider_specific_body": {
                key: value for key, value in runtime.request_body.items()
                if key not in {"temperature", "top_p", "max_tokens"}
            },
            "workers": runtime.workers,
            "timeout_seconds": runtime.timeout_seconds,
            "transport_max_retries": runtime.max_retries,
            "parse_retry_limit": 1,
        }
        for alias, runtime in runtimes.items()
    }
    calls = run_calls(rows, runtimes, args.output / "missing1_llm_calls.jsonl")
    validation_rows = build_validation_rows(rows, calls, runtime_metadata)
    write_jsonl(args.output / "missing1_llm_validation.jsonl", validation_rows)
    pass_rows = [row for row in validation_rows if row["overall_status"] == "PASS_STRICT"]
    review_rows = [row for row in validation_rows if row["overall_status"] != "PASS_STRICT"]
    write_jsonl(args.output / "missing1_validation_pass.jsonl", pass_rows)
    write_jsonl(args.output / "missing1_validation_review.jsonl", review_rows)
    report = build_report(validation_rows, structure, runtime_metadata)
    write_json(args.output / "missing1_validation_report.json", report)
    (args.output / "missing1_manual_review.md").write_text(
        build_manual_review(validation_rows), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
