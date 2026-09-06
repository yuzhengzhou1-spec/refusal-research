#!/usr/bin/env python3
"""Type-aware v3 compatibility layer for :mod:`filter_core_gold`.

The underlying v2 QUERY_ONLY and CLEAN_FULL prompts are unchanged and retain
their request IDs. Only COUNTERFACTUAL uses the v3 request namespace, so a v2
pilot can be upgraded without repeating already-paid calls.
"""

from __future__ import annotations

import concurrent.futures
import hashlib
import json
import re
import threading
from typing import Any

import evaluate_eplus as api
import filter_core_gold as base


PIPELINE_VERSION = "causal_core_gold.qwen_default.v3"
V2_VERSION = "causal_core_gold.qwen_default.v2"


def version_for_stage(stage: str) -> str:
    return PIPELINE_VERSION if stage == "COUNTERFACTUAL" else V2_VERSION


def call_identifier(sid: str, model: str, stage: str) -> str:
    return f"{sid}::{model}::{stage}::{version_for_stage(stage)}"


def request_hash(
    runtime: api.RuntimeModel, messages: list[dict[str, str]], stage: str
) -> str:
    payload = {
        "model": runtime.model_id,
        "messages": messages,
        "body": runtime.request_body,
        "stage": stage,
        "pipeline": version_for_stage(stage),
    }
    return hashlib.sha256(json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")).hexdigest()


def semantic_type(question_text: str, answer_text: str) -> str:
    question = base.normalize(question_text)
    answer = base.normalize(answer_text)
    if re.search(r"\b(1[0-9]{3}|20[0-9]{2})\b", answer_text) or any(
        month.casefold() in answer_text.casefold() for month in base.MONTHS
    ):
        return "DATE"
    if re.fullmatch(r"[+-]?\d[\d.,]*(?:\s*%|\s+[A-Za-z]+)?", answer_text.strip()):
        return "NUMBER"
    if any(token in question.split() for token in ("who", "whom", "whose")):
        return "PERSON"
    if "language" in question or "nationality" in question:
        return "LANGUAGE_OR_NATIONALITY"
    if any(token in question for token in (
        "which country", "what country", "which city", "what city", "where ",
        " location", "which place", "what place",
    )):
        return "LOCATION"
    if any(token in question for token in (
        "political party", "organization", "organisation", "association",
        "community", "company", "institution", "university",
    )):
        return "ORGANIZATION"
    if any(token in question for token in (
        "film", "movie", "album", "book", "novel", "song", "television",
        "show", "series", "game",
    )):
        return "WORK"
    if answer in {"yes", "no"}:
        return "BOOLEAN"
    return "OTHER"


def transform_numbers(value: str, is_date: bool) -> str:
    def replace(match: re.Match[str]) -> str:
        token = match.group(0)
        if "," in token and len(token.split(",", 1)[1]) == 3:
            return f"{int(token.replace(',', '')) + 7:,}"
        if "." in token or "," in token:
            separator = "." if "." in token else ","
            whole, fraction = token.split(separator, 1)
            return str(int(whole) + 7) + separator + fraction
        number = int(token)
        if is_date and 1000 <= number <= 2099:
            return str(1900 + ((number - 1900 + 17) % 200))
        if is_date and 1 <= number <= 31:
            return str(((number + 8 - 1) % 28) + 1)
        return str(number + 7)

    return re.sub(r"\d+(?:[.,]\d+)?", replace, value)


def counterfactual_answer(original: str, sid: str, question_text: str = "") -> str:
    value = str(original).strip()
    if not value:
        raise ValueError("empty answer")
    kind = semantic_type(question_text, value)
    if kind == "DATE":
        result = value
        month_match = re.search(
            r"\b(" + "|".join(base.MONTHS) + r")\b", result, re.IGNORECASE
        )
        if month_match:
            current = next(
                index for index, month in enumerate(base.MONTHS)
                if month.casefold() == month_match.group(1).casefold()
            )
            replacement = base.MONTHS[(current + 5) % 12]
            result = result[:month_match.start()] + replacement + result[month_match.end():]
        return transform_numbers(result, is_date=True)
    if kind == "NUMBER":
        return transform_numbers(value, is_date=False)
    typed = {
        "PERSON": "Maren Valeris",
        "LANGUAGE_OR_NATIONALITY": "Lumerian",
        "LOCATION": "Selora",
        "ORGANIZATION": "Averon Society",
        "WORK": "The Lumerian Chronicle",
    }
    if kind in typed:
        return typed[kind]
    return base.counterfactual_answer(value, sid)


def build_counterfactual(row: dict[str, Any]) -> dict[str, Any]:
    original_answers = base.answers(row)
    kind = semantic_type(base.query(row), original_answers[0])
    replacement = counterfactual_answer(
        original_answers[0], base.sample_id(row), base.query(row)
    )
    changed_evidence, replacements = base.replace_recursive(
        base.evidence(row), original_answers, replacement
    )
    rendered = base.shared.evidence_text(changed_evidence)
    remaining = [
        alias for alias in original_answers if base.phrase_in_text(alias, rendered)
    ]
    if replacements <= 0:
        raise ValueError("no answer occurrence was replaced in evidence")
    if remaining:
        raise ValueError(f"original answer aliases remain after replacement: {remaining}")
    if not base.phrase_in_text(replacement, rendered):
        raise ValueError("counterfactual answer is absent after replacement")
    return {
        "sample_id": base.sample_id(row),
        "dataset": base.dataset(row),
        "query": base.query(row),
        "original_answers": original_answers,
        "counterfactual_answer": replacement,
        "semantic_type": kind,
        "replacement_strategy": "type_aware_format_preserving_v3",
        "counterfactual_evidence": changed_evidence,
        "replacement_count": replacements,
        "diagnostic_only": True,
    }


def run_stage(
    rows: list[dict[str, Any]],
    models: list[str],
    runtimes: dict[str, api.RuntimeModel],
    stage: str,
    calls_path: Any,
    counterfactuals: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    calls = base.latest_calls(calls_path)
    tasks: list[tuple[dict[str, Any], api.RuntimeModel, dict[str, Any] | None]] = []
    for row in rows:
        for model in models:
            identifier = call_identifier(base.sample_id(row), model, stage)
            counterfactual = counterfactuals.get(base.sample_id(row))
            if stage == "QUERY_ONLY":
                messages = base.query_messages(row)
            elif stage == "CLEAN_FULL":
                messages = base.evidence_messages(base.query(row), base.evidence(row))
            elif stage == "COUNTERFACTUAL" and counterfactual is not None:
                messages = base.evidence_messages(
                    base.query(row), counterfactual["counterfactual_evidence"]
                )
            else:
                raise ValueError(f"missing counterfactual for {base.sample_id(row)}")
            expected_hash = request_hash(runtimes[model], messages, stage)
            existing = calls.get(identifier, {})
            if (
                existing.get("status") in {"SUCCESS", "PARSE_ERROR"}
                and existing.get("request_hash") == expected_hash
            ):
                continue
            tasks.append((row, runtimes[model], counterfactual))
    if not tasks:
        return calls
    lock = threading.Lock()
    workers = sum(runtimes[model].workers for model in models)
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(base.evaluate_call, row, runtime, stage, counterfactual)
            for row, runtime, counterfactual in tasks
        ]
        for completed, future in enumerate(concurrent.futures.as_completed(futures), 1):
            result = future.result()
            base.append_call(calls_path, result, lock)
            calls[result["call_id"]] = result
            if completed % 25 == 0 or completed == len(tasks):
                print(json.dumps({
                    "stage": stage,
                    "completed_this_run": completed,
                    "required_this_run": len(tasks),
                }, ensure_ascii=False), flush=True)
    return calls


def activate() -> None:
    base.PIPELINE_VERSION = PIPELINE_VERSION
    base.call_identifier = call_identifier
    base.request_hash = request_hash
    base.counterfactual_answer = counterfactual_answer
    base.build_counterfactual = build_counterfactual
    base.run_stage = run_stage


activate()

# Re-export the patched public surface for the verifier and tests.
for _name in dir(base):
    if not _name.startswith("_") and _name not in globals():
        globals()[_name] = getattr(base, _name)


if __name__ == "__main__":
    raise SystemExit(base.main())
