#!/usr/bin/env python3
"""Causal, parameter-knowledge-clean core-gold filtering.

Default admission uses Qwen3-8B only. DeepSeek remains available through
``--models`` for optional audits. The funnel is deliberately sequential:

1. QUERY_ONLY: discard a sample when any required model answers correctly.
2. CLEAN_FULL: require every required model to answer from supporting evidence.
3. COUNTERFACTUAL: require every required model to follow a type-compatible
   counterfactual answer inserted into a diagnostic copy of the FULL evidence.

Counterfactual copies and model outputs are audit artifacts only. They are never
included as training examples.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import dataclasses
import hashlib
import json
import os
import re
import threading
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable

import evaluate_eplus as api
import validate_missing1_llm as shared


PIPELINE_VERSION = "causal_core_gold.qwen_default.v2"
DEFAULT_MODELS = ("qwen3_8b",)
SUPPORTED_MODELS = ("qwen3_8b", "deepseek_v4_flash", "deepseek_v4_pro")
STAGES = ("QUERY_ONLY", "CLEAN_FULL", "COUNTERFACTUAL")

QUERY_ONLY_SYSTEM = (
    "Answer the user's factual question. Return only the shortest answer span, "
    "without explanation, citations, or introductory words."
)
QUERY_ONLY_USER = "Question:\n{query}\n\nAnswer:"
EVIDENCE_SYSTEM = (
    "Use only the evidence supplied by the user. Do not browse, retrieve, or use "
    "facts remembered from training. Decide whether the evidence uniquely supports "
    "an answer. Return exactly one JSON object with these fields and no markdown: "
    "answerable (boolean), answer (string), missing_description (string), reason "
    "(string). If answerable is true, give the shortest supported answer and leave "
    "missing_description empty. If false, leave answer empty."
)
EVIDENCE_USER = (
    "Question:\n{query}\n\nEvidence:\n{evidence}\n\n"
    "Judge and answer using only the supplied evidence."
)

PSEUDO_WORDS = (
    "Maren", "Valeris", "Corwin", "Selora", "Tavian", "Norwyn", "Elara",
    "Caldren", "Lumeria", "Averon", "Soralis", "Belvar", "Nerissa", "Ordel",
)
MONTHS = (
    "January", "February", "March", "April", "May", "June", "July", "August",
    "September", "October", "November", "December",
)


def load_rows(path: Path) -> list[dict[str, Any]]:
    return shared.load_jsonl(path)


def atomic_json(path: Path, value: Any) -> None:
    shared.write_json(path, value)


def atomic_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    shared.write_jsonl(path, rows)


def answers(row: dict[str, Any]) -> list[str]:
    return [str(value) for value in row.get("答案", []) if str(value).strip()]


def query(row: dict[str, Any]) -> str:
    return str(row.get("问题") or "").strip()


def evidence(row: dict[str, Any]) -> list[dict[str, Any]]:
    return list(row.get("完整上下文") or [])


def sample_id(row: dict[str, Any]) -> str:
    return str(row.get("样本ID") or "")


def dataset(row: dict[str, Any]) -> str:
    return str(row.get("来源") or "")


def normalize(value: Any) -> str:
    return shared.normalize_phrase(value)


def phrase_in_text(phrase: str, text: str) -> bool:
    needle = normalize(phrase)
    haystack = f" {normalize(text)} "
    return bool(needle) and f" {needle} " in haystack


def clean_structure_check(rows: list[dict[str, Any]]) -> dict[str, Any]:
    errors: list[str] = []
    ids: list[str] = []
    for index, row in enumerate(rows, 1):
        sid = sample_id(row)
        ids.append(sid)
        if not sid or not query(row) or not answers(row) or not evidence(row):
            errors.append(f"row {index}: missing ID, query, answers, or evidence")
            continue
        support_ids = {
            str(unit.get("文档ID") or "") for unit in row.get("关键证据单元", [])
        }
        context_ids = {
            str(document.get("文档ID") or "") for document in evidence(row)
        }
        if not support_ids or support_ids != context_ids:
            errors.append(f"{sid}: context is not exactly the supporting-document set")
        if any(not document.get("是否支撑文档") for document in evidence(row)):
            errors.append(f"{sid}: context contains a non-support document")
        if any(phrase_in_text(alias, query(row)) for alias in answers(row)):
            errors.append(f"{sid}: answer alias occurs in query")
    if len(ids) != len(set(ids)):
        errors.append("duplicate sample IDs")
    return {
        "rows": len(rows),
        "error_count": len(errors),
        "errors": errors[:200],
        "passed": not errors,
    }


def deterministic_index(sid: str, offset: int, modulo: int) -> int:
    digest = hashlib.sha256(f"{sid}:{offset}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % modulo


def counterfactual_answer(original: str, sid: str) -> str:
    value = str(original).strip()
    if not value:
        raise ValueError("empty answer")

    month_match = re.search(r"\b(" + "|".join(MONTHS) + r")\b", value, re.I)
    year_match = re.search(r"\b(1[0-9]{3}|20[0-9]{2})\b", value)
    if month_match or year_match:
        result = value
        if month_match:
            current = next(
                index for index, month in enumerate(MONTHS)
                if month.casefold() == month_match.group(1).casefold()
            )
            replacement_month = MONTHS[(current + 5) % 12]
            result = result[:month_match.start()] + replacement_month + result[month_match.end():]
        if year_match:
            year = int(year_match.group(1))
            replacement_year = 1900 + ((year - 1900 + 17) % 126)
            if replacement_year == year:
                replacement_year += 1
            result = re.sub(r"\b" + re.escape(year_match.group(1)) + r"\b", str(replacement_year), result, count=1)
        day_match = re.search(r"\b([12]?[0-9]|3[01])\b", result)
        if day_match and (month_match or year_match):
            day = int(day_match.group(1))
            replacement_day = ((day + 8 - 1) % 28) + 1
            result = result[:day_match.start()] + str(replacement_day) + result[day_match.end():]
        if normalize(result) != normalize(value):
            return result

    if re.fullmatch(r"[+-]?\d+(?:[.,]\d+)?(?:\s*%|\s+[A-Za-z]+)?", value):
        number_match = re.search(r"\d+(?:[.,]\d+)?", value)
        assert number_match is not None
        token = number_match.group(0)
        if "." in token or "," in token:
            separator = "." if "." in token else ","
            whole, fraction = token.split(separator, 1)
            replacement = str(int(whole) + 7) + separator + fraction
        else:
            replacement = str(int(token) + 7)
        return value[:number_match.start()] + replacement + value[number_match.end():]

    word_count = max(1, len(re.findall(r"[\w]+", value, flags=re.UNICODE)))
    chosen: list[str] = []
    for offset in range(word_count):
        candidate = PSEUDO_WORDS[deterministic_index(sid, offset, len(PSEUDO_WORDS))]
        while candidate in chosen:
            candidate = PSEUDO_WORDS[(PSEUDO_WORDS.index(candidate) + 1) % len(PSEUDO_WORDS)]
        chosen.append(candidate)
    if value.isupper() and word_count == 1:
        length = max(2, min(8, len(re.sub(r"\W", "", value))))
        alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        return "".join(alphabet[deterministic_index(sid, index, 26)] for index in range(length))
    return " ".join(chosen)


def replace_alias(text: str, alias: str, replacement: str) -> tuple[str, int]:
    if not alias:
        return text, 0
    pattern = re.compile(r"(?<!\w)" + re.escape(alias) + r"(?!\w)", re.IGNORECASE)
    return pattern.subn(lambda _: replacement, text)


def replace_recursive(value: Any, aliases: list[str], replacement: str) -> tuple[Any, int]:
    if isinstance(value, str):
        output = value
        total = 0
        for alias in sorted(set(aliases), key=len, reverse=True):
            output, count = replace_alias(output, alias, replacement)
            total += count
        return output, total
    if isinstance(value, list):
        output_list = []
        total = 0
        for item in value:
            changed, count = replace_recursive(item, aliases, replacement)
            output_list.append(changed)
            total += count
        return output_list, total
    if isinstance(value, dict):
        output_dict = {}
        total = 0
        for key, item in value.items():
            changed, count = replace_recursive(item, aliases, replacement)
            output_dict[key] = changed
            total += count
        return output_dict, total
    return value, 0


def build_counterfactual(row: dict[str, Any]) -> dict[str, Any]:
    original_answers = answers(row)
    replacement = counterfactual_answer(original_answers[0], sample_id(row))
    changed_evidence, replacements = replace_recursive(
        evidence(row), original_answers, replacement
    )
    rendered = shared.evidence_text(changed_evidence)
    remaining = [alias for alias in original_answers if phrase_in_text(alias, rendered)]
    if replacements <= 0:
        raise ValueError("no answer occurrence was replaced in evidence")
    if remaining:
        raise ValueError(f"original answer aliases remain after replacement: {remaining}")
    if not phrase_in_text(replacement, rendered):
        raise ValueError("counterfactual answer is absent after replacement")
    return {
        "sample_id": sample_id(row),
        "dataset": dataset(row),
        "query": query(row),
        "original_answers": original_answers,
        "counterfactual_answer": replacement,
        "counterfactual_evidence": changed_evidence,
        "replacement_count": replacements,
        "diagnostic_only": True,
    }


def query_messages(row: dict[str, Any]) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": QUERY_ONLY_SYSTEM},
        {"role": "user", "content": QUERY_ONLY_USER.format(query=query(row))},
    ]


def evidence_messages(question: str, documents: list[dict[str, Any]]) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": EVIDENCE_SYSTEM},
        {"role": "user", "content": EVIDENCE_USER.format(
            query=question,
            evidence=shared.render_evidence(documents),
        )},
    ]


def call_identifier(sid: str, model: str, stage: str) -> str:
    return f"{sid}::{model}::{stage}::{PIPELINE_VERSION}"


def request_hash(runtime: api.RuntimeModel, messages: list[dict[str, str]], stage: str) -> str:
    payload = {
        "model": runtime.model_id,
        "messages": messages,
        "body": runtime.request_body,
        "stage": stage,
        "pipeline": PIPELINE_VERSION,
    }
    return hashlib.sha256(json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")).hexdigest()


def evaluate_call(
    row: dict[str, Any],
    runtime: api.RuntimeModel,
    stage: str,
    counterfactual: dict[str, Any] | None,
) -> dict[str, Any]:
    if stage == "QUERY_ONLY":
        messages = query_messages(row)
        expected = answers(row)
        parser: Callable[[str], Any] = api.extract_prediction
        call_input: dict[str, Any] = {"query": query(row)}
    elif stage == "CLEAN_FULL":
        messages = evidence_messages(query(row), evidence(row))
        expected = answers(row)
        parser = shared.parse_model_json
        call_input = {"query": query(row), "evidence": evidence(row)}
    elif stage == "COUNTERFACTUAL" and counterfactual is not None:
        messages = evidence_messages(query(row), counterfactual["counterfactual_evidence"])
        expected = [counterfactual["counterfactual_answer"]]
        parser = shared.parse_model_json
        call_input = {
            "query": query(row),
            "evidence": counterfactual["counterfactual_evidence"],
            "expected_counterfactual_answer": counterfactual["counterfactual_answer"],
        }
    else:
        raise ValueError(stage)

    attempts: list[dict[str, Any]] = []
    parsed: Any = None
    status = "PARSE_ERROR"
    for attempt in (1, 2):
        try:
            payload, latency = api.call_chat(runtime, messages)
            raw = api.response_content(payload)
            parse_error = None
            try:
                parsed = parser(raw)
                if isinstance(parsed, str) and not parsed.strip():
                    raise ValueError("empty answer")
            except Exception as exc:
                parsed = None
                parse_error = str(exc)
            attempts.append({
                "attempt": attempt,
                "raw_response": raw,
                "parsed_response": parsed,
                "parse_error": parse_error,
                "request_error": None,
                "returned_model_id": payload.get("model"),
                "latency_seconds": round(latency, 4),
                "usage": api.usage_fields(payload),
            })
            if parsed is not None:
                status = "SUCCESS"
                break
        except Exception as exc:
            attempts.append({
                "attempt": attempt,
                "raw_response": "",
                "parsed_response": None,
                "parse_error": None,
                "request_error": str(exc)[:1200],
                "returned_model_id": None,
                "latency_seconds": None,
                "usage": {"输入tokens": None, "输出tokens": None, "总tokens": None},
            })

    predicted = ""
    answerable: bool | None = None
    if status == "SUCCESS":
        if stage == "QUERY_ONLY":
            predicted = str(parsed).strip()
        else:
            answerable = bool(parsed["answerable"])
            predicted = str(parsed["answer"]).strip()
    correct = status == "SUCCESS" and shared.strict_answer_match(predicted, expected)
    return {
        "call_id": call_identifier(sample_id(row), runtime.alias, stage),
        "sample_id": sample_id(row),
        "dataset": dataset(row),
        "model_alias": runtime.alias,
        "requested_model_id": runtime.model_id,
        "provider": runtime.provider,
        "base_url": runtime.base_url,
        "stage": stage,
        "pipeline_version": PIPELINE_VERSION,
        "request_body": dict(runtime.request_body),
        "request_hash": request_hash(runtime, messages, stage),
        "input": call_input,
        "attempts": attempts,
        "status": status,
        "parsed_response": parsed,
        "answerable": answerable,
        "predicted_answer": predicted,
        "answer_correct": bool(correct),
    }


def latest_calls(path: Path) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    if path.exists():
        for row in load_rows(path):
            latest[str(row["call_id"])] = row
    return latest


def append_call(path: Path, row: dict[str, Any], lock: threading.Lock) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
    with lock:
        with path.open("a", encoding="utf-8", newline="") as handle:
            handle.write(encoded)
            handle.flush()


def run_stage(
    rows: list[dict[str, Any]],
    models: list[str],
    runtimes: dict[str, api.RuntimeModel],
    stage: str,
    calls_path: Path,
    counterfactuals: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    calls = latest_calls(calls_path)
    tasks: list[tuple[dict[str, Any], api.RuntimeModel, dict[str, Any] | None]] = []
    for row in rows:
        for model in models:
            identifier = call_identifier(sample_id(row), model, stage)
            if calls.get(identifier, {}).get("status") in {"SUCCESS", "PARSE_ERROR"}:
                continue
            tasks.append((row, runtimes[model], counterfactuals.get(sample_id(row))))
    if not tasks:
        return calls
    lock = threading.Lock()
    workers = sum(runtimes[model].workers for model in models)
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(evaluate_call, row, runtime, stage, counterfactual)
            for row, runtime, counterfactual in tasks
        ]
        for completed, future in enumerate(concurrent.futures.as_completed(futures), 1):
            result = future.result()
            append_call(calls_path, result, lock)
            calls[result["call_id"]] = result
            if completed % 25 == 0 or completed == len(tasks):
                print(json.dumps({
                    "stage": stage,
                    "completed_this_run": completed,
                    "required_this_run": len(tasks),
                }, ensure_ascii=False), flush=True)
    return calls


def stage_pass(
    row: dict[str, Any],
    models: list[str],
    stage: str,
    calls: dict[str, dict[str, Any]],
) -> bool:
    stage_calls = [calls[call_identifier(sample_id(row), model, stage)] for model in models]
    if any(call["status"] != "SUCCESS" for call in stage_calls):
        return False
    if stage == "QUERY_ONLY":
        return all(not call["answer_correct"] for call in stage_calls)
    return all(call["answer_correct"] for call in stage_calls)


def runtime_from_config(
    config: dict[str, Any], alias: str, workers: int | None, max_tokens: int
) -> api.RuntimeModel:
    placeholder = argparse.Namespace(
        provider=None, model_id=None, base_url=None, api_key_env=None, workers=workers
    )
    runtime = api.resolve_runtime(config, alias, placeholder)
    body = {
        "temperature": 0,
        "top_p": 1,
        "max_tokens": max_tokens,
        **config["模型"][alias].get("extra_body", {}),
    }
    return dataclasses.replace(runtime, request_body=body)


def screening_rows(
    rows: list[dict[str, Any]],
    models: list[str],
    calls: dict[str, dict[str, Any]],
    counterfactuals: dict[str, dict[str, Any]],
    construction_errors: dict[str, str],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in rows:
        sid = sample_id(row)
        summaries: dict[str, dict[str, Any]] = {}
        for stage in STAGES:
            summaries[stage] = {}
            for model in models:
                call = calls.get(call_identifier(sid, model, stage))
                if call is not None:
                    summaries[stage][model] = {
                        key: call.get(key) for key in (
                            "status", "answerable", "predicted_answer", "answer_correct",
                            "request_hash", "requested_model_id", "provider",
                        )
                    }
        query_clean = bool(summaries["QUERY_ONLY"]) and stage_pass(
            row, models, "QUERY_ONLY", calls
        )
        full_pass = query_clean and bool(summaries["CLEAN_FULL"]) and stage_pass(
            row, models, "CLEAN_FULL", calls
        )
        cf_pass = full_pass and sid in counterfactuals and bool(summaries["COUNTERFACTUAL"]) and stage_pass(
            row, models, "COUNTERFACTUAL", calls
        )
        if not query_clean:
            decision = "REJECT_PARAMETRIC_KNOWLEDGE_OR_QUERY_ERROR"
        elif not full_pass:
            decision = "REJECT_CLEAN_FULL_FAILURE"
        elif sid in construction_errors:
            decision = "REJECT_COUNTERFACTUAL_CONSTRUCTION"
        elif not cf_pass:
            decision = "REJECT_COUNTERFACTUAL_NOT_FOLLOWED"
        else:
            decision = "CORE_QUALIFIED"
        output.append({
            "sample_id": sid,
            "dataset": dataset(row),
            "query": query(row),
            "answers": answers(row),
            "required_models": models,
            "query_only_knowledge_clean": query_clean,
            "clean_full_pass": full_pass,
            "counterfactual_follow_pass": cf_pass,
            "counterfactual_answer": counterfactuals.get(sid, {}).get("counterfactual_answer"),
            "counterfactual_construction_error": construction_errors.get(sid),
            "decision": decision,
            "stages": summaries,
        })
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--secrets-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--models", nargs="+", choices=SUPPORTED_MODELS, default=list(DEFAULT_MODELS))
    parser.add_argument("--workers", type=int)
    parser.add_argument("--max-tokens", type=int, default=384)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--target", type=int, default=2000)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()

    rows = load_rows(args.input)
    if args.limit > 0:
        rows = rows[:args.limit]
    structure = clean_structure_check(rows)
    args.output.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output / "core_gold_preflight.json", structure)
    if not structure["passed"]:
        raise RuntimeError(f"preflight failed: {structure['errors'][:20]}")
    if args.prepare_only:
        print(json.dumps(structure, ensure_ascii=False))
        return 0

    api.load_secrets_file(args.secrets_file)
    config = api.load_json(args.config)
    runtimes = {
        model: runtime_from_config(config, model, args.workers, args.max_tokens)
        for model in args.models
    }
    calls_path = args.output / "core_gold_calls.jsonl"
    empty_counterfactuals: dict[str, dict[str, Any]] = {}

    calls = run_stage(
        rows, args.models, runtimes, "QUERY_ONLY", calls_path, empty_counterfactuals
    )
    query_clean = [row for row in rows if stage_pass(row, args.models, "QUERY_ONLY", calls)]
    calls = run_stage(
        query_clean, args.models, runtimes, "CLEAN_FULL", calls_path, empty_counterfactuals
    )
    full_pass = [row for row in query_clean if stage_pass(row, args.models, "CLEAN_FULL", calls)]

    counterfactuals: dict[str, dict[str, Any]] = {}
    construction_errors: dict[str, str] = {}
    for row in full_pass:
        try:
            counterfactuals[sample_id(row)] = build_counterfactual(row)
        except ValueError as exc:
            construction_errors[sample_id(row)] = str(exc)
    atomic_jsonl(
        args.output / "core_gold_counterfactuals.jsonl",
        [counterfactuals[sample_id(row)] for row in full_pass if sample_id(row) in counterfactuals],
    )
    counterfactual_rows = [row for row in full_pass if sample_id(row) in counterfactuals]
    calls = run_stage(
        counterfactual_rows, args.models, runtimes, "COUNTERFACTUAL", calls_path, counterfactuals
    )

    screenings = screening_rows(
        rows, args.models, calls, counterfactuals, construction_errors
    )
    atomic_jsonl(args.output / "core_gold_screening.jsonl", screenings)
    screening_by_id = {row["sample_id"]: row for row in screenings}
    qualified = [row for row in rows if screening_by_id[sample_id(row)]["decision"] == "CORE_QUALIFIED"]
    qualified.sort(key=lambda row: hashlib.sha256(
        f"{PIPELINE_VERSION}:{sample_id(row)}".encode("utf-8")
    ).hexdigest())
    final_rows = qualified[:args.target] if args.target > 0 else qualified
    overflow_rows = qualified[len(final_rows):]
    for row in final_rows:
        row["模式版本"] = "causal_core_gold.zh.v2"
        row["验证"].update({
            "模型已检查": True,
            "模型回答正确": True,
            "参数知识已检查": True,
            "参数知识命中": False,
            "反事实已检查": True,
            "反事实跟随": True,
        })
        row["元数据"].update({
            "核心黄金筛选版本": PIPELINE_VERSION,
            "核心黄金筛选模型": list(args.models),
            "参数知识探针": "query_only_direct_answer",
            "反事实策略": "type_compatible_answer_substitution",
        })
        row["核心黄金筛选"] = screening_by_id[sample_id(row)]
    atomic_jsonl(args.output / "core_gold_final.jsonl", final_rows)
    atomic_jsonl(args.output / "core_gold_overflow.jsonl", overflow_rows)
    atomic_jsonl(
        args.output / "core_gold_rejected.jsonl",
        [row for row in screenings if row["decision"] != "CORE_QUALIFIED"],
    )

    decision_counts = Counter(row["decision"] for row in screenings)
    by_dataset: dict[str, Any] = {}
    for name, items in defaultdict(list, {
        name: [row for row in screenings if row["dataset"] == name]
        for name in sorted({row["dataset"] for row in screenings})
    }).items():
        by_dataset[name] = {
            "candidates": len(items),
            "query_knowledge_clean": sum(row["query_only_knowledge_clean"] for row in items),
            "clean_full_pass": sum(row["clean_full_pass"] for row in items),
            "counterfactual_follow_pass": sum(row["counterfactual_follow_pass"] for row in items),
        }
    report = {
        "stage": "causal parameter-knowledge-clean core gold filtering",
        "pipeline_version": PIPELINE_VERSION,
        "required_models": list(args.models),
        "default_policy": "Qwen3-8B only; DeepSeek interfaces retained for optional audit",
        "candidate_rows": len(rows),
        "query_knowledge_clean": len(query_clean),
        "clean_full_pass": len(full_pass),
        "counterfactual_constructible": len(counterfactuals),
        "qualified_total": len(qualified),
        "target": args.target,
        "final_rows": len(final_rows),
        "target_reached": args.target <= 0 or len(final_rows) >= args.target,
        "overflow_rows": len(overflow_rows),
        "decision_counts": dict(decision_counts),
        "by_dataset": by_dataset,
        "runtime": {
            model: {
                "model_id": runtime.model_id,
                "provider": runtime.provider,
                "base_url": runtime.base_url,
                "request_body": runtime.request_body,
            }
            for model, runtime in runtimes.items()
        },
        "scope_notes": [
            "The final set is screened against the required models only; parameter knowledge is model-specific.",
            "Counterfactual evidence is diagnostic-only and is not training data.",
            "No distractor documents are included in candidate or final FULL evidence.",
        ],
    }
    atomic_json(args.output / "core_gold_report.json", report)
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
