#!/usr/bin/env python3
"""Build v2.3 MISSING evidence-state families with split editor/clarifier calls."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import re
import threading
import time
from collections import Counter, defaultdict
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any

from evaluation import evaluate_eplus as shared


PIPELINE_VERSION = "missing_family.split_responsibility.v2_5_2"
EDIT_PROMPT_VERSION = "missing_family.editor.answer_free_gap.v2_2"
CLARIFY_PROMPT_VERSION = "missing_family.clarification.grounded_gap.v3_3_2"
MODEL_ALIAS = "deepseek_v4_pro"

EDITOR_SYSTEM_PROMPT = """Create one simple MISSING-information evidence variant. Your only job is to edit evidence. Do not generate dialogue and do not explain hidden reasoning.

1. Identify one complete evidence claim, relation, value, or condition that is necessary to answer the question under all of its stated conditions.
2. Make only that information unavailable, using one small local edit in exactly one evidence document.
3. Delete the supporting fact, relation, attribute, value, or clause. Do not merely erase an answer string while retaining fragments that can be joined with another document to recover the same fully conditioned claim. The edited document must remain grammatical and natural.
4. Preserve all unrelated text and preserve every document ID, title, count, and order.
5. Information written in the question is not evidence and must not be treated as evidential support.
6. The remaining evidence may still mention the answer entity, a question entity, or partial related facts. This is allowed. The edit is valid when the remaining evidence no longer independently supports every condition needed for the answer.
7. Do not remove other mentions merely because they resemble or contain the answer. Do not clean all answer occurrences.
8. Before returning, inspect all remaining documents together. If an unchanged statement plus the retained part of the edited statement can reconstruct the same subject-relation-value and all query conditions, the deletion is ineffective. Remove the complete local relation clause instead, or set constructible to false.
   In particular, deleting only an award/result predicate is ineffective when the edited evidence still links the subject to the accepted event and another remaining document satisfies the event descriptors in the question. If two documents independently expose complementary parts that still identify the answer and one-document editing cannot break both, set constructible to false.
9. Do not replace the removed claim with a vague or generalized sentence that still supplies the same relation or condition.
10. Return clarification_target as a short answer-free description of the information slot to request. It must not contain any accepted answer, alias, or missing value. Name only the known entity and required relation/attribute, for example "the release year of Film X".
11. Return restoring_information as one complete factual sentence directly and explicitly supported by the original edited document. It must be equivalent to the removed source claim, not a stronger inference. A building being in a city does not prove the city is a county seat; an institution being in a country does not prove its namesake is from that country.
12. Do not add entities, distractors, ambiguity, contradiction, a second gap, or any fact not explicitly present in FULL evidence.

If FULL evidence does not explicitly contain the needed claim, or no one-claim, one-document minimal edit can satisfy these rules, set constructible to false. Return JSON only, with no markdown or commentary."""

CLARIFY_SYSTEM_PROMPT = """Write one concise, evidence-aware clarification for a fixed answer-free information target.

First decide whether remaining_evidence already directly and uniquely supplies clarification_target. Treat the original question only as a request, never as evidence. If the target is already resolved, set target_resolved_by_remaining_evidence to true, explain why in resolution_reason, leave all support and clarification fields empty, and stop.

Otherwise select exactly one relevant, answer-free support span from remaining_evidence. Return its document ID and copy the shortest complete supporting clause or sentence verbatim as known_support_span. The span must occur exactly in that document's title or text. The clarification's known-support statement may paraphrase only this span; it must not import a date, entity, condition, or relation merely from original_question.

The response must contain three connected parts:
1. Briefly state one relevant fact or clue that the remaining context does establish and that advances the original question. Mention only information explicitly present in remaining_evidence, and keep it answer-free.
2. Contrast that known support with the precise unresolved slot in clarification_target. Explain that the current context does not establish, identify, or specify that relation/value. Do not claim that all information is absent when partial related information remains.
3. Ask exactly one focused question requesting that unresolved relation/value. Preserve the target's known entity, relation, attribute, time, and other conditions.

Prefer a natural pattern such as: "The context establishes [relevant known clue], but it does not identify [answer-free target]. Could you provide [the needed relation/value]?"

The remaining context may still mention a person, place, organization, date, number, title, or other value that could be the original answer or could fill clarification_target. Never copy such a candidate value into the known-support summary. Instead summarize an orthogonal clue one step earlier in the chain. For example, say that the context identifies the relevant film or role, not the remaining candidate person's name.

A retained list may be incomplete. Never say "which of those", "which of these", "which listed person", or otherwise assert that the missing value must be one of the visible list items. Refer neutrally to the unresolved person, item, or relation.

Do not merely restate the original question. Do not reveal or guess the accepted answer. Do not answer the original question, mention deletion/editing/withholding of evidence, invent a clue, request multiple information slots, or ask the user to resend the whole context. Prefer two sentences and at most 70 words. Return JSON only."""

EDIT_SCHEMA = {
    "constructible": True,
    "skip_reason": "",
    "clarification_target": "",
    "restoring_information": "",
    "edit_type": "REMOVE_RELATION",
    "edited_document_id": "",
    "original_span": "",
    "replacement_span": "",
    "modified_evidence": [{"document_id": "", "title": "", "text": ""}],
}
CLARIFY_SCHEMA = {
    "target_resolved_by_remaining_evidence": False,
    "resolution_reason": "",
    "known_support_document_id": "",
    "known_support_span": "",
    "assistant_clarification": "",
}

SURFACE_ARTIFACT = re.compile(
    r"\b(?:and|or|of|with|like)\s*[.!?](?:\s|$)|[.!?]{2,}|\s{3,}", re.IGNORECASE
)


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=root / "core_gold_qwen_v4_output" / "screening" / "core_gold_final.jsonl")
    parser.add_argument("--overflow", type=Path, default=root / "core_gold_qwen_v4_output" / "screening" / "core_gold_overflow.jsonl")
    parser.add_argument("--screening", type=Path, default=root / "core_gold_qwen_v4_output" / "screening" / "core_gold_screening.jsonl")
    parser.add_argument("--config", type=Path, default=root / "evaluation" / "config" / "models.json")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target", type=int, default=0, help="0 means construct from every selected seed")
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
    parser.add_argument("--retry-api-errors", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def stable_key(seed: int, sample_id: str) -> str:
    return hashlib.sha256(f"{seed}:{sample_id}".encode()).hexdigest()


def read_jsonl_if_exists(path: Path) -> list[dict[str, Any]]:
    return shared.load_jsonl(path) if path.exists() else []


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def append_jsonl(path: Path, row: dict[str, Any], lock: threading.Lock) -> None:
    line = json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
    with lock, path.open("a", encoding="utf-8", newline="") as handle:
        handle.write(line)
        handle.flush()


def drop_retryable_api_errors(calls: dict[str, dict[str, Any]]) -> list[str]:
    retry_ids = [
        sid for sid, call in calls.items() if call.get("status") == "API_ERROR"
    ]
    for sid in retry_ids:
        del calls[sid]
    return retry_ids


def select_rows(rows: list[dict[str, Any]], limit: int, seed: int) -> list[dict[str, Any]]:
    if limit <= 0:
        return rows
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row["来源"])].append(row)
    selected: list[dict[str, Any]] = []
    for dataset in sorted(groups):
        ordered = sorted(groups[dataset], key=lambda x: stable_key(seed, str(x["样本ID"])))
        selected.extend(ordered[:limit])
    return selected


def preflight(rows: list[dict[str, Any]]) -> dict[str, Any]:
    errors: list[str] = []
    seen: set[str] = set()
    counts: Counter[str] = Counter()
    for row in rows:
        sid = str(row.get("样本ID") or "")
        counts[str(row.get("来源") or "")] += 1
        if not sid:
            errors.append("<missing-id>: missing 样本ID")
        elif sid in seen:
            errors.append(f"{sid}: duplicate sample ID")
        seen.add(sid)
        for field in ("来源", "问题", "答案", "完整上下文", "核心黄金筛选"):
            if not row.get(field):
                errors.append(f"{sid}: missing {field}")
        gate = row.get("核心黄金筛选") or {}
        if gate.get("query_only_knowledge_clean") is not True:
            errors.append(f"{sid}: query-only gate is not clean")
        if gate.get("clean_full_pass") is not True:
            errors.append(f"{sid}: FULL gate did not pass")
        if gate.get("required_models") != ["qwen3_8b"]:
            errors.append(f"{sid}: unexpected required_models")
    return {
        "rows": len(rows),
        "dataset_counts": dict(sorted(counts.items())),
        "passed": not errors,
        "errors": errors,
    }


def attach_screening_gate(
    row: dict[str, Any], screening_by_id: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    if row.get("核心黄金筛选"):
        return row
    sid = str(row.get("样本ID") or "")
    record = screening_by_id.get(sid)
    if not record or record.get("decision") != "CORE_QUALIFIED":
        return row
    enriched = dict(row)
    enriched["核心黄金筛选"] = {
        "pipeline_version": "causal_core_gold.qwen_default.v4",
        "required_models": record.get("required_models"),
        "query_only_knowledge_clean": record.get("query_only_knowledge_clean"),
        "clean_full_pass": record.get("clean_full_pass"),
        "counterfactual_follow_pass": record.get("counterfactual_follow_pass"),
        "counterfactual_answer": record.get("counterfactual_answer"),
        "decision": record.get("decision"),
        "stages": record.get("stages"),
        "metadata_rehydrated_from_screening": True,
    }
    return enriched


def compact_evidence(sample: dict[str, Any]) -> list[dict[str, str]]:
    return [
        {
            "document_id": str(doc.get("文档ID") or ""),
            "title": str(doc.get("标题") or ""),
            "text": str(doc.get("文本") or ""),
        }
        for doc in sample["完整上下文"]
    ]


def editor_messages(sample: dict[str, Any]) -> list[dict[str, str]]:
    payload = {
        "question": sample["问题"],
        "accepted_answers": sample["答案"],
        "full_evidence": compact_evidence(sample),
        "required_output_schema": EDIT_SCHEMA,
    }
    return [
        {"role": "system", "content": EDITOR_SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False, indent=2)},
    ]


def clarification_safe_evidence(
    sample: dict[str, Any], edit: dict[str, Any]
) -> list[dict[str, str]]:
    """Hide answer-bearing documents from the clarifier-only evidence view."""
    safe: list[dict[str, str]] = []
    for document in edit["modified_evidence"]:
        title = clean(document.get("title"))
        text = clean(document.get("text"))
        if (
            potential_answer_mention(title, sample)
            or potential_answer_mention(text, sample)
        ):
            title = ""
            text = ""
        safe.append(
            {
                "document_id": clean(document.get("document_id")),
                "title": title,
                "text": text,
            }
        )
    return safe


def clarification_messages(sample: dict[str, Any], edit: dict[str, Any]) -> list[dict[str, str]]:
    payload = {
        "original_question": sample["问题"],
        "clarification_target": edit["clarification_target"],
        "remaining_evidence": clarification_safe_evidence(sample, edit),
        "required_output_schema": CLARIFY_SCHEMA,
    }
    return [
        {"role": "system", "content": CLARIFY_SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False, indent=2)},
    ]


def parse_json_object(raw: str) -> dict[str, Any]:
    text = raw.strip()
    fence = chr(96) * 3
    if text.startswith(fence):
        text = text[3:].lstrip()
        if text.lower().startswith("json"):
            text = text[4:].lstrip()
        if text.endswith(fence):
            text = text[:-3].rstrip()
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            raise
        value = json.loads(text[start:end + 1])
    if not isinstance(value, dict):
        raise ValueError("response is not a JSON object")
    return value


def clean(value: Any) -> str:
    return str(value or "").strip()


def response_usage(payload: dict[str, Any]) -> dict[str, Any]:
    usage = payload.get("usage") or {}
    return {
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "total_tokens": usage.get("total_tokens"),
    }


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


def accepted_answers(sample: dict[str, Any]) -> list[str]:
    return [clean(x) for x in sample["答案"] if clean(x)]


def answer_leaked(text: str, sample: dict[str, Any]) -> bool:
    normalized_text = f" {shared.normalize_answer(text)} "
    return any(
        bool(shared.normalize_answer(answer))
        and f" {shared.normalize_answer(answer)} " in normalized_text
        for answer in accepted_answers(sample)
    )


def potential_answer_mention(text: str, sample: dict[str, Any]) -> bool:
    """Conservatively catch multi-token answers separated by a middle token."""
    if answer_leaked(text, sample):
        return True
    text_tokens = re.findall(r"[0-9a-zÀ-ž]+", shared.normalize_answer(text))
    for answer in accepted_answers(sample):
        answer_tokens = re.findall(
            r"[0-9a-zÀ-ž]+", shared.normalize_answer(answer)
        )
        if len(answer_tokens) < 2:
            continue
        previous = -1
        matched = True
        for token in answer_tokens:
            try:
                position = text_tokens.index(token, previous + 1)
            except ValueError:
                matched = False
                break
            if previous >= 0 and position - previous - 1 > 2:
                matched = False
                break
            previous = position
        if matched:
            return True
    return False


CONTENT_STOPWORDS = {
    "a", "an", "the", "of", "to", "in", "on", "at", "for", "and", "or",
    "is", "was", "were", "are", "be", "being", "that", "which", "who", "what",
    "when", "where", "whether", "information", "identity", "name", "exact",
}
RELATION_ALIASES = {
    "direct": {"director", "directed", "directs"},
    "own": {"owner", "owned", "owns", "ownership"},
    "release": {"release", "released"},
    "form": {"formed", "formation"},
    "found": {"founded", "founder"},
    "locate": {"located", "location"},
    "seat": {"seat"},
    "birth": {"born", "birth"},
    "death": {"died", "death"},
    "population": {"population"},
    "date": {"date"},
    "year": {"year"},
    "member": {"member", "membership"},
    "captain": {"captain", "captained"},
    "headquarters": {"headquarters", "headquartered"},
    "employ": {"employ", "employed", "employer"},
    "publish": {"publisher", "published"},
    "sibling": {"sibling"},
    "minister": {"minister"},
    "origin": {"origin"},
}


def word_tokens(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[0-9A-Za-zÀ-ž]+", text.lower())
        if token not in CONTENT_STOPWORDS
    }


def relation_tags(text: str) -> set[str]:
    tokens = set(re.findall(r"[0-9A-Za-zÀ-ž]+", text.lower()))
    return {
        tag
        for tag, aliases in RELATION_ALIASES.items()
        if tokens.intersection(aliases)
    }


def has_repeated_sentence(text: str) -> bool:
    seen: set[str] = set()
    for sentence in re.split(r"(?<=[.!?])\s+", text):
        normalized = " ".join(re.findall(r"[0-9A-Za-zÀ-ž]+", sentence.lower()))
        if len(normalized.split()) < 4:
            continue
        if normalized in seen:
            return True
        seen.add(normalized)
    return False


def edit_errors(sample: dict[str, Any], result: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if not isinstance(result.get("constructible"), bool):
        return ["constructible must be boolean"]
    if result["constructible"] is False:
        if not clean(result.get("skip_reason")):
            errors.append("unconstructible result has no skip_reason")
        return errors

    for field in (
        "clarification_target",
        "restoring_information",
        "edit_type",
        "edited_document_id",
        "original_span",
    ):
        if not clean(result.get(field)):
            errors.append(f"missing or empty {field}")
    target = clean(result.get("clarification_target"))
    if target and answer_leaked(target, sample):
        errors.append("clarification_target contains an accepted answer")
    if result.get("edit_type") not in {"REMOVE_FACT", "REMOVE_VALUE", "REMOVE_RELATION"}:
        errors.append("invalid edit_type")

    before = compact_evidence(sample)
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
        if clean(modified.get("document_id")) != original["document_id"]:
            errors.append(f"document ID changed at index {index}")
        if clean(modified.get("title")) != original["title"].strip():
            errors.append(f"document title changed at index {index}")
        modified_text = clean(modified.get("text"))
        if modified_text != original["text"].strip():
            changed.append(original["document_id"])
            if SURFACE_ARTIFACT.search(modified_text):
                errors.append(f"surface artifact in changed document {original['document_id']}")
            if has_repeated_sentence(modified_text):
                errors.append(f"repeated sentence in changed document {original['document_id']}")

    edited_id = clean(result.get("edited_document_id"))
    if len(changed) != 1:
        errors.append(f"expected exactly one changed document, found {len(changed)}")
    elif changed[0] != edited_id:
        errors.append("edited_document_id does not match changed document")

    before_by_id = {doc["document_id"]: doc["text"] for doc in before}
    after_by_id = {
        clean(doc.get("document_id")): clean(doc.get("text"))
        for doc in after
        if isinstance(doc, dict)
    }
    original_span = clean(result.get("original_span"))
    replacement_span = clean(result.get("replacement_span"))
    if edited_id in before_by_id and original_span not in before_by_id[edited_id]:
        errors.append("original_span absent from original document")
    if edited_id in after_by_id and replacement_span and replacement_span not in after_by_id[edited_id]:
        errors.append("replacement_span absent from modified document")
    if original_span == replacement_span:
        errors.append("original_span and replacement_span are identical")
    if clean(result.get("restoring_information")) == replacement_span:
        errors.append("restoring_information is identical to replacement_span")
    restoring = clean(result.get("restoring_information"))
    target_relations = relation_tags(target)
    restoring_relations = relation_tags(restoring)
    source_relations = relation_tags(original_span)
    # Only relations with a known high-risk entailment gap are checked lexically.
    # General relation paraphrases (e.g. "1981 single" -> "release year") are
    # too varied for a sound word-level validator and remain prompt-controlled.
    target_origin = bool(
        re.search(r"\borigin\b|\bcountry\b.{0,60}\bfrom\b", target, re.IGNORECASE)
    )
    restoring_origin = bool(
        re.search(r"\b(?:origin|from)\b", restoring, re.IGNORECASE)
    )
    source_origin = bool(
        re.search(r"\b(?:origin|from)\b", original_span, re.IGNORECASE)
    )
    sensitive_relations = target_relations.intersection({"seat"})
    if sensitive_relations - restoring_relations:
        errors.append("restoring_information does not express a sensitive target relation")
    if sensitive_relations - source_relations:
        errors.append("sensitive target relation is absent from original_span")
    if target_origin and not restoring_origin:
        errors.append("restoring_information does not express the origin relation")
    if target_origin and not source_origin:
        errors.append("origin relation is absent from original_span")
    if (
        re.search(r"\bevent\b", target, re.IGNORECASE)
        and replacement_span
        and answer_leaked(replacement_span, sample)
    ):
        errors.append("accepted event remains explicitly identified in replacement_span")
    return errors


def clarification_errors(
    sample: dict[str, Any], result: dict[str, Any], edit: dict[str, Any]
) -> list[str]:
    if not isinstance(result, dict):
        return ["clarification response is not an object"]
    resolved = result.get("target_resolved_by_remaining_evidence")
    errors: list[str] = []
    if not isinstance(resolved, bool):
        errors.append("target_resolved_by_remaining_evidence must be boolean")
        resolved = False
    text = clean(result.get("assistant_clarification"))
    reason = clean(result.get("resolution_reason"))
    support_document_id = clean(result.get("known_support_document_id"))
    support_span = clean(result.get("known_support_span"))
    if resolved:
        if not reason:
            errors.append("resolved target has no resolution_reason")
        if support_document_id or support_span or text:
            errors.append("resolved target must not include support or clarification")
        return errors
    if reason:
        errors.append("unresolved target must have empty resolution_reason")
    if not support_document_id:
        errors.append("known_support_document_id is empty")
    if not support_span:
        errors.append("known_support_span is empty")
    safe_by_id = {
        document["document_id"]: document
        for document in clarification_safe_evidence(sample, edit)
    }
    support_document = safe_by_id.get(support_document_id)
    if support_document_id and support_document is None:
        errors.append("known_support_document_id is absent from remaining_evidence")
    elif support_span and support_document is not None:
        if (
            support_span not in support_document["title"]
            and support_span not in support_document["text"]
        ):
            errors.append("known_support_span is not an exact remaining-evidence span")
        if potential_answer_mention(support_span, sample):
            errors.append("known_support_span contains a potential accepted answer")
    if not text:
        errors.append("assistant_clarification is empty")
    if text and potential_answer_mention(text, sample):
        errors.append("assistant_clarification leaks an accepted answer")
    if text and text.count("?") != 1:
        errors.append("assistant_clarification must ask exactly one question")
    if text and re.search(
        r"\b(?:and|or)\s+(?:who|what|which|when|where|how)\b",
        text,
        re.IGNORECASE,
    ):
        errors.append("assistant_clarification asks multiple information slots")
    if text and re.search(
        r"\b(?:which|one)\s+of\s+(?:those|these|the\s+listed)\b"
        r"|\bwhich\s+listed\s+(?:person|people|actor|item|candidate)\b",
        text,
        re.IGNORECASE,
    ):
        errors.append("assistant_clarification assumes an incomplete list contains the answer")
    if text and re.search(
        r"\b(?:deleted|removed|withheld|omitted)\b"
        r"|\b(?:evidence|context|document|passage|text)\b.{0,30}\bedit(?:ed|ing)?\b"
        r"|\b(?:evidence|context)\s+(?:is|was)\s+missing\b",
        text,
        re.IGNORECASE,
    ):
        errors.append("assistant_clarification mentions evidence manipulation")
    if text and not re.search(
        r"\b(?:context|available information|information provided|provided information)\b",
        text,
        re.IGNORECASE,
    ):
        errors.append("assistant_clarification lacks known-support framing")
    if text and not re.search(
        r"\b(?:but|however|yet|while|although)\b", text, re.IGNORECASE
    ):
        errors.append("assistant_clarification lacks support-gap contrast")
    if text and not re.search(
        r"\b(?:does not|doesn't|do not|don't|is not|isn't|unclear|unspecified|not specified|not establish|not identify|not provide)\b",
        text,
        re.IGNORECASE,
    ):
        errors.append("assistant_clarification does not state the unresolved gap")
    if text and not re.search(
        r"\b(?:provide|confirm|clarify|identify|specify|tell|state|name)\b",
        text,
        re.IGNORECASE,
    ):
        errors.append("assistant_clarification does not request the missing relation")
    if text and len(re.findall(r"[0-9A-Za-zÀ-ž]+", text)) > 70:
        errors.append("assistant_clarification exceeds 70 words")
    target_value = clean(edit.get("clarification_target"))
    target_tokens = word_tokens(target_value)
    clarification_tokens = word_tokens(text)
    target_relation_tags = relation_tags(target_value)
    clarification_relation_tags = relation_tags(text)
    if target_relation_tags:
        if not target_relation_tags.intersection(clarification_relation_tags):
            errors.append("assistant_clarification changes the target relation")
    if target_tokens:
        coverage = len(target_tokens.intersection(clarification_tokens)) / len(target_tokens)
        if coverage < 0.25:
            errors.append(f"assistant_clarification target coverage too low: {coverage:.3f}")
    return errors


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
                candidate = parse_json_object(raw)
                structure = validate(sample, candidate)
                if not structure:
                    parsed = candidate
                    if stage == "EDIT" and candidate["constructible"] is False:
                        status = "SKIPPED"
                    elif (
                        stage == "CLARIFICATION"
                        and candidate.get("target_resolved_by_remaining_evidence") is True
                    ):
                        status = "TARGET_RESOLVED"
                    else:
                        status = "SUCCESS"
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
        if parsed is not None:
            break
        if status == "STRUCTURE_ERROR":
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
        "created_at": now_iso(),
    }


def edit_one(sample: dict[str, Any], runtime: shared.RuntimeModel) -> dict[str, Any]:
    messages = editor_messages(sample)
    return call_stage(sample, runtime, "EDIT", messages, edit_errors, EDIT_PROMPT_VERSION)


def clarify_one(
    sample: dict[str, Any], edit_call: dict[str, Any], runtime: shared.RuntimeModel
) -> dict[str, Any]:
    messages = clarification_messages(sample, edit_call["parsed_response"])
    def validate(sample_value: dict[str, Any], result: dict[str, Any]) -> list[str]:
        return clarification_errors(sample_value, result, edit_call["parsed_response"])

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
    raw = ""
    for attempt in reversed(call.get("attempts") or []):
        if clean(attempt.get("raw_response")):
            raw = clean(attempt.get("raw_response"))
            break
    if not raw:
        return call
    updated = dict(call)
    try:
        candidate = parse_json_object(raw)
        errors = validate(sample, candidate)
        if errors:
            status = "STRUCTURE_ERROR"
        elif stage == "EDIT" and candidate.get("constructible") is False:
            status = "SKIPPED"
        elif (
            stage == "CLARIFICATION"
            and candidate.get("target_resolved_by_remaining_evidence") is True
        ):
            status = "TARGET_RESOLVED"
        else:
            status = "SUCCESS"
        updated.update(
            {
                "pipeline_version": PIPELINE_VERSION,
                "parsed_response": candidate,
                "status": status,
                "errors": errors,
                "revalidated_at": now_iso(),
            }
        )
    except Exception as exc:
        updated.update(
            {
                "pipeline_version": PIPELINE_VERSION,
                "parsed_response": None,
                "status": "PARSE_ERROR",
                "errors": [str(exc)[:1000]],
                "revalidated_at": now_iso(),
            }
        )
    return updated


def build_family(
    sample: dict[str, Any],
    edit_call: dict[str, Any],
    clarify_call: dict[str, Any],
) -> dict[str, Any]:
    edit = edit_call["parsed_response"]
    clarification = clarify_call["parsed_response"]["assistant_clarification"]
    answers = accepted_answers(sample)
    screening = sample["核心黄金筛选"]
    return {
        "schema_version": "evidence_state_family.v1",
        "family_id": f"{sample['样本ID']}::MISSING::v1",
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
        "full_state": {"state": "FULL", "evidence": compact_evidence(sample)},
        "initial_state": {
            "state": "MISSING",
            "evidence": edit["modified_evidence"],
            "missing_information": edit["clarification_target"],
            "clarification_grounding": {
                "document_id": clarify_call["parsed_response"]["known_support_document_id"],
                "span": clarify_call["parsed_response"]["known_support_span"],
            },
            "minimal_edit": {
                "edit_type": edit["edit_type"],
                "document_id": edit["edited_document_id"],
                "original_span": edit["original_span"],
                "replacement_span": edit.get("replacement_span", ""),
            },
        },
        "restoration": {
            "restoring_information": edit["restoring_information"],
            "restores_original_support": True,
        },
        "trajectory": [
            {
                "turn": 1,
                "role": "assistant",
                "action": "REQUEST_INFORMATION",
                "content": clarification,
            },
            {
                "turn": 2,
                "role": "user",
                "action": "PROVIDE_INFORMATION",
                "content": edit["restoring_information"],
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
        f"[{d['document_id']}] {d['title']}\n{d['text']}"
        for d in family["full_state"]["evidence"]
    )
    missing = "\n\n".join(
        f"[{d['document_id']}] {d['title']}\n{d['text']}"
        for d in family["initial_state"]["evidence"]
    )
    turns = "\n".join(
        f"- {t['role']} / {t['action']}: {t['content']}"
        for t in family["trajectory"]
    )
    edit = family["initial_state"]["minimal_edit"]
    return f"""## {family['family_id']}

- dataset: {family['dataset']}
- query: {family['query']}
- answer: {family['answer']}
- missing information: {family['initial_state']['missing_information']}
- restoring information: {family['restoration']['restoring_information']}
- edit: {edit['edit_type']} in {edit['document_id']}
- original span: {edit['original_span']}
- replacement span: {edit['replacement_span']}

### FULL evidence

{full}

### MISSING evidence

{missing}

### One-clarification trajectory

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
    return {
        "pipeline_version": PIPELINE_VERSION,
        "prompt_versions": {
            "editor": EDIT_PROMPT_VERSION,
            "clarification": CLARIFY_PROMPT_VERSION,
        },
        "source_rows": len(rows),
        "candidate_rows": len(candidates),
        "target": args.target or len(candidates),
        "constructed_families": len(families),
        "target_reached": len(families) >= (args.target or len(candidates)),
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
            "query_is_not_evidence": True,
            "answer_entity_may_remain": True,
            "partial_related_information_may_remain": True,
            "remove_one_necessary_claim": True,
            "clarification_target_is_answer_free": True,
            "clarifier_view_removes_direct_answer_mentions": True,
            "clarifier_view_removes_near_name_answer_mentions": True,
            "clarifier_view_hides_entire_answer_bearing_documents": True,
            "clarification_known_support_requires_exact_source_span": True,
            "remaining_evidence_target_resolution_is_a_rejection": True,
            "restoring_information_cannot_strengthen_source_claim": True,
            "exactly_one_changed_document": True,
            "user_reply_equals_restoring_information": True,
            "final_answer_is_canonical_and_deterministic": True,
            "repair_stage_used": False,
            "qwen_missing_behavior_used_as_filter": False,
        },
        "created_at": now_iso(),
    }


def run_parallel(
    samples: list[dict[str, Any]],
    worker: Any,
    workers: int,
    path: Path,
    calls: dict[str, dict[str, Any]],
    lock: threading.Lock,
) -> None:
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        future_map = {executor.submit(worker, sample): sample for sample in samples}
        for future in concurrent.futures.as_completed(future_map):
            sample = future_map[future]
            try:
                call = future.result()
            except Exception as exc:
                call = {
                    "sample_id": sample["样本ID"],
                    "dataset": sample["来源"],
                    "status": "WORKER_ERROR",
                    "errors": [str(exc)[:1200]],
                    "created_at": now_iso(),
                }
            calls[str(call["sample_id"])] = call
            append_jsonl(path, call, lock)
            print(f"{call['sample_id']}\t{call['status']}", flush=True)


def main() -> int:
    args = parse_args()
    primary = shared.load_jsonl(args.input)
    overflow = shared.load_jsonl(args.overflow) if args.overflow.exists() else []
    screening_rows = shared.load_jsonl(args.screening)
    screening_by_id = {str(row["sample_id"]): row for row in screening_rows}
    primary = [attach_screening_gate(row, screening_by_id) for row in primary]
    overflow = [attach_screening_gate(row, screening_by_id) for row in overflow]
    combined: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in primary + overflow:
        sid = str(row.get("样本ID") or "")
        if sid not in seen:
            combined.append(row)
            seen.add(sid)

    check = preflight(combined)
    if check["errors"]:
        raise ValueError(f"seed preflight failed: {check['errors'][:20]}")
    # Engineering pilots must sample only from the 2,000-row primary gold pool.
    # Overflow is eligible solely as a full-run replacement after primary failures.
    if args.limit_per_dataset > 0:
        candidates = select_rows(primary, args.limit_per_dataset, args.seed)
    else:
        candidates = combined
    goal = args.target or len(candidates)
    if goal > len(candidates):
        raise ValueError(f"target {goal} exceeds {len(candidates)} available candidates")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        args.output_dir / "missing_family_preflight.json",
        {
            **check,
            "primary_rows": len(primary),
            "overflow_rows": len(overflow),
            "screening_rows": len(screening_rows),
            "rehydrated_overflow_gates": sum(
                bool((row.get("核心黄金筛选") or {}).get("metadata_rehydrated_from_screening"))
                for row in overflow
            ),
            "unique_candidates": len(combined),
            "selected_candidates": len(candidates),
            "selected_dataset_counts": dict(sorted(Counter(x["来源"] for x in candidates).items())),
            "target": goal,
            "pipeline_version": PIPELINE_VERSION,
            "editor_prompt_version": EDIT_PROMPT_VERSION,
            "clarification_prompt_version": CLARIFY_PROMPT_VERSION,
        },
    )
    if args.dry_run:
        write_jsonl(
            args.output_dir / "missing_family_editor_prompts.jsonl",
            [
                {"sample_id": x["样本ID"], "dataset": x["来源"], "messages": editor_messages(x)}
                for x in candidates
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
    edit_path = args.output_dir / "missing_edit_calls.jsonl"
    clarify_path = args.output_dir / "missing_clarification_calls.jsonl"
    edit_calls = {
        str(row["sample_id"]): row for row in read_jsonl_if_exists(edit_path)
    }
    clarify_calls = {
        str(row["sample_id"]): row for row in read_jsonl_if_exists(clarify_path)
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
            append_jsonl(edit_path, updated, lock)
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
            append_jsonl(clarify_path, updated, lock)

    if args.retry_api_errors:
        retry_edit_ids = drop_retryable_api_errors(edit_calls)
        retry_clarify_ids = drop_retryable_api_errors(clarify_calls)
        print(
            f"retrying_api_errors edit={len(retry_edit_ids)} "
            f"clarification={len(retry_clarify_ids)}",
            flush=True,
        )

    while True:
        success_ids = [
            sid
            for sid in order
            if edit_calls.get(sid, {}).get("status") == "SUCCESS"
            and clarify_calls.get(sid, {}).get("status") == "SUCCESS"
        ]
        if len(success_ids) >= goal:
            break

        pending_clarifications = [
            sample_by_id[sid]
            for sid in order
            if edit_calls.get(sid, {}).get("status") == "SUCCESS"
            and sid not in clarify_calls
        ]
        if pending_clarifications:
            def clarification_worker(sample: dict[str, Any]) -> dict[str, Any]:
                return clarify_one(sample, edit_calls[str(sample["样本ID"])], runtime)

            run_parallel(
                pending_clarifications,
                clarification_worker,
                workers,
                clarify_path,
                clarify_calls,
                lock,
            )
            continue

        unattempted = [sample_by_id[sid] for sid in order if sid not in edit_calls]
        if not unattempted:
            break
        needed = goal - len(success_ids)
        batch_size = min(len(unattempted), max(needed, min(50, max(1, (needed + 9) // 10))))
        edit_batch = unattempted[:batch_size]
        run_parallel(
            edit_batch,
            lambda sample: edit_one(sample, runtime),
            workers,
            edit_path,
            edit_calls,
            lock,
        )

    success_ids = [
        sid
        for sid in order
        if edit_calls.get(sid, {}).get("status") == "SUCCESS"
        and clarify_calls.get(sid, {}).get("status") == "SUCCESS"
    ][:goal]
    families = [
        build_family(sample_by_id[sid], edit_calls[sid], clarify_calls[sid])
        for sid in success_ids
    ]
    write_jsonl(args.output_dir / "missing_families.jsonl", families)
    report = build_report(
        args, combined, candidates, families, edit_calls, clarify_calls, runtime
    )
    write_json(args.output_dir / "missing_family_report.json", report)

    review_families: list[dict[str, Any]] = []
    review_counts: Counter[str] = Counter()
    for family in families:
        dataset = family["dataset"]
        if review_counts[dataset] < args.review_per_dataset:
            review_families.append(family)
            review_counts[dataset] += 1
    review = "# MISSING family split-responsibility review\n\n" + "\n\n".join(
        markdown_sample(family) for family in review_families
    )
    (args.output_dir / "missing_family_review.md").write_text(review + "\n", encoding="utf-8")
    print(
        f"constructed={len(families)} target={goal} "
        f"edit_calls={len(edit_calls)} clarification_calls={len(clarify_calls)}",
        flush=True,
    )
    return 0 if len(families) >= goal else 2


if __name__ == "__main__":
    raise SystemExit(main())
