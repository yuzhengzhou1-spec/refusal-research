#!/usr/bin/env python3
"""构造 HotpotQA、MuSiQue-Ans 与 NQ 代理 FULL（E+）统一母样本。"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import random
import re
import shutil
import tempfile
import unicodedata
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator


SCHEMA_VERSION = "parent_pool.zh.v1"
DATASET_SEED_OFFSETS = {
    "hotpotqa": 101,
    "musique": 202,
    "natural_questions": 303,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--candidate-count", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260821)
    parser.add_argument("--answer-max-tokens", type=int, default=12)
    parser.add_argument("--answer-max-chars", type=int, default=80)
    return parser.parse_args()


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def normalize_text(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).casefold()
    text = re.sub(r"[^\w]+", " ", text, flags=re.UNICODE)
    return " ".join(text.split())


def contains_phrase(text: str, phrase: str) -> bool:
    haystack = normalize_text(text)
    needle = normalize_text(phrase)
    if not haystack or not needle:
        return False
    return f" {needle} " in f" {haystack} "


def any_answer_in(text: str, answers: list[str]) -> bool:
    return any(contains_phrase(text, answer) for answer in answers if answer)


def dedupe_strings(values: Iterable[Any]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value or "").strip()
        key = normalize_text(text)
        if text and key and key not in seen:
            seen.add(key)
            result.append(text)
    return result


def answer_too_long(answer: str, max_tokens: int, max_chars: int) -> bool:
    return len(answer) > max_chars or len(answer.split()) > max_tokens


def answer_type(answer: str) -> str:
    normalized = normalize_text(answer)
    if re.fullmatch(r"\d+(?:[.,]\d+)?", answer.strip()):
        return "数值"
    if re.search(r"\b(?:1[0-9]{3}|20[0-9]{2})\b", answer):
        return "日期或年份"
    if len(normalized.split()) <= 6:
        return "短实体"
    return "短文本"


def split_sentences(text: str) -> list[str]:
    clean = " ".join(str(text or "").split())
    if not clean:
        return []
    pieces = re.split(r"(?<=[.!?])\s+(?=[\"'(\[]?[A-Z0-9])", clean)
    return [piece.strip() for piece in pieces if piece.strip()]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            handle.write(text)
        os.replace(temp_name, path)
    except Exception:
        if os.path.exists(temp_name):
            os.unlink(temp_name)
        raise


def write_json(path: Path, value: Any) -> None:
    atomic_write_text(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    text = "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows)
    atomic_write_text(path, text)


def stream_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{line_number}: JSON解析失败") from exc


def stream_zip_jsonl(zip_path: Path, member: str) -> Iterator[dict[str, Any]]:
    with zipfile.ZipFile(zip_path) as archive:
        with archive.open(member) as raw:
            with io.TextIOWrapper(raw, encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, 1):
                    if line.strip():
                        try:
                            yield json.loads(line)
                        except json.JSONDecodeError as exc:
                            raise ValueError(f"{zip_path}!{member}:{line_number}: JSON解析失败") from exc


def reservoir_sample(
    records: Iterable[dict[str, Any]],
    eligible: Callable[[dict[str, Any]], bool],
    count: int,
    seed: int,
) -> tuple[list[dict[str, Any]], int, int]:
    rng = random.Random(seed)
    sample: list[dict[str, Any]] = []
    total = 0
    eligible_seen = 0
    for record in records:
        total += 1
        if not eligible(record):
            continue
        eligible_seen += 1
        if len(sample) < count:
            sample.append(record)
        else:
            index = rng.randrange(eligible_seen)
            if index < count:
                sample[index] = record
    rng.shuffle(sample)
    return sample, total, eligible_seen


def rejection(source: str, source_id: str, reasons: list[str]) -> dict[str, Any]:
    return {
        "来源": source,
        "原始样本ID": source_id,
        "主要原因": reasons[0],
        "全部原因": reasons,
    }


def infer_hotpot_hops(
    question: str,
    answer: str,
    support_titles: list[str],
    support_text_by_title: dict[str, str],
) -> tuple[dict[str, int], str, str]:
    if len(support_titles) != 2:
        return {}, "未解析", "支撑文档数不是2"

    mentioned_in_question = [
        title for title in support_titles if contains_phrase(question, title)
    ]
    if len(mentioned_in_question) == 1:
        first = mentioned_in_question[0]
        second = support_titles[1] if support_titles[0] == first else support_titles[0]
        return {first: 1, second: 2}, "已解析", "问题显式提及第一跳标题"

    link_targets: list[str] = []
    for title in support_titles:
        other = support_titles[1] if support_titles[0] == title else support_titles[0]
        if contains_phrase(support_text_by_title.get(other, ""), title):
            link_targets.append(title)
    if len(set(link_targets)) == 1:
        second = link_targets[0]
        first = support_titles[1] if support_titles[0] == second else support_titles[0]
        return {first: 1, second: 2}, "已解析", "第一跳证据链接到第二跳标题"

    answer_docs = [
        title
        for title in support_titles
        if contains_phrase(support_text_by_title.get(title, ""), answer)
    ]
    if len(answer_docs) == 1:
        second = answer_docs[0]
        first = support_titles[1] if support_titles[0] == second else support_titles[0]
        return {first: 1, second: 2}, "已解析", "唯一答案所在支撑文档作为末跳"

    return {}, "未解析", "无法从问题、标题链接或答案位置唯一恢复"


def validate_common(
    question: str,
    answers: list[str],
    context_text: str,
    max_tokens: int,
    max_chars: int,
) -> list[str]:
    reasons: list[str] = []
    if not question.strip():
        reasons.append("EMPTY_QUESTION")
    if not answers:
        reasons.append("MISSING_ANSWER")
        return reasons
    canonical = answers[0]
    if answer_too_long(canonical, max_tokens, max_chars):
        reasons.append("ANSWER_TOO_LONG")
    if contains_phrase(question, canonical):
        reasons.append("ANSWER_IN_QUESTION")
    if not any_answer_in(context_text, answers):
        reasons.append("ANSWER_NOT_IN_FULL_CONTEXT")
    return reasons


def build_hotpot(
    record: dict[str, Any],
    max_tokens: int,
    max_chars: int,
) -> tuple[dict[str, Any] | None, list[str]]:
    source_id = str(record.get("id", "")).strip()
    question = str(record.get("question", "")).strip()
    answer = str(record.get("answer", "")).strip()
    answers = dedupe_strings([answer])
    reasons: list[str] = []

    context = record.get("context") or {}
    titles = list(context.get("title") or [])
    sentence_groups = list(context.get("sentences") or [])
    if len(titles) != len(sentence_groups) or not titles:
        reasons.append("PARSE_ERROR")
        return None, reasons

    title_to_index: dict[str, int] = {}
    context_full: list[dict[str, Any]] = []
    for index, (title, sentences) in enumerate(zip(titles, sentence_groups)):
        title_text = str(title)
        if title_text not in title_to_index:
            title_to_index[title_text] = index
        clean_sentences = [str(sentence) for sentence in (sentences or [])]
        context_full.append(
            {
                "文档ID": f"hp_doc_{index}",
                "标题": title_text,
                "文本": "".join(clean_sentences).strip(),
                "句子": clean_sentences,
                "是否支撑文档": False,
                "原始位置": index,
            }
        )

    sf = record.get("supporting_facts") or {}
    sf_titles = list(sf.get("title") or [])
    sf_ids = list(sf.get("sent_id") or [])
    if len(sf_titles) != len(sf_ids) or not sf_titles:
        reasons.append("EMPTY_SUPPORT")

    support_units_raw: list[tuple[str, int, str]] = []
    broken_pointer = False
    for title, sentence_id in zip(sf_titles, sf_ids):
        title_text = str(title)
        if title_text not in title_to_index:
            broken_pointer = True
            continue
        doc_index = title_to_index[title_text]
        try:
            sentence_index = int(sentence_id)
            sentence = context_full[doc_index]["句子"][sentence_index]
        except (TypeError, ValueError, IndexError):
            broken_pointer = True
            continue
        support_units_raw.append((title_text, sentence_index, sentence))
        context_full[doc_index]["是否支撑文档"] = True
    if broken_pointer:
        reasons.append("BROKEN_SUPPORT_POINTER")

    support_titles = list(dict.fromkeys(title for title, _, _ in support_units_raw))
    if len(support_titles) != 2 or not (2 <= len(support_units_raw) <= 4):
        reasons.append("INCOMPLETE_CHAIN")

    all_context = "\n".join(doc["标题"] + "\n" + doc["文本"] for doc in context_full)
    reasons.extend(validate_common(question, answers, all_context, max_tokens, max_chars))
    support_text = "\n".join(text for _, _, text in support_units_raw)
    if answers and not any_answer_in(support_text, answers):
        reasons.append("ANSWER_NOT_IN_SUPPORT")

    support_normalized = {normalize_text(text) for _, _, text in support_units_raw if text}
    duplicate_found = False
    distractor_answer_docs = 0
    for doc in context_full:
        if doc["是否支撑文档"]:
            continue
        if any_answer_in(doc["文本"], answers):
            distractor_answer_docs += 1
        if any(normalize_text(sentence) in support_normalized for sentence in doc["句子"] if sentence):
            duplicate_found = True
    if distractor_answer_docs >= 2:
        reasons.append("DISTRACTOR_ANSWER_LEAKAGE")
    if duplicate_found:
        reasons.append("DUPLICATE_SUPPORT_DISTRACTOR")

    support_text_by_title = {
        title: " ".join(text for unit_title, _, text in support_units_raw if unit_title == title)
        for title in support_titles
    }
    hop_map, hop_status, hop_rule = infer_hotpot_hops(
        question, answer, support_titles, support_text_by_title
    )
    if hop_status != "已解析":
        reasons.append("UNRESOLVED_HOP_ORDER")

    if reasons:
        return None, list(dict.fromkeys(reasons))

    support_units = []
    for title, sentence_id, text in support_units_raw:
        doc_index = title_to_index[title]
        support_units.append(
            {
                "单元ID": f"hp_doc_{doc_index}_sent_{sentence_id}",
                "文档ID": f"hp_doc_{doc_index}",
                "句子ID": sentence_id,
                "文本": text,
                "跳次": hop_map.get(title),
                "支撑来源": "gold_annotation",
                "必要性": "未验证",
            }
        )

    parent = {
        "模式版本": SCHEMA_VERSION,
        "样本ID": f"hotpotqa_{source_id}",
        "来源": "hotpotqa",
        "原始样本ID": source_id,
        "问题": question,
        "答案": answers,
        "完整上下文": context_full,
        "关键证据单元": support_units,
        "问题分解": [],
        "跳数": 2,
        "推理类型": "bridge",
        "验证": {
            "结构有效": True,
            "答案位于完整上下文": True,
            "答案位于支撑证据": True,
            "支撑链完整": True,
            "跳序状态": hop_status,
            "跳序规则": hop_rule,
            "模型已检查": False,
            "模型回答正确": None,
            "人工已检查": False,
        },
        "元数据": {
            "原始划分": "train",
            "答案类型": answer_type(answer),
            "问题答案泄漏": False,
            "干扰证据答案文档数": distractor_answer_docs,
            "数据配置": "distractor",
            "难度": record.get("level"),
            "原始类型": record.get("type"),
            "代理证据": False,
        },
    }
    return parent, []


def parse_dependencies(question: str) -> list[int]:
    return sorted({int(value) for value in re.findall(r"#(\d+)", str(question or ""))})


def build_musique(
    record: dict[str, Any],
    max_tokens: int,
    max_chars: int,
) -> tuple[dict[str, Any] | None, list[str]]:
    source_id = str(record.get("id", "")).strip()
    question = str(record.get("question", "")).strip()
    answers = dedupe_strings([record.get("answer"), *(record.get("answer_aliases") or [])])
    reasons: list[str] = []

    paragraphs = list(record.get("paragraphs") or [])
    context_full: list[dict[str, Any]] = []
    index_to_doc: dict[int, dict[str, Any]] = {}
    for position, paragraph in enumerate(paragraphs):
        try:
            paragraph_idx = int(paragraph.get("idx"))
        except (TypeError, ValueError):
            reasons.append("PARSE_ERROR")
            continue
        doc = {
            "文档ID": f"mq_para_{paragraph_idx}",
            "标题": str(paragraph.get("title", "")),
            "文本": str(paragraph.get("paragraph_text", "")).strip(),
            "句子": [],
            "是否支撑文档": bool(paragraph.get("is_supporting")),
            "原始位置": position,
            "段落ID": paragraph_idx,
        }
        context_full.append(doc)
        index_to_doc[paragraph_idx] = doc

    decomposition_raw = list(record.get("question_decomposition") or [])
    if len(decomposition_raw) != 2:
        reasons.append("INCOMPLETE_CHAIN")

    support_units: list[dict[str, Any]] = []
    decomposition: list[dict[str, Any]] = []
    broken_pointer = False
    for hop, step in enumerate(decomposition_raw, 1):
        try:
            paragraph_idx = int(step.get("paragraph_support_idx"))
        except (TypeError, ValueError):
            broken_pointer = True
            continue
        doc = index_to_doc.get(paragraph_idx)
        if doc is None or not doc["是否支撑文档"] or not doc["文本"]:
            broken_pointer = True
            continue
        step_question = str(step.get("question", "")).strip()
        step_answer = str(step.get("answer", "")).strip()
        dependencies = parse_dependencies(step_question)
        decomposition.append(
            {
                "跳次": hop,
                "子问题": step_question,
                "子答案": step_answer,
                "依赖跳次": dependencies,
                "支撑文档ID": doc["文档ID"],
            }
        )
        support_units.append(
            {
                "单元ID": doc["文档ID"],
                "文档ID": doc["文档ID"],
                "句子ID": None,
                "文本": doc["文本"],
                "跳次": hop,
                "支撑来源": "gold_annotation",
                "必要性": "未验证",
                "标注粒度": "段落",
            }
        )
    if broken_pointer:
        reasons.append("BROKEN_SUPPORT_POINTER")
    if len({unit["文档ID"] for unit in support_units}) != 2:
        reasons.append("INCOMPLETE_CHAIN")
    if len(decomposition) == 2 and 1 not in decomposition[1]["依赖跳次"]:
        reasons.append("INCOMPLETE_CHAIN")

    all_context = "\n".join(doc["标题"] + "\n" + doc["文本"] for doc in context_full)
    reasons.extend(validate_common(question, answers, all_context, max_tokens, max_chars))
    support_text = "\n".join(unit["文本"] for unit in support_units)
    if answers and not any_answer_in(support_text, answers):
        reasons.append("ANSWER_NOT_IN_SUPPORT")

    support_doc_ids = {unit["文档ID"] for unit in support_units}
    support_normalized = {normalize_text(unit["文本"]) for unit in support_units}
    distractor_answer_docs = 0
    duplicate_found = False
    for doc in context_full:
        if doc["文档ID"] in support_doc_ids:
            continue
        if any_answer_in(doc["文本"], answers):
            distractor_answer_docs += 1
        if normalize_text(doc["文本"]) in support_normalized:
            duplicate_found = True
    if distractor_answer_docs >= 2:
        reasons.append("DISTRACTOR_ANSWER_LEAKAGE")
    if duplicate_found:
        reasons.append("DUPLICATE_SUPPORT_DISTRACTOR")

    if reasons:
        return None, list(dict.fromkeys(reasons))

    parent = {
        "模式版本": SCHEMA_VERSION,
        "样本ID": f"musique_{source_id}",
        "来源": "musique",
        "原始样本ID": source_id,
        "问题": question,
        "答案": answers,
        "完整上下文": context_full,
        "关键证据单元": support_units,
        "问题分解": decomposition,
        "跳数": 2,
        "推理类型": "composition",
        "验证": {
            "结构有效": True,
            "答案位于完整上下文": True,
            "答案位于支撑证据": True,
            "支撑链完整": True,
            "跳序状态": "已解析",
            "跳序规则": "官方 question_decomposition 顺序",
            "模型已检查": False,
            "模型回答正确": None,
            "人工已检查": False,
        },
        "元数据": {
            "原始划分": "train",
            "答案类型": answer_type(answers[0]),
            "问题答案泄漏": False,
            "干扰证据答案文档数": distractor_answer_docs,
            "数据配置": "MuSiQue-Ans",
            "代理证据": False,
            "原始answerable": bool(record.get("answerable")),
        },
    }
    return parent, []


def extract_title_and_body(context: str) -> tuple[str, str]:
    match = re.match(r"\s*Title:\s*([^\r\n]+)[\r\n]+(.*)", context, flags=re.IGNORECASE | re.DOTALL)
    if not match:
        return "", context.strip()
    return match.group(1).strip(), match.group(2).strip()


def build_nq_proxy(
    record: dict[str, Any],
    max_tokens: int,
    max_chars: int,
) -> tuple[dict[str, Any] | None, list[str]]:
    source_id = str(record.get("source_id", "")).strip()
    question = str(record.get("original_query", "")).strip()
    context = str(record.get("original_context", "")).strip()
    answers = dedupe_strings(record.get("original_answers") or [])
    reasons = validate_common(question, answers, context, max_tokens, max_chars)

    title, body = extract_title_and_body(context)
    sentences = split_sentences(body)
    candidate_indexes = [
        index for index, sentence in enumerate(sentences) if any_answer_in(sentence, answers)
    ]
    if not candidate_indexes:
        reasons.append("ANSWER_SENTENCE_NOT_FOUND")

    if answers and re.fullmatch(r"\d{1,4}", normalize_text(answers[0])):
        reasons.append("GENERIC_NUMERIC_ANSWER")

    if reasons:
        return None, list(dict.fromkeys(reasons))

    context_full = [
        {
            "文档ID": f"nq_proxy_{source_id}",
            "标题": title,
            "文本": body,
            "句子": sentences,
            "是否支撑文档": True,
            "原始位置": 0,
        }
    ]
    support_units = [
        {
            "单元ID": f"nq_proxy_{source_id}_sent_{index}",
            "文档ID": f"nq_proxy_{source_id}",
            "句子ID": index,
            "文本": sentences[index],
            "跳次": 1,
            "支撑来源": "proxy_answer_sentence",
            "必要性": "未验证",
            "标注粒度": "候选句",
        }
        for index in candidate_indexes
    ]
    parent = {
        "模式版本": SCHEMA_VERSION,
        "样本ID": f"natural_questions_proxy_{source_id}",
        "来源": "natural_questions",
        "原始样本ID": source_id,
        "问题": question,
        "答案": answers,
        "完整上下文": context_full,
        "关键证据单元": support_units,
        "问题分解": [],
        "跳数": 1,
        "推理类型": "single_hop",
        "验证": {
            "结构有效": True,
            "答案位于完整上下文": True,
            "答案位于支撑证据": True,
            "支撑链完整": True,
            "跳序状态": "不适用",
            "跳序规则": "单跳",
            "模型已检查": False,
            "模型回答正确": None,
            "人工已检查": False,
        },
        "元数据": {
            "原始划分": "未知",
            "答案类型": answer_type(answers[0]),
            "问题答案泄漏": False,
            "干扰证据答案文档数": 0,
            "数据配置": "RefusalBench-NQ original_context 代理",
            "代理证据": True,
            "证据可靠度": "低于原始NQ answer span",
            "原始NQ字段缺失": [
                "document_tokens",
                "long_answer_candidates",
                "annotations.long_answer",
                "annotations.short_answers",
            ],
        },
    }
    return parent, []


def validate_parent_schema(record: dict[str, Any]) -> list[str]:
    required = [
        "模式版本",
        "样本ID",
        "来源",
        "原始样本ID",
        "问题",
        "答案",
        "完整上下文",
        "关键证据单元",
        "问题分解",
        "跳数",
        "推理类型",
        "验证",
        "元数据",
    ]
    errors = [f"缺少字段:{key}" for key in required if key not in record]
    if not isinstance(record.get("答案"), list) or not record.get("答案"):
        errors.append("答案必须是非空数组")
    if not isinstance(record.get("完整上下文"), list) or not record.get("完整上下文"):
        errors.append("完整上下文必须是非空数组")
    if not isinstance(record.get("关键证据单元"), list) or not record.get("关键证据单元"):
        errors.append("关键证据单元必须是非空数组")
    return errors


def build_dataset(
    name: str,
    candidates: list[dict[str, Any]],
    builder: Callable[[dict[str, Any], int, int], tuple[dict[str, Any] | None, list[str]]],
    output_root: Path,
    max_tokens: int,
    max_chars: int,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    reason_counts: Counter[str] = Counter()
    schema_errors = 0

    for record in candidates:
        parent, reasons = builder(record, max_tokens, max_chars)
        source_id = str(record.get("id") or record.get("source_id") or "")
        if parent is not None:
            errors = validate_parent_schema(parent)
            if errors:
                schema_errors += 1
                reasons = ["SCHEMA_ERROR", *errors]
                parent = None
        if parent is None:
            unique_reasons = list(dict.fromkeys(reasons or ["UNKNOWN_REJECTION"]))
            reason_counts.update(unique_reasons)
            rejected.append(rejection(name, source_id, unique_reasons))
        else:
            accepted.append(parent)

    write_jsonl(output_root / "data" / f"{name}_parent_filtered.jsonl", accepted)
    write_jsonl(output_root / "rejected" / f"{name}_rejected.jsonl", rejected)
    review_count = min(50, len(accepted))
    review_rng = random.Random(seed + DATASET_SEED_OFFSETS[name] + 7000)
    review_rows = review_rng.sample(accepted, review_count) if review_count else []
    write_review_csv(output_root / "review" / f"{name}_manual_review.csv", review_rows)

    stats = {
        "候选数": len(candidates),
        "保留数": len(accepted),
        "拒绝数": len(rejected),
        "保留率": round(len(accepted) / len(candidates), 6) if candidates else 0.0,
        "Schema错误数": schema_errors,
        "拒绝原因计数": dict(reason_counts.most_common()),
        "人工抽检数": review_count,
    }
    return accepted, rejected, stats


def write_review_csv(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    fields = [
        "样本ID",
        "来源",
        "问题",
        "答案",
        "完整上下文",
        "关键证据单元",
        "完整上下文可可靠回答",
        "标准答案正确",
        "证据相关",
        "推理链完整",
        "存在等价证据",
        "删除后可能不可回答",
        "问题有歧义",
        "审核备注",
    ]
    try:
        with os.fdopen(fd, "w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for record in records:
                writer.writerow(
                    {
                        "样本ID": record["样本ID"],
                        "来源": record["来源"],
                        "问题": record["问题"],
                        "答案": json.dumps(record["答案"], ensure_ascii=False),
                        "完整上下文": json.dumps(record["完整上下文"], ensure_ascii=False),
                        "关键证据单元": json.dumps(record["关键证据单元"], ensure_ascii=False),
                        "完整上下文可可靠回答": "",
                        "标准答案正确": "",
                        "证据相关": "",
                        "推理链完整": "",
                        "存在等价证据": "",
                        "删除后可能不可回答": "",
                        "问题有歧义": "",
                        "审核备注": "",
                    }
                )
        os.replace(temp_name, path)
    except Exception:
        if os.path.exists(temp_name):
            os.unlink(temp_name)
        raise


def schema_document() -> dict[str, Any]:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "parent_pool.zh.v1",
        "title": "FULL母样本中文Schema",
        "type": "object",
        "required": [
            "模式版本",
            "样本ID",
            "来源",
            "原始样本ID",
            "问题",
            "答案",
            "完整上下文",
            "关键证据单元",
            "问题分解",
            "跳数",
            "推理类型",
            "验证",
            "元数据",
        ],
        "properties": {
            "模式版本": {"const": SCHEMA_VERSION},
            "样本ID": {"type": "string", "minLength": 1},
            "来源": {
                "enum": ["hotpotqa", "musique", "natural_questions"]
            },
            "原始样本ID": {"type": "string", "minLength": 1},
            "问题": {"type": "string", "minLength": 1},
            "答案": {
                "type": "array",
                "minItems": 1,
                "items": {"type": "string", "minLength": 1},
            },
            "完整上下文": {"type": "array", "minItems": 1},
            "关键证据单元": {"type": "array", "minItems": 1},
            "问题分解": {"type": "array"},
            "跳数": {"type": "integer", "minimum": 1},
            "推理类型": {
                "enum": ["single_hop", "bridge", "comparison", "composition"]
            },
            "验证": {"type": "object"},
            "元数据": {"type": "object"},
        },
        "additionalProperties": False,
    }


def ensure_clean_output(output_root: Path) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    for relative in ["data", "rejected", "review", "reports", "schema", "manifests", "input_snapshot"]:
        directory = output_root / relative
        if directory.exists():
            shutil.rmtree(directory)
        directory.mkdir(parents=True, exist_ok=True)


def write_checksums(output_root: Path) -> None:
    lines = []
    excluded = {
        output_root / "manifests" / "output_checksums.sha256",
    }
    for path in sorted(output_root.rglob("*")):
        if path.is_file() and path not in excluded:
            relative = path.relative_to(output_root).as_posix()
            lines.append(f"{sha256_file(path)}  {relative}")
    atomic_write_text(output_root / "manifests" / "output_checksums.sha256", "\n".join(lines) + "\n")


def main() -> int:
    args = parse_args()
    source_root = args.source_root.resolve()
    output_root = args.output_root.resolve()
    hotpot_path = source_root / "raw" / "hotpotqa_full.jsonl"
    musique_zip = source_root / "_musique_v1.0.zip"
    musique_member = "data/musique_ans_v1.0_train.jsonl"
    nq_proxy_path = source_root / "raw" / "refusalbench_nq_full.jsonl"
    nq_open_path = source_root / "raw" / "nq_open_full.jsonl"

    required_paths = [hotpot_path, musique_zip, nq_proxy_path, nq_open_path]
    missing = [str(path) for path in required_paths if not path.exists()]
    if missing:
        raise FileNotFoundError("缺少输入文件: " + ", ".join(missing))

    ensure_clean_output(output_root)
    started_at = now_iso()

    hotpot_candidates, hotpot_total, hotpot_eligible = reservoir_sample(
        stream_jsonl(hotpot_path),
        lambda row: row.get("type") == "bridge"
        and normalize_text(row.get("answer")) not in {"yes", "no"},
        args.candidate_count,
        args.seed + DATASET_SEED_OFFSETS["hotpotqa"],
    )

    musique_candidates, musique_total, musique_eligible = reservoir_sample(
        stream_zip_jsonl(musique_zip, musique_member),
        lambda row: bool(row.get("answerable"))
        and str(row.get("id", "")).startswith("2hop__"),
        args.candidate_count,
        args.seed + DATASET_SEED_OFFSETS["musique"],
    )

    unique_nq: dict[str, dict[str, Any]] = {}
    nq_total = 0
    for row in stream_jsonl(nq_proxy_path):
        nq_total += 1
        source_id = str(row.get("source_id", ""))
        if source_id and source_id not in unique_nq:
            unique_nq[source_id] = row
    nq_records = list(unique_nq.values())
    nq_rng = random.Random(args.seed + DATASET_SEED_OFFSETS["natural_questions"])
    nq_rng.shuffle(nq_records)
    nq_candidates = nq_records[: args.candidate_count]
    nq_eligible = len(nq_records)

    write_jsonl(output_root / "input_snapshot" / "hotpotqa_candidates.jsonl", hotpot_candidates)
    write_jsonl(output_root / "input_snapshot" / "musique_candidates.jsonl", musique_candidates)
    write_jsonl(output_root / "input_snapshot" / "natural_questions_proxy_candidates.jsonl", nq_candidates)

    all_accepted: list[dict[str, Any]] = []
    stats: dict[str, Any] = {}
    for name, candidates, builder in [
        ("hotpotqa", hotpot_candidates, build_hotpot),
        ("musique", musique_candidates, build_musique),
        ("natural_questions", nq_candidates, build_nq_proxy),
    ]:
        accepted, _, dataset_stats = build_dataset(
            name,
            candidates,
            builder,
            output_root,
            args.answer_max_tokens,
            args.answer_max_chars,
            args.seed,
        )
        all_accepted.extend(accepted)
        stats[name] = dataset_stats

    write_jsonl(output_root / "data" / "parent_pool_full_merged.jsonl", all_accepted)
    write_json(output_root / "schema" / "parent_pool.schema.json", schema_document())

    stats["合计"] = {
        "候选数": sum(item["候选数"] for item in stats.values()),
        "保留数": sum(item["保留数"] for item in stats.values()),
        "拒绝数": sum(item["拒绝数"] for item in stats.values()),
    }
    stats["说明"] = {
        "模型E+评测": "未执行；尚未提供基线模型",
        "Natural Questions": (
            "仅有100个唯一 RefusalBench-NQ original_context，作为代理母样本；"
            "不能替代原始NQ gold long/short answer标注"
        ),
    }
    write_json(output_root / "reports" / "dataset_statistics.json", stats)

    manifest = {
        "构建开始": started_at,
        "构建结束": now_iso(),
        "配置": {
            "随机种子": args.seed,
            "每数据集候选量": args.candidate_count,
            "答案最大空白词数": args.answer_max_tokens,
            "答案最大字符数": args.answer_max_chars,
        },
        "输入": [
            {
                "数据集": "hotpotqa",
                "路径": str(hotpot_path),
                "SHA256": sha256_file(hotpot_path),
                "原始记录数": hotpot_total,
                "首版范围内记录数": hotpot_eligible,
                "划分": "train",
            },
            {
                "数据集": "musique",
                "路径": str(musique_zip),
                "压缩包成员": musique_member,
                "SHA256": sha256_file(musique_zip),
                "原始记录数": musique_total,
                "首版范围内记录数": musique_eligible,
                "划分": "train",
            },
            {
                "数据集": "natural_questions",
                "路径": str(nq_proxy_path),
                "SHA256": sha256_file(nq_proxy_path),
                "原始派生记录数": nq_total,
                "唯一原始上下文数": nq_eligible,
                "划分": "未知",
                "代理来源": True,
            },
            {
                "数据集": "nq_open",
                "路径": str(nq_open_path),
                "SHA256": sha256_file(nq_open_path),
                "用途": "已盘点但未使用；缺少证据字段",
            },
        ],
    }
    write_json(output_root / "manifests" / "input_manifest.json", manifest)

    validation_lines = [
        "# FULL（E+）母样本构建报告",
        "",
        f"- 构建时间：{manifest['构建结束']}",
        f"- 随机种子：{args.seed}",
        f"- 候选上限：每数据集 {args.candidate_count} 条",
        f"- 合计：候选 {stats['合计']['候选数']}，保留 {stats['合计']['保留数']}，拒绝 {stats['合计']['拒绝数']}",
        "",
        "## 分数据集结果",
        "",
    ]
    for name in ["hotpotqa", "musique", "natural_questions"]:
        item = stats[name]
        validation_lines.extend(
            [
                f"- {name}：候选 {item['候选数']}，保留 {item['保留数']}，"
                f"拒绝 {item['拒绝数']}，保留率 {item['保留率']:.2%}；"
                f"Schema 错误 {item['Schema错误数']}。",
            ]
        )
    validation_lines.extend(
        [
            "",
            "## 验证边界",
            "",
            "- 已完成结构、答案定位、支撑指针、推理链和泄漏规则校验。",
            "- 尚未执行人工抽检；已生成每数据集最多 50 条抽检表。",
            "- 尚未执行基线模型 Acc(q,E+)；未提供模型调用配置。",
            "- Natural Questions 使用代理上下文，需原始 NQ 数据替换后才能达到 gold 证据验收标准。",
            "",
        ]
    )
    atomic_write_text(
        output_root / "reports" / "validation_report.md",
        "\n".join(validation_lines),
    )

    build_config = {
        "项目版本": "parent_pool_v1",
        "随机种子": args.seed,
        "每数据集候选量": args.candidate_count,
        "答案最大空白词数": args.answer_max_tokens,
        "答案最大字符数": args.answer_max_chars,
        "字段语言": "zh",
        "报告语言": "zh",
        "原始文本翻译": False,
    }
    write_json(output_root / "build_config.json", build_config)

    progress = [
        "# 项目进度",
        "",
        "- 状态：第一阶段自动构建完成，等待人工/模型验收",
        (
            f"- 已完成：源文件盘点；候选抽样；统一中文 schema；FULL 母样本、"
            f"拒绝记录、统计、校验报告和人工抽检表生成。共保留 {stats['合计']['保留数']} 条。"
        ),
        "- 当前：自动规则校验已完成。",
        "- 下一步：人工审核每个数据集抽检表；配置基线模型并测 Acc(q,E+)。",
        (
            "- 仍需数据：原始 Natural Questions（document_tokens、"
            "long_answer_candidates、annotations）以替换当前代理 NQ。"
        ),
        "",
    ]
    atomic_write_text(output_root / "项目进度.md", "\n".join(progress))
    write_checksums(output_root)
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
