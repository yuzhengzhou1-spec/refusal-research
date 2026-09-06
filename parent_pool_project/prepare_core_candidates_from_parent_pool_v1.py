#!/usr/bin/env python3
"""Prepare supporting-only core-gold candidates from a v1 parent-pool JSONL.

This is the on-disk adapter for the earliest unified parent pool.  It does not
call a model and never modifies its input.  Official raw HotpotQA/MuSiQue data
remain supported by ``build_core_gold_candidates_v2.py``; this adapter exists so
the currently retained v1 parent pool can be used as an auditable expansion
source when the original download directory is unavailable.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent
EVALUATION_ROOT = PROJECT_ROOT / "evaluation"
if str(EVALUATION_ROOT) not in sys.path:
    sys.path.insert(0, str(EVALUATION_ROOT))

import validate_missing1_llm as shared  # noqa: E402
import sanitize_core_gold_candidates as sanitation


PIPELINE_VERSION = "parent_pool_to_core_candidate.v1"
ALLOWED_DATASETS = ("hotpotqa", "musique")


def stable_key(seed: int, dataset: str, sample_id: str) -> str:
    return hashlib.sha256(
        f"{seed}:{PIPELINE_VERSION}:{dataset}:{sample_id}".encode("utf-8")
    ).hexdigest()


def collect_screened_ids(paths: list[Path]) -> set[str]:
    result: set[str] = set()
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(path)
        for row in shared.load_jsonl(path):
            value = row.get("sample_id") or row.get("样本ID")
            if value:
                result.add(str(value))
    return result


def reject_reason(row: dict[str, Any]) -> str | None:
    dataset = str(row.get("来源") or "")
    if dataset not in ALLOWED_DATASETS:
        return "UNSUPPORTED_DATASET"
    if int(row.get("跳数") or 0) != 2:
        return "NOT_TWO_HOP"
    if not row.get("问题") or not row.get("答案") or not row.get("完整上下文"):
        return "MISSING_REQUIRED_FIELD"
    support_ids = {
        str(unit.get("文档ID") or "")
        for unit in row.get("关键证据单元", [])
        if unit.get("文档ID")
    }
    if len(support_ids) != 2:
        return "SUPPORT_DOCUMENT_COUNT_NOT_TWO"
    context_ids = {
        str(document.get("文档ID") or "")
        for document in row.get("完整上下文", [])
    }
    if not support_ids.issubset(context_ids):
        return "SUPPORT_DOCUMENT_MISSING"
    query = str(row.get("问题") or "")
    aliases = [str(value) for value in row.get("答案", []) if str(value).strip()]
    if any(sanitation.phrase_in_text(alias, query) for alias in aliases):
        return "ANSWER_ALIAS_IN_QUERY"
    if shared.normalize_phrase(aliases[0] if aliases else "") in {"yes", "no"}:
        return "YES_NO_ANSWER"
    return None


def supporting_only(row: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(row)
    support_ids = {
        str(unit.get("文档ID") or "") for unit in result["关键证据单元"]
    }
    result["完整上下文"] = [
        document for document in result["完整上下文"]
        if str(document.get("文档ID") or "") in support_ids
    ]
    for document in result["完整上下文"]:
        document["是否支撑文档"] = True
    result["模式版本"] = "core_gold_candidate.zh.v2"
    result.setdefault("验证", {}).update({
        "模型已检查": False,
        "模型回答正确": None,
        "参数知识已检查": False,
        "反事实已检查": False,
    })
    result.setdefault("元数据", {}).update({
        "上下文模式": "仅官方支撑文档",
        "包含干扰文档": False,
        "核心黄金候选": True,
        "候选选择规则": "stable_sha256_prefix_from_parent_pool",
        "候选适配版本": PIPELINE_VERSION,
    })
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--exclude-screening", type=Path, action="append", default=[])
    parser.add_argument("--per-dataset", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260828)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.per_dataset < 0:
        raise ValueError("--per-dataset must be non-negative")
    excluded = collect_screened_ids(args.exclude_screening)
    accepted_by_dataset: dict[str, list[dict[str, Any]]] = defaultdict(list)
    rejected: list[dict[str, Any]] = []
    input_rows = 0
    for row in shared.load_jsonl(args.input):
        input_rows += 1
        sid = str(row.get("样本ID") or "")
        dataset = str(row.get("来源") or "")
        reason = "ALREADY_SCREENED" if sid in excluded else reject_reason(row)
        if reason:
            rejected.append({"sample_id": sid, "dataset": dataset, "reason": reason})
            continue
        accepted_by_dataset[dataset].append(supporting_only(row))

    selected: list[dict[str, Any]] = []
    eligible_counts: dict[str, int] = {}
    for dataset in ALLOWED_DATASETS:
        ordered = sorted(
            accepted_by_dataset.get(dataset, []),
            key=lambda row: stable_key(args.seed, dataset, str(row["样本ID"])),
        )
        eligible_counts[dataset] = len(ordered)
        selected.extend(ordered[:args.per_dataset] if args.per_dataset else ordered)
    if len({str(row["样本ID"]) for row in selected}) != len(selected):
        raise RuntimeError("duplicate sample IDs after parent-pool adaptation")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "core_gold_candidates_sanitized.jsonl"
    shared.write_jsonl(output_path, selected)
    shared.write_jsonl(args.output_dir / "core_gold_candidate_rejections.jsonl", rejected)
    report = {
        "pipeline_version": PIPELINE_VERSION,
        "input": str(args.input.resolve()),
        "input_rows": input_rows,
        "excluded_screened_ids": len(excluded),
        "eligible_by_dataset": eligible_counts,
        "selected_by_dataset": dict(Counter(str(row["来源"]) for row in selected)),
        "selected_total": len(selected),
        "rejection_counts": dict(Counter(row["reason"] for row in rejected)),
        "output": str(output_path.resolve()),
        "model_calls": 0,
        "input_modified": False,
        "context_policy": "official supporting documents only",
    }
    shared.write_json(args.output_dir / "core_gold_candidate_report.json", report)
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
