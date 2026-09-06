#!/usr/bin/env python3
"""Build deterministic supporting-only candidates for causal core-gold filtering.

This builder intentionally ignores the legacy gold pool. It reads the original
HotpotQA and MuSiQue sources, reuses the validated parent-sample parsers, removes
all distractor documents, and selects a stable prefix per dataset. Increasing
``--per-dataset`` therefore preserves every previously selected candidate.
"""

from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Iterable

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import build_parent_pool as parent  # noqa: E402


DATASETS = ("hotpotqa", "musique")
MUSIQUE_MEMBER = "data/musique_ans_v1.0_train.jsonl"


def stable_rank(seed: int, dataset: str, source_id: str) -> str:
    raw = f"{seed}:causal-core-gold-v2:{dataset}:{source_id}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def stable_take(
    records: Iterable[dict[str, Any]],
    dataset: str,
    eligible: Callable[[dict[str, Any]], bool],
    count: int,
    seed: int,
) -> tuple[list[dict[str, Any]], int, int]:
    """Return the ``count`` smallest stable hashes using bounded memory."""
    heap: list[tuple[int, str, dict[str, Any]]] = []
    total = 0
    eligible_count = 0
    for row in records:
        total += 1
        if not eligible(row):
            continue
        eligible_count += 1
        source_id = str(row.get("id") or row.get("source_id") or "")
        rank_hex = stable_rank(seed, dataset, source_id)
        rank_int = int(rank_hex, 16)
        item = (-rank_int, source_id, row)
        if len(heap) < count:
            heapq.heappush(heap, item)
        elif rank_int < -heap[0][0]:
            heapq.heapreplace(heap, item)
    selected = [(-negative, source_id, row) for negative, source_id, row in heap]
    selected.sort(key=lambda item: (item[0], item[1]))
    return [row for _, _, row in selected], total, eligible_count


def supporting_only(sample: dict[str, Any]) -> dict[str, Any]:
    support_ids = {
        str(unit.get("文档ID") or "") for unit in sample["关键证据单元"]
    }
    clean_docs = [
        document for document in sample["完整上下文"]
        if str(document.get("文档ID") or "") in support_ids
    ]
    clean_ids = {str(document.get("文档ID") or "") for document in clean_docs}
    if clean_ids != support_ids:
        raise ValueError(f"{sample['样本ID']}: support documents are incomplete")
    output = json.loads(json.dumps(sample, ensure_ascii=False))
    output["完整上下文"] = clean_docs
    output["模式版本"] = "core_gold_candidate.zh.v2"
    output["验证"].update({
        "模型已检查": False,
        "模型回答正确": None,
        "参数知识已检查": False,
        "反事实已检查": False,
    })
    output["元数据"].update({
        "上下文模式": "仅官方支撑文档",
        "包含干扰文档": False,
        "核心黄金候选": True,
        "候选选择规则": "stable_sha256_prefix",
    })
    return output


def build_selected(
    dataset: str,
    selected: list[dict[str, Any]],
    builder: Callable[[dict[str, Any], int, int], tuple[dict[str, Any] | None, list[str]]],
    max_tokens: int,
    max_chars: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for raw in selected:
        sample, reasons = builder(raw, max_tokens, max_chars)
        source_id = str(raw.get("id") or raw.get("source_id") or "")
        if sample is None:
            rejected.append({
                "dataset": dataset,
                "source_id": source_id,
                "reasons": reasons,
            })
            continue
        errors = parent.validate_parent_schema(sample)
        if errors:
            rejected.append({
                "dataset": dataset,
                "source_id": source_id,
                "reasons": ["SCHEMA_ERROR", *errors],
            })
            continue
        accepted.append(supporting_only(sample))
    return accepted, rejected


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-dataset", type=int, required=True)
    parser.add_argument("--seed", type=int, default=20260821)
    parser.add_argument("--answer-max-tokens", type=int, default=12)
    parser.add_argument("--answer-max-chars", type=int, default=80)
    args = parser.parse_args()
    if args.per_dataset <= 0:
        raise ValueError("--per-dataset must be positive")

    hotpot_path = args.source_root / "raw" / "hotpotqa_full.jsonl"
    musique_zip = args.source_root / "_musique_v1.0.zip"
    for path in (hotpot_path, musique_zip):
        if not path.exists():
            raise FileNotFoundError(path)

    hotpot_raw, hotpot_total, hotpot_eligible = stable_take(
        parent.stream_jsonl(hotpot_path),
        "hotpotqa",
        lambda row: row.get("type") == "bridge"
        and parent.normalize_text(row.get("answer")) not in {"yes", "no"},
        args.per_dataset,
        args.seed,
    )
    musique_raw, musique_total, musique_eligible = stable_take(
        parent.stream_zip_jsonl(musique_zip, MUSIQUE_MEMBER),
        "musique",
        lambda row: bool(row.get("answerable"))
        and str(row.get("id", "")).startswith("2hop__"),
        args.per_dataset,
        args.seed,
    )

    hotpot, hotpot_rejected = build_selected(
        "hotpotqa", hotpot_raw, parent.build_hotpot,
        args.answer_max_tokens, args.answer_max_chars,
    )
    musique, musique_rejected = build_selected(
        "musique", musique_raw, parent.build_musique,
        args.answer_max_tokens, args.answer_max_chars,
    )
    accepted = hotpot + musique
    rejected = hotpot_rejected + musique_rejected
    if len({row["样本ID"] for row in accepted}) != len(accepted):
        raise RuntimeError("duplicate candidate sample IDs")

    args.output.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output / "core_gold_candidates.jsonl", accepted)
    write_jsonl(args.output / "core_gold_candidate_rejections.jsonl", rejected)
    report = {
        "stage": "causal core-gold candidate construction",
        "seed": args.seed,
        "per_dataset_requested": args.per_dataset,
        "context_mode": "official supporting documents only",
        "datasets": {
            "hotpotqa": {
                "raw_rows": hotpot_total,
                "eligible_rows": hotpot_eligible,
                "selected": len(hotpot_raw),
                "accepted": len(hotpot),
                "rejected": len(hotpot_rejected),
            },
            "musique": {
                "raw_rows": musique_total,
                "eligible_rows": musique_eligible,
                "selected": len(musique_raw),
                "accepted": len(musique),
                "rejected": len(musique_rejected),
            },
        },
        "accepted_total": len(accepted),
        "rejection_reasons": dict(Counter(
            reason for row in rejected for reason in row["reasons"]
        )),
        "excluded_datasets": {
            "natural_questions": (
                "Current NQ inputs lack original gold evidence spans; proxy evidence is excluded "
                "from the high-confidence causal core."
            )
        },
    }
    (args.output / "core_gold_candidate_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
