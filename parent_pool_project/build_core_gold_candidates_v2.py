#!/usr/bin/env python3
"""Launch the core candidate builder with clean answer-type metadata labels."""

from __future__ import annotations

import re

import build_parent_pool as parent


def clean_answer_type(answer: str) -> str:
    normalized = parent.normalize_text(answer)
    if re.fullmatch(r"\d+(?:[.,]\d+)?", answer.strip()):
        return "\u6570\u503c"  # 数值
    if re.search(r"\b(?:1[0-9]{3}|20[0-9]{2})\b", answer):
        return "\u65e5\u671f\u6216\u5e74\u4efd"  # 日期或年份
    if len(normalized.split()) <= 6:
        return "\u77ed\u5b9e\u4f53"  # 短实体
    return "\u77ed\u6587\u672c"  # 短文本


parent.answer_type = clean_answer_type

import build_core_gold_candidates as builder  # noqa: E402

builder.parent.answer_type = clean_answer_type


if __name__ == "__main__":
    raise SystemExit(builder.main())
