#!/usr/bin/env python3
"""Recursion-safe v4 launcher for the type-aware core-gold pipeline."""

from __future__ import annotations

import re

import filter_core_gold as base
import filter_core_gold_v3 as v3


PIPELINE_VERSION = "causal_core_gold.qwen_default.v4"
v3.PIPELINE_VERSION = PIPELINE_VERSION
base.PIPELINE_VERSION = PIPELINE_VERSION


def generic_fallback(value: str, sid: str) -> str:
    word_count = max(1, len(re.findall(r"[\w]+", value, flags=re.UNICODE)))
    chosen: list[str] = []
    for offset in range(word_count):
        candidate = base.PSEUDO_WORDS[
            base.deterministic_index(sid, offset, len(base.PSEUDO_WORDS))
        ]
        while candidate in chosen:
            candidate = base.PSEUDO_WORDS[
                (base.PSEUDO_WORDS.index(candidate) + 1) % len(base.PSEUDO_WORDS)
            ]
        chosen.append(candidate)
    if value.isupper() and word_count == 1:
        length = max(2, min(8, len(re.sub(r"\W", "", value))))
        alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        return "".join(
            alphabet[base.deterministic_index(sid, index, 26)]
            for index in range(length)
        )
    return " ".join(chosen)


def counterfactual_answer(original: str, sid: str, question_text: str = "") -> str:
    value = str(original).strip()
    if not value:
        raise ValueError("empty answer")
    kind = v3.semantic_type(question_text, value)
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
        return v3.transform_numbers(result, is_date=True)
    if kind == "NUMBER":
        return v3.transform_numbers(value, is_date=False)
    typed = {
        "PERSON": "Maren Valeris",
        "LANGUAGE_OR_NATIONALITY": "Lumerian",
        "LOCATION": "Selora",
        "ORGANIZATION": "Averon Society",
        "WORK": "The Lumerian Chronicle",
    }
    return typed.get(kind) or generic_fallback(value, sid)


# v3.build_counterfactual resolves this module global at call time.
v3.counterfactual_answer = counterfactual_answer
base.counterfactual_answer = counterfactual_answer


if __name__ == "__main__":
    raise SystemExit(base.main())
