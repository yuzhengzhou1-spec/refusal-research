#!/usr/bin/env python3
"""Frozen independent six-dimension semantic gate for CONFLICT families."""

import validate_conflict_families_v1 as base


PIPELINE_VERSION = "conflict_semantic_gate.v1_3_frozen"
PROMPT_VERSION = "conflict_semantic_gate.atomic_necessity.v1_3_frozen"
SYSTEM_PROMPT = """You are an independent strict semantic quality gate for a synthetic CONFLICT evidence sample. Judge only; do not repair the sample or use external knowledge.

The appended sentence is intentionally synthetic and does not need external support. Test six conditions:
1. same_atomic_fact: both claims assign different values to the same subject, relation, scope, time, and conditions.
2. mutually_exclusive: both values cannot coexist under that scope. Lists, memberships, siblings, guest appearances, collaborators, repeated events, awards at different editions, and proximity relations are normally multi-valued and fail.
3. necessary_for_query: the disputed fact directly supplies the answer or an indispensable unique bridge. Query wording is not evidence.
4. changes_query_answer: using the original versus conflicting claim changes the answer candidate or breaks the indispensable bridge.
5. not_resolvable_without_user: unaffected evidence, titles, query constraints, arithmetic, dates, or deterministic relations cannot independently choose one value.
6. resolution_restores_answer: confirming the original claim restores a complete evidence-supported route to the accepted answer.

Fail fragments, weak proxies, incidental attributes, and any conflict where both claims still lead to the same answer. Set valid true only when all six booleans are true. Always provide a concise non-empty reason. Return JSON only."""
SCHEMA = {
    "valid": True,
    "same_atomic_fact": True,
    "mutually_exclusive": True,
    "necessary_for_query": True,
    "changes_query_answer": True,
    "not_resolvable_without_user": True,
    "resolution_restores_answer": True,
    "failure_codes": [],
    "reason": "Brief evidence-based justification.",
}
DIMENSIONS = (
    "same_atomic_fact",
    "mutually_exclusive",
    "necessary_for_query",
    "changes_query_answer",
    "not_resolvable_without_user",
    "resolution_restores_answer",
)


def configure() -> None:
    base.PIPELINE_VERSION = PIPELINE_VERSION
    base.PROMPT_VERSION = PROMPT_VERSION
    base.SYSTEM_PROMPT = SYSTEM_PROMPT
    base.SCHEMA = SCHEMA
    base.DIMENSIONS = DIMENSIONS


if __name__ == "__main__":
    configure()
    raise SystemExit(base.main())
