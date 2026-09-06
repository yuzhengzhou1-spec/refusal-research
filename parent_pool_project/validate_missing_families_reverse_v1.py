#!/usr/bin/env python3
"""Frozen adversarial counterexample gate for MISSING families."""

import validate_missing_families_v1 as base


PIPELINE_VERSION = "missing_reverse_semantic_gate.v1_1_frozen"
PROMPT_VERSION = "missing_reverse_gate.direct_defect_search.v1_1_frozen"
SYSTEM_PROMPT = """Act as an adversarial counterexample finder for one synthetic MISSING-information family. Judge only from the supplied fields; do not repair the sample and do not use outside or parametric knowledge.

Your job is to try to invalidate the sample, not to agree with its constructor. Search all MISSING documents jointly for five defects. Every defect field means exactly "this defect exists":
1. remaining_answer_chain_found: true only if surviving evidence itself still directly or through a complete chain establishes missing_information or the accepted answer under every query condition. A bare entity mention is insufficient. Query wording is never a premise and cannot complete a chain.
2. unsupported_known_support_found: true only if the clarification's opening factual claim is not fully entailed by the single recorded clarification_grounding span.
3. query_promoted_to_evidence_found: true only if the clarification presents a fact found only in the query as something the context establishes. Query constraints may appear in the unresolved request without triggering this defect.
4. gap_defect_found: true only if the clarification requests the wrong or multiple slots, leaks the answer, treats an incomplete list as exhaustive, or otherwise misstates the edit gap.
5. restoration_defect_found: true only if restoring_information is stronger than original_span or fails to restore a complete evidence-supported route to the accepted answer.

Set valid to true if and only if all five defect booleans are false. Do not mark a defect merely because the query plus world knowledge suggests an answer. If a defect is true, quote the decisive supplied text in reason. If no defect is found, briefly state why the missing relation remains unavailable and restoration is sufficient. Always return a non-empty reason. Return JSON only."""
SCHEMA = {
    "valid": True,
    "remaining_answer_chain_found": False,
    "unsupported_known_support_found": False,
    "query_promoted_to_evidence_found": False,
    "gap_defect_found": False,
    "restoration_defect_found": False,
    "failure_codes": [],
    "reason": "Brief adversarial justification.",
}
DIMENSIONS = (
    "remaining_answer_chain_found",
    "unsupported_known_support_found",
    "query_promoted_to_evidence_found",
    "gap_defect_found",
    "restoration_defect_found",
)


def derived_valid(result):
    return not any(result[field] for field in DIMENSIONS)


def dimension_failed(result, field):
    return result.get(field) is True


def configure() -> None:
    base.PIPELINE_VERSION = PIPELINE_VERSION
    base.PROMPT_VERSION = PROMPT_VERSION
    base.SYSTEM_PROMPT = SYSTEM_PROMPT
    base.SCHEMA = SCHEMA
    base.DIMENSIONS = DIMENSIONS
    base.derived_valid = derived_valid
    base.dimension_failed = dimension_failed


if __name__ == "__main__":
    configure()
    raise SystemExit(base.main())
