from __future__ import annotations

import sys
import unittest
from pathlib import Path

EVALUATION = Path(__file__).resolve().parents[1]
if str(EVALUATION) not in sys.path:
    sys.path.insert(0, str(EVALUATION))

import filter_core_gold_v4 as pipeline  # noqa: E402


class CoreGoldV4Tests(unittest.TestCase):
    def test_other_type_has_nonrecursive_fallback(self) -> None:
        replacement = pipeline.counterfactual_answer(
            "professional association football", "fallback", "What sport was played?"
        )
        self.assertTrue(replacement)
        self.assertNotEqual(replacement.casefold(), "professional association football")

    def test_query_and_full_still_reuse_v2(self) -> None:
        self.assertEqual(pipeline.v3.version_for_stage("QUERY_ONLY"), pipeline.v3.V2_VERSION)
        self.assertEqual(pipeline.v3.version_for_stage("CLEAN_FULL"), pipeline.v3.V2_VERSION)
        self.assertEqual(pipeline.v3.version_for_stage("COUNTERFACTUAL"), pipeline.PIPELINE_VERSION)


if __name__ == "__main__":
    unittest.main()
