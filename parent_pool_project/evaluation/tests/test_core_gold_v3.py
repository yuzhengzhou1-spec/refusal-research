from __future__ import annotations

import sys
import unittest
from pathlib import Path

EVALUATION = Path(__file__).resolve().parents[1]
if str(EVALUATION) not in sys.path:
    sys.path.insert(0, str(EVALUATION))

import filter_core_gold_v3 as pipeline  # noqa: E402


class CoreGoldV3Tests(unittest.TestCase):
    def test_qwen_remains_default_and_deepseek_optional(self) -> None:
        self.assertEqual(pipeline.DEFAULT_MODELS, ("qwen3_8b",))
        self.assertIn("deepseek_v4_flash", pipeline.SUPPORTED_MODELS)

    def test_stage_versions_reuse_query_and_full_only(self) -> None:
        self.assertEqual(pipeline.version_for_stage("QUERY_ONLY"), pipeline.V2_VERSION)
        self.assertEqual(pipeline.version_for_stage("CLEAN_FULL"), pipeline.V2_VERSION)
        self.assertEqual(pipeline.version_for_stage("COUNTERFACTUAL"), pipeline.PIPELINE_VERSION)

    def test_date_range_replaces_every_year(self) -> None:
        replacement = pipeline.counterfactual_answer("1977 and 1985", "range")
        self.assertNotIn("1977", replacement)
        self.assertNotIn("1985", replacement)

    def test_nationality_is_type_compatible(self) -> None:
        replacement = pipeline.counterfactual_answer(
            "Scottish", "nationality", "What nationality was the writer?"
        )
        self.assertEqual(replacement, "Lumerian")

    def test_thousands_format_is_preserved(self) -> None:
        self.assertEqual(
            pipeline.counterfactual_answer("273,282", "number", "How many units?"),
            "273,289",
        )


if __name__ == "__main__":
    unittest.main()
