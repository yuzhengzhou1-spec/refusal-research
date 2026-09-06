import json
import tempfile
import unittest
from pathlib import Path

import evaluate_family_qwen_baseline_v1 as baseline
import run_family_pipeline_v1 as runner
import select_missing_high_quality_v1 as selector
import validate_conflict_families_v1 as conflict_forward
import validate_missing_families_reverse_v1 as missing_reverse


class FamilyFreezeTests(unittest.TestCase):
    def test_blind_selection_is_balanced_and_excludes_development(self):
        rows = []
        for dataset in ("hotpotqa", "musique"):
            for index in range(5):
                rows.append({"样本ID": f"{dataset}-{index}", "来源": dataset})
        selected = runner.select_blind_rows(rows, {"hotpotqa-0", "musique-0"}, 7, 2)
        ids = {row["样本ID"] for row in selected}
        self.assertEqual(len(selected), 4)
        self.assertFalse(ids.intersection({"hotpotqa-0", "musique-0"}))

    def test_collect_ids_supports_call_and_family_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ids.jsonl"
            path.write_text(
                json.dumps({"sample_id": "a"}) + "\n"
                + json.dumps({"family": {"source_sample_id": "b"}}) + "\n",
                encoding="utf-8",
            )
            self.assertEqual(runner.collect_ids(path), {"a", "b"})

    def test_missing_selector_requires_both_gates(self):
        passed = {"status": "PASS"}
        failed = {"status": "FAIL"}
        self.assertEqual(selector.exclusion_reasons(passed, passed), [])
        self.assertIn("REVERSE_SEMANTIC_GATE_NOT_PASS", selector.exclusion_reasons(passed, failed))

    def test_missing_reverse_gate_is_adversarial_variant(self):
        self.assertIn("remaining_answer_chain_found", missing_reverse.DIMENSIONS)
        self.assertIn("counterexample", missing_reverse.SYSTEM_PROMPT.lower())
        self.assertTrue(missing_reverse.derived_valid({field: False for field in missing_reverse.DIMENSIONS}))

    def test_conflict_forward_dimensions_are_frozen(self):
        self.assertIn("single_disputed_slot", conflict_forward.DIMENSIONS)
        self.assertEqual(len(conflict_forward.DIMENSIONS), 7)

    def test_baseline_response_contract(self):
        self.assertEqual(baseline.response_errors({"action": "ANSWER", "answer": "x", "clarification": "", "reason": "r"}), [])
        errors = baseline.response_errors({"action": "REQUEST_INFORMATION", "answer": "x", "clarification": "?", "reason": "r"})
        self.assertIn("request action has non-empty answer", errors)


if __name__ == "__main__":
    unittest.main()
