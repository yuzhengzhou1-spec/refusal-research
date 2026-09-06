from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path


MODULE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_ROOT))

import validate_missing1_llm as validator


class Missing1ValidatorTests(unittest.TestCase):
    def sample(self) -> dict:
        full = [{"文档ID": "d1", "标题": "City", "文本": "Paris is in France.", "句子": ["Paris is in France."]}]
        return {
            "pair_id": "s1__missing1_001",
            "source_sample_id": "s1",
            "dataset": "nq",
            "state": "MISSING",
            "missing_subtype": "MISSING_ANSWER",
            "query": "Which city is in France?",
            "answer": "Paris",
            "answer_aliases": [],
            "full_evidence": full,
            "insufficient_evidence": [{"文档ID": "d1", "标题": "City", "文本": "", "句子": []}],
            "removed_evidence": {
                "unit_id": "u1",
                "document_id": "d1",
                "sentence_id": 0,
                "text": "Paris is in France.",
                "role": "answer",
                "granularity": "sentence",
            },
        }

    def test_parse_exact_json_and_fence(self) -> None:
        raw = '{"answerable":false,"answer":"","missing_description":"city relation","reason":"not stated"}'
        parsed = validator.parse_model_json(raw)
        self.assertFalse(parsed["answerable"])
        fenced = f"```json\n{raw}\n```"
        self.assertEqual(validator.parse_model_json(fenced), parsed)

    def test_parse_rejects_wrong_schema(self) -> None:
        with self.assertRaises(ValueError):
            validator.parse_model_json('{"answerable":"false","answer":""}')

    def test_restore_is_exact(self) -> None:
        sample = self.sample()
        self.assertEqual(validator.restore_evidence(sample), sample["full_evidence"])

    def test_answer_normalization(self) -> None:
        self.assertTrue(validator.strict_answer_match("15,848", ["15848"]))
        self.assertTrue(validator.strict_answer_match("The New York", ["New York"]))

    def test_direct_answer_remaining_is_ineffective(self) -> None:
        sample = self.sample()
        sample["insufficient_evidence"][0]["标题"] = "Paris"
        support = validator.deterministic_support_check(sample)
        self.assertTrue(support["complete_support_detected"])

    def test_bridge_answer_presence_is_not_complete_support(self) -> None:
        sample = self.sample()
        sample["missing_subtype"] = "MISSING_BRIDGE"
        sample["insufficient_evidence"][0]["标题"] = "Paris"
        support = validator.deterministic_support_check(sample)
        self.assertFalse(support["complete_support_detected"])

    def test_gap_rule(self) -> None:
        sample = self.sample()
        parsed = {
            "answerable": False,
            "answer": "",
            "missing_description": "The evidence does not identify Paris or its relation to France.",
            "reason": "missing relation",
        }
        self.assertEqual(validator.gap_match(sample, parsed)["label"], "GAP_MATCH")

    def test_messages_do_not_include_hidden_labels(self) -> None:
        sample = self.sample()
        messages = validator.build_messages(sample["query"], sample["insufficient_evidence"])
        serialized = json.dumps(messages)
        self.assertNotIn("removed_evidence", serialized)
        self.assertNotIn("MISSING_ANSWER", serialized)
        self.assertNotIn(sample["answer"], serialized)


if __name__ == "__main__":
    unittest.main()
