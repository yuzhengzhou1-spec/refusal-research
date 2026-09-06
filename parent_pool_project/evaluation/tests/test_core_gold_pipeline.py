from __future__ import annotations

import sys
import unittest
from pathlib import Path

EVALUATION = Path(__file__).resolve().parents[1]
PROJECT = EVALUATION.parent
for path in (str(EVALUATION), str(PROJECT)):
    if path not in sys.path:
        sys.path.insert(0, path)

import build_core_gold_candidates as candidates  # noqa: E402
import filter_core_gold as pipeline  # noqa: E402


def sample(answer: str = "Benedict Arnold") -> dict:
    return {
        "模式版本": "core_gold_candidate.zh.v2",
        "样本ID": "hotpotqa_x1",
        "来源": "hotpotqa",
        "原始样本ID": "x1",
        "问题": "Who was the general described in the documents?",
        "答案": [answer],
        "完整上下文": [
            {
                "文档ID": "hp_doc_1",
                "标题": answer,
                "文本": f"{answer} was the general described in the documents.",
                "句子": [f"{answer} was the general described in the documents."],
                "是否支撑文档": True,
            }
        ],
        "关键证据单元": [
            {"文档ID": "hp_doc_1", "文本": f"{answer} was the general."}
        ],
        "问题分解": [],
        "跳数": 1,
        "推理类型": "single_hop",
        "验证": {},
        "元数据": {"包含干扰文档": False},
    }


class CandidateTests(unittest.TestCase):
    def test_stable_take_is_prefix_stable(self) -> None:
        rows = [{"id": str(index)} for index in range(100)]
        ten, _, _ = candidates.stable_take(rows, "hotpotqa", lambda _: True, 10, 7)
        twenty, _, _ = candidates.stable_take(rows, "hotpotqa", lambda _: True, 20, 7)
        self.assertEqual([row["id"] for row in ten], [row["id"] for row in twenty[:10]])

    def test_supporting_only_removes_distractor(self) -> None:
        row = sample()
        row["完整上下文"].append({
            "文档ID": "hp_doc_9", "标题": "noise", "文本": "noise",
            "句子": ["noise"], "是否支撑文档": False,
        })
        clean = candidates.supporting_only(row)
        self.assertEqual([doc["文档ID"] for doc in clean["完整上下文"]], ["hp_doc_1"])
        self.assertFalse(clean["元数据"]["包含干扰文档"])


class CounterfactualTests(unittest.TestCase):
    def test_entity_counterfactual_replaces_title_and_text(self) -> None:
        variant = pipeline.build_counterfactual(sample())
        rendered = pipeline.shared.evidence_text(variant["counterfactual_evidence"])
        self.assertNotIn("Benedict Arnold", rendered)
        self.assertIn(variant["counterfactual_answer"], rendered)
        self.assertGreaterEqual(variant["replacement_count"], 2)
        self.assertTrue(variant["diagnostic_only"])

    def test_date_counterfactual_changes_date_naturally(self) -> None:
        answer = "March 10, 1949"
        replacement = pipeline.counterfactual_answer(answer, "date_sample")
        self.assertNotEqual(pipeline.normalize(answer), pipeline.normalize(replacement))
        self.assertRegex(replacement, r"[A-Za-z]+\s+\d{1,2},\s+\d{4}")

    def test_numeric_counterfactual_preserves_suffix(self) -> None:
        self.assertEqual(pipeline.counterfactual_answer("42 percent", "n"), "49 percent")

    def test_clean_structure_rejects_answer_in_query(self) -> None:
        row = sample()
        row["问题"] = "Was Benedict Arnold the general?"
        result = pipeline.clean_structure_check([row])
        self.assertFalse(result["passed"])
        self.assertIn("answer alias occurs in query", result["errors"][0])

    def test_default_model_is_qwen_only(self) -> None:
        self.assertEqual(pipeline.DEFAULT_MODELS, ("qwen3_8b",))
        self.assertIn("deepseek_v4_flash", pipeline.SUPPORTED_MODELS)


if __name__ == "__main__":
    unittest.main()
