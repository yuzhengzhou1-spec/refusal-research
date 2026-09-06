from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import prepare_core_candidates_from_parent_pool_v1 as adapter
import run_family_target_v1 as target


def parent_row() -> dict:
    return {
        "模式版本": "parent_pool.zh.v1",
        "样本ID": "hotpotqa_example",
        "来源": "hotpotqa",
        "问题": "Which city contains the tower described by the architect?",
        "答案": ["Selora"],
        "完整上下文": [
            {"文档ID": "d1", "标题": "Architect", "文本": "The architect designed Tower X.", "是否支撑文档": True},
            {"文档ID": "noise", "标题": "Noise", "文本": "Unrelated.", "是否支撑文档": False},
            {"文档ID": "d2", "标题": "Tower X", "文本": "Tower X is in Selora.", "是否支撑文档": True},
        ],
        "关键证据单元": [
            {"文档ID": "d1", "文本": "The architect designed Tower X."},
            {"文档ID": "d2", "文本": "Tower X is in Selora."},
        ],
        "跳数": 2,
        "验证": {},
        "元数据": {},
    }


def family_row(state: str) -> dict:
    if state == "missing":
        actions = ["REQUEST_INFORMATION", "PROVIDE_INFORMATION", "ANSWER"]
    else:
        actions = ["REQUEST_CONFLICT_RESOLUTION", "CONFIRM_INFORMATION", "ANSWER"]
    return {
        "source_sample_id": "hotpotqa_example",
        "dataset": "hotpotqa",
        "query": "Question?",
        "answer": "Answer",
        "seed_gate": {
            "query_only_answer_correct": False,
            "full_answer_correct": True,
        },
        "full_state": {"state": "FULL", "evidence": [{"document_id": "d1"}]},
        "initial_state": {"state": state.upper(), "evidence": [{"document_id": "d1"}]},
        "trajectory": [
            {"turn": index + 1, "action": action}
            for index, action in enumerate(actions)
        ],
    }


def test_parent_adapter_removes_distractors_without_mutating_input() -> None:
    row = parent_row()
    assert adapter.reject_reason(row) is None
    converted = adapter.supporting_only(row)
    assert [document["文档ID"] for document in converted["完整上下文"]] == ["d1", "d2"]
    assert len(row["完整上下文"]) == 3
    assert converted["元数据"]["包含干扰文档"] is False


def test_parent_adapter_rejects_answer_leakage() -> None:
    row = parent_row()
    row["问题"] = "Is the answer Selora?"
    assert adapter.reject_reason(row) == "ANSWER_ALIAS_IN_QUERY"


def test_target_pass_validation_accepts_both_frozen_states() -> None:
    assert target.validate_pass(family_row("missing"), "missing") == []
    assert target.validate_pass(family_row("conflict"), "conflict") == []


def test_target_pass_validation_rejects_dirty_query_gate() -> None:
    row = family_row("missing")
    row["seed_gate"]["query_only_answer_correct"] = True
    assert "query-only gate is not clean" in target.validate_pass(row, "missing")


def test_api_error_count_reads_nested_reports(tmp_path: Path) -> None:
    report_dir = tmp_path / "audit"
    report_dir.mkdir()
    (report_dir / "gate_report.json").write_text(
        json.dumps({"status_counts": {"API_ERROR": 3, "WORKER_ERROR": 1}}),
        encoding="utf-8",
    )
    assert target.api_error_count(tmp_path) == 4
