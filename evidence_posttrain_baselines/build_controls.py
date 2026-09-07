from __future__ import annotations

import argparse
import json
import re

from common import load_experiment, read_jsonl, resolve_root_path, write_jsonl

# 每类派生态的期望行为先冻结于此; 评测侧按expected_action/expected_answer判分
EXPECTATIONS = {
    "MISSING_IRRELEVANT_DELETED": "ANSWER",   # 删掉无关文档, 证据仍充分
    "CONFLICT_IRRELEVANT_ADDED": "ANSWER",    # 无关冲突不改变本题可答性
    "MISSING_RESTORED": "ANSWER",             # 缺口补回, 立即恢复作答
    "CONFLICT_RESOLVED": "ANSWER",            # 冲突消解, 立即恢复作答
    "FULL_COUNTERFACTUAL": "ANSWER",          # 证据改值后应跟随证据而非记忆
}


def reverse_missing_edit(record: dict) -> list[dict] | None:
    """把MISSING的最小编辑逆向: replacement_span换回original_span。"""
    edit = record["initial_state"].get("minimal_edit", {})
    if not edit.get("replacement_span"):
        return None
    docs = []
    for doc in record["initial_state"]["evidence"]:
        text = dict(doc)
        if doc["document_id"] == edit["document_id"]:
            if edit["replacement_span"] not in doc["text"]:
                return None
            text["text"] = doc["text"].replace(edit["replacement_span"], edit["original_span"])
        docs.append(text)
    return docs


def resolve_conflict_edit(record: dict) -> list[dict] | None:
    """删除CONFLICT追加的冲突句。"""
    edit = record["initial_state"].get("minimal_edit", {})
    appended = edit.get("appended_text")
    if not appended:
        return None
    docs = []
    for doc in record["initial_state"]["evidence"]:
        text = dict(doc)
        if doc["document_id"] == edit["document_id"]:
            if appended not in doc["text"]:
                return None
            text["text"] = doc["text"].replace(appended, "").rstrip()
        docs.append(text)
    return docs


def counterfactual_value(answer: str) -> str | None:
    """对年份/数值型答案生成反事实值(保持格式, 只改数字); 实体型答案跳过。"""
    years = re.findall(r"\b(1[5-9]\d{2}|20\d{2})\b", answer)
    if years:
        year = years[-1]
        return answer.replace(year, str(int(year) + 3), 1)
    numbers = re.findall(r"\b\d+(?:\.\d+)?\b", answer)
    if numbers:
        return answer.replace(numbers[-1], str(float(numbers[-1]) + 7 if "." in numbers[-1] else int(numbers[-1]) + 7), 1)
    return None


def build_controls(records: list[dict]) -> tuple[list[dict], list[dict]]:
    controls, skipped = [], []
    conflicts = sorted(r["family_id"] for r in records if r["experiment_state"] == "CONFLICT")
    for record in records:
        fid, gold = record["family_id"], record["answer"]
        variants = [gold, *record.get("answer_aliases", [])]

        # 删无关文档: 非编辑目标且不含答案的文档才视为无关
        non_key = [d for d in record["full_state"]["evidence"]
                   if d["document_id"] != record["initial_state"]["minimal_edit"].get("document_id")
                   and not any(v in d["text"] for v in variants if v)]
        if len(record["full_state"]["evidence"]) >= 3 and non_key:
            victim = non_key[-1]
            controls.append({"control_id": f"{fid}::MISSING_IRRELEVANT_DELETED", "family_id": fid,
                             "control_type": "MISSING_IRRELEVANT_DELETED", "expected_action": "ANSWER",
                             "expected_answer": gold, "query": record["query"],
                             "evidence": [d for d in record["full_state"]["evidence"] if d["document_id"] != victim["document_id"]]})
        else:
            skipped.append({"family_id": fid, "control": "MISSING_IRRELEVANT_DELETED", "reason": "NO_IRRELEVANT_DOC"})

        # 加无关冲突: 借用另一家族的冲突句(与本题无关)
        if conflicts:
            donor = conflicts[(conflicts.index(record["family_id"]) + 1) % len(conflicts)] if record["family_id"] in conflicts else conflicts[0]
            donor_edit = next(r for r in records if r["family_id"] == donor)["initial_state"]["minimal_edit"]
            if donor_edit.get("appended_text"):
                evidence = [dict(d) for d in record["full_state"]["evidence"]]
                host = evidence[-1]
                host["text"] = host["text"] + " " + donor_edit["appended_text"]
                controls.append({"control_id": f"{fid}::CONFLICT_IRRELEVANT_ADDED", "family_id": fid,
                                 "control_type": "CONFLICT_IRRELEVANT_ADDED", "expected_action": "ANSWER",
                                 "expected_answer": gold, "query": record["query"], "evidence": evidence})
            else:
                skipped.append({"family_id": fid, "control": "CONFLICT_IRRELEVANT_ADDED", "reason": "DONOR_NO_APPEND"})

        restored = reverse_missing_edit(record) if record["experiment_state"] == "MISSING" else None
        if restored:
            controls.append({"control_id": f"{fid}::MISSING_RESTORED", "family_id": fid,
                             "control_type": "MISSING_RESTORED", "expected_action": "ANSWER",
                             "expected_answer": gold, "query": record["query"], "evidence": restored})
        else:
            skipped.append({"family_id": fid, "control": "MISSING_RESTORED", "reason": "NOT_MISSING_OR_SPAN_MISSING"})

        resolved = resolve_conflict_edit(record) if record["experiment_state"] == "CONFLICT" else None
        if resolved:
            controls.append({"control_id": f"{fid}::CONFLICT_RESOLVED", "family_id": fid,
                             "control_type": "CONFLICT_RESOLVED", "expected_action": "ANSWER",
                             "expected_answer": gold, "query": record["query"], "evidence": resolved})
        else:
            skipped.append({"family_id": fid, "control": "CONFLICT_RESOLVED", "reason": "NOT_CONFLICT_OR_APPEND_MISSING"})

        # 反事实: 只对年份/数值型答案做, 实体型跳过
        new_value = counterfactual_value(gold)
        if new_value and any(v in d["text"] for d in record["full_state"]["evidence"] for v in variants if v):
            evidence = [dict(d) for d in record["full_state"]["evidence"]]
            for doc in evidence:
                for v in variants:
                    if v:
                        doc["text"] = doc["text"].replace(v, new_value)
            controls.append({"control_id": f"{fid}::FULL_COUNTERFACTUAL", "family_id": fid,
                             "control_type": "FULL_COUNTERFACTUAL", "expected_action": "ANSWER",
                             "expected_answer": new_value, "query": record["query"], "evidence": evidence})
        else:
            skipped.append({"family_id": fid, "control": "FULL_COUNTERFACTUAL", "reason": "NO_NUMERIC_ANSWER_OR_ABSENT_IN_EVIDENCE"})
    return controls, skipped


def main() -> None:
    parser = argparse.ArgumentParser(description="从现有家族机械构造反事实对照组(不扩主题)")
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--split", choices=["train", "dev", "test"], default="test")
    args = parser.parse_args()
    experiment, _, _ = load_experiment(args.config)
    processed = resolve_root_path(experiment["paths"]["processed_dir"])
    records = read_jsonl(processed / f"families_{args.split}.jsonl")
    controls, skipped = build_controls(records)
    write_jsonl(processed / f"controls_{args.split}.jsonl", controls)
    write_jsonl(processed / f"controls_{args.split}_skipped.jsonl", skipped)
    from collections import Counter
    print(json.dumps({"controls": len(controls), "by_type": dict(Counter(c["control_type"] for c in controls)), "skipped": len(skipped)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
