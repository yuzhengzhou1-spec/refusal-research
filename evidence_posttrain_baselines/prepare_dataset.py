from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from common import (
    ROOT,
    canonical_response,
    load_experiment,
    make_initial_messages,
    normalized_trajectory,
    parse_response,
    read_jsonl,
    resolve_root_path,
    write_jsonl,
)


REQUIRED = {
    "source_sample_id", "family_id", "dataset", "query", "answer", "answer_aliases",
    "full_state", "initial_state", "trajectory",
}


def validate(records: list[dict[str, Any]]) -> None:
    seen = set()
    for record in records:
        missing = REQUIRED - record.keys()
        if missing:
            raise ValueError(f"{record.get('family_id')} 缺少字段：{sorted(missing)}")
        if record["family_id"] in seen:
            raise ValueError(f"family_id 重复：{record['family_id']}")
        seen.add(record["family_id"])
        if "evidence" not in record["full_state"] or "evidence" not in record["initial_state"]:
            raise ValueError(f"{record['family_id']} 缺少 evidence")
        normalized_trajectory(record)


def state_of(record: dict[str, Any]) -> str:
    state = str(record["initial_state"].get("state", "")).upper()
    if "MISSING" in state or "missing_information" in record["initial_state"]:
        return "MISSING"
    if "CONFLICT" in state or "conflict" in record["initial_state"]:
        return "CONFLICT"
    family = str(record.get("family_id", "")).upper()
    if "MISSING" in family:
        return "MISSING"
    if "CONFLICT" in family:
        return "CONFLICT"
    raise ValueError(f"无法判断状态：{record['family_id']}")


def split_groups(records: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, str]:
    """按源问题分组，并搜索状态/数据集分布最平衡的确定性切分。"""
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[str(record["source_sample_id"])].append(record)
    ids = sorted(groups)
    ratios = [config["train_ratio"], config["dev_ratio"], config["test_ratio"]]
    names = ["train", "dev", "test"]
    train_size = round(len(ids) * ratios[0])
    dev_size = round(len(ids) * ratios[1])
    sizes = [train_size, dev_size, len(ids) - train_size - dev_size]
    strata = Counter((state_of(row), row["dataset"]) for row in records)

    best_score = float("inf")
    best_assignment: dict[str, str] = {}
    for trial in range(int(config.get("search_trials", 2000))):
        shuffled = ids.copy()
        random.Random(int(config["seed"]) + trial).shuffle(shuffled)
        chunks, cursor = {}, 0
        for name, size in zip(names, sizes):
            chunks[name] = shuffled[cursor:cursor + size]
            cursor += size
        score = 0.0
        for index, name in enumerate(names):
            observed = Counter(
                (state_of(row), row["dataset"])
                for group_id in chunks[name]
                for row in groups[group_id]
            )
            for stratum, total in strata.items():
                target = total * ratios[index]
                score += ((observed[stratum] - target) / max(1.0, target)) ** 2
        if score < best_score:
            best_score = score
            best_assignment = {group_id: name for name, chunk in chunks.items() for group_id in chunk}
    return best_assignment


def full_signature(record: dict[str, Any]) -> str:
    value = {"query": record["query"], "evidence": record["full_state"]["evidence"], "answer": record["answer"]}
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


def make_dialogues(record: dict[str, Any], task: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    trajectory = normalized_trajectory(record)
    state_messages = make_initial_messages(record, task, full=False) + trajectory
    full_messages = make_initial_messages(record, task, full=True) + [
        {"role": "assistant", "content": canonical_response("ANSWER", str(record["answer"]))}
    ]
    metadata = {
        "family_id": record["family_id"], "source_sample_id": record["source_sample_id"],
        "dataset": record["dataset"], "state": state_of(record),
    }
    return {"messages": state_messages, "metadata": metadata}, {"messages": full_messages, "metadata": {**metadata, "state": "FULL"}}


def main() -> None:
    parser = argparse.ArgumentParser(description="校验并划分样本家族，生成SFT与评测数据")
    parser.add_argument("--config", default="configs/experiment.yaml")
    args = parser.parse_args()
    experiment, _, task = load_experiment(args.config)
    paths = experiment["paths"]
    missing = read_jsonl(resolve_root_path(paths["missing_source"]))
    conflict = read_jsonl(resolve_root_path(paths["conflict_source"]))
    records = missing + conflict
    validate(records)
    assignment = split_groups(records, experiment["split"])
    output = resolve_root_path(paths["processed_dir"])
    output.mkdir(parents=True, exist_ok=True)

    split_records: dict[str, list[dict[str, Any]]] = {name: [] for name in ("train", "dev", "test")}
    for record in records:
        name = assignment[str(record["source_sample_id"])]
        enriched = {**record, "experiment_state": state_of(record), "split": name}
        split_records[name].append(enriched)

    signatures: dict[str, str] = {}
    summary: dict[str, Any] = {"total_families": len(records), "unique_source_samples": len(assignment), "splits": {}}
    for name, rows in split_records.items():
        write_jsonl(output / f"families_{name}.jsonl", rows)
        dialogues, initial_dialogues, full_dialogues = [], [], []
        for row in rows:
            state_dialogue, full_dialogue = make_dialogues(row, task)
            dialogues.append(state_dialogue)
            initial_dialogues.append({
                "messages": state_dialogue["messages"][:3],
                "metadata": {**state_dialogue["metadata"], "training_view": "initial_action_only"},
            })
            source_id = str(row["source_sample_id"])
            signature = full_signature(row)
            if source_id in signatures and signatures[source_id] != signature:
                raise ValueError(f"同源样本的FULL状态不一致：{source_id}")
            if source_id not in signatures:
                signatures[source_id] = signature
                full_dialogues.append(full_dialogue)
        write_jsonl(output / f"sft_{name}.jsonl", dialogues + full_dialogues)
        write_jsonl(output / f"sft_initial_only_{name}.jsonl", initial_dialogues + full_dialogues)
        write_jsonl(output / f"sft_full_only_{name}.jsonl", full_dialogues)
        summary["splits"][name] = {
            "families": len(rows),
            "source_samples": len({str(row["source_sample_id"]) for row in rows}),
            "states": dict(Counter(row["experiment_state"] for row in rows)),
            "datasets": dict(Counter(row["dataset"] for row in rows)),
            "sft_dialogues": len(dialogues) + len(full_dialogues),
        }

    source_splits = defaultdict(set)
    for source_id, name in assignment.items():
        source_splits[name].add(source_id)
    if any(source_splits[a] & source_splits[b] for a, b in (("train", "dev"), ("train", "test"), ("dev", "test"))):
        raise AssertionError("source_sample_id 发生跨集合泄漏")
    (output / "split_manifest.json").write_text(
        json.dumps({"assignment": assignment, "summary": summary}, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
