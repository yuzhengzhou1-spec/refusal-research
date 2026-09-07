from __future__ import annotations

import argparse
import json

from analyze_predictions import clarification_quality
from common import (
    answer_is_correct,
    load_experiment,
    load_yaml,
    make_initial_messages,
    normalized_trajectory,
    read_jsonl,
    resolve_output_artifact,
    resolve_root_path,
    write_jsonl,
)


def pair(prompt: list[dict], chosen_text: str, rejected_text: str, record: dict, category: str) -> dict:
    return {
        "prompt": prompt,
        "chosen": [{"role": "assistant", "content": chosen_text}],
        "rejected": [{"role": "assistant", "content": rejected_text}],
        "metadata": {
            "family_id": record["family_id"], "source_sample_id": record["source_sample_id"],
            "dataset": record["dataset"], "state": record["experiment_state"], "category": category,
        },
    }


def build_under_refusal(predictions: list[dict], families: dict, task: dict) -> tuple[list[dict], list[dict]]:
    """该问没问: chosen=gold澄清首步, rejected=模型硬答/错误动作。"""
    pairs, skipped = [], []
    for prediction in predictions:
        record = families.get(prediction["family_id"])
        if record is None:
            continue
        if prediction["scores"]["initial_action_correct"]:
            continue
        attempts = prediction["initial"].get("raw_attempts", [])
        if not attempts or not prediction["initial"].get("parsed"):
            skipped.append({"family_id": prediction["family_id"], "reason": "UNPARSEABLE_REJECTED"})
            continue
        chosen = normalized_trajectory(record)[0]["content"]
        if chosen.strip() == attempts[-1].strip():
            skipped.append({"family_id": prediction["family_id"], "reason": "IDENTICAL_PAIR"})
            continue
        pairs.append(pair(make_initial_messages(record, task, full=False), chosen, attempts[-1], record, "UNDER_REFUSAL"))
    return pairs, skipped


def build_over_refusal_guard(overrefusal_predictions: list[dict], families: dict, task: dict) -> tuple[list[dict], list[dict]]:
    """过度拒答保护: B4在FULL态仍请求信息, chosen=gold ANSWER, rejected=模型的请求。"""
    pairs, skipped = [], []
    for prediction in overrefusal_predictions:
        record = families.get(prediction["family_id"])
        if record is None:
            continue
        action = (prediction["full"].get("parsed") or {}).get("action")
        attempts = prediction["full"].get("raw_attempts", [])
        if action in (None, "ANSWER") or not attempts:
            continue  # FULL态规范作答的不构成保护样本; 解析失败不采(rejected语义不清)
        answer_turn = normalized_trajectory(record)[-1]  # 轨迹固定assistant-user-assistant, 末轮即gold答案
        if answer_turn["content"].strip() == attempts[-1].strip():
            skipped.append({"family_id": prediction["family_id"], "reason": "IDENTICAL_PAIR"})
            continue
        pairs.append(pair(make_initial_messages(record, task, full=True), answer_turn["content"], attempts[-1], record, "OVER_REFUSAL_GUARD"))
    return pairs, skipped


def build_precise_clarification(predictions: list[dict], families: dict, task: dict) -> list[dict]:
    """动作对了但澄清质量差(未瞄准缺口/泄露答案): chosen=gold澄清, rejected=模型的含糊澄清。"""
    pairs = []
    for prediction in predictions:
        record = families.get(prediction["family_id"])
        if record is None or not prediction["scores"]["initial_action_correct"]:
            continue
        quality = clarification_quality(prediction, record)
        if not quality["evaluated"] or (quality["targeted"] and not quality["leaks_answer"]):
            continue
        attempts = prediction["initial"].get("raw_attempts", [])
        if not attempts:
            continue
        chosen = normalized_trajectory(record)[0]["content"]
        if chosen.strip() == attempts[-1].strip():
            continue
        pairs.append(pair(make_initial_messages(record, task, full=False), chosen, attempts[-1], record, "PRECISE_CLARIFICATION"))
    return pairs


def build_timely_recovery(predictions: list[dict], families: dict, task: dict) -> list[dict]:
    """用户已补充信息后应及时作答: chosen=gold恢复答案轮, rejected=模型续轮仍在问/答错。"""
    pairs = []
    for prediction in predictions:
        record = families.get(prediction["family_id"])
        if record is None or not prediction["scores"]["initial_action_correct"]:
            continue
        continuation = prediction["continuation"]
        if continuation.get("status") == "NOT_RUN" or not continuation.get("raw_attempts"):
            continue
        parsed = continuation.get("parsed") or {}
        answer = parsed.get("answer", "")
        if parsed.get("action") == "ANSWER" and answer_is_correct(answer, record):
            continue  # 恢复成功, 不构成样本
        initial_messages = make_initial_messages(record, task, full=False)
        turns = normalized_trajectory(record)
        prompt = initial_messages + [
            {"role": "assistant", "content": prediction["initial"]["raw_attempts"][-1]},
            {"role": "user", "content": turns[1]["content"]},
        ]
        answer_turn = turns[-1]  # 末轮即gold恢复答案
        if answer_turn["content"].strip() == continuation["raw_attempts"][-1].strip():
            continue
        pairs.append(pair(prompt, answer_turn["content"], continuation["raw_attempts"][-1], record, "TIMELY_RECOVERY"))
    return pairs


def main() -> None:
    parser = argparse.ArgumentParser(description="四类DPO偏好对: 请求不足/拒答过度/澄清精度/恢复时效")
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--train-config", default="configs/training/dpo_lora.yaml")
    parser.add_argument("--predictions", help="覆盖训练配置中的source_predictions(B3在train分割)")
    args = parser.parse_args()
    experiment, _, task = load_experiment(args.config)
    training = load_yaml(args.train_config)
    processed = resolve_root_path(experiment["paths"]["processed_dir"])
    families = {row["family_id"]: row for row in read_jsonl(processed / "families_train.jsonl")}
    predictions = read_jsonl(resolve_output_artifact(experiment, args.predictions or training["source_predictions"]))
    overrefusal_predictions = (
        read_jsonl(resolve_output_artifact(experiment, training["overrefusal_predictions"]))
        if training.get("overrefusal_predictions") else []
    )
    cap = int(training.get("max_per_category", 0))  # 0=不设限; 正数=各类上限(under:over按1:1配平)

    under, skipped = build_under_refusal(predictions, families, task)
    over, over_skipped = build_over_refusal_guard(overrefusal_predictions, families, task)
    precise = build_precise_clarification(predictions, families, task)
    recovery = build_timely_recovery(predictions, families, task)
    skipped += over_skipped

    def cap_to(items: list[dict]) -> list[dict]:
        return sorted(items, key=lambda p: p["metadata"]["family_id"])[:cap] if cap else items

    pairs = cap_to(under) + cap_to(over) + cap_to(precise) + cap_to(recovery)
    write_jsonl(processed / "dpo_train.jsonl", pairs)
    write_jsonl(processed / "dpo_skipped.jsonl", skipped)
    by_category = {category: sum(1 for p in pairs if p["metadata"]["category"] == category) for category in
                   ("UNDER_REFUSAL", "OVER_REFUSAL_GUARD", "PRECISE_CLARIFICATION", "TIMELY_RECOVERY")}
    print(json.dumps({"preference_pairs": len(pairs), "by_category": by_category, "available_before_cap": {
        "UNDER_REFUSAL": len(under), "OVER_REFUSAL_GUARD": len(over),
        "PRECISE_CLARIFICATION": len(precise), "TIMELY_RECOVERY": len(recovery),
    }, "skipped": len(skipped)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
