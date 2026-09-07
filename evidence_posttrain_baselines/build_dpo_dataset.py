from __future__ import annotations

import argparse
import json

from common import load_experiment, load_yaml, make_initial_messages, normalized_trajectory, read_jsonl, resolve_output_artifact, resolve_root_path, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="把基座模型在训练集上的真实错误转换为DPO偏好对")
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--train-config", default="configs/training/dpo_lora.yaml")
    parser.add_argument("--predictions", help="覆盖训练配置中的source_predictions")
    args = parser.parse_args()
    experiment, _, task = load_experiment(args.config)
    training = load_yaml(args.train_config)
    processed = resolve_root_path(experiment["paths"]["processed_dir"])
    families = {row["family_id"]: row for row in read_jsonl(processed / "families_train.jsonl")}
    predictions = read_jsonl(resolve_output_artifact(experiment, args.predictions or training["source_predictions"]))
    pairs, skipped = [], []
    for prediction in predictions:
        record = families.get(prediction["family_id"])
        if record is None:
            skipped.append({"family_id": prediction["family_id"], "reason": "NOT_IN_TRAIN_SPLIT"})
            continue
        if prediction["scores"]["initial_action_correct"]:
            skipped.append({"family_id": prediction["family_id"], "reason": "BASE_RESPONSE_ALREADY_CORRECT"})
            continue
        attempts = prediction["initial"].get("raw_attempts", [])
        if not attempts or not prediction["initial"].get("parsed"):
            skipped.append({"family_id": prediction["family_id"], "reason": "UNPARSEABLE_REJECTED"})
            continue
        chosen = normalized_trajectory(record)[0]["content"]
        rejected = attempts[-1]
        if chosen.strip() == rejected.strip():
            skipped.append({"family_id": prediction["family_id"], "reason": "IDENTICAL_PAIR"})
            continue
        pairs.append({
            "prompt": make_initial_messages(record, task, full=False),
            "chosen": [{"role": "assistant", "content": chosen}],
            "rejected": [{"role": "assistant", "content": rejected}],
            "metadata": {"family_id": record["family_id"], "source_sample_id": record["source_sample_id"], "dataset": record["dataset"], "state": record["experiment_state"]},
        })
    write_jsonl(processed / "dpo_train.jsonl", pairs)
    write_jsonl(processed / "dpo_skipped.jsonl", skipped)
    print(json.dumps({"preference_pairs": len(pairs), "skipped": len(skipped)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
