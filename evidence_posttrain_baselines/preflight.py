from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path

from common import load_experiment, model_path, read_jsonl, resolve_root_path


def version_of(package: str) -> str:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return "NOT_INSTALLED"


def main() -> None:
    parser = argparse.ArgumentParser(description="训练前检查配置、数据、权重和关键依赖")
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--tokenizer-check", action="store_true", help="加载本地tokenizer并检查长度")
    args = parser.parse_args()
    experiment, model, _ = load_experiment(args.config)
    processed = resolve_root_path(experiment["paths"]["processed_dir"])
    weights = model_path(experiment, model)
    report = {
        "active_model": model["name"], "model_path": str(weights),
        "model_weights_ready": (weights / "config.json").is_file(),
        "processed_dir": str(processed),
        "data_files": {name: (processed / name).exists() for name in ("families_train.jsonl", "families_dev.jsonl", "families_test.jsonl", "sft_train.jsonl", "sft_dev.jsonl")},
        "packages": {name: version_of(name) for name in ("torch", "transformers", "datasets", "trl", "peft", "accelerate", "vllm", "openai")},
    }
    if args.tokenizer_check:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(weights, trust_remote_code=model["trust_remote_code"])
        lengths = []
        for split in ("train", "dev", "test"):
            path = processed / f"sft_{split}.jsonl"
            if path.exists():
                for row in read_jsonl(path):
                    tokenized = tokenizer.apply_chat_template(row["messages"], tokenize=True)
                    lengths.append(len(tokenized))
        report["token_lengths"] = {"count": len(lengths), "max": max(lengths, default=0), "configured_max": 4096, "truncated": sum(length > 4096 for length in lengths)}
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
