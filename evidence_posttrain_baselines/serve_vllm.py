from __future__ import annotations

import argparse
import shlex
import subprocess

from common import load_experiment, model_path


def parse_adapters(entries: list[str] | None, served_model_name: str) -> list[tuple[str, str]]:
    """把 --adapter 条目解析为(别名, 路径)列表; 条目形如 name=path 或纯路径(别名自动派生)。"""
    adapters: list[tuple[str, str]] = []
    for entry in entries or []:
        if "=" in entry:
            alias, path = entry.split("=", 1)
            adapters.append((alias.strip(), path.strip()))
        else:
            from pathlib import Path

            adapters.append((f"{served_model_name}-{Path(entry.strip()).name}", entry.strip()))
    return adapters


def build_command(experiment, model, adapters: list[tuple[str, str]] | None = None, port: int = 8000, max_lora_rank: int | None = None) -> list[str]:
    config = model["vllm"]
    command = [
        "vllm", "serve", str(model_path(experiment, model)),
        "--served-model-name", model["served_model_name"],
        "--tensor-parallel-size", str(config["tensor_parallel_size"]),
        "--dtype", str(config["dtype"]),
        "--max-model-len", str(config["max_model_len"]),
        "--gpu-memory-utilization", str(config["gpu_memory_utilization"]),
        "--port", str(port),
    ]
    if model.get("trust_remote_code"):
        command.append("--trust-remote-code")
    if adapters:
        command.extend(["--enable-lora", "--lora-modules", *[f"{alias}={path}" for alias, path in adapters]])
        if max_lora_rank:
            command.extend(["--max-lora-rank", str(max_lora_rank)])
    return command


def main() -> None:
    parser = argparse.ArgumentParser(description="从统一配置启动vLLM服务")
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--adapter", action="append", help="LoRA适配器, 可重复传入; 形如 name=path 或纯路径(别名自动派生)")
    parser.add_argument("--adapter-name", help="单个适配器时的别名(兼容旧用法)")
    parser.add_argument("--run", action="store_true", help="默认只打印命令；指定后真正启动")
    args = parser.parse_args()
    experiment, model, _ = load_experiment(args.config)
    inference = experiment["inference"]
    entries = list(args.adapter or [])
    if not entries and inference.get("adapter_path"):
        legacy_name = args.adapter_name or inference.get("adapter_name") or f"{model['served_model_name']}-lora"
        entries = [f"{legacy_name}={inference['adapter_path']}"]
    if len(entries) == 1 and args.adapter_name and "=" not in entries[0]:
        entries = [f"{args.adapter_name}={entries[0]}"]
    adapters = parse_adapters(entries, model["served_model_name"])
    command = build_command(
        experiment, model, adapters,
        port=int(inference.get("port", 8000)), max_lora_rank=inference.get("max_lora_rank"),
    )
    print(shlex.join(command))
    if args.run:
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
