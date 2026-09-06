from __future__ import annotations

import argparse
import shlex
import subprocess

from common import load_experiment, model_path


def build_command(experiment, model, adapter: str | None, adapter_name: str | None) -> list[str]:
    config = model["vllm"]
    command = [
        "vllm", "serve", str(model_path(experiment, model)),
        "--served-model-name", model["served_model_name"],
        "--tensor-parallel-size", str(config["tensor_parallel_size"]),
        "--dtype", str(config["dtype"]),
        "--max-model-len", str(config["max_model_len"]),
        "--gpu-memory-utilization", str(config["gpu_memory_utilization"]),
    ]
    if model.get("trust_remote_code"):
        command.append("--trust-remote-code")
    if adapter:
        alias = adapter_name or f"{model['served_model_name']}-lora"
        command.extend(["--enable-lora", "--lora-modules", f"{alias}={adapter}"])
    return command


def main() -> None:
    parser = argparse.ArgumentParser(description="从统一配置启动vLLM服务")
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--adapter", help="可选的LoRA适配器目录")
    parser.add_argument("--adapter-name", help="适配器在OpenAI接口中的模型名")
    parser.add_argument("--run", action="store_true", help="默认只打印命令；指定后真正启动")
    args = parser.parse_args()
    experiment, model, _ = load_experiment(args.config)
    inference = experiment["inference"]
    adapter = args.adapter or inference.get("adapter_path")
    adapter_name = args.adapter_name or inference.get("adapter_name")
    command = build_command(experiment, model, adapter, adapter_name)
    print(shlex.join(command))
    if args.run:
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
