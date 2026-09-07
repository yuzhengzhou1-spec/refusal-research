from __future__ import annotations

import argparse
import asyncio
import json
from collections import defaultdict
from typing import Any

from common import answer_is_correct, load_experiment, make_user_content, read_jsonl, resolve_root_path, write_jsonl
from run_vllm_eval import call_json


async def evaluate_control(client: Any, semaphore: asyncio.Semaphore, control: dict[str, Any], model: dict[str, Any], task: dict[str, Any], inference: dict[str, Any], retries: int) -> dict[str, Any]:
    messages = [
        {"role": "system", "content": task["system_prompt"]},
        {"role": "user", "content": make_user_content(control["query"], control["evidence"], task["response_schema"])},
    ]
    result = await call_json(client, semaphore, messages, model, inference, retries)
    parsed = result["parsed"] or {}
    action = parsed.get("action")
    gold = {"answer": control["expected_answer"], "answer_aliases": []}
    return {
        "control_id": control["control_id"], "family_id": control["family_id"],
        "control_type": control["control_type"], "expected_action": control["expected_action"],
        "expected_answer": control["expected_answer"],
        "model": model["name"], "raw_attempts": result["raw_attempts"], "parsed": result["parsed"],
        "scores": {
            "action_correct": action == control["expected_action"],
            "answer_correct": action == "ANSWER" and answer_is_correct(parsed.get("answer", ""), gold),
            "success": action == control["expected_action"] and (control["expected_action"] != "ANSWER" or answer_is_correct(parsed.get("answer", ""), gold)),
        },
    }


async def async_main(args: argparse.Namespace) -> None:
    from openai import AsyncOpenAI

    experiment, model, task = load_experiment(args.config)
    served_name = args.served_model_name or experiment["inference"].get("served_model_name_override")
    if served_name:
        model = {**model, "served_model_name": served_name}
    processed = resolve_root_path(experiment["paths"]["processed_dir"])
    controls = read_jsonl(processed / f"controls_{args.split}.jsonl")
    inference = experiment["inference"]
    client = AsyncOpenAI(base_url=inference["base_url"], api_key=inference["api_key"], timeout=inference["timeout_seconds"])
    semaphore = asyncio.Semaphore(int(inference["concurrency"]))
    jobs = [evaluate_control(client, semaphore, c, model, task, inference, int(experiment["task"]["parse_retry"])) for c in controls]
    results = await asyncio.gather(*jobs)
    output = resolve_root_path(experiment["paths"]["output_dir"]) / args.run_name
    if output.exists():
        raise FileExistsError(f"输出目录已存在：{output}")
    output.mkdir(parents=True)
    write_jsonl(output / "controls_predictions.jsonl", results)
    by_type: dict[str, dict[str, Any]] = {}
    for ctype in sorted({r["control_type"] for r in results}):
        rows = [r for r in results if r["control_type"] == ctype]
        by_type[ctype] = {
            "count": len(rows),
            "action_correct": sum(r["scores"]["action_correct"] for r in rows) / len(rows),
            "answer_correct": sum(r["scores"]["answer_correct"] for r in rows) / len(rows),
            "success": sum(r["scores"]["success"] for r in rows) / len(rows),
        }
    report = {"run_name": args.run_name, "model": model, "split": args.split, "by_control_type": by_type,
              "overall_success": sum(r["scores"]["success"] for r in results) / len(results)}
    (output / "controls_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="反事实对照组评测: 模型是否真识别证据充分性而非模板匹配")
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--split", choices=["train", "dev", "test"], default="test")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--served-model-name", help="评测LoRA时覆盖vLLM中的模型别名")
    asyncio.run(async_main(parser.parse_args()))


if __name__ == "__main__":
    main()
