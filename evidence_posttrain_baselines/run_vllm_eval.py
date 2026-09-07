from __future__ import annotations

import argparse
import asyncio
import json
from collections import Counter
from datetime import datetime
from typing import Any

from common import (
    answer_is_correct,
    load_experiment,
    make_initial_messages,
    normalized_trajectory,
    parse_response,
    read_jsonl,
    resolve_root_path,
    write_jsonl,
)


def expected_action(record: dict[str, Any]) -> str:
    return "REQUEST_INFORMATION" if record["experiment_state"] == "MISSING" else "REQUEST_CONFIRMATION"


# 冻结协议的正式JSON Schema, 与task_prompt.yaml的response_schema示例及parse_response约定一致;
# --guided-json时经vLLM structured outputs(outlines/xgrammar)强制输出合法, 用于剥离协议学习成分
GUIDED_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "action": {"type": "string", "enum": ["ANSWER", "REQUEST_INFORMATION", "REQUEST_CONFIRMATION"]},
        "answer": {"type": "string"},
        "clarification": {"type": "string"},
        "reason": {"type": "string"},
    },
    "required": ["action", "answer", "clarification", "reason"],
    "additionalProperties": False,
}


async def call_json(client: Any, semaphore: asyncio.Semaphore, messages: list[dict[str, str]], model: dict[str, Any], inference: dict[str, Any], retries: int, guided_json: bool = False) -> dict[str, Any]:
    attempts = []
    for _ in range(retries + 1):
        try:
            kwargs: dict[str, Any] = {
                "model": model["served_model_name"], "messages": messages,
                "temperature": inference["temperature"], "top_p": inference["top_p"],
                "max_tokens": inference["max_tokens"], "seed": inference["seed"],
            }
            extra_body: dict[str, Any] = {}
            if model.get("chat_template_kwargs"):
                extra_body["chat_template_kwargs"] = model["chat_template_kwargs"]
            if guided_json:
                # vLLM 0.19的guided_json已被静默忽略, 结构化输出走OpenAI风格response_format
                kwargs["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {"name": "evidence_protocol", "strict": True, "schema": GUIDED_JSON_SCHEMA},
                }
            if extra_body:
                kwargs["extra_body"] = extra_body
            async with semaphore:
                response = await client.chat.completions.create(**kwargs)
            raw = response.choices[0].message.content or ""
            attempts.append(raw)
            try:
                return {"status": "OK", "raw_attempts": attempts, "parsed": parse_response(raw)}
            except (ValueError, json.JSONDecodeError):
                continue
        except Exception as error:  # 保留接口异常，便于断点排查
            attempts.append(f"__API_ERROR__:{type(error).__name__}:{error}")
    return {"status": "PARSE_OR_API_ERROR", "raw_attempts": attempts, "parsed": None}


async def evaluate_one(client: Any, semaphore: asyncio.Semaphore, record: dict[str, Any], model: dict[str, Any], task: dict[str, Any], inference: dict[str, Any], retries: int, guided_json: bool = False) -> dict[str, Any]:
    full_messages = make_initial_messages(record, task, full=True)
    initial_messages = make_initial_messages(record, task, full=False)
    full_result, initial_result = await asyncio.gather(
        call_json(client, semaphore, full_messages, model, inference, retries, guided_json),
        call_json(client, semaphore, initial_messages, model, inference, retries, guided_json),
    )
    turns = normalized_trajectory(record)
    user_reply = turns[1]["content"]
    continuation_result = {"status": "NOT_RUN", "raw_attempts": [], "parsed": None}
    if initial_result["parsed"] and initial_result["raw_attempts"]:
        continuation_messages = initial_messages + [
            {"role": "assistant", "content": initial_result["raw_attempts"][-1]},
            {"role": "user", "content": user_reply},
        ]
        continuation_result = await call_json(client, semaphore, continuation_messages, model, inference, retries, guided_json)

    full_parsed = full_result["parsed"] or {}
    initial_parsed = initial_result["parsed"] or {}
    continuation_parsed = continuation_result["parsed"] or {}
    full_correct = full_parsed.get("action") == "ANSWER" and answer_is_correct(full_parsed.get("answer", ""), record)
    action_correct = initial_parsed.get("action") == expected_action(record)
    recovery_correct = continuation_parsed.get("action") == "ANSWER" and answer_is_correct(continuation_parsed.get("answer", ""), record)
    return {
        "family_id": record["family_id"], "source_sample_id": record["source_sample_id"],
        "dataset": record["dataset"], "state": record["experiment_state"],
        "model": model["name"], "expected_action": expected_action(record),
        "full": full_result, "initial": initial_result, "continuation": continuation_result,
        "scores": {
            "full_answer_correct": full_correct,
            "initial_action_correct": action_correct,
            "initial_direct_answer": initial_parsed.get("action") == "ANSWER",
            "recovery_answer_correct": recovery_correct,
            "end_to_end_success": action_correct and recovery_correct,
        },
    }


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    metrics = list(next(iter(results))["scores"]) if results else []

    def block(rows: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "count": len(rows),
            **{metric: sum(bool(row["scores"][metric]) for row in rows) / len(rows) if rows else 0.0 for metric in metrics},
            "initial_actions": dict(Counter((row["initial"].get("parsed") or {}).get("action", "ERROR") for row in rows)),
        }

    grouped_state = {state: block([row for row in results if row["state"] == state]) for state in sorted({row["state"] for row in results})}
    grouped_dataset = {dataset: block([row for row in results if row["dataset"] == dataset]) for dataset in sorted({row["dataset"] for row in results})}
    return {"overall": block(results), "by_state": grouped_state, "by_dataset": grouped_dataset}


async def async_main(args: argparse.Namespace) -> None:
    from openai import AsyncOpenAI

    experiment, model, task = load_experiment(args.config)
    served_name = args.served_model_name or experiment["inference"].get("served_model_name_override")
    if served_name:
        model = {**model, "served_model_name": served_name}
    processed = resolve_root_path(experiment["paths"]["processed_dir"])
    records = read_jsonl(processed / f"families_{args.split}.jsonl")
    inference = experiment["inference"]
    client = AsyncOpenAI(base_url=inference["base_url"], api_key=inference["api_key"], timeout=inference["timeout_seconds"])
    semaphore = asyncio.Semaphore(int(inference["concurrency"]))
    jobs = [evaluate_one(client, semaphore, row, model, task, inference, int(experiment["task"]["parse_retry"]), args.guided_json) for row in records]
    results = await asyncio.gather(*jobs)
    run_name = args.run_name or f"{model['name']}_{args.split}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output = resolve_root_path(experiment["paths"]["output_dir"]) / run_name
    if output.exists():
        raise FileExistsError(f"输出目录已存在：{output}")
    output.mkdir(parents=True)
    write_jsonl(output / "predictions.jsonl", results)
    report = {
        "run_name": run_name, "model": model, "split": args.split,
        "inference": inference, "metrics": summarize(results),
    }
    (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="通过vLLM运行FULL、初始动作和真实续轮评测")
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--split", choices=["train", "dev", "test"], default="test")
    parser.add_argument("--run-name")
    parser.add_argument("--served-model-name", help="评测LoRA时覆盖vLLM中的模型别名")
    parser.add_argument("--guided-json", action="store_true", help="vLLM结构化输出强制JSON协议(基线对照: 剥离格式学习成分)")
    asyncio.run(async_main(parser.parse_args()))


if __name__ == "__main__":
    main()
