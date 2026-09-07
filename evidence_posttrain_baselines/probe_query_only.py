from __future__ import annotations

import argparse
import asyncio
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

from common import (
    answer_is_correct,
    load_experiment,
    make_user_content,
    normalize_answer,
    parse_response,
    read_jsonl,
    resolve_root_path,
    write_jsonl,
)


async def probe_one(client: Any, semaphore: asyncio.Semaphore, record: dict[str, Any], model: dict[str, Any], task: dict[str, Any], inference: dict[str, Any], retries: int) -> dict[str, Any]:
    """只给query不给任何证据。若模型仍能产出规范答案,说明参数知识可解,该家族对此模型被污染。"""
    messages = [
        {"role": "system", "content": task["system_prompt"]},
        {"role": "user", "content": make_user_content(record["query"], [], task["response_schema"])},
    ]
    attempts: list[str] = []
    parsed = None
    for _ in range(retries + 1):
        try:
            kwargs: dict[str, Any] = {
                "model": model["served_model_name"], "messages": messages,
                "temperature": inference["temperature"], "top_p": inference["top_p"],
                "max_tokens": inference["max_tokens"], "seed": inference["seed"],
            }
            if model.get("chat_template_kwargs"):
                kwargs["extra_body"] = {"chat_template_kwargs": model["chat_template_kwargs"]}
            async with semaphore:
                response = await client.chat.completions.create(**kwargs)
            raw = response.choices[0].message.content or ""
            attempts.append(raw)
            try:
                parsed = parse_response(raw)
                break
            except (ValueError, json.JSONDecodeError):
                continue
        except Exception as error:
            attempts.append(f"__API_ERROR__:{type(error).__name__}:{error}")
    answer = (parsed or {}).get("answer", "")
    json_correct = parsed is not None and (parsed.get("action") == "ANSWER") and answer_is_correct(answer, record)
    gold_variants = [record["answer"], *record.get("answer_aliases", [])]
    raw_text = normalize_answer(" ".join(a for a in attempts if not a.startswith("__API_ERROR__")))
    mentions = any(normalize_answer(v) in raw_text for v in gold_variants if v)
    return {
        "family_id": record["family_id"], "source_sample_id": record["source_sample_id"],
        "dataset": record["dataset"], "state": record["experiment_state"],
        "model": model["name"],
        "raw_attempts": attempts,
        "parsed_action": (parsed or {}).get("action"),
        "answers_correctly_in_json": json_correct,
        "mentions_answer_in_text": mentions,
        "parametric_contamination": json_correct or mentions,
    }


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    def block(rows: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "count": len(rows),
            "answers_correctly_in_json": sum(r["answers_correctly_in_json"] for r in rows),
            "mentions_answer_in_text": sum(r["mentions_answer_in_text"] for r in rows),
            "parametric_contamination": sum(r["parametric_contamination"] for r in rows),
            "initial_actions": dict(Counter(r["parsed_action"] or "ERROR" for r in rows)),
        }

    return {
        "overall": block(results),
        "by_state": {state: block([r for r in results if r["state"] == state]) for state in sorted({r["state"] for r in results})},
        "by_dataset": {dataset: block([r for r in results if r["dataset"] == dataset]) for dataset in sorted({r["dataset"] for r in results})},
        "contaminated_family_ids": sorted(r["family_id"] for r in results if r["parametric_contamination"]),
    }


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
    jobs = [probe_one(client, semaphore, row, model, task, inference, int(experiment["task"]["parse_retry"])) for row in records]
    results = await asyncio.gather(*jobs)
    run_name = args.run_name or f"{model['name']}_query_only_{args.split}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output = resolve_root_path(experiment["paths"]["output_dir"]) / run_name
    if output.exists():
        raise FileExistsError(f"输出目录已存在：{output}")
    output.mkdir(parents=True)
    write_jsonl(output / "query_only_predictions.jsonl", results)
    report = {"run_name": run_name, "model": model, "split": args.split, "inference": inference, "summary": summarize(results)}
    (output / "query_only_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report["summary"]["overall"], ensure_ascii=False, indent=2))
    print(f"污染家族: {len(report['summary']['contaminated_family_ids'])}/{len(results)}")
    print(f"报告: {output / 'query_only_report.json'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="query-only参数知识探针:不给证据时模型能否直接答对")
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--split", choices=["train", "dev", "test"], default="test")
    parser.add_argument("--run-name")
    parser.add_argument("--served-model-name", help="评测LoRA时覆盖vLLM中的模型别名")
    asyncio.run(async_main(parser.parse_args()))


if __name__ == "__main__":
    main()
