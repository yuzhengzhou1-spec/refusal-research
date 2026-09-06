#!/usr/bin/env python3
"""Evaluate FULL (E+) samples through OpenAI-compatible APIs or vLLM."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import random
import re
import threading
import time
import unicodedata
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable


PROMPT_VERSION = "closed_context_answer_only.v2"
SYSTEM_PROMPT = (
    "You are being evaluated on closed-context question answering. "
    "Use only the supplied evidence. Do not use outside knowledge. "
    "The evidence is sufficient. Return only the shortest answer span, "
    "without explanation, citations, or introductory words."
)
ABSTAIN_PATTERNS = (
    "cannot answer",
    "can't answer",
    "insufficient",
    "not enough information",
    "无法回答",
    "信息不足",
    "无法确定",
)


@dataclass(frozen=True)
class RuntimeModel:
    alias: str
    provider: str
    model_id: str
    base_url: str
    api_key_env: str
    api_key: str
    workers: int
    timeout_seconds: float
    max_retries: int
    request_body: dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--models",
        nargs="+",
        default=["deepseek_v4_flash"],
        help="Model aliases to evaluate; defaults to the lower-cost DeepSeek V4 Flash.",
    )
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument(
        "--dataset",
        choices=["all", "hotpotqa", "musique", "natural_questions"],
        default="all",
    )
    parser.add_argument("--seed", type=int, default=20260821)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--provider")
    parser.add_argument("--base-url")
    parser.add_argument("--model-id")
    parser.add_argument("--api-key-env")
    parser.add_argument(
        "--secrets-file",
        type=Path,
        default=Path(__file__).resolve().parent / "secrets" / ".env.local",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--context-mode",
        choices=["supporting_only", "full"],
        default="supporting_only",
        help="Send only evidence-bearing paragraphs by default; use full only to reproduce legacy runs.",
    )
    return parser.parse_args()


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
    return rows


def load_secrets_file(path: Path) -> set[str]:
    """Load missing environment variables from a local key-value file."""
    loaded: set[str] = set()
    if not path.exists():
        return loaded
    with path.open(encoding="utf-8-sig") as handle:
        for line_number, raw_line in enumerate(handle, 1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                raise ValueError(f"{path}:{line_number}: expected NAME=VALUE")
            name, value = line.split("=", 1)
            name = name.strip()
            value = value.strip()
            if not re.fullmatch(r"[A-Z][A-Z0-9_]*", name):
                raise ValueError(f"{path}:{line_number}: invalid variable name")
            if value and not os.getenv(name):
                os.environ[name] = value
                loaded.add(name)
    return loaded


def select_samples(
    rows: list[dict[str, Any]],
    dataset: str,
    limit: int,
    seed: int,
) -> list[dict[str, Any]]:
    selected = rows if dataset == "all" else [
        row for row in rows if row["来源"] == dataset
    ]
    selected = list(selected)
    if 0 < limit < len(selected):
        random.Random(seed).shuffle(selected)
        selected = selected[:limit]
    return selected


def select_context_documents(
    sample: dict[str, Any],
    context_mode: str = "supporting_only",
) -> list[dict[str, Any]]:
    documents = list(sample["完整上下文"])
    if context_mode == "full":
        return documents
    if context_mode != "supporting_only":
        raise ValueError(f"unknown context mode: {context_mode}")

    support_ids = {
        str(unit.get("文档ID") or "").strip()
        for unit in sample.get("关键证据单元", [])
        if str(unit.get("文档ID") or "").strip()
    }
    if not support_ids:
        raise ValueError(f"sample {sample['样本ID']} has no support document IDs")
    selected = [
        document
        for document in documents
        if str(document.get("文档ID") or "").strip() in support_ids
    ]
    selected_ids = {
        str(document.get("文档ID") or "").strip()
        for document in selected
    }
    missing = sorted(support_ids - selected_ids)
    if missing:
        raise ValueError(
            f"sample {sample['样本ID']} references missing support documents: {missing}"
        )
    return selected


def render_context(
    sample: dict[str, Any],
    context_mode: str = "supporting_only",
) -> str:
    blocks = []
    for index, doc in enumerate(select_context_documents(sample, context_mode), 1):
        title = str(doc.get("标题") or "").strip()
        text = str(doc.get("文本") or "").strip()
        header = f"[Document {index}]"
        if title:
            header += f" {title}"
        blocks.append(f"{header}\n{text}")
    return "\n\n".join(blocks)


def build_messages(
    sample: dict[str, Any],
    context_mode: str = "supporting_only",
) -> list[dict[str, str]]:
    user = (
        f"Question:\n{sample['问题']}\n\n"
        f"Evidence:\n{render_context(sample, context_mode)}\n\n"
        "Answer:"
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def request_hash(
    model_id: str,
    messages: list[dict[str, str]],
    body: dict[str, Any],
) -> str:
    payload = {
        "prompt_version": PROMPT_VERSION,
        "model": model_id,
        "messages": messages,
        "body": body,
    }
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def normalize_answer(text: Any) -> str:
    value = unicodedata.normalize("NFKC", str(text or "")).casefold()
    value = re.sub(r"(?<=\d),(?=\d)", "", value)
    value = value.replace("'", "").replace("’", "")
    value = "".join(
        " " if unicodedata.category(char).startswith(("P", "S")) else char
        for char in value
    )
    tokens = [token for token in value.split() if token not in {"a", "an", "the"}]
    return " ".join(tokens)


def token_f1(prediction: str, reference: str) -> float:
    prediction_tokens = normalize_answer(prediction).split()
    reference_tokens = normalize_answer(reference).split()
    if not prediction_tokens or not reference_tokens:
        return float(prediction_tokens == reference_tokens)
    common = Counter(prediction_tokens) & Counter(reference_tokens)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(prediction_tokens)
    recall = overlap / len(reference_tokens)
    return 2 * precision * recall / (precision + recall)


def extract_prediction(content: Any) -> str:
    text = str(content or "").strip()
    if not text:
        return ""
    fence = chr(96) * 3
    fenced = text
    if fenced.startswith(fence):
        fenced = fenced[len(fence):].lstrip()
        for language in ("json", "text"):
            if fenced.lower().startswith(language):
                fenced = fenced[len(language):].lstrip()
                break
        if fenced.endswith(fence):
            fenced = fenced[:-len(fence)].rstrip()
    try:
        parsed = json.loads(fenced)
        if isinstance(parsed, dict) and "answer" in parsed:
            return str(parsed["answer"]).strip()
    except json.JSONDecodeError:
        pass
    first_line = next((line.strip() for line in text.splitlines() if line.strip()), "")
    first_line = re.sub(
        r"^(?:answer|final answer|答案)\s*[:：]\s*",
        "",
        first_line,
        flags=re.IGNORECASE,
    )
    return first_line.strip().strip('"')


def score_prediction(prediction: str, answers: list[str]) -> dict[str, Any]:
    normalized_prediction = normalize_answer(prediction)
    canonical_match = int(normalized_prediction == normalize_answer(answers[0]))
    alias_match = int(
        any(normalized_prediction == normalize_answer(answer) for answer in answers)
    )
    best_f1 = max((token_f1(prediction, answer) for answer in answers), default=0.0)
    lowered = prediction.casefold()
    return {
        "标准答案EM": canonical_match,
        "别名匹配": alias_match,
        "最佳TokenF1": round(best_f1, 6),
        "是否拒答": any(pattern in lowered for pattern in ABSTAIN_PATTERNS),
    }


def endpoint(base_url: str) -> str:
    return base_url.rstrip("/") + "/chat/completions"


def retry_delay(attempt: int, retry_after: str | None) -> float:
    if retry_after:
        try:
            return min(float(retry_after), 60.0)
        except ValueError:
            pass
    return min(2 ** attempt, 30) + random.random()


def call_chat(
    runtime: RuntimeModel,
    messages: list[dict[str, str]],
) -> tuple[dict[str, Any], float]:
    body = {"model": runtime.model_id, "messages": messages, **runtime.request_body}
    encoded = json.dumps(body, ensure_ascii=False).encode("utf-8")
    headers = {
        "Authorization": f"Bearer {runtime.api_key}",
        "Content-Type": "application/json",
        "User-Agent": "parent-pool-eplus-eval/1.0",
    }
    last_error: Exception | None = None
    for attempt in range(runtime.max_retries + 1):
        started = time.perf_counter()
        request = urllib.request.Request(
            endpoint(runtime.base_url),
            data=encoded,
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(
                request,
                timeout=runtime.timeout_seconds,
            ) as response:
                payload = json.loads(response.read().decode("utf-8"))
            return payload, time.perf_counter() - started
        except urllib.error.HTTPError as exc:
            message = exc.read().decode("utf-8", errors="replace")[:1000]
            last_error = RuntimeError(f"HTTP {exc.code}: {message}")
            retryable = exc.code in {408, 409, 429, 500, 502, 503, 504}
            if not retryable or attempt >= runtime.max_retries:
                break
            time.sleep(retry_delay(attempt, exc.headers.get("Retry-After")))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            last_error = exc
            if attempt >= runtime.max_retries:
                break
            time.sleep(retry_delay(attempt, None))
    raise RuntimeError(str(last_error or "unknown API error"))


def response_content(payload: dict[str, Any]) -> str:
    choices = payload.get("choices") or []
    if not choices:
        raise ValueError("response has no choices")
    content = (choices[0].get("message") or {}).get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            str(item.get("text", "")) for item in content if isinstance(item, dict)
        )
    return str(content or "")


def usage_fields(payload: dict[str, Any]) -> dict[str, int | None]:
    usage = payload.get("usage") or {}
    return {
        "输入tokens": usage.get("prompt_tokens"),
        "输出tokens": usage.get("completion_tokens"),
        "总tokens": usage.get("total_tokens"),
    }


def evaluate_one(
    sample: dict[str, Any],
    runtime: RuntimeModel,
    context_mode: str,
) -> dict[str, Any]:
    messages = build_messages(sample, context_mode)
    requested_at = now_iso()
    common = {
        "样本ID": sample["样本ID"],
        "来源": sample["来源"],
        "问题": sample["问题"],
        "答案": sample["答案"],
        "模型别名": runtime.alias,
        "请求模型ID": runtime.model_id,
        "服务商": runtime.provider,
        "提示版本": PROMPT_VERSION,
        "上下文模式": context_mode,
        "请求哈希": request_hash(
            runtime.model_id,
            messages,
            runtime.request_body,
        ),
        "请求时间": requested_at,
    }
    try:
        payload, latency = call_chat(runtime, messages)
        raw_content = response_content(payload)
        prediction = extract_prediction(raw_content)
        return {
            **common,
            "返回模型ID": payload.get("model"),
            "原始输出": raw_content,
            "预测答案": prediction,
            **score_prediction(prediction, sample["答案"]),
            **usage_fields(payload),
            "延迟秒": round(latency, 4),
            "状态": "成功",
            "错误": None,
        }
    except Exception as exc:
        return {
            **common,
            "返回模型ID": None,
            "原始输出": "",
            "预测答案": "",
            "标准答案EM": 0,
            "别名匹配": 0,
            "最佳TokenF1": 0.0,
            "是否拒答": False,
            "输入tokens": None,
            "输出tokens": None,
            "总tokens": None,
            "延迟秒": None,
            "状态": "错误",
            "错误": str(exc)[:1200],
        }


def resolve_runtime(
    config: dict[str, Any],
    alias: str,
    args: argparse.Namespace,
) -> RuntimeModel:
    model_config = dict(config["模型"][alias])
    provider_name = args.provider or model_config["服务商"]
    provider = dict(config["服务商"][provider_name])
    defaults = dict(config["默认参数"])
    api_key_env = args.api_key_env or provider["api_key_env"]
    api_key = os.getenv(api_key_env, "")
    if not api_key and provider.get("api_key_optional"):
        api_key = "EMPTY"
    if not api_key:
        raise RuntimeError(
            f"missing credential environment variable: {api_key_env}. "
            "Secrets must not be stored in project files."
        )
    request_body = {
        "temperature": defaults["temperature"],
        "top_p": defaults["top_p"],
        "max_tokens": defaults["max_tokens"],
        **model_config.get("extra_body", {}),
    }
    return RuntimeModel(
        alias=alias,
        provider=provider_name,
        model_id=args.model_id or model_config["model_id"],
        base_url=args.base_url or provider["base_url"],
        api_key_env=api_key_env,
        api_key=api_key,
        workers=args.workers or int(model_config.get("workers", 1)),
        timeout_seconds=float(defaults["timeout_seconds"]),
        max_retries=int(defaults["max_retries"]),
        request_body=request_body,
    )


def latest_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    latest: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for row in load_jsonl(path):
        sample_id = row["样本ID"]
        if sample_id not in latest:
            order.append(sample_id)
        latest[sample_id] = row
    return [latest[sample_id] for sample_id in order]


def completed_ids(
    path: Path,
    prompt_version: str | None = None,
    context_mode: str | None = None,
) -> set[str]:
    return {
        row["样本ID"]
        for row in latest_rows(path)
        if row["状态"] == "成功"
        and (prompt_version is None or row.get("提示版本") == prompt_version)
        and (context_mode is None or row.get("上下文模式") == context_mode)
    }


def append_jsonl(
    path: Path,
    record: dict[str, Any],
    lock: threading.Lock,
) -> None:
    line = json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
    with lock:
        with path.open("a", encoding="utf-8", newline="") as handle:
            handle.write(line)
            handle.flush()


def metrics(items: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(items)
    successful = [item for item in items if item["状态"] == "成功"]
    success_count = len(successful)
    return {
        "样本数": total,
        "成功请求数": success_count,
        "请求成功率": round(success_count / total, 6) if total else 0.0,
        "Acc_Eplus_全部样本": (
            round(sum(item["别名匹配"] for item in items) / total, 6)
            if total
            else 0.0
        ),
        "Acc_Eplus_成功请求": (
            round(sum(item["别名匹配"] for item in successful) / success_count, 6)
            if success_count
            else 0.0
        ),
        "标准答案EM": (
            round(sum(item["标准答案EM"] for item in items) / total, 6)
            if total
            else 0.0
        ),
        "平均TokenF1": (
            round(sum(item["最佳TokenF1"] for item in items) / total, 6)
            if total
            else 0.0
        ),
        "拒答率": (
            round(sum(bool(item["是否拒答"]) for item in items) / total, 6)
            if total
            else 0.0
        ),
        "输入tokens": sum(item["输入tokens"] or 0 for item in items),
        "输出tokens": sum(item["输出tokens"] or 0 for item in items),
        "错误数": total - success_count,
    }


def aggregate(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    all_rows = list(rows)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in all_rows:
        grouped[row["来源"]].append(row)
    return {
        "总体": metrics(all_rows),
        "分数据集": {
            name: metrics(items) for name, items in sorted(grouped.items())
        },
    }


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    os.replace(temporary, path)


def run_model(
    runtime: RuntimeModel,
    samples: list[dict[str, Any]],
    output_root: Path,
    overwrite: bool,
    context_mode: str,
) -> dict[str, Any]:
    model_root = output_root / runtime.alias
    model_root.mkdir(parents=True, exist_ok=True)
    predictions_path = model_root / "predictions.jsonl"
    if overwrite and predictions_path.exists():
        predictions_path.unlink()
    done = completed_ids(predictions_path, PROMPT_VERSION, context_mode)
    pending = [sample for sample in samples if sample["样本ID"] not in done]
    lock = threading.Lock()
    started = now_iso()

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=runtime.workers
    ) as executor:
        futures = {
            executor.submit(evaluate_one, sample, runtime, context_mode): sample["样本ID"]
            for sample in pending
        }
        for index, future in enumerate(concurrent.futures.as_completed(futures), 1):
            append_jsonl(predictions_path, future.result(), lock)
            if index % 25 == 0 or index == len(pending):
                print(
                    json.dumps(
                        {
                            "模型": runtime.alias,
                            "本轮完成": index,
                            "本轮总数": len(pending),
                            "累计已有": len(done) + index,
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )

    selected_ids = {sample["样本ID"] for sample in samples}
    rows = [
        row
        for row in latest_rows(predictions_path)
        if row["样本ID"] in selected_ids
    ]
    report = {
        "模型别名": runtime.alias,
        "模型ID": runtime.model_id,
        "服务商": runtime.provider,
        "base_url": runtime.base_url,
        "凭据环境变量": runtime.api_key_env,
        "提示版本": PROMPT_VERSION,
        "上下文模式": context_mode,
        "构建开始": started,
        "构建结束": now_iso(),
        "并发数": runtime.workers,
        "指标": aggregate(rows),
    }
    write_json(model_root / "metrics.json", report)
    return report


def main() -> int:
    args = parse_args()
    load_secrets_file(args.secrets_file)
    config = load_json(args.config)
    rows = load_jsonl(args.input)
    samples = select_samples(rows, args.dataset, args.limit, args.seed)
    if not samples:
        raise RuntimeError("no samples selected")
    args.output_root.mkdir(parents=True, exist_ok=True)
    reports = []
    for alias in args.models:
        if alias not in config["模型"]:
            raise KeyError(f"unknown model alias: {alias}")
        runtime = resolve_runtime(config, alias, args)
        reports.append(
            run_model(
                runtime,
                samples,
                args.output_root,
                args.overwrite,
                args.context_mode,
            )
        )
    write_json(
        args.output_root / "run_summary.json",
        {
            "生成时间": now_iso(),
            "输入": str(args.input.resolve()),
            "样本数": len(samples),
            "数据集筛选": args.dataset,
            "随机种子": args.seed,
            "上下文模式": args.context_mode,
            "模型报告": reports,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
