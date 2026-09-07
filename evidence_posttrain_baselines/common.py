from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable

import yaml


ROOT = Path(__file__).resolve().parent
ALLOWED_ACTIONS = {"ANSWER", "REQUEST_INFORMATION", "REQUEST_CONFIRMATION"}


def load_yaml(path: str | Path) -> dict[str, Any]:
    """读取 YAML；相对路径一律以实验目录为基准。"""
    file_path = Path(path)
    if not file_path.is_absolute():
        file_path = ROOT / file_path
    with file_path.open("r", encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def load_experiment(path: str | Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """同时载入实验、模型和任务提示配置。"""
    experiment = load_yaml(path)
    model = load_yaml(Path("configs/models") / f"{experiment['active_model']}.yaml")
    task = load_yaml(experiment["task"]["prompt_file"])
    return experiment, model, task


def resolve_root_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (ROOT / path).resolve()


def model_path(experiment: dict[str, Any], model: dict[str, Any]) -> Path:
    root = resolve_root_path(experiment["paths"]["model_root"])
    return (root / model["local_path"]).resolve()


def resolve_output_artifact(experiment: dict[str, Any], value: str) -> Path:
    """运行产物(预测jsonl、LoRA适配器)按output_dir解析;本地相对outputs与服务器绝对实验盘均成立。"""
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (resolve_root_path(experiment["paths"]["output_dir"]) / path).resolve()


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records = []
    with Path(path).open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"{path} 第 {line_number} 行不是合法 JSON") from error
    return records


def write_jsonl(path: str | Path, records: Iterable[dict[str, Any]]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")


def make_user_content(query: str, evidence: Any, schema: dict[str, Any]) -> str:
    """所有模型和训练阶段共用同一种任务输入。"""
    payload = {"query": query, "evidence": evidence, "response_schema": schema}
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def make_initial_messages(record: dict[str, Any], task: dict[str, Any], full: bool = False) -> list[dict[str, str]]:
    state = record["full_state"] if full else record["initial_state"]
    return [
        {"role": "system", "content": task["system_prompt"]},
        {"role": "user", "content": make_user_content(record["query"], state["evidence"], task["response_schema"])},
    ]


def canonical_response(action: str, answer: str = "", clarification: str = "", reason: str = "") -> str:
    if action == "REQUEST_CONFLICT_RESOLUTION":
        action = "REQUEST_CONFIRMATION"
    value = {"action": action, "answer": answer, "clarification": clarification, "reason": reason}
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def parse_response(raw: str) -> dict[str, str]:
    """仅做确定性 JSON 清理，不调用模型修复。"""
    text = raw.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        text = fenced.group(1)
    value = json.loads(text)
    if not isinstance(value, dict):
        raise ValueError("模型输出不是 JSON 对象")
    action = str(value.get("action", "")).strip().upper()
    if action == "REQUEST_CONFLICT_RESOLUTION":
        action = "REQUEST_CONFIRMATION"
    if action not in ALLOWED_ACTIONS:
        raise ValueError(f"未知动作：{action}")
    return {
        "action": action,
        "answer": str(value.get("answer", "")).strip(),
        "clarification": str(value.get("clarification", "")).strip(),
        "reason": str(value.get("reason", "")).strip(),
    }


def normalize_answer(value: Any) -> str:
    """答案匹配只做轻量格式归一，不做语义猜测。"""
    text = str(value or "").strip().casefold()
    text = re.sub(r"[\s\W_]+", " ", text, flags=re.UNICODE)
    return text.strip()


def answer_is_correct(prediction: str, record: dict[str, Any]) -> bool:
    aliases = [record.get("answer", ""), *record.get("answer_aliases", [])]
    target = normalize_answer(prediction)
    return bool(target) and target in {normalize_answer(item) for item in aliases if item is not None}


def role_of(turn: dict[str, Any]) -> str:
    role = str(turn.get("role", turn.get("speaker", ""))).lower()
    mapping = {"assistant": "assistant", "model": "assistant", "user": "user", "human": "user"}
    if role not in mapping:
        raise ValueError(f"无法识别轨迹角色：{role}")
    return mapping[role]


def content_of(turn: dict[str, Any]) -> str:
    """兼容生产数据中字符串内容和结构化动作两种轨迹表示。"""
    role = role_of(turn)
    content = turn.get("content")
    if role == "user":
        if isinstance(content, str):
            return content
        for key in ("user_reply", "message", "text", "information", "confirmation"):
            if isinstance(turn.get(key), str):
                return turn[key]
        if content is not None:
            return json.dumps(content, ensure_ascii=False)
        raise ValueError("用户轨迹缺少文本内容")

    if isinstance(content, str):
        try:
            parsed = parse_response(content)
            return canonical_response(**parsed)
        except (ValueError, json.JSONDecodeError, TypeError):
            pass
    action = str(turn.get("action", content.get("action", "") if isinstance(content, dict) else "")).upper()
    source = content if isinstance(content, dict) else turn
    plain_content = content if isinstance(content, str) else ""
    return canonical_response(
        action=action,
        answer=str(source.get("answer", plain_content if action == "ANSWER" else "")),
        clarification=str(source.get("clarification", source.get("question", plain_content if action != "ANSWER" else ""))),
        reason=str(source.get("reason", "")),
    )


def normalized_trajectory(record: dict[str, Any]) -> list[dict[str, str]]:
    turns = [{"role": role_of(turn), "content": content_of(turn)} for turn in record.get("trajectory", [])]
    if len(turns) != 3 or [turn["role"] for turn in turns] != ["assistant", "user", "assistant"]:
        raise ValueError(f"{record.get('family_id')} 的轨迹必须是 assistant-user-assistant 三轮")
    return turns
