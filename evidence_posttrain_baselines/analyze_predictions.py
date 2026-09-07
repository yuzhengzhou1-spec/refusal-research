from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from common import answer_is_correct, load_experiment, normalize_answer, read_jsonl, resolve_root_path

# 简明英文停用词表: 只为内容词召回服务, 不追求语言学完备
STOPWORDS = set("a an the of to in on for and or is was were be been being this that these those it its his her their with as at by from which who whom what when where how did does do not no".split())


def content_words(text: str) -> set[str]:
    return {w for w in normalize_answer(text).split() if w and w not in STOPWORDS}


def parsed_action(block: dict[str, Any]) -> str | None:
    return (block.get("parsed") or {}).get("action")


def gap_reference(record: dict[str, Any]) -> str:
    """该家族澄清应指向的信息: MISSING用缺失信息锚点, CONFLICT用冲突双方claim。"""
    state = record["initial_state"]
    if record["experiment_state"] == "MISSING":
        parts = [state.get("missing_information") or "", (state.get("clarification_grounding") or {}).get("span") or ""]
    else:
        parts = [state.get("conflicting_claim") or "", state.get("original_claim") or "", state.get("conflict_target") or ""]
    return " ".join(p for p in parts if p)


def clarification_quality(row: dict[str, Any], record: dict[str, Any]) -> dict[str, Any]:
    """按状态评澄清: MISSING看缺口瞄准与答案泄露; CONFLICT看是否同时提到两个冲突值(确认式澄清的应尽义务, 提及gold值不算泄露)。"""
    parsed = row["initial"].get("parsed") or {}
    clarification = parsed.get("clarification", "") or ""
    if parsed.get("action") not in ("REQUEST_INFORMATION", "REQUEST_CONFIRMATION") or not clarification:
        return {"evaluated": False, "targeted": False, "gap_recall": 0.0, "leaks_answer": False, "mentions_both_values": False}
    said = content_words(clarification)
    reference = content_words(gap_reference(record))
    result: dict[str, Any] = {
        "evaluated": bool(reference),
        "targeted": (len(reference & said) / len(reference)) >= 0.5 if reference else None,
        "gap_recall": round(len(reference & said) / len(reference), 4) if reference else None,
        "leaks_answer": False,
        "mentions_both_values": None,
    }
    if row["state"] == "MISSING":
        gold_variants = [record["answer"], *record.get("answer_aliases", [])]
        result["leaks_answer"] = any(normalize_answer(v) in normalize_answer(clarification) for v in gold_variants if v)
    else:
        state = record["initial_state"]
        text = normalize_answer(clarification)
        result["mentions_both_values"] = all(
            normalize_answer(v) in text for v in (state.get("original_value"), state.get("conflicting_value")) if v
        )
    return result


def initial_error_taxonomy(row: dict[str, Any], record: dict[str, Any]) -> str:
    """初始轮错误三分类: format / action / ok; 硬答且答对单独标记参数记忆型(under-refusal但有真知识)。"""
    action = parsed_action(row["initial"])
    if action is None:
        return "FORMAT_ERROR"
    if action == row["expected_action"]:
        return "OK"
    if action == "ANSWER":
        answer = (row["initial"].get("parsed") or {}).get("answer", "")
        return "ACTION_ERROR_DIRECT_CORRECT" if answer_is_correct(answer, record) else "ACTION_ERROR"
    return "ACTION_ERROR"


def full_error_taxonomy(row: dict[str, Any]) -> str:
    action = parsed_action(row["full"])
    if action is None:
        return "FORMAT_ERROR"
    if action == "ANSWER":
        return "OK" if row["scores"]["full_answer_correct"] else "ANSWER_ERROR"
    return "ACTION_ERROR"  # 证据齐全却请求信息/确认 = over-refusal


def cluster_bootstrap(rows: list[dict[str, Any]], metric_fn, iterations: int = 10000, seed: int = 20260901) -> list[float]:
    """按source_sample_id聚类重采样(同源家族相关, 不能当独立样本), 返回[2.5%,50%,97.5%]分位。"""
    clusters: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        clusters[row["source_sample_id"]].append(row)
    keys = sorted(clusters)
    rng = random.Random(seed)
    stats = []
    for _ in range(iterations):
        sample = [row for key in (rng.choice(keys) for _ in range(len(keys))) for row in clusters[key]]
        stats.append(metric_fn(sample))
    stats.sort()
    def pct(p: float) -> float:
        return stats[min(len(stats) - 1, int(p * len(stats)))]
    return [round(pct(0.025), 4), round(pct(0.5), 4), round(pct(0.975), 4)]


def analyze_run(run_dir: Path, families_by_id: dict[str, dict[str, Any]]) -> dict[str, Any]:
    report = json.loads((run_dir / "report.json").read_text(encoding="utf-8"))
    rows = read_jsonl(run_dir / "predictions.jsonl")
    for row in rows:
        row["expected_action"] = "REQUEST_INFORMATION" if row["state"] == "MISSING" else "REQUEST_CONFIRMATION"

    # FULL按源问题去重: 同一source的MISSING/CONFLICT两行携带同一FULL探测, 重复计会虚增精度
    full_rows: dict[str, dict[str, Any]] = {}
    for row in rows:
        full_rows.setdefault(row["source_sample_id"], row)
    dedup = list(full_rows.values())

    rate = lambda subset, pred: (sum(1 for r in subset if pred(r)) / len(subset)) if subset else None

    clarification = [clarification_quality(row, families_by_id[row["family_id"]]) for row in rows]
    evaluated_clar = [c for c in clarification if c["evaluated"]]
    evaluated_missing = [c for c, r in zip(clarification, rows) if c["evaluated"] and r["state"] == "MISSING"]
    evaluated_conflict = [c for c, r in zip(clarification, rows) if c["evaluated"] and r["state"] == "CONFLICT"]

    confusion: dict[str, Counter] = {}
    for true_state, pool, block in (("FULL", dedup, "full"), ("MISSING", rows, "initial"), ("CONFLICT", rows, "initial")):
        counter = Counter()
        for row in pool:
            if row["state"] != true_state and true_state != "FULL":
                continue
            action = parsed_action(row[block]) or "FORMAT_ERROR"
            counter[action] += 1
        confusion[true_state] = dict(counter)

    metrics = {
        "count_families": len(rows),
        "count_sources": len(full_rows),
        "full_answer_correct_dedup": rate(dedup, lambda r: r["scores"]["full_answer_correct"]),
        "json_valid_rate": {
            "initial": rate(rows, lambda r: parsed_action(r["initial"]) is not None),
            "full": rate(dedup, lambda r: parsed_action(r["full"]) is not None),
            "continuation_attempted": rate([r for r in rows if r["continuation"]["status"] != "NOT_RUN"], lambda r: parsed_action(r["continuation"]) is not None),
        },
        "initial_taxonomy": dict(Counter(initial_error_taxonomy(row, families_by_id[row["family_id"]]) for row in rows)),
        "full_taxonomy_dedup": dict(Counter(full_error_taxonomy(row) for row in dedup)),
        "over_refusal_rate": rate(dedup, lambda r: parsed_action(r["full"]) not in (None, "ANSWER")),
        "under_refusal_rate": rate(rows, lambda r: parsed_action(r["initial"]) == "ANSWER"),
        "clarification": {
            "evaluated": len(evaluated_clar),
            "targeted_rate": rate(evaluated_clar, lambda c: c["targeted"] is True),
            "mean_gap_recall": (sum(c["gap_recall"] for c in evaluated_clar if c["gap_recall"] is not None) / max(1, sum(1 for c in evaluated_clar if c["gap_recall"] is not None))) if evaluated_clar else None,
            "missing_leak_rate": rate(evaluated_missing, lambda c: c["leaks_answer"] is True),
            "conflict_both_values_rate": rate(evaluated_conflict, lambda c: c["mentions_both_values"] is True),
        },
        "confusion_matrix_true_x_predicted": confusion,
        "bootstrap95_ci": {
            "full_answer_correct_dedup": cluster_bootstrap(dedup, lambda s: sum(bool(r["scores"]["full_answer_correct"]) for r in s) / len(s) if s else 0),
            "initial_action_correct": cluster_bootstrap(rows, lambda s: sum(bool(r["scores"]["initial_action_correct"]) for r in s) / len(s) if s else 0),
            "recovery_answer_correct": cluster_bootstrap(rows, lambda s: sum(bool(r["scores"]["recovery_answer_correct"]) for r in s) / len(s) if s else 0),
            "end_to_end_success": cluster_bootstrap(rows, lambda s: sum(bool(r["scores"]["end_to_end_success"]) for r in s) / len(s) if s else 0),
        },
    }
    return {"run_name": report["run_name"], "model": report["model"]["name"], "split": report["split"], "upgraded_metrics": metrics}


def markdown_table(analyses: list[dict[str, Any]]) -> str:
    lines = ["| 运行 | n家族/源 | full(去重) | full 95%CI | JSON合法 初始/FULL | over-ref | under-ref | 瞄准 | MISSING泄露 | CONFLICT双值 | 动作 95%CI | e2e 95%CI |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for a in analyses:
        m = a["upgraded_metrics"]
        ci = m["bootstrap95_ci"]
        fmt = lambda v: "-" if v is None else f"{v:.2f}"
        fmt_ci = lambda v: f"[{v[0]:.2f},{v[2]:.2f}]" if v else "-"
        c = m["clarification"]
        lines.append(
            f"| {a['run_name']} | {m['count_families']}/{m['count_sources']} "
            f"| {m['full_answer_correct_dedup']:.3f} | {fmt_ci(ci['full_answer_correct_dedup'])} "
            f"| {m['json_valid_rate']['initial']:.2f}/{m['json_valid_rate']['full']:.2f} "
            f"| {m['over_refusal_rate']:.2f} | {m['under_refusal_rate']:.2f} "
            f"| {fmt(c['targeted_rate'])} | {fmt(c['missing_leak_rate'])} | {fmt(c['conflict_both_values_rate'])} "
            f"| {fmt_ci(ci['initial_action_correct'])} | {fmt_ci(ci['end_to_end_success'])} |"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="离线升级评测口径: 不调用模型, 重算已有predictions")
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("runs", nargs="+", help="baselines下的运行目录名")
    parser.add_argument("--markdown", help="汇总markdown输出路径")
    args = parser.parse_args()
    experiment, _, _ = load_experiment(args.config)
    output_root = resolve_root_path(experiment["paths"]["output_dir"])
    processed = resolve_root_path(experiment["paths"]["processed_dir"])
    families_cache: dict[str, dict[str, dict[str, Any]]] = {}
    analyses = []
    for name in args.runs:
        run_dir = output_root / name
        split = json.loads((run_dir / "report.json").read_text(encoding="utf-8"))["split"]
        if split not in families_cache:
            families_cache[split] = {r["family_id"]: r for r in read_jsonl(processed / f"families_{split}.jsonl")}
        analysis = analyze_run(run_dir, families_cache[split])
        (run_dir / "analysis.json").write_text(json.dumps(analysis, ensure_ascii=False, indent=2), encoding="utf-8")
        analyses.append(analysis)
        print(f"analyzed {name}")
    table = markdown_table(analyses)
    print()
    print(table)
    if args.markdown:
        Path(args.markdown).write_text("# 升级评测口径汇总\n\n" + table + "\n", encoding="utf-8")
        print(f"\n汇总已写入 {args.markdown}")


if __name__ == "__main__":
    main()
