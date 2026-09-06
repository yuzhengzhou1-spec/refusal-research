from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


METRICS = ["full_answer_correct", "initial_action_correct", "initial_direct_answer", "recovery_answer_correct", "end_to_end_success"]


def percent(value: Any) -> str:
    return f"{100 * float(value):.2f}%"


def main() -> None:
    parser = argparse.ArgumentParser(description="汇总多个vLLM评测报告")
    parser.add_argument("reports", nargs="+", help="一个或多个report.json")
    parser.add_argument("--output", default="outputs/comparison.md")
    args = parser.parse_args()
    reports = [json.loads(Path(path).read_text(encoding="utf-8")) for path in args.reports]
    header = ["运行", "模型", *METRICS]
    lines = ["# 实验结果对比", "", "| " + " | ".join(header) + " |", "| " + " | ".join(["---"] * len(header)) + " |"]
    for report in reports:
        overall = report["metrics"]["overall"]
        row = [report["run_name"], report["model"]["name"], *[percent(overall[name]) for name in METRICS]]
        lines.append("| " + " | ".join(row) + " |")
    lines.extend(["", "说明：`initial_direct_answer` 越低越好，其余指标越高越好。"])
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(output.resolve())


if __name__ == "__main__":
    main()

