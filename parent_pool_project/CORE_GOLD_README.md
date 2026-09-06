# Qwen3-8B 因果核心黄金集 v4

本链路不继承旧黄金池，从原始 HotpotQA 和 MuSiQue 数据重新构造只含官方支撑文档的候选，并使用 Qwen3-8B 单模型进行三阶段严格筛选。

## 核心准入条件

1. `QUERY_ONLY`：只输入 query，普通最短答案提示；Qwen3-8B 不能严格匹配标准答案或认可别名。
2. `CLEAN_FULL`：只输入 query 和官方支撑文档；Qwen3-8B 必须严格回答正确。
3. `COUNTERFACTUAL`：在诊断副本中以类型一致方式替换答案事实；Qwen3-8B 必须严格跟随反事实证据。

DeepSeek 配置和调用接口仍然保留，可通过 `--models deepseek_v4_flash` 或多模型列表进行后续抽检，但不参与默认准入。

## 最终结果

- 原始稳定抽样：HotpotQA 4,300，MuSiQue 4,300。
- 结构与官方证据链检查后：7,561。
- query 答案别名泄漏净化后：7,558。
- query-only 知识未命中：7,017。
- Clean-FULL 严格正确：3,695。
- 可安全构造类型一致反事实：3,599。
- 三阶段全部通过：2,613。
- 最终核心集：2,000。
- overflow：613。

最终 2,000 条包括 HotpotQA 1,352 条和 MuSiQue 648 条，全部为 2-hop、2 个官方支撑文档，不含干扰段落。

## 主要文件

- `core_gold_qwen_v4_output/core_gold_candidates_sanitized.jsonl`
- `core_gold_qwen_v4_output/screening/core_gold_final.jsonl`
- `core_gold_qwen_v4_output/screening/core_gold_overflow.jsonl`
- `core_gold_qwen_v4_output/screening/core_gold_rejected.jsonl`
- `core_gold_qwen_v4_output/screening/core_gold_screening.jsonl`
- `core_gold_qwen_v4_output/screening/core_gold_calls.jsonl`
- `core_gold_qwen_v4_output/screening/core_gold_report.json`
- `core_gold_qwen_v4_output/screening/core_gold_audit.json`

反事实文件仅用于证据依赖诊断，明确标记为 `diagnostic_only`，不属于训练或最终测试证据。

## 复现入口

1. `build_core_gold_candidates_v2.py`：从原始数据稳定抽样并删除干扰段落。
2. `sanitize_core_gold_candidates.py`：删除 query 中含答案别名的候选。
3. `evaluation/filter_core_gold_v4.py`：默认 Qwen3-8B 三阶段漏斗；支持可选 DeepSeek。
4. `evaluation/compact_core_gold_calls.py`：压缩为当前漏斗使用的调用记录。
5. `evaluation/verify_core_gold_v4.py`：独立离线审计。

当前核心集只保证已针对 Qwen3-8B 做参数知识去污染。若后续使用其他模型并希望得到同样保证，需要对相应模型至少补做 query-only 检查。
