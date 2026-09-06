# 从最初版数据到高质量证据状态家族：链路 v1

更新日期：2026-08-28

## 1. 目标与边界

本链路将可用数据源按需转换为参数知识去污染的核心黄金样本，再使用已经冻结的构造与语义门生成 MISSING 或 CONFLICT 样本家族。它解决两个问题：

1. 当前先汇总并补齐各 100 条高质量 MISSING / CONFLICT；
2. 未来不再受现有 2,000 条主黄金池限制，可从更早的数据源继续筛选新的核心黄金候选。

链路不会修改任何输入 JSONL，不会把 Qwen 的 MISSING/CONFLICT 行为用作样本准入条件，也不会在余额或 API 错误时更换构造模型。

## 2. 数据流

```text
官方原始 HotpotQA / MuSiQue  ── build_core_gold_candidates_v2.py ──┐
                                                                  ├─ supporting-only 候选
最初版统一母池 v1 ─ prepare_core_candidates_from_parent_pool_v1.py ─┘
                                      │
                                      ├─ 规则门：2-hop、2个官方支撑文档、答案不泄漏于query
                                      ▼
                         Qwen3-8B QUERY_ONLY（答对则剔除）
                                      ▼
                         Qwen3-8B CLEAN_FULL（答对才保留）
                                      ▼
                         Qwen3-8B COUNTERFACTUAL（跟随才保留）
                                      ▼
                              核心黄金候选池
                         ┌────────────┴────────────┐
                         ▼                         ▼
               冻结 MISSING 流水线        冻结 CONFLICT 流水线
               编辑→澄清→双语义门         冲突追加→澄清→双语义门
                         └────────────┬────────────┘
                                      ▼
                   run_family_target_v1.py 按PASS目标续批
                                      ▼
                       精确N条主文件 + 未选PASS surplus
```

## 3. 两个源入口

### 3.1 官方原始文件入口

入口脚本：`build_core_gold_candidates_v2.py`

要求 source root 中存在：

- `raw/hotpotqa_full.jsonl`
- `_musique_v1.0.zip`

该入口直接使用原始 HotpotQA bridge 与 MuSiQue answerable 2-hop 记录，并删除所有非官方支撑文档。当前磁盘中原先的 `F:\zyz\shujv_preview` 已不存在，因此本轮没有重新下载或伪造原始文件；脚本接口和统一编排入口仍保留。

### 3.2 当前可执行的最初版统一母池入口

现存输入：`F:\zyz\outputs\parent_pool_full_merged.jsonl`

入口脚本：`prepare_core_candidates_from_parent_pool_v1.py`

该脚本：

- 只接受 HotpotQA / MuSiQue；
- 只接受 2-hop、恰好两个支撑文档的记录；
- 删除干扰文档但不改问题、答案、支撑文本或关键证据单元；
- 在模型调用前剔除 query 中直接出现答案或别名的记录；
- 可排除已经出现在旧 `core_gold_screening.jsonl` 中的 ID；
- 使用稳定 SHA-256 排序按需抽样，扩大批量时不会随机漂移。

本轮实际检查结果：最初版母池共 1,851 条；排除已筛记录和 NQ 后，尚有 HotpotQA 791 条、MuSiQue 631 条可作为新候选。

## 4. 核心黄金门

核心准入仍由 `evaluation/filter_core_gold_v4.py` 执行，默认且仅要求 Qwen3-8B：

| 阶段 | 输入 | 通过条件 | 用途 |
|---|---|---|---|
| QUERY_ONLY | 只给 query，普通最短问答 | 不能严格匹配答案或别名 | 排除目标模型的参数知识命中 |
| CLEAN_FULL | query + 两个官方支撑文档 | 严格答对 | 确认证据足以支撑回答 |
| COUNTERFACTUAL | query + 类型一致的诊断性反事实证据 | 跟随反事实答案 | 加强证据依赖核验 |

反事实副本只用于诊断，不进入训练集或最终测试证据。

### 4.1 本轮新ID真实冒烟

目录：`source_to_family_smoke_v1/`

- 输入：HotpotQA 20 + MuSiQue 20，均未出现在旧 7,558 条 screening 中；
- QUERY_ONLY 未命中：40/40；
- CLEAN_FULL 通过：22/40；
- COUNTERFACTUAL 通过并成为新核心黄金：17/40；
- 新核心黄金分布：HotpotQA 11、MuSiQue 6。

这17条保存在 `source_to_family_smoke_v1/core_screening/core_gold_final.jsonl`，没有并入或覆盖原 2,000 条主黄金池。

## 5. 冻结家族生产

统一单批入口：`run_family_pipeline_v1.py`

冻结协议：`evidence_family_protocol_v1.json`，版本 `evidence_family_freeze.v1.0.1`。

### MISSING

- 构造模型：DeepSeek V4 Pro；
- 构造器：`missing_family.split_responsibility.v2_5_2`；
- 正向门：`missing_semantic_gate.v1_2`；
- 反向门：`missing_reverse_semantic_gate.v1_1_frozen`；
- 只有编辑、澄清、正向门和反向门均成功的记录才进入冻结PASS文件。

### CONFLICT

- 构造模型：DeepSeek V4 Pro；
- 构造器：`conflict_family.append_assertion.v1_2`；
- 语义门A：`conflict_semantic_gate.v1_5_frozen`；
- 语义门B：`conflict_semantic_gate.v1_3_frozen`；
- 两个值必须指向同一个单值槽位并形成直接冲突，澄清必须呈现两值并请求确认。

所有模型调用均为 temperature 0、top_p 1、thinking disabled。Qwen 行为评测不参与数据准入。

## 6. 精确目标与自动补批

入口脚本：`run_family_target_v1.py`

该脚本接受：

- 已确认的历史PASS文件；
- 一个或多个核心黄金 seed pool；
- 对应的 screening 文件；
- 已尝试ID排除文件；
- 目标PASS数、每批每数据集候选数和最大批数。

每批继续调用冻结的 `run_family_pipeline_v1.py`，合并时按 `source_sample_id` 去重；达到目标后用稳定 SHA-256 选择精确 N 条，额外PASS写入 surplus。若审计报告出现 API_ERROR / WORKER_ERROR 且目标尚未达到，脚本停止继续开新批，保留全部中间输出供原地续跑。

## 7. 本轮100+100状态

### 7.1 MISSING：完成

主文件：`family_100_v1/missing/missing_100_high_quality.jsonl`

- 100条，100个唯一 source ID；
- HotpotQA 63，MuSiQue 37；
- 来源：开发阶段用户确认PASS 33条、独立盲测冻结PASS 41条、本轮新增冻结PASS 26条；
- 尚有2条合格 surplus 未删除；
- SHA-256：`5c2b58e38f7e0d334153c62e903764de9b35d4d61e6f027f8e1a3a953cecc1f1`。

本轮80个新种子构造出58个结构合格家族；正向门57条PASS。反向门执行到MuSiQue中途时DeepSeek返回HTTP 402，30条被标记API_ERROR并排除，最终新增26条PASS。因此100条内容有效，但分布暂时偏向HotpotQA。余额恢复后可原地重跑同一命令，API错误样本应继续裁决，再重新稳定汇总以改善分布。

### 7.2 CONFLICT：完成

主文件：`family_100_v1/conflict/conflict_100_high_quality.jsonl`

- 100条，100个唯一 source ID；
- HotpotQA 47，MuSiQue 53；
- 两批各尝试80个全新种子，分别新增41条和40条共识PASS；
- 加上原有54条后共有135条可用PASS，稳定选择100条进入主文件；
- 35条额外PASS保存在`conflict_surplus_high_quality.jsonl`；
- 两批最终审计报告中的API/worker错误均为0；
- SHA-256：`7df7967b07eb0cf36f5641aaaf9a4c470893ae2cbd9d68d239336b892d1a33e3`。

第一批语义门A曾出现33条间歇性API_ERROR。使用不改变提示词、模型和解码设置的`--retry-api-errors`开关，仅重试错误ID并保留已有PASS/FAIL，重试后33条全部得到真实裁决。旧`conflict_partial_54_of_100.jsonl`已被主文件取代，并在2026-08-29旧版本清理中删除。

## 8. 复现与续跑命令

### 8.1 从最初版母池跑到新核心黄金

```powershell
python run_source_to_family_smoke_v1.py `
  --parent-pool-input F:\zyz\outputs\parent_pool_full_merged.jsonl `
  --output-dir source_to_family_smoke_v1 `
  --exclude-screening core_gold_qwen_v4_output\screening\core_gold_screening.jsonl `
  --per-dataset 20 --family-per-dataset 2 --workers 2 --core-only
```

移除 `--core-only` 后会复用已经保存的Qwen调用，并继续对新黄金样本执行MISSING和CONFLICT端到端冒烟。

### 8.2 CONFLICT目标入口

项目默认参数已经封装为一条命令：

```powershell
python run_family_100_v1.py --state conflict
```

该入口会调用 `run_family_target_v1.py --state conflict --target 100`，并自动传入：

- 现有两个CONFLICT PASS文件；
- `core_gold_final.jsonl`、`core_gold_overflow.jsonl`；
- 可选的新17条核心黄金；
- 旧blind seed排除文件；
- `--batch-per-dataset 40 --max-waves 4`。

当前目标已经完成。该命令可用于复现或断点续跑；命令、输入哈希、所选ID、各批PASS和最终来源都会写入 `conflict_target_report.json`。

## 9. 验证与不可误读项

- 新增3个脚本均通过 `py_compile`；
- 5个新增纯断言回归测试通过；当前Python环境未安装pytest，因此测试以直接调用测试函数的方式执行；
- MISSING最终文件和CONFLICT partial文件均完成行数、唯一ID、状态、query-only门、FULL门和SHA-256核验；
- 本轮没有修改原2,000条黄金、613条overflow、旧screening或旧家族文件；
- “代码链路已连接”不等于“官方原始下载文件当前存在”；
- MISSING与CONFLICT主文件均已达到100条；旧partial文件仅作为审计快照保留。
