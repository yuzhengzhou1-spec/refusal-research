# Python文件说明与保留审计

更新日期：2026-08-30

## 1. 审计结论

当前项目共有42个Python文件：项目根目录20个、`evaluation/`目录8个、`evaluation/tests/`目录14个。

2026-08-30按“现行入口、现行依赖、冻结复现、独立审计、回归测试”五类逐项检查后，没有发现可以在不破坏当前链路的前提下直接删除的Python文件，因此本轮没有删除Python源码。此前已经删除的旧MISSING-1构造器、旧必要性验证器、旧试验脚本和对应测试不再列入本文档。

保留原则：

- 被当前生产入口直接调用或导入；
- 为当前正式数据和报告提供可复现路径；
- 是冻结版本的独立语义门或兼容层；
- 是现行代码的回归测试；
- 虽然名称较旧，但仍提供当前链路使用的共享函数。

## 2. 建议优先使用的入口

| 任务 | 入口 |
|---|---|
| 从最初母池抽取新核心黄金候选 | `prepare_core_candidates_from_parent_pool_v1.py` |
| 从官方原始HotpotQA/MuSiQue构造候选 | `build_core_gold_candidates_v2.py` |
| 执行当前Qwen核心黄金三阶段筛选 | `evaluation/filter_core_gold_v4.py` |
| 构造单批MISSING或CONFLICT家族 | `run_family_pipeline_v1.py` |
| 按目标PASS数量持续生产 | `run_family_target_v1.py` |
| 复现当前100条目标入口 | `run_family_100_v1.py` |
| 从母池到家族做端到端冒烟 | `run_source_to_family_smoke_v1.py` |
| 运行Qwen家族行为基线 | `evaluate_family_qwen_baseline_v1.py` |

## 3. 项目根目录：生产与编排文件

| 文件 | 简单说明 | 保留原因 |
|---|---|---|
| `build_parent_pool.py` | 将HotpotQA、MuSiQue和NQ转换并合并为统一FULL母池。 | 原始数据接入层；也被核心黄金候选构造器导入。 |
| `build_core_gold_candidates.py` | 从母池稳定抽样，只保留官方支撑文档，生成核心黄金候选。 | `build_core_gold_candidates_v2.py`复用其主体实现。 |
| `build_core_gold_candidates_v2.py` | 为候选构造器注入当前干净的答案类型规则并启动主体。 | 当前官方原始数据入口；删除会破坏v2复现命令。 |
| `sanitize_core_gold_candidates.py` | 规则化移除query直接泄露答案的候选。 | 当前核心黄金候选清洗阶段，被端到端入口调用。 |
| `finalize_core_gold_metadata.py` | 确定性重算核心黄金中的非语义字段`元数据.答案类型`。 | 当前核心黄金产物包含对应修复报告，保留用于精确复现和审计。 |
| `prepare_core_candidates_from_parent_pool_v1.py` | 从现存统一母池抽取未筛选ID，删除干扰文档并生成候选。 | 当前“最初母池→新核心黄金”扩展入口。 |
| `build_missing_families_split_v2_3.py` | 分离编辑器与澄清器，构造MISSING家族；内部冻结版本为`v2_5_2`。 | 当前MISSING构造主体，被多个入口、验证器和基线共享。文件名保留是为了兼容历史manifest。 |
| `build_conflict_families_v1.py` | 通过原子冲突追加构造CONFLICT家族并生成澄清轨迹。 | 当前CONFLICT构造主体。 |
| `validate_missing_families_v1.py` | 对MISSING执行正向语义门，判断缺口、必要性和恢复一致性。 | 当前冻结流水线的第一道MISSING语义门。 |
| `validate_missing_families_reverse_v1.py` | 以直接缺陷搜索方式执行MISSING反向语义门。 | 当前MISSING双门共识的独立反向门。 |
| `validate_conflict_families_v1.py` | 对CONFLICT执行当前v1.5六维语义审计。 | 当前CONFLICT主语义门。 |
| `validate_conflict_families_v1_3_frozen.py` | 复用验证主体但冻结为独立v1.3提示与判定维度。 | CONFLICT要求两个独立冻结门共识，不能与v1.5合并。 |
| `select_missing_high_quality_v1.py` | 合并MISSING结构结果、正向门和反向门，只输出共识PASS。 | 当前MISSING最终选择器。 |
| `select_conflict_high_quality_v1.py` | 合并CONFLICT结构规则和双语义门，过滤非原子或非必要冲突。 | 当前CONFLICT最终选择器。 |
| `run_family_pipeline_v1.py` | 固定选样、排除开发ID，依次运行构造、双门和选择器。 | 当前单批冻结生产总入口。 |
| `run_family_target_v1.py` | 汇总已有PASS并逐批补充，直到获得精确目标数量。 | 当前按PASS目标扩展数据的编排入口。 |
| `run_family_100_v1.py` | 封装当前项目默认路径，复现100条MISSING或CONFLICT目标。 | 当前100+100的一键复现入口。 |
| `run_source_to_family_smoke_v1.py` | 编排“母池→候选→核心黄金→MISSING/CONFLICT”的小规模全链路冒烟。 | 用于验证上游扩展链路没有断裂。 |
| `evaluate_family_qwen_baseline_v1.py` | 用统一三动作JSON协议评测Qwen初始动作和用户补充后的恢复。 | 当前Qwen基线及原始提示词所在文件。 |
| `generate_family_freeze_report_v1.py` | 汇总盲测质量门、Qwen基线、哈希和协议修订，生成封板报告。 | 用于重建当前冻结报告。 |

## 4. `evaluation/`：核心黄金筛选与共享评测

| 文件 | 简单说明 | 保留原因 |
|---|---|---|
| `evaluation/evaluate_eplus.py` | 提供OpenAI兼容API调用、模型配置、证据渲染和FULL评测基础能力。 | 多个构造器、验证器和筛选器共同依赖的底层模块。 |
| `evaluation/filter_core_gold.py` | 实现QUERY_ONLY、CLEAN_FULL、COUNTERFACTUAL三阶段核心黄金漏斗。 | 当前v3/v4只覆盖部分行为，主体仍在此文件中。 |
| `evaluation/filter_core_gold_v3.py` | 在核心漏斗主体上增加答案类型感知的反事实替换。 | 当前v4直接复用其类型判断与数字变换。 |
| `evaluation/filter_core_gold_v4.py` | 为核心黄金漏斗注入当前v4版本、递归安全的兜底替换并启动主体。 | 当前正式核心黄金筛选入口。 |
| `evaluation/verify_core_gold.py` | 不调用模型，独立检查核心黄金阶段状态、答案匹配、哈希和输出一致性。 | 当前核心黄金的离线审计主体。 |
| `evaluation/verify_core_gold_v4.py` | 激活v4命名空间后运行独立审计主体。 | 当前v4产物的正式验证入口。 |
| `evaluation/compact_core_gold_calls.py` | 只保留最终漏斗实际使用的最新模型调用，压缩调用日志。 | 当前核心黄金目录已有对应压缩报告，保留用于复现日志整理。 |
| `evaluation/validate_missing1_llm.py` | 历史名称的三条件验证模块，同时提供JSONL、证据渲染、JSON解析、标准化和严格答案匹配函数。 | 旧MISSING-1 CLI已不再使用，但当前核心黄金筛选和母池扩展仍导入其中的共享函数，现阶段不能直接删除。 |

## 5. `evaluation/tests/`：回归测试

| 文件 | 简单说明 |
|---|---|
| `evaluation/tests/test_build_conflict_families_v1.py` | 检查冲突必须原子、保留原断言、澄清覆盖双方值等构造硬规则。 |
| `evaluation/tests/test_build_missing_families_split_v2_3.py` | 检查缺口编辑、答案泄露、澄清目标、恢复信息和重试行为。 |
| `evaluation/tests/test_core_gold_pipeline.py` | 检查核心候选稳定抽样、支撑文档模式和基础反事实构造。 |
| `evaluation/tests/test_core_gold_v3.py` | 检查v3类型判断、日期范围、数值格式和模型默认配置。 |
| `evaluation/tests/test_core_gold_v4.py` | 检查v4非递归兜底和对v2阶段的兼容。 |
| `evaluation/tests/test_evaluate_eplus.py` | 检查API适配、断点续跑、证据模式、答案评分和配置安全。 |
| `evaluation/tests/test_family_freeze_v1.py` | 检查盲测选样、开发集排除、双门共识和Qwen响应契约。 |
| `evaluation/tests/test_finalize_core_gold_metadata.py` | 检查答案类型元数据的确定性修复。 |
| `evaluation/tests/test_sanitize_core_gold_candidates.py` | 检查答案别名泄露清洗，避免部分单词误杀。 |
| `evaluation/tests/test_select_conflict_high_quality_v1.py` | 检查CONFLICT最终选择器的语义目标和答案匹配规则。 |
| `evaluation/tests/test_source_family_chain_v1.py` | 检查母池扩展与目标累积链路的ID稳定性和去重。 |
| `evaluation/tests/test_validate_conflict_families_v1.py` | 检查CONFLICT语义门维度及程序最终裁决。 |
| `evaluation/tests/test_validate_missing_families_v1.py` | 检查MISSING语义门对完整残余链和query-only知识的拒收。 |
| `evaluation/tests/test_validate_missing1_llm.py` | 检查共享JSON解析、严格答案匹配、证据恢复和旧三条件验证规则。 |

这些测试不参与数据生产，但它们冻结了大量不容易从文件名看出的质量规则。删除测试不会立即改变数据，却会使后续修改无法确认是否破坏现有协议，因此全部保留。

## 6. 看似重复但暂不合并的文件

### 核心黄金`v2/v3/v4`

`build_core_gold_candidates_v2.py`、`filter_core_gold_v3.py`和`filter_core_gold_v4.py`都是较小的兼容层，但当前正式manifest和文档引用这些入口。现在合并会改变导入关系、版本命名和复现命令，收益小于风险。

### 两个CONFLICT验证器

`validate_conflict_families_v1.py`和`validate_conflict_families_v1_3_frozen.py`不是简单重复。当前准入规则要求两个独立提示版本共识，后者必须保持独立冻结。

### 旧名`validate_missing1_llm.py`

该文件的旧三条件验证CLI已经退出当前生产流程，但其中的通用I/O、证据渲染、JSON解析和答案匹配仍被现行核心黄金代码使用。未来可以先把共享函数迁移到独立工具模块，再删除旧CLI；在迁移和等价测试完成前直接删除会使当前入口报错。

## 7. 后续清理规则

以后只有同时满足以下条件的Python文件才可删除：

1. 不被任何现行入口、模块或测试导入；
2. 不被正式manifest、复现命令或当前报告引用；
3. 已有功能等价的新实现；
4. 删除后全部回归测试通过；
5. 对应说明文档同步更新。
