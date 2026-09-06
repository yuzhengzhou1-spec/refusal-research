# Evidence-Supported Interactive QA

本仓库用于构建证据状态样本家族，并研究大模型在FULL、MISSING和CONFLICT状态下的回答、请求补充和请求确认行为。

## 目录

- `parent_pool_project/`：数据筛选与样本家族生产。
- `evidence_posttrain_baselines/`：vLLM推理、SFT-LoRA、DPO-LoRA和统一评测。
- `evidence_posttrain_methods/`：论文方法与后续消融。
- `model_weights/`：本地模型目录骨架；实际权重不进入Git。
- `ops/`：本机推送、服务器更新、环境初始化和实验溯源工具。

## 从哪里开始

- 项目结构：[POSTTRAIN_WORKSPACE.md](POSTTRAIN_WORKSPACE.md)
- 本机到服务器工作流：[SYNC_AND_SERVER_GUIDE.md](SYNC_AND_SERVER_GUIDE.md)
- 基线运行：[evidence_posttrain_baselines/README.md](evidence_posttrain_baselines/README.md)
- 数据生产：[parent_pool_project/SOURCE_TO_HIGH_QUALITY_FAMILY_PIPELINE_V1.md](parent_pool_project/SOURCE_TO_HIGH_QUALITY_FAMILY_PIPELINE_V1.md)

提交前运行：

```powershell
python ops/check_repository.py
```

服务器上的正式实验必须通过 `ops/run_with_manifest.py` 或 `ops/run_experiment.sh` 启动，以保存代码提交、配置和硬件环境。

