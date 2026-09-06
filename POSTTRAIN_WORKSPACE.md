# 论文后训练工作区

三个目录的职责已经分离：

| 目录 | 唯一职责 |
| --- | --- |
| `parent_pool_project/` | 从原始数据筛选种子，并生产 FULL/MISSING/CONFLICT 样本家族 |
| `evidence_posttrain_baselines/` | 冻结切分，运行零样本、SFT、DPO、vLLM推理和论文基线评测 |
| `evidence_posttrain_methods/` | 研究论文拟提出的方法；不得另建不兼容的测试切分 |
| `model_weights/` | 存放底座权重和适配器，不存代码与数据 |

标准数据流：

```text
原始数据
  → parent_pool_project 生产样本家族
  → evidence_posttrain_baselines/prepare_dataset.py 冻结同源无泄漏切分
  → 零样本评测 / SFT-LoRA / DPO-LoRA
  → 同一 vLLM 评测协议
  → evidence_posttrain_methods 中的方法对照
```

任何论文实验都从 `evidence_posttrain_baselines/configs/experiment.yaml` 开始，不再把训练或评测脚本放回数据生产目录。

