# 证据充分性后训练：基线实验工程

本目录只负责论文实验，不生产 MISSING/CONFLICT 数据。上游数据仍由 `../parent_pool_project` 生成；模型权重放在 `../model_weights`。

## 当前研究设计

主底座为 Qwen2.5-7B-Instruct 与 Llama-3.1-8B-Instruct，Qwen3-8B 和 Mistral-7B-Instruct-v0.3 为辅助对照。选择依据见 [MODEL_SELECTION.md](MODEL_SELECTION.md)。

必须报告的比较组：

| 编号 | 方法 | 目的 |
| --- | --- | --- |
| B0 | 零样本提示 + vLLM | 测量未经训练的直接作答倾向 |
| B1 | 仅 FULL 问答 SFT-LoRA | 控制“只是学会回答格式”的收益 |
| B2 | FULL + 初始动作 SFT-LoRA | 测量动作监督，不训练续轮恢复 |
| B3 | FULL + 完整一次澄清轨迹 SFT-LoRA | 核心监督微调基线 |
| B4 | B3 后接 DPO-LoRA | 使用基座模型真实错误作为 rejected response |

当前200条用于打通链路和小规模先导实验，不应被表述为论文最终训练规模。切分按 `source_sample_id` 进行；同一问题的 MISSING/CONFLICT 永远不会跨 train/dev/test。

## 一次完整运行

以下命令均在本目录执行。

```bash
python prepare_dataset.py --config configs/experiment.yaml
python preflight.py --config configs/experiment.yaml --tokenizer-check
python serve_vllm.py --config configs/experiment.yaml --run
python run_vllm_eval.py --config configs/experiment.yaml --split test --run-name qwen25_zero_shot
accelerate launch train_sft_lora.py --config configs/experiment.yaml --variant full_trajectory
python serve_vllm.py --config configs/experiment.yaml --adapter outputs/qwen2_5_7b_instruct/sft_lora_full_trajectory/final_adapter --adapter-name qwen25-sft --run
python run_vllm_eval.py --config configs/experiment.yaml --served-model-name qwen25-sft --split test --run-name qwen25_sft
```

DPO 需要先让未经DPO的模型在训练切分上产生真实错误：

```bash
python run_vllm_eval.py --config configs/experiment.yaml --split train --run-name qwen25_sft_train
python build_dpo_dataset.py --config configs/experiment.yaml --predictions outputs/qwen25_sft_train/predictions.jsonl
accelerate launch train_dpo_lora.py --config configs/experiment.yaml --sft-adapter outputs/qwen2_5_7b_instruct/sft_lora_full_trajectory/final_adapter
```

比较多个结果：

```bash
python compare_runs.py outputs/qwen25_zero_shot/report.json outputs/qwen25_sft/report.json
```

## 切换模型

通常只修改 `configs/experiment.yaml` 的 `active_model`：

```yaml
active_model: llama3_1_8b_instruct
```

可选值为 `qwen2_5_7b_instruct`、`llama3_1_8b_instruct`、`qwen3_8b`、`mistral_7b_instruct_v03`。若权重目录名不同，修改对应的 `configs/models/*.yaml` 中 `local_path`；其余脚本无需改动。

SFT训练视图由 `configs/training/sft_lora.yaml` 的 `dataset_variant` 控制。DPO使用的训练集预测和SFT适配器分别由 `configs/training/dpo_lora.yaml` 的 `source_predictions`、`sft_adapter` 控制。部署适配器时也可以直接填写总配置中的 `inference.adapter_path`、`adapter_name` 和 `served_model_name_override`；命令行参数仅用于临时覆盖。

## 指标定义

- `full_answer_correct`：完整证据下是否回答正确，用于监控过度拒答和能力退化。
- `initial_action_correct`：MISSING 是否请求信息、CONFLICT 是否请求确认。
- `initial_direct_answer`：证据不足/冲突时仍直接回答，越低越好。
- `recovery_answer_correct`：收到补充或确认后是否恢复正确答案。
- `end_to_end_success`：初始动作正确且续轮答案正确，是论文主指标候选。

报告中同时保存初始动作计数和每条原始输出，因此可以直接检查“动作错误时具体做了什么”。

## 文件逐项说明

| 文件 | 功能 |
| --- | --- |
| `configs/experiment.yaml` | 总入口；选择模型、源数据、切分和vLLM连接参数 |
| `configs/task_prompt.yaml` | 所有底座共用的任务提示与JSON协议 |
| `configs/models/*.yaml` | 模型权重位置、服务名、vLLM参数及Qwen3非思考开关 |
| `configs/training/sft_lora.yaml` | SFT-LoRA超参数 |
| `configs/training/dpo_lora.yaml` | DPO-LoRA超参数 |
| `common.py` | 配置、JSONL、消息构造、解析和答案匹配公共函数 |
| `prepare_dataset.py` | 结构检查、同源无泄漏切分及SFT数据生成 |
| `preflight.py` | 检查环境、权重、数据文件和token长度 |
| `serve_vllm.py` | 从模型配置打印或启动vLLM命令 |
| `run_vllm_eval.py` | 运行FULL、初始动作和真实续轮评测，保存逐条输出与汇总 |
| `train_sft_lora.py` | TRL+PEFT的SFT-LoRA入口，支持完整轨迹与消融视图 |
| `build_dpo_dataset.py` | 将训练集上的真实错误变成chosen/rejected偏好对 |
| `train_dpo_lora.py` | 从SFT适配器继续DPO-LoRA |
| `compare_runs.py` | 把多个报告汇总为Markdown对比表 |
| `tests/test_processed_data.py` | 检查同源泄漏、200条轨迹动作和消融数据完整性 |
| `requirements/inference.txt` | vLLM推理环境依赖 |
| `requirements/train.txt` | Transformers/TRL/PEFT训练环境依赖 |
| `data/processed/` | 预处理产物；可重建，不手工编辑 |
| `outputs/` | 模型适配器、逐条预测和报告 |

## 安装建议

建议在Linux机器上使用Python 3.11和独立虚拟环境。PyTorch应按机器CUDA版本先从官方源安装，再安装其余依赖：

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements/train.txt
pip install -r requirements/inference.txt
accelerate config
```

CUDA/PyTorch/vLLM存在严格版本耦合；`cu128` 仅为示例，实际以 NVIDIA 驱动和 vLLM 官方安装页为准。训练和推理最好使用两个虚拟环境，以减少依赖冲突。
