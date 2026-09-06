# 底座模型选择结论

## 结论

论文主表建议使用两个底座：

1. `Qwen2.5-7B-Instruct`：主开发底座。许可证宽松、结构化输出稳定、中文和英文能力兼顾，也有成熟的 SFT/DPO/GRPO 配方。
2. `Llama-3.1-8B-Instruct`：主复现底座。近年的公开后训练工作大量基于 Llama 3.1 8B，适合与已有论文横向比较；使用前需接受 Meta 许可证并取得权重访问权限。

保留两个辅助底座：

- `Qwen3-8B`：与项目早期 Qwen3-8B 结果连续对比，但不作为唯一主结果。其 thinking/non-thinking 双模式会引入额外变量，评测时固定 `enable_thinking=false`。
- `Mistral-7B-Instruct-v0.3`：Apache-2.0 的跨模型家族稳健性检查。资源不足时可不进入主表。

暂不把 `Gemma-2-9B-it` 设为必跑底座：参数规模略大、权重受访问条款约束，且在本课题中提供的信息增量小于第二个主底座。

## 为什么不只用 Qwen3-8B

Qwen3-8B 可以继续作为基线，但它发布时间更晚，思考模式的开关会影响输出长度、JSON 遵循和拒答行为。仅用它会同时改变“模型家族”和“推理模式”，不利于把论文收益归因于训练方法。Qwen2.5-7B-Instruct 与 Llama-3.1-8B-Instruct 的组合更便于复现和横向对照。

## Base 还是 Instruct

主实验使用 Instruct 版本。本任务要求模型理解动作协议、输出 JSON、进行一次澄清并继续回答，本质是对已有助手模型做行为后训练。若使用纯 Base 模型，首先学会聊天格式本身就会消耗数据，并混入与论文目标无关的变量。

只有在回答“方法是否能从纯预训练模型学出交互行为”这一附加研究问题时，才增加 Base 版本；它不应替代主实验。

## 与公开实践的对应关系

- Ai2 的 Tülu 3 以 Llama 3.1 为底座，公开了 SFT、偏好优化和带可验证奖励的强化学习流程：<https://allenai.org/tulu>、<https://allenai.github.io/open-instruct/>。
- Hugging Face Alignment Handbook 给出了 SFT→DPO、LoRA/QLoRA 和多卡训练配方：<https://github.com/huggingface/alignment-handbook>。
- TRL 官方提供 SFT、DPO、GRPO 训练器以及 PEFT、vLLM 集成：<https://huggingface.co/docs/trl/sft_trainer>、<https://huggingface.co/docs/trl/main/dpo_trainer>、<https://huggingface.co/docs/trl/grpo_trainer>。
- Open-R1 的公开 GRPO 配方直接覆盖 Qwen2.5 系列：<https://github.com/huggingface/open-r1>。
- 模型卡：<https://huggingface.co/Qwen/Qwen2.5-7B-Instruct>、<https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct>、<https://huggingface.co/Qwen/Qwen3-8B>、<https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3>。

## 4×4090 的使用建议

- LoRA-SFT：优先单卡单实验，4张卡并行跑模型、随机种子或消融；当前200条数据没有必要为吞吐量启用复杂的 ZeRO。
- LoRA-DPO：先用单卡或两卡验证显存；脚本采用 PEFT 模型并由 TRL 在参考计算时关闭适配器，避免复制完整8B参考模型。
- 全参数微调：当前数据规模不值得，且会显著增加过拟合和工程成本。
- GRPO/PPO：第一阶段不跑。先证明 SFT 与 DPO 的收益，并冻结可自动判定的奖励；否则 RL 只会放大奖励漏洞。

