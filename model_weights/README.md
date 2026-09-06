# 模型权重目录

此目录只保存本地底座权重和LoRA适配器，不提交大文件。默认目录结构：

```text
model_weights/
├── Qwen2.5-7B-Instruct/
├── Llama-3.1-8B-Instruct/
├── Qwen3-8B/
├── Mistral-7B-Instruct-v0.3/
└── adapters/
```

目录名与 `../evidence_posttrain_baselines/configs/models/*.yaml` 的 `local_path` 对应。若使用其他挂载位置，只需修改模型配置中的 `local_path` 或总配置中的 `model_root`。

Llama-3.1 权重需要先在 Hugging Face 接受 Meta 许可证。下载示例：

```bash
huggingface-cli download Qwen/Qwen2.5-7B-Instruct --local-dir Qwen2.5-7B-Instruct
huggingface-cli download meta-llama/Llama-3.1-8B-Instruct --local-dir Llama-3.1-8B-Instruct
```

