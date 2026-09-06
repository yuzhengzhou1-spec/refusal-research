from __future__ import annotations

import argparse

from common import load_experiment, load_yaml, resolve_root_path


def main() -> None:
    parser = argparse.ArgumentParser(description="从SFT适配器继续执行DPO-LoRA")
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--train-config", default="configs/training/dpo_lora.yaml")
    parser.add_argument("--sft-adapter", help="覆盖训练配置中的sft_adapter")
    args = parser.parse_args()

    import torch
    from datasets import load_dataset
    from peft import AutoPeftModelForCausalLM
    from transformers import AutoTokenizer
    from trl import DPOConfig, DPOTrainer

    experiment, model_config, _ = load_experiment(args.config)
    training = load_yaml(args.train_config)
    processed = resolve_root_path(experiment["paths"]["processed_dir"])
    dataset = load_dataset("json", data_files={"train": str(processed / "dpo_train.jsonl")})["train"]
    adapter_path = resolve_root_path(args.sft_adapter or training["sft_adapter"])
    model = AutoPeftModelForCausalLM.from_pretrained(adapter_path, is_trainable=True, dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(adapter_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    output = resolve_root_path(experiment["paths"]["output_dir"]) / model_config["name"] / training["output_subdir"]
    train_args = DPOConfig(
        output_dir=str(output), num_train_epochs=training["num_train_epochs"], learning_rate=training["learning_rate"],
        per_device_train_batch_size=training["per_device_train_batch_size"], per_device_eval_batch_size=training["per_device_eval_batch_size"],
        gradient_accumulation_steps=training["gradient_accumulation_steps"], max_length=training["max_length"],
        beta=training["beta"], loss_type=[training["loss_type"]],
        logging_steps=training["logging_steps"], save_steps=training["save_steps"], warmup_ratio=training["warmup_ratio"],
        bf16=training["bf16"], tf32=training["tf32"], gradient_checkpointing=training["gradient_checkpointing"],
        gradient_checkpointing_kwargs={"use_reentrant": False}, report_to=["tensorboard"], seed=training["seed"], save_total_limit=2,
    )
    # 不复制8B参考模型；TRL计算参考分数时会临时关闭当前PEFT适配器。
    trainer = DPOTrainer(model=model, ref_model=None, args=train_args, train_dataset=dataset, processing_class=tokenizer)
    trainer.train()
    trainer.save_model(str(output / "final_adapter"))
    tokenizer.save_pretrained(str(output / "final_adapter"))


if __name__ == "__main__":
    main()
