from __future__ import annotations

import argparse
from pathlib import Path

from common import load_experiment, load_yaml, model_path, resolve_root_path


def main() -> None:
    parser = argparse.ArgumentParser(description="使用TRL和PEFT执行SFT-LoRA")
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--train-config", default="configs/training/sft_lora.yaml")
    parser.add_argument("--variant", choices=["full_trajectory", "initial_only", "full_only"], help="覆盖训练配置中的dataset_variant")
    parser.add_argument("--seed", type=int, help="覆盖训练配置中的seed(多种子复验)")
    parser.add_argument("--output-suffix", help="输出目录后缀(多种子复验时区分, 如 _seed20260902)")
    args = parser.parse_args()

    import torch
    from datasets import load_dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    experiment, model_config, _ = load_experiment(args.config)
    training = load_yaml(args.train_config)
    if args.seed is not None:
        training["seed"] = args.seed
    variant = args.variant or training["dataset_variant"]
    base_path = model_path(experiment, model_config)
    processed = resolve_root_path(experiment["paths"]["processed_dir"])
    prefix = {"full_trajectory": "sft", "initial_only": "sft_initial_only", "full_only": "sft_full_only"}[variant]
    dataset = load_dataset("json", data_files={
        "train": str(processed / f"{prefix}_train.jsonl"),
        "validation": str(processed / f"{prefix}_dev.jsonl"),
    })
    tokenizer = AutoTokenizer.from_pretrained(base_path, trust_remote_code=model_config["trust_remote_code"])
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    # assistant_only_loss需要带{% generation %}标记的训练模板; 个别模型(如Llama-3.1)原生模板
    # 不含标记且TRL无法自动补丁。从模型配置注入标记版模板, 并断言渲染输出与原模板逐字符一致。
    if model_config.get("train_chat_template"):
        template_path = (Path(__file__).parent / model_config["train_chat_template"]).resolve()
        marked = template_path.read_text(encoding="utf-8")
        original_template = tokenizer.chat_template
        sample = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]
        tokenizer.chat_template = marked
        assert tokenizer.apply_chat_template(sample, tokenize=False) == tokenizer.apply_chat_template(
            sample, tokenize=False, chat_template=original_template
        ), "标记版训练模板渲染结果与原模板不一致"
        print(f"已注入训练模板: {template_path}")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_path, dtype=torch.bfloat16,
        trust_remote_code=model_config["trust_remote_code"], attn_implementation="sdpa",
    )
    lora = training["lora"]
    peft_config = LoraConfig(
        r=lora["r"], lora_alpha=lora["alpha"], lora_dropout=lora["dropout"],
        target_modules=lora["target_modules"], task_type="CAUSAL_LM", bias="none",
    )
    output = resolve_root_path(experiment["paths"]["output_dir"]) / model_config["name"] / f"{training['output_subdir']}_{variant}{args.output_suffix or ''}"
    # TRL 1.12的SFTConfig只接受warmup_steps，按总步数把比例换算回去
    steps_per_epoch = -(-len(dataset["train"]) // (training["per_device_train_batch_size"] * training["gradient_accumulation_steps"]))
    warmup_steps = max(1, round(float(training["warmup_ratio"]) * steps_per_epoch * int(training["num_train_epochs"])))
    train_args = SFTConfig(
        output_dir=str(output), num_train_epochs=training["num_train_epochs"], learning_rate=training["learning_rate"],
        per_device_train_batch_size=training["per_device_train_batch_size"], per_device_eval_batch_size=training["per_device_eval_batch_size"],
        gradient_accumulation_steps=training["gradient_accumulation_steps"], max_length=training["max_length"],
        logging_steps=training["logging_steps"], eval_strategy="steps", eval_steps=training["eval_steps"],
        save_steps=training["save_steps"], warmup_steps=warmup_steps, lr_scheduler_type=training["lr_scheduler_type"],
        bf16=training["bf16"], tf32=training["tf32"], gradient_checkpointing=training["gradient_checkpointing"],
        gradient_checkpointing_kwargs={"use_reentrant": False}, assistant_only_loss=training["assistant_only_loss"],
        packing=training["packing"], report_to=["tensorboard"], seed=training["seed"], save_total_limit=2,
        load_best_model_at_end=True, metric_for_best_model="eval_loss",
    )
    trainer = SFTTrainer(
        model=base_model, args=train_args, train_dataset=dataset["train"], eval_dataset=dataset["validation"],
        processing_class=tokenizer, peft_config=peft_config,
    )
    trainer.train()
    trainer.save_model(str(output / "final_adapter"))
    tokenizer.save_pretrained(str(output / "final_adapter"))


if __name__ == "__main__":
    main()
