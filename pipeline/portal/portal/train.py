"""Source LoRA training: fine-tune a base model on a task with LoRA."""

from __future__ import annotations

from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig as PeftLoraConfig
from peft import TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    set_seed,
)

from portal.artifacts import save_adapter
from portal.config import TrainConfig


def train_source_lora(config: TrainConfig, output_dir: Path) -> Path:
    """Train a LoRA adapter on the source model and save it as a content-addressed artifact.

    Returns the artifact directory containing the PEFT adapter.
    """
    set_seed(config.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(config.source_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        config.source_model,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        trust_remote_code=True,
    )

    peft_config = PeftLoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=config.lora.rank,
        lora_alpha=config.lora.alpha,
        lora_dropout=config.lora.dropout,
        target_modules=config.lora.target_modules,
    )
    model = get_peft_model(model, peft_config)

    ds = load_dataset(config.dataset_name, split=config.dataset_split)
    if config.max_samples is not None:
        ds = ds.select(range(min(config.max_samples, len(ds))))

    def tokenize(example: dict) -> dict:
        text = example.get("text") or example.get("input", "")
        return tokenizer(
            text,
            truncation=True,
            max_length=config.max_seq_length,
            padding="max_length",
        )

    ds = ds.map(tokenize, batched=False, remove_columns=ds.column_names)
    ds.set_format("torch")

    training_dir = output_dir / config.task_name / "_training_runs" / config.content_hash()
    args = TrainingArguments(
        output_dir=str(training_dir),
        num_train_epochs=config.num_epochs,
        per_device_train_batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        bf16=(device == "cuda"),
        logging_steps=10,
        save_strategy="no",
        report_to="none",
        seed=config.seed,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=ds,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )
    trainer.train()

    artifact_dir = save_adapter(
        model,
        config=config.model_dump(),
        output_dir=output_dir,
        kind="source",
    )
    return artifact_dir
