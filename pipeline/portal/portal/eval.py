"""Adapter evaluation: benchmark a LoRA adapter on a task."""

from __future__ import annotations

from pathlib import Path

import torch
from datasets import load_dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

from portal.artifacts import load_adapter_path, save_eval_results
from portal.config import EvalConfig


def evaluate_adapter(
    adapter_dir: Path,
    config: EvalConfig,
    output_dir: Path,
) -> Path:
    """Evaluate a LoRA adapter on the given benchmark.

    Returns the artifact directory containing eval results JSON.
    """
    set_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(config.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        trust_remote_code=True,
    )

    adapter_path = load_adapter_path(adapter_dir)
    model = PeftModel.from_pretrained(base_model, str(adapter_path))
    model.eval()
    model.to(device)

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

    total_loss = 0.0
    total_tokens = 0
    num_batches = 0

    with torch.no_grad():
        for start in range(0, len(ds), config.batch_size):
            end = min(start + config.batch_size, len(ds))
            batch = _collate_slice(ds, start, end, device)
            outputs = model(**batch)

            mask = batch["attention_mask"]
            tokens_in_batch = mask.sum().item()
            total_loss += outputs.loss.item() * tokens_in_batch
            total_tokens += tokens_in_batch
            num_batches += 1

    avg_loss = total_loss / max(total_tokens, 1)
    perplexity = torch.exp(torch.tensor(avg_loss)).item()

    metrics = {
        "loss": round(avg_loss, 6),
        "perplexity": round(perplexity, 4),
        "num_samples": len(ds),
        "num_batches": num_batches,
    }

    eval_config = {
        "task_name": config.task_name,
        "model_name": config.model_name,
        "dataset_name": config.dataset_name,
        "dataset_split": config.dataset_split,
        "adapter_dir": str(adapter_dir),
    }

    result_dir = save_eval_results(metrics, eval_config, output_dir)

    print(f"  [eval] loss={metrics['loss']}  ppl={metrics['perplexity']}  "
          f"samples={metrics['num_samples']}")
    return result_dir


def _collate_slice(ds, start: int, end: int, device: str) -> dict[str, torch.Tensor]:
    """Collate a contiguous slice of the dataset into a batch."""
    batch = {}
    for key in ds[0].keys():
        tensors = [ds[i][key] for i in range(start, end)]
        batch[key] = torch.stack(tensors).to(device)
    batch["labels"] = batch["input_ids"].clone()
    return batch
