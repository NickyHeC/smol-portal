"""Adapter evaluation: benchmark a LoRA adapter on a task."""

from __future__ import annotations

from pathlib import Path

import torch
from datasets import load_dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

from portal.artifacts import load_adapter_path, save_eval_results
from portal.config import EvalConfig
from portal.cuda import causal_lm_load_kwargs, configure_cuda_for_smolvm


def evaluate_adapter(
    adapter_dir: Path,
    config: EvalConfig,
    output_dir: Path,
) -> Path:
    """Evaluate a LoRA adapter on the given benchmark.

    Returns the artifact directory containing eval results JSON.
    """
    set_seed(42)
    configure_cuda_for_smolvm()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Eval is forward-only, so the fused-SDPA backward concern (#597) doesn't
    # apply. Pin the math SDPA backend regardless of PORTAL_SKIP_CUDA_SMOLVM so
    # the reported metric is reproducible and independent of the training flag.
    if torch.cuda.is_available():
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)

    tokenizer = AutoTokenizer.from_pretrained(config.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        **causal_lm_load_kwargs(),
    )

    adapter_path = load_adapter_path(adapter_dir)
    model = PeftModel.from_pretrained(base_model, str(adapter_path))
    model.eval()

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

            # Weight each batch's mean loss by the number of tokens it actually
            # scored (labels != -100), matching the loss denominator so the
            # aggregate is a true token-level average.
            scored_tokens = (batch["labels"] != -100).sum().item()
            if scored_tokens == 0:
                continue
            total_loss += outputs.loss.item() * scored_tokens
            total_tokens += scored_tokens
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

    print(
        f"  [eval] loss={metrics['loss']}  ppl={metrics['perplexity']}  "
        f"samples={metrics['num_samples']}"
    )
    return result_dir


def _collate_slice(ds, start: int, end: int, device: str) -> dict[str, torch.Tensor]:
    """Collate a contiguous slice of the dataset into a batch."""
    batch = {}
    for key in ds[0].keys():
        tensors = [ds[i][key] for i in range(start, end)]
        batch[key] = torch.stack(tensors).to(device)
    labels = batch["input_ids"].clone()
    # Ignore padding in the loss: pad_token == eos_token, so unmasked pads would
    # score meaningless targets and corrupt perplexity.
    if "attention_mask" in batch:
        labels[batch["attention_mask"] == 0] = -100
    batch["labels"] = labels
    return batch
