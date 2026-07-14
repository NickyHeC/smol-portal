"""Slim converter: project a task latent into a target model's LoRA adapter.

The converter is a small MLP trained to map (task_latent, target_model_info) → LoRA weights
for the target model, using limited calibration data to guide the projection.

Architecture:
    task_latent (z) → Converter MLP → target LoRA weights (per-layer A, B matrices)

The converter is trained per target model by:
    1. Loading the task latent from the source model
    2. Initializing random target LoRA weights
    3. Training the converter to produce LoRA weights that minimize loss on calibration data
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from datasets import load_dataset
from peft import LoraConfig as PeftLoraConfig
from peft import TaskType, get_peft_model
from torch.func import functional_call
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

from portal.artifacts import load_task_latent, save_adapter
from portal.config import ConverterConfig, LatentMode, content_hash
from portal.cuda import causal_lm_load_kwargs, configure_cuda_for_smolvm
from portal.data import extract_text


class LatentToLoraConverter(nn.Module):
    """MLP that maps a task latent to flattened LoRA weight vectors."""

    def __init__(self, latent_dim: int, output_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


def _get_lora_param_shapes(model: nn.Module) -> list[tuple[str, torch.Size]]:
    """Collect names and shapes of LoRA parameters from a PeftModel."""
    shapes = []
    for name, param in model.named_parameters():
        if "lora_" in name and param.requires_grad:
            shapes.append((name, param.shape))
    return shapes


def apply_latent_mode(latent: torch.Tensor, mode: LatentMode, seed: int) -> torch.Tensor:
    """Transform the task latent for ablation experiments.

    ``real`` passes it through unchanged. The other modes replace or scramble
    the latent so we can measure how much the source task information actually
    matters to the converter. Deterministic given ``seed``.
    """
    if mode == LatentMode.REAL:
        return latent
    if mode == LatentMode.ZERO:
        return torch.zeros_like(latent)

    generator = torch.Generator(device="cpu").manual_seed(seed)
    if mode == LatentMode.RANDOM:
        noise = torch.randn(latent.shape, generator=generator, dtype=latent.dtype)
        return noise.to(latent.device)
    if mode == LatentMode.SHUFFLED:
        flat = latent.flatten().cpu()
        perm = torch.randperm(flat.numel(), generator=generator)
        return flat[perm].reshape(latent.shape).to(latent.device)

    raise ValueError(f"Unknown latent mode: {mode!r}")


def convert_latent_to_adapter(
    latent_dir: Path,
    task_name: str,
    config: ConverterConfig,
    output_dir: Path,
    lora_rank: int = 16,
    lora_target_modules: list[str] | None = None,
) -> Path:
    """Train a converter to project the task latent into target LoRA weights.

    Returns the artifact directory containing the target adapter.
    """
    set_seed(config.seed)
    configure_cuda_for_smolvm()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if config.calibration_dataset is None:
        raise ValueError(
            "ConverterConfig.calibration_dataset is required — pass --cal-dataset "
            "(normally the task dataset). It previously fell back to the target "
            "model id, which is not a dataset and silently loaded nothing useful. "
            "`portal port` sets this for you from --dataset."
        )

    task_latent, latent_meta = load_task_latent(latent_dir)
    task_latent = apply_latent_mode(task_latent, config.latent_mode, seed=config.seed)
    latent_dim = task_latent.shape[0]
    task_latent = task_latent.to(device).unsqueeze(0)

    if lora_target_modules is None:
        lora_target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"]

    tokenizer = AutoTokenizer.from_pretrained(config.target_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        config.target_model,
        **causal_lm_load_kwargs(),
    )

    peft_config = PeftLoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_rank,
        lora_alpha=lora_rank * 2,
        lora_dropout=0.0,
        target_modules=lora_target_modules,
    )
    peft_model = get_peft_model(base_model, peft_config)
    lora_shapes = _get_lora_param_shapes(peft_model)
    total_lora_params = sum(s.numel() for _, s in lora_shapes)

    ds = load_dataset(config.calibration_dataset, split=config.calibration_split)
    ds = ds.select(range(min(config.calibration_samples, len(ds))))

    def tokenize(example: dict) -> dict:
        text = extract_text(example)
        return tokenizer(text, truncation=True, max_length=512, padding="max_length")

    ds = ds.map(tokenize, batched=False, remove_columns=ds.column_names)
    ds.set_format("torch")

    converter = LatentToLoraConverter(
        latent_dim=latent_dim,
        output_dim=total_lora_params,
        hidden_dim=config.hidden_dim,
    ).to(device)

    optimizer = torch.optim.Adam(converter.parameters(), lr=config.learning_rate)

    converter.train()
    batch = _collate_batch(ds, batch_size=min(4, len(ds)), device=device)
    for epoch in range(config.num_epochs):
        predicted_weights = converter(task_latent).squeeze(0)

        # Map predicted weights to the LoRA parameter names, keeping them in the
        # autograd graph. `functional_call` runs the forward pass with these
        # tensors substituted for the model's own parameters, so gradients flow
        # back to the converter. Assigning to `param.data` (the previous
        # approach) detaches the graph and the converter never learns.
        overrides = {}
        offset = 0
        for name, shape in lora_shapes:
            numel = shape.numel()
            overrides[name] = predicted_weights[offset : offset + numel].reshape(shape)
            offset += numel

        outputs = functional_call(peft_model, overrides, args=(), kwargs=batch)
        loss = outputs.loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  [converter] epoch {epoch + 1}/{config.num_epochs}  loss={loss.item():.4f}")

    # Final weight injection and save
    converter.eval()
    with torch.no_grad():
        final_weights = converter(task_latent).squeeze(0)
        offset = 0
        for name, shape in lora_shapes:
            numel = shape.numel()
            param_data = final_weights[offset : offset + numel].reshape(shape)
            _set_param_by_name(peft_model, name, param_data)
            offset += numel

    artifact_config = {
        "task_name": task_name,
        "target_model": config.target_model,
        "source_latent_hash": content_hash(latent_meta),
        "converter_config": config.model_dump(),
    }
    return save_adapter(peft_model, config=artifact_config, output_dir=output_dir, kind="target")


def _set_param_by_name(model: nn.Module, name: str, data: torch.Tensor) -> None:
    """Set a named parameter's data in-place."""
    parts = name.split(".")
    obj = model
    for part in parts[:-1]:
        obj = getattr(obj, part)
    param = getattr(obj, parts[-1])
    param.data.copy_(data.detach())


def _collate_batch(ds, batch_size: int, device: str) -> dict[str, torch.Tensor]:
    """Build a simple batch dict from the first N examples."""
    batch = {}
    indices = range(min(batch_size, len(ds)))
    for key in ds[0].keys():
        tensors = [ds[i][key] for i in indices]
        batch[key] = torch.stack(tensors).to(device)
    labels = batch["input_ids"].clone()
    # Don't score padding: pad_token == eos_token here, so unmasked pads would
    # dominate the loss with meaningless targets.
    if "attention_mask" in batch:
        labels[batch["attention_mask"] == 0] = -100
    batch["labels"] = labels
    return batch
