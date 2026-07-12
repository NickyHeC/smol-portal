"""Hypernetwork: extract base-agnostic task latents from a trained LoRA adapter.

Architecture (HypeLoRA-inspired):
    A small encoder network maps the flattened LoRA weight matrices (A, B per layer)
    into a compact task latent vector. The decoder (used only during training) maps
    the latent back to reconstructed LoRA weights. At inference time, only the
    encoder output (the task latent) is kept.

    LoRA weights → flatten → Encoder → task latent (z) → Decoder → reconstructed LoRA
                                         ↓
                              saved as artifact (.safetensors)
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from safetensors.torch import load_file

from portal.artifacts import save_task_latent
from portal.config import HypernetConfig, content_hash
from portal.cuda import configure_cuda_for_smolvm


class LoraAutoencoder(nn.Module):
    """Autoencoder that compresses LoRA weights into a task latent."""

    def __init__(self, input_dim: int, latent_dim: int, hidden_dim: int, num_layers: int):
        super().__init__()
        encoder_layers: list[nn.Module] = []
        decoder_layers: list[nn.Module] = []

        dims = [input_dim] + [hidden_dim] * (num_layers - 1) + [latent_dim]
        for i in range(len(dims) - 1):
            encoder_layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                encoder_layers.append(nn.GELU())
        self.encoder = nn.Sequential(*encoder_layers)

        rev_dims = list(reversed(dims))
        for i in range(len(rev_dims) - 1):
            decoder_layers.append(nn.Linear(rev_dims[i], rev_dims[i + 1]))
            if i < len(rev_dims) - 2:
                decoder_layers.append(nn.GELU())
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return z, x_hat


def _flatten_lora_weights(adapter_dir: Path) -> torch.Tensor:
    """Load LoRA adapter weights and flatten into a single vector."""
    safetensors_path = adapter_dir / "adapter_model.safetensors"
    if not safetensors_path.exists():
        raise FileNotFoundError(f"No adapter_model.safetensors in {adapter_dir}")

    state_dict = load_file(str(safetensors_path))
    weight_tensors = []
    for key in sorted(state_dict.keys()):
        if "lora_" in key:
            weight_tensors.append(state_dict[key].flatten().float())

    if not weight_tensors:
        raise ValueError(f"No LoRA weight tensors found in {adapter_dir}")

    return torch.cat(weight_tensors)


def extract_task_latent(
    adapter_dir: Path,
    source_model: str,
    task_name: str,
    config: HypernetConfig,
    output_dir: Path,
) -> Path:
    """Train a LoRA autoencoder and extract the task latent.

    Returns the artifact directory containing the task latent.
    """
    torch.manual_seed(config.seed)
    configure_cuda_for_smolvm()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    lora_weights = _flatten_lora_weights(adapter_dir)
    input_dim = lora_weights.shape[0]

    autoencoder = LoraAutoencoder(
        input_dim=input_dim,
        latent_dim=config.latent_dim,
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers,
    ).to(device)

    lora_weights = lora_weights.to(device).unsqueeze(0)  # [1, input_dim]
    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=config.learning_rate)
    loss_fn = nn.MSELoss()

    autoencoder.train()
    for epoch in range(config.num_epochs):
        optimizer.zero_grad()
        z, x_hat = autoencoder(lora_weights)
        loss = loss_fn(x_hat, lora_weights)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  [hypernet] epoch {epoch + 1}/{config.num_epochs}  loss={loss.item():.6f}")

    autoencoder.eval()
    with torch.no_grad():
        task_latent, _ = autoencoder(lora_weights)
        task_latent = task_latent.squeeze(0).cpu()  # [latent_dim]

    meta = {
        "source_model": source_model,
        "task_name": task_name,
        "latent_dim": config.latent_dim,
        "input_dim": input_dim,
        "config": config.model_dump(),
        "config_hash": content_hash(config.model_dump()),
    }

    return save_task_latent(task_latent, meta, output_dir)
