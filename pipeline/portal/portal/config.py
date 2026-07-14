"""Shared types, config schemas, and constants for PorTAL."""

from __future__ import annotations

import hashlib
import json
from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_OUTPUT_DIR = Path("artifacts")
LATENT_FILENAME = "task_latent.safetensors"
LATENT_META_FILENAME = "task_latent_meta.json"
ADAPTER_DIR_NAME = "adapter"
EVAL_FILENAME = "eval_results.json"


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class ArtifactKind(StrEnum):
    TASK_LATENT = "task_latent"
    ADAPTER = "adapter"
    EVAL = "eval"


# ---------------------------------------------------------------------------
# Training config
# ---------------------------------------------------------------------------


class LoraConfig(BaseModel):
    """LoRA hyperparameters for source-model fine-tuning."""

    rank: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: list[str] = Field(
        default_factory=lambda: ["q_proj", "v_proj", "k_proj", "o_proj"]
    )


class TrainConfig(BaseModel):
    """Full config for training a source LoRA adapter."""

    source_model: str
    task_name: str
    dataset_name: str
    dataset_split: str = "train"
    max_samples: int | None = None
    lora: LoraConfig = Field(default_factory=LoraConfig)
    learning_rate: float = 2e-4
    num_epochs: int = 3
    batch_size: int = 4
    max_seq_length: int = 512
    seed: int = 42

    def content_hash(self) -> str:
        """Deterministic hash of this config for content-addressing."""
        blob = json.dumps(self.model_dump(), sort_keys=True, default=str)
        return hashlib.sha256(blob.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Hypernetwork / latent config
# ---------------------------------------------------------------------------


class HypernetConfig(BaseModel):
    """Config for the hypernetwork that extracts base-agnostic task latents."""

    latent_dim: int = 256
    hidden_dim: int = 512
    num_layers: int = 3
    learning_rate: float = 1e-3
    num_epochs: int = 50
    seed: int = 42


# ---------------------------------------------------------------------------
# Converter config
# ---------------------------------------------------------------------------


class LatentMode(StrEnum):
    """Latent-ablation modes for the converter (see ROADMAP Phase A2).

    Used to test whether the source task latent actually contributes: if
    ``zero``/``random``/``shuffled`` reach the same target performance as
    ``real`` after converter training, the converter is learning the task from
    calibration data and ignoring the latent.
    """

    REAL = "real"
    ZERO = "zero"
    RANDOM = "random"
    SHUFFLED = "shuffled"


class ConverterConfig(BaseModel):
    """Config for the slim converter that projects a task latent into a target model."""

    target_model: str
    calibration_dataset: str | None = None
    calibration_split: str = "train"
    calibration_samples: int = 256
    hidden_dim: int = 512
    learning_rate: float = 1e-3
    num_epochs: int = 30
    latent_mode: LatentMode = LatentMode.REAL
    seed: int = 42


# ---------------------------------------------------------------------------
# Eval config
# ---------------------------------------------------------------------------


class EvalConfig(BaseModel):
    """Config for adapter evaluation."""

    model_name: str
    task_name: str
    dataset_name: str
    dataset_split: str = "test"
    max_samples: int | None = None
    batch_size: int = 8
    max_seq_length: int = 512


# ---------------------------------------------------------------------------
# Top-level port config (wires everything together)
# ---------------------------------------------------------------------------


class PortConfig(BaseModel):
    """Full end-to-end config for `portal port`.

    Per-stage sizing knobs (below) let the CLI drive smoke-sized or real runs
    directly, matching the `port_e2e.py` reference driver. Their defaults equal
    the individual stage-config defaults, so a plain `portal port` is unchanged
    (and produces the same content-addressed artifacts as before). Providing an
    explicit `train`/`hypernet`/`converter` config overrides the knobs for that
    stage.
    """

    source_model: str
    target_model: str
    task_name: str
    dataset_name: str
    output_dir: Path = DEFAULT_OUTPUT_DIR
    train: TrainConfig | None = None
    hypernet: HypernetConfig | None = None
    converter: ConverterConfig | None = None
    eval_split: str = "test"
    skip_train: bool = False

    # --- per-stage sizing knobs (defaults mirror the stage configs) ---
    calibration_dataset: str | None = None  # defaults to dataset_name
    max_samples: int | None = None
    max_seq_length: int = 512
    batch_size: int = 4
    lora_rank: int = 16
    train_epochs: int = 3
    extract_epochs: int = 50
    convert_epochs: int = 30
    cal_samples: int = 256
    latent_dim: int = 256
    hidden_dim: int = 512
    latent_mode: LatentMode = LatentMode.REAL
    seed: int = 42

    def build_train_config(self) -> TrainConfig:
        if self.train is not None:
            return self.train
        return TrainConfig(
            source_model=self.source_model,
            task_name=self.task_name,
            dataset_name=self.dataset_name,
            max_samples=self.max_samples,
            lora=LoraConfig(rank=self.lora_rank),
            num_epochs=self.train_epochs,
            batch_size=self.batch_size,
            max_seq_length=self.max_seq_length,
            seed=self.seed,
        )

    def build_hypernet_config(self) -> HypernetConfig:
        if self.hypernet is not None:
            return self.hypernet
        return HypernetConfig(
            latent_dim=self.latent_dim,
            hidden_dim=self.hidden_dim,
            num_epochs=self.extract_epochs,
            seed=self.seed,
        )

    def build_converter_config(self) -> ConverterConfig:
        if self.converter is not None:
            return self.converter
        return ConverterConfig(
            target_model=self.target_model,
            calibration_dataset=self.calibration_dataset or self.dataset_name,
            calibration_samples=self.cal_samples,
            hidden_dim=self.hidden_dim,
            num_epochs=self.convert_epochs,
            latent_mode=self.latent_mode,
            seed=self.seed,
        )

    def build_eval_config(self, model_name: str) -> EvalConfig:
        return EvalConfig(
            model_name=model_name,
            task_name=self.task_name,
            dataset_name=self.dataset_name,
            dataset_split=self.eval_split,
            max_samples=self.max_samples,
            batch_size=self.batch_size,
            max_seq_length=self.max_seq_length,
        )


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def content_hash(data: dict[str, Any]) -> str:
    """SHA-256 prefix of a JSON-serialised dict, for content-addressing artifacts."""
    blob = json.dumps(data, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]
