"""Artifact persistence: save, load, validate, and content-address PorTAL artifacts.

Artifact layout on disk (content-addressed):

    {output_dir}/
    └── {task_name}/
        ├── source_lora_{config_hash}/
        │   └── adapter/              # PEFT-compatible LoRA checkpoint
        │       ├── adapter_model.safetensors
        │       └── adapter_config.json
        ├── task_latent_{config_hash}/
        │   ├── task_latent.safetensors
        │   └── task_latent_meta.json
        ├── target_lora_{config_hash}/
        │   └── adapter/
        │       ├── adapter_model.safetensors
        │       └── adapter_config.json
        └── eval_{config_hash}/
            └── eval_results.json
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import safetensors.torch as st
import torch

from portal.config import (
    ADAPTER_DIR_NAME,
    EVAL_FILENAME,
    LATENT_FILENAME,
    LATENT_META_FILENAME,
    ArtifactKind,
    content_hash,
)


# ---------------------------------------------------------------------------
# Task latents
# ---------------------------------------------------------------------------


def save_task_latent(
    latent: torch.Tensor,
    meta: dict[str, Any],
    output_dir: Path,
) -> Path:
    """Save a task latent tensor + metadata. Returns the artifact directory."""
    artifact_dir = _artifact_dir(output_dir, meta, ArtifactKind.TASK_LATENT)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    st.save_file({"task_latent": latent}, artifact_dir / LATENT_FILENAME)

    meta_with_ts = {**meta, "created_at": _now_iso()}
    (artifact_dir / LATENT_META_FILENAME).write_text(
        json.dumps(meta_with_ts, indent=2, default=str)
    )
    return artifact_dir


def load_task_latent(artifact_dir: Path) -> tuple[torch.Tensor, dict[str, Any]]:
    """Load a task latent and its metadata from an artifact directory."""
    tensors = st.load_file(str(artifact_dir / LATENT_FILENAME))
    meta = json.loads((artifact_dir / LATENT_META_FILENAME).read_text())
    return tensors["task_latent"], meta


# ---------------------------------------------------------------------------
# Adapters (PEFT-compatible LoRA checkpoints)
# ---------------------------------------------------------------------------


def save_adapter(
    model,  # PeftModel
    config: dict[str, Any],
    output_dir: Path,
    *,
    kind: str = "source",
) -> Path:
    """Save a PEFT adapter. Returns the adapter directory."""
    tag = "source_lora" if kind == "source" else "target_lora"
    artifact_dir = _artifact_dir(output_dir, config, tag=tag)
    adapter_dir = artifact_dir / ADAPTER_DIR_NAME
    adapter_dir.mkdir(parents=True, exist_ok=True)

    model.save_pretrained(str(adapter_dir))

    meta = {**config, "kind": kind, "created_at": _now_iso()}
    (artifact_dir / "adapter_meta.json").write_text(
        json.dumps(meta, indent=2, default=str)
    )
    return artifact_dir


def load_adapter_path(artifact_dir: Path) -> Path:
    """Return the PEFT adapter directory inside an artifact dir."""
    adapter_dir = artifact_dir / ADAPTER_DIR_NAME
    if not adapter_dir.exists():
        raise FileNotFoundError(f"No adapter directory at {adapter_dir}")
    return adapter_dir


# ---------------------------------------------------------------------------
# Eval results
# ---------------------------------------------------------------------------


def save_eval_results(
    results: dict[str, Any],
    config: dict[str, Any],
    output_dir: Path,
) -> Path:
    """Save evaluation results as JSON. Returns the artifact directory."""
    artifact_dir = _artifact_dir(output_dir, config, ArtifactKind.EVAL)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "config": config,
        "metrics": results,
        "created_at": _now_iso(),
    }
    (artifact_dir / EVAL_FILENAME).write_text(json.dumps(payload, indent=2, default=str))
    return artifact_dir


def load_eval_results(artifact_dir: Path) -> dict[str, Any]:
    """Load evaluation results from an artifact directory."""
    return json.loads((artifact_dir / EVAL_FILENAME).read_text())


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


def find_artifact(output_dir: Path, task_name: str, kind: str, config_hash: str) -> Path | None:
    """Look up an existing content-addressed artifact directory."""
    candidate = output_dir / task_name / f"{kind}_{config_hash}"
    return candidate if candidate.exists() else None


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _artifact_dir(
    output_dir: Path,
    config: dict[str, Any],
    kind: ArtifactKind | str = ArtifactKind.TASK_LATENT,
    *,
    tag: str | None = None,
) -> Path:
    """Build a content-addressed artifact directory path."""
    task_name = config.get("task_name", "unknown_task")
    label = tag or (kind.value if isinstance(kind, ArtifactKind) else kind)
    h = content_hash(config)
    return output_dir / task_name / f"{label}_{h}"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
