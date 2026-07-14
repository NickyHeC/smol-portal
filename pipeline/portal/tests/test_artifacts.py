"""Tests for artifact persistence and content-addressing."""

from __future__ import annotations

import torch

from portal.artifacts import (
    find_artifact,
    load_eval_results,
    load_task_latent,
    save_eval_results,
    save_task_latent,
)
from portal.config import content_hash


def test_task_latent_roundtrip(tmp_path):
    latent = torch.randn(256)
    meta = {"source_model": "test-model", "task_name": "test_task", "latent_dim": 256}

    artifact_dir = save_task_latent(latent, meta, tmp_path)

    assert artifact_dir.exists()
    assert (artifact_dir / "task_latent.safetensors").exists()
    assert (artifact_dir / "task_latent_meta.json").exists()

    loaded_latent, loaded_meta = load_task_latent(artifact_dir)
    assert loaded_latent.shape == latent.shape
    assert torch.allclose(loaded_latent, latent, atol=1e-6)
    assert loaded_meta["source_model"] == "test-model"
    assert "created_at" in loaded_meta


def test_latent_records_runtime_but_hash_excludes_it(tmp_path):
    latent = torch.randn(64)
    meta = {"source_model": "m", "task_name": "t", "latent_dim": 64}

    artifact_dir = save_task_latent(latent, meta, tmp_path)

    # Provenance manifest is recorded in the written metadata…
    _, loaded_meta = load_task_latent(artifact_dir)
    assert "runtime" in loaded_meta
    assert "packages" in loaded_meta["runtime"]

    # …but the directory hash is computed over the caller's meta only, so it
    # stays stable across machines/library versions (idempotent reruns).
    assert artifact_dir.name.endswith(content_hash(meta))


def test_eval_results_roundtrip(tmp_path):
    metrics = {"loss": 0.42, "perplexity": 1.52, "num_samples": 100}
    config = {"task_name": "test_task", "model_name": "test-model", "dataset_name": "test-ds"}

    artifact_dir = save_eval_results(metrics, config, tmp_path)

    assert artifact_dir.exists()
    assert (artifact_dir / "eval_results.json").exists()

    loaded = load_eval_results(artifact_dir)
    assert loaded["metrics"]["loss"] == 0.42
    assert loaded["config"]["task_name"] == "test_task"
    assert "created_at" in loaded


def test_content_addressing_is_deterministic(tmp_path):
    meta = {"source_model": "m", "task_name": "t", "latent_dim": 128}
    h1 = content_hash(meta)
    h2 = content_hash(meta)
    assert h1 == h2
    assert len(h1) == 16


def test_content_addressing_changes_with_input():
    h1 = content_hash({"task_name": "a"})
    h2 = content_hash({"task_name": "b"})
    assert h1 != h2


def test_find_artifact_exists(tmp_path):
    meta = {"task_name": "my_task"}
    h = content_hash(meta)
    artifact_dir = tmp_path / "my_task" / f"task_latent_{h}"
    artifact_dir.mkdir(parents=True)

    found = find_artifact(tmp_path, "my_task", "task_latent", h)
    assert found == artifact_dir


def test_find_artifact_missing(tmp_path):
    found = find_artifact(tmp_path, "no_task", "task_latent", "deadbeef")
    assert found is None
