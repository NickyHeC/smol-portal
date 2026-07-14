"""Direct target-LoRA baseline.

The PorTAL claim ("recover ~94–98% of direct LoRA accuracy") is only
interpretable against the thing it's compared to: a LoRA trained *directly* on
the target model for the same task. This trains that baseline and evaluates it
on the same split ``portal eval`` uses, so its metrics are directly comparable
to a ported adapter's.

Composition only — it reuses the already-validated ``train_source_lora`` and
``evaluate_adapter`` paths, just pointed at the target model.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from rich.console import Console

from portal.config import EvalConfig, LoraConfig, TrainConfig

console = Console()


def run_direct_lora_baseline(
    target_model: str,
    task_name: str,
    dataset_name: str,
    output_dir: Path,
    *,
    rank: int = 16,
    num_epochs: int = 3,
    max_samples: int | None = None,
    max_seq_length: int = 512,
    batch_size: int = 4,
    eval_split: str = "test",
    seed: int = 42,
) -> dict[str, Any]:
    """Train a LoRA directly on the target model and evaluate it.

    Returns a dict with the adapter and eval artifact directories.
    """
    from portal.eval import evaluate_adapter
    from portal.train import train_source_lora

    console.print("\n[bold]Baseline 1/2:[/] Training direct LoRA on target…")
    train_cfg = TrainConfig(
        source_model=target_model,
        task_name=f"{task_name}__baseline",
        dataset_name=dataset_name,
        max_samples=max_samples,
        lora=LoraConfig(rank=rank),
        num_epochs=num_epochs,
        batch_size=batch_size,
        max_seq_length=max_seq_length,
        seed=seed,
    )
    adapter_dir = train_source_lora(train_cfg, output_dir)
    console.print(f"  → {adapter_dir}")

    console.print("\n[bold]Baseline 2/2:[/] Evaluating direct-LoRA baseline…")
    eval_cfg = EvalConfig(
        model_name=target_model,
        task_name=f"{task_name}__baseline",
        dataset_name=dataset_name,
        dataset_split=eval_split,
        max_samples=max_samples,
        batch_size=batch_size,
        max_seq_length=max_seq_length,
    )
    eval_dir = evaluate_adapter(adapter_dir=adapter_dir, config=eval_cfg, output_dir=output_dir)
    console.print(f"  → {eval_dir}")

    return {"adapter_dir": adapter_dir, "eval_dir": eval_dir}
