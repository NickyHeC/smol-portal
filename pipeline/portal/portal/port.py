"""End-to-end porting pipeline: train → extract → convert → eval."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from rich.console import Console

from portal.config import PortConfig

console = Console()


def run_port_pipeline(
    config: PortConfig,
    *,
    source_adapter_dir: Path | None = None,
) -> dict[str, Any]:
    """Run the full PorTAL pipeline.

    Steps:
        1. Train source LoRA (or reuse existing)
        2. Extract task latent via hypernetwork
        3. Convert task latent → target adapter via slim converter
        4. Evaluate target adapter

    Returns a dict of artifact paths and eval metrics.
    """
    from portal.converter import convert_latent_to_adapter
    from portal.eval import evaluate_adapter
    from portal.hypernetwork import extract_task_latent
    from portal.train import train_source_lora

    # --- Step 1: Train source LoRA ---
    if config.skip_train and source_adapter_dir is not None:
        console.print(f"  [dim]Skipping training, reusing {source_adapter_dir}[/]")
        adapter_dir = source_adapter_dir
    elif config.skip_train:
        raise ValueError("--skip-train requires --source-adapter-dir")
    else:
        console.print("\n[bold]Step 1/4:[/] Training source LoRA…")
        train_cfg = config.build_train_config()
        adapter_dir = train_source_lora(train_cfg, config.output_dir)
        console.print(f"  → {adapter_dir}")

    from portal.artifacts import load_adapter_path

    adapter_path = load_adapter_path(adapter_dir)

    # --- Step 2: Extract task latent ---
    console.print("\n[bold]Step 2/4:[/] Extracting task latent…")
    latent_dir = extract_task_latent(
        adapter_dir=adapter_path,
        source_model=config.source_model,
        task_name=config.task_name,
        config=config.hypernet,
        output_dir=config.output_dir,
    )
    console.print(f"  → {latent_dir}")

    # --- Step 3: Convert to target adapter ---
    console.print("\n[bold]Step 3/4:[/] Converting task latent → target adapter…")
    converter_cfg = config.build_converter_config()
    target_adapter_dir = convert_latent_to_adapter(
        latent_dir=latent_dir,
        task_name=config.task_name,
        config=converter_cfg,
        output_dir=config.output_dir,
    )
    console.print(f"  → {target_adapter_dir}")

    # --- Step 4: Evaluate target adapter ---
    console.print("\n[bold]Step 4/4:[/] Evaluating target adapter…")
    eval_cfg = config.build_eval_config(config.target_model)
    eval_dir = evaluate_adapter(
        adapter_dir=target_adapter_dir,
        config=eval_cfg,
        output_dir=config.output_dir,
    )
    console.print(f"  → {eval_dir}")

    return {
        "source_adapter_dir": adapter_dir,
        "latent_dir": latent_dir,
        "target_adapter_dir": target_adapter_dir,
        "eval_dir": eval_dir,
    }
