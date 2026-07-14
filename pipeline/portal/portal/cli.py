"""PorTAL CLI — typer entry point.

Commands:
    portal train   — Train a LoRA adapter on a source model for a given task.
    portal extract — Extract a base-agnostic task latent from a trained LoRA.
    portal convert — Project a task latent into a target model's LoRA adapter.
    portal eval    — Evaluate an adapter on a benchmark.
    portal baseline— Train + eval a direct LoRA on the target (comparison point).
    portal port    — End-to-end: train → extract → convert → eval.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from portal import __version__
from portal.config import DEFAULT_OUTPUT_DIR, LatentMode

app = typer.Typer(
    name="portal",
    help="PorTAL: learn a task once, port to any base model.",
    no_args_is_help=True,
    rich_markup_mode="rich",
)
console = Console()


def _version_callback(value: bool) -> None:
    if value:
        console.print(f"portal {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: Annotated[
        bool | None,
        typer.Option("--version", "-V", callback=_version_callback, is_eager=True),
    ] = None,
) -> None:
    """PorTAL: Portable Task Adapters."""


# ---------------------------------------------------------------------------
# portal train
# ---------------------------------------------------------------------------


@app.command()
def train(
    model: Annotated[str, typer.Option("--model", "-m", help="Source model name or HF id.")],
    task: Annotated[str, typer.Option("--task", "-t", help="Task name (used for artifact dirs).")],
    dataset: Annotated[str, typer.Option("--dataset", "-d", help="HuggingFace dataset id.")],
    output_dir: Annotated[Path, typer.Option("--output-dir", "-o")] = DEFAULT_OUTPUT_DIR,
    rank: Annotated[int, typer.Option(help="LoRA rank.")] = 16,
    epochs: Annotated[int, typer.Option(help="Training epochs.")] = 3,
    lr: Annotated[float, typer.Option(help="Learning rate.")] = 2e-4,
    batch_size: Annotated[int, typer.Option(help="Batch size.")] = 4,
    max_seq_length: Annotated[int, typer.Option(help="Max sequence length.")] = 512,
    max_samples: Annotated[int | None, typer.Option(help="Limit dataset rows.")] = None,
    seed: Annotated[int, typer.Option(help="Random seed.")] = 42,
) -> None:
    """Train a LoRA adapter on a source model for a given task."""
    from portal.config import LoraConfig, TrainConfig
    from portal.train import train_source_lora

    cfg = TrainConfig(
        source_model=model,
        task_name=task,
        dataset_name=dataset,
        lora=LoraConfig(rank=rank),
        learning_rate=lr,
        num_epochs=epochs,
        batch_size=batch_size,
        max_seq_length=max_seq_length,
        max_samples=max_samples,
        seed=seed,
    )
    console.print(f"[bold]Training LoRA on [cyan]{model}[/] for task [green]{task}[/]…")
    result_dir = train_source_lora(cfg, output_dir)
    console.print(f"[bold green]✓[/] Adapter saved to {result_dir}")


# ---------------------------------------------------------------------------
# portal extract
# ---------------------------------------------------------------------------


@app.command()
def extract(
    adapter_dir: Annotated[
        Path, typer.Option("--adapter-dir", "-a", help="Path to trained LoRA adapter.")
    ],
    model: Annotated[
        str, typer.Option("--model", "-m", help="Source model the adapter was trained on.")
    ],
    task: Annotated[str, typer.Option("--task", "-t", help="Task name.")],
    output_dir: Annotated[Path, typer.Option("--output-dir", "-o")] = DEFAULT_OUTPUT_DIR,
    latent_dim: Annotated[int, typer.Option(help="Task latent dimensionality.")] = 256,
    epochs: Annotated[int, typer.Option(help="Hypernetwork training epochs.")] = 50,
    seed: Annotated[int, typer.Option(help="Random seed.")] = 42,
) -> None:
    """Extract a base-agnostic task latent from a trained LoRA adapter."""
    from portal.config import HypernetConfig
    from portal.hypernetwork import extract_task_latent

    cfg = HypernetConfig(latent_dim=latent_dim, num_epochs=epochs, seed=seed)
    console.print(f"[bold]Extracting task latent from [cyan]{adapter_dir}[/]…")
    result_dir = extract_task_latent(
        adapter_dir=adapter_dir,
        source_model=model,
        task_name=task,
        config=cfg,
        output_dir=output_dir,
    )
    console.print(f"[bold green]✓[/] Task latent saved to {result_dir}")


# ---------------------------------------------------------------------------
# portal convert
# ---------------------------------------------------------------------------


@app.command()
def convert(
    latent_dir: Annotated[
        Path, typer.Option("--latent-dir", "-l", help="Path to task latent artifact.")
    ],
    target: Annotated[str, typer.Option("--target", "-t", help="Target model name or HF id.")],
    task: Annotated[str, typer.Option("--task", help="Task name.")],
    calibration_dataset: Annotated[
        str,
        typer.Option("--cal-dataset", help="Calibration dataset HF id (the task dataset)."),
    ],
    output_dir: Annotated[Path, typer.Option("--output-dir", "-o")] = DEFAULT_OUTPUT_DIR,
    calibration_samples: Annotated[int, typer.Option("--cal-samples")] = 256,
    epochs: Annotated[int, typer.Option(help="Converter training epochs.")] = 30,
    latent_mode: Annotated[
        LatentMode,
        typer.Option("--latent-mode", help="Latent ablation mode (real|zero|random|shuffled)."),
    ] = LatentMode.REAL,
    seed: Annotated[int, typer.Option(help="Random seed.")] = 42,
) -> None:
    """Project a task latent into a target model's LoRA adapter."""
    from portal.config import ConverterConfig
    from portal.converter import convert_latent_to_adapter

    cfg = ConverterConfig(
        target_model=target,
        calibration_dataset=calibration_dataset,
        calibration_samples=calibration_samples,
        num_epochs=epochs,
        latent_mode=latent_mode,
        seed=seed,
    )
    console.print(f"[bold]Converting task latent → [cyan]{target}[/] adapter…")
    result_dir = convert_latent_to_adapter(
        latent_dir=latent_dir,
        task_name=task,
        config=cfg,
        output_dir=output_dir,
    )
    console.print(f"[bold green]✓[/] Target adapter saved to {result_dir}")


# ---------------------------------------------------------------------------
# portal eval
# ---------------------------------------------------------------------------


@app.command(name="eval")
def eval_cmd(
    adapter_dir: Annotated[
        Path, typer.Option("--adapter-dir", "-a", help="Path to LoRA adapter to evaluate.")
    ],
    model: Annotated[
        str, typer.Option("--model", "-m", help="Base model the adapter was made for.")
    ],
    task: Annotated[str, typer.Option("--task", "-t", help="Task name.")],
    dataset: Annotated[str, typer.Option("--dataset", "-d", help="HuggingFace dataset id.")],
    output_dir: Annotated[Path, typer.Option("--output-dir", "-o")] = DEFAULT_OUTPUT_DIR,
    split: Annotated[str, typer.Option(help="Dataset split for eval.")] = "test",
    batch_size: Annotated[int, typer.Option(help="Batch size.")] = 8,
    max_samples: Annotated[int | None, typer.Option(help="Limit eval rows.")] = None,
    seed: Annotated[int, typer.Option(help="Random seed.")] = 42,
) -> None:
    """Evaluate an adapter on a benchmark."""
    from portal.config import EvalConfig
    from portal.eval import evaluate_adapter

    cfg = EvalConfig(
        model_name=model,
        task_name=task,
        dataset_name=dataset,
        dataset_split=split,
        max_samples=max_samples,
        batch_size=batch_size,
    )
    console.print(f"[bold]Evaluating [cyan]{model}[/] + adapter on [green]{task}[/]…")
    result_dir = evaluate_adapter(
        adapter_dir=adapter_dir,
        config=cfg,
        output_dir=output_dir,
    )
    console.print(f"[bold green]✓[/] Eval results saved to {result_dir}")


# ---------------------------------------------------------------------------
# portal baseline  (direct target LoRA — the comparison point for `port`)
# ---------------------------------------------------------------------------


@app.command()
def baseline(
    model: Annotated[
        str, typer.Option("--model", "-m", help="Target model to train a direct LoRA on.")
    ],
    task: Annotated[str, typer.Option("--task", "-t", help="Task name.")],
    dataset: Annotated[str, typer.Option("--dataset", "-d", help="HuggingFace dataset id.")],
    output_dir: Annotated[Path, typer.Option("--output-dir", "-o")] = DEFAULT_OUTPUT_DIR,
    rank: Annotated[int, typer.Option(help="LoRA rank.")] = 16,
    epochs: Annotated[int, typer.Option(help="Training epochs.")] = 3,
    batch_size: Annotated[int, typer.Option(help="Batch size.")] = 4,
    max_seq_length: Annotated[int, typer.Option(help="Max sequence length.")] = 512,
    max_samples: Annotated[int | None, typer.Option(help="Limit dataset rows.")] = None,
    split: Annotated[str, typer.Option(help="Dataset split for eval.")] = "test",
    seed: Annotated[int, typer.Option(help="Random seed.")] = 42,
) -> None:
    """Train + evaluate a direct LoRA on the target model (the baseline for `port`)."""
    from portal.baseline import run_direct_lora_baseline

    console.print(f"[bold]Direct-LoRA baseline on [cyan]{model}[/] for task [green]{task}[/]…")
    results = run_direct_lora_baseline(
        target_model=model,
        task_name=task,
        dataset_name=dataset,
        output_dir=output_dir,
        rank=rank,
        num_epochs=epochs,
        batch_size=batch_size,
        max_seq_length=max_seq_length,
        max_samples=max_samples,
        eval_split=split,
        seed=seed,
    )
    console.print("\n[bold green]✓ Baseline complete.[/]")
    console.print(f"  Adapter      : {results['adapter_dir']}")
    console.print(f"  Eval results : {results['eval_dir']}")


# ---------------------------------------------------------------------------
# portal port  (end-to-end)
# ---------------------------------------------------------------------------


@app.command()
def port(
    source: Annotated[str, typer.Option("--from", help="Source model name or HF id.")],
    target: Annotated[str, typer.Option("--to", help="Target model name or HF id.")],
    task: Annotated[str, typer.Option("--task", "-t", help="Task name.")],
    dataset: Annotated[str, typer.Option("--dataset", "-d", help="HuggingFace dataset id.")],
    output_dir: Annotated[Path, typer.Option("--output-dir", "-o")] = DEFAULT_OUTPUT_DIR,
    skip_train: Annotated[
        bool, typer.Option("--skip-train", help="Skip source LoRA training (reuse existing).")
    ] = False,
    source_adapter_dir: Annotated[
        Path | None,
        typer.Option("--source-adapter-dir", help="Existing source adapter to reuse."),
    ] = None,
    seed: Annotated[int, typer.Option(help="Random seed.")] = 42,
) -> None:
    """End-to-end: train source LoRA → extract task latent → convert to target → eval."""
    from portal.config import PortConfig
    from portal.port import run_port_pipeline

    cfg = PortConfig(
        source_model=source,
        target_model=target,
        task_name=task,
        dataset_name=dataset,
        output_dir=output_dir,
        skip_train=skip_train,
    )
    console.print(f"[bold]Porting [cyan]{source}[/] → [cyan]{target}[/] on task [green]{task}[/]…")
    results = run_port_pipeline(cfg, source_adapter_dir=source_adapter_dir)
    console.print("\n[bold green]✓ Pipeline complete.[/]")
    console.print(f"  Source adapter : {results['source_adapter_dir']}")
    console.print(f"  Task latent    : {results['latent_dir']}")
    console.print(f"  Target adapter : {results['target_adapter_dir']}")
    console.print(f"  Eval results   : {results['eval_dir']}")
