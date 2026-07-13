#!/usr/bin/env python3
"""Reference end-to-end PorTAL run (train -> extract -> convert -> eval).

This mirrors ``portal port`` but exposes the smoke-sizing knobs the CLI does not
yet surface (sample counts, sequence length, per-stage epochs). Use it as a
template for driving the pipeline programmatically, e.g. from a coding agent.

Intended to run *inside* a CUDA-enabled smolvm guest (see README.md). Install
portal in the VM first, then run this script. Example wrapper:

    smolvm machine run --net --cuda --mem 16384 \
      -e HF_HOME=/tmp/hf \
      -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False \
      --image ./portal-cuda.tar -- \
      sh -c 'pip install -q "portal @ ${PORTAL_ZIP}" \
        typer rich pydantic safetensors "datasets>=3.0,<4" accelerate \
        "transformers>=4.45,<4.52" "peft>=0.14,<0.18" && \
      python3 examples/smolvm/port_e2e.py --source Qwen/Qwen3-0.6B \
        --target TinyLlama/TinyLlama-1.1B-Chat-v1.0 --dataset stanfordnlp/imdb'

Notes:
- Gated models (e.g. google/gemma-3-1b-it) need an accepted license and
  ``-e HF_TOKEN=...`` on the ``machine run`` line.
- Fused SDPA (smolvm >= 1.5.2): add ``-e PORTAL_SKIP_CUDA_SMOLVM=1``. Default is
  math SDPA, which is safe on all supported smolvm versions.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="Qwen/Qwen3-0.6B", help="Source model HF id.")
    parser.add_argument(
        "--target",
        default="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        help="Target model HF id (ungated by default).",
    )
    parser.add_argument("--task", default="imdb-port", help="Task name (used for artifact dirs).")
    parser.add_argument("--dataset", default="stanfordnlp/imdb", help="HuggingFace dataset id.")
    parser.add_argument("--output-dir", type=Path, default=Path("/tmp/port-artifacts"))
    parser.add_argument("--max-samples", type=int, default=64, help="Train/eval rows.")
    parser.add_argument("--max-seq-length", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--rank", type=int, default=8, help="LoRA rank.")
    parser.add_argument("--train-epochs", type=int, default=1)
    parser.add_argument("--extract-epochs", type=int, default=10)
    parser.add_argument("--convert-epochs", type=int, default=20)
    parser.add_argument("--cal-samples", type=int, default=64, help="Converter calibration rows.")
    parser.add_argument("--latent-dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=128)
    args = parser.parse_args()

    # Imported here so ``--help`` works without torch/transformers installed.
    import portal.converter as pconv
    import portal.eval as pe
    import portal.hypernetwork as ph
    import portal.train as pt
    from portal.artifacts import load_adapter_path, load_eval_results
    from portal.config import (
        ConverterConfig,
        EvalConfig,
        HypernetConfig,
        LoraConfig,
        TrainConfig,
    )

    out = args.output_dir

    print("Step 1/4: train source LoRA")
    adapter_artifact = pt.train_source_lora(
        TrainConfig(
            source_model=args.source,
            task_name=args.task,
            dataset_name=args.dataset,
            max_samples=args.max_samples,
            num_epochs=args.train_epochs,
            batch_size=args.batch_size,
            max_seq_length=args.max_seq_length,
            lora=LoraConfig(rank=args.rank, alpha=args.rank * 2),
        ),
        out,
    )
    print("  ->", adapter_artifact)

    print("Step 2/4: extract task latent")
    latent_dir = ph.extract_task_latent(
        load_adapter_path(adapter_artifact),
        args.source,
        args.task,
        HypernetConfig(
            num_epochs=args.extract_epochs,
            latent_dim=args.latent_dim,
            hidden_dim=args.hidden_dim,
            num_layers=2,
        ),
        out,
    )
    print("  ->", latent_dir)

    print("Step 3/4: convert latent -> target adapter (loss should decrease)")
    target_adapter = pconv.convert_latent_to_adapter(
        latent_dir,
        args.task,
        ConverterConfig(
            target_model=args.target,
            calibration_dataset=args.dataset,
            calibration_samples=args.cal_samples,
            num_epochs=args.convert_epochs,
            hidden_dim=args.hidden_dim,
        ),
        out,
        lora_rank=args.rank,
    )
    print("  ->", target_adapter)

    print("Step 4/4: evaluate target adapter")
    eval_dir = pe.evaluate_adapter(
        target_adapter,
        EvalConfig(
            model_name=args.target,
            task_name=args.task,
            dataset_name=args.dataset,
            dataset_split="test",
            max_samples=args.max_samples,
            batch_size=args.batch_size,
            max_seq_length=args.max_seq_length,
        ),
        out,
    )
    print("  ->", eval_dir)

    metrics = load_eval_results(eval_dir).get("metrics", {})
    print("eval metrics:", json.dumps(metrics))
    print("port e2e ok")


if __name__ == "__main__":
    main()
