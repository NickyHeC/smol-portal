#!/usr/bin/env python3
"""T4 smoke: tiny ``PortalAdapterRefitter`` under smolvm (or bare metal).

Loads a published source artifact (shared latents + core), freezes them, and
trains a fresh alignment on a small target base for 1 epoch / few examples.

Not accuracy-claiming — hosting plumbing for the train/refit path.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


DEFAULT_DATASET = "RampPublic/portallib-tasks"
DEFAULT_DATASET_REVISION = "ffc3c0e44f529bf64a5ae62ed5db090952db97ea"
DEFAULT_SOURCE = "RampPublic/portal-qwen3-1.7b"
DEFAULT_SOURCE_REVISION = "v0.2.0"
# Small Qwen3 target so A10 can refit without H200-class VRAM.
DEFAULT_TARGET_ID = "Qwen/Qwen3-0.6B"
DEFAULT_TARGET_REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"


def _apply_hosting_knobs(*, force_fp32: bool, math_sdpa: bool) -> None:
    import torch

    if (
        math_sdpa
        and hasattr(torch.backends, "cuda")
        and hasattr(torch.backends.cuda, "enable_flash_sdp")
    ):
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
        print("sdpa: math only", flush=True)
    os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
    if force_fp32:
        print("dtype: fp32 (hosting-safe)", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--dataset-revision", default=DEFAULT_DATASET_REVISION)
    parser.add_argument("--source", default=DEFAULT_SOURCE)
    parser.add_argument("--source-revision", default=DEFAULT_SOURCE_REVISION)
    parser.add_argument("--target-id", default=DEFAULT_TARGET_ID)
    parser.add_argument("--target-revision", default=DEFAULT_TARGET_REVISION)
    parser.add_argument("--refit-max-examples", type=int, default=8)
    parser.add_argument("--eval-max-examples", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("/tmp/portallib-refit-smoke")
    )
    parser.add_argument(
        "--hosting-safe",
        action=argparse.BooleanOptionalAction,
        default=os.environ.get("PORTALLIB_HOST") == "smolvm",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    try:
        import portallib
        from portallib import (
            BaseModelSpec,
            PortalAdapterRefitter,
            PortalModel,
            PortalTrainingConfig,
            load_base,
            load_dataset,
            runtime_device,
        )
    except ImportError as exc:
        raise SystemExit("portallib not importable") from exc

    print(
        json.dumps(
            {
                "portallib": getattr(portallib, "__version__", None),
                "source": args.source,
                "target": args.target_id,
                "refit_max_examples": args.refit_max_examples,
                "epochs": args.epochs,
                "hosting_safe": args.hosting_safe,
                "host": os.environ.get("PORTALLIB_HOST", "unknown"),
            }
        ),
        flush=True,
    )

    dataset = load_dataset(args.dataset, revision=args.dataset_revision)
    if args.dry_run:
        print("dry-run ok", flush=True)
        sys.exit(0)

    import torch

    device, dtype = runtime_device("auto", "float32" if args.hosting_safe else "auto")
    print(f"cuda_available={torch.cuda.is_available()} device={device}", flush=True)
    if args.hosting_safe:
        _apply_hosting_knobs(force_fp32=True, math_sdpa=device.type == "cuda")

    source = PortalModel.from_pretrained(args.source, revision=args.source_revision)
    target = load_base(
        BaseModelSpec(
            args.target_id,
            args.target_revision,
            dtype=dtype,
            device_map="cuda" if args.hosting_safe and device.type == "cuda" else None,
        ),
        device=device,
        dtype=dtype,
    )

    config = PortalTrainingConfig.from_portal_config(
        source.config,
        refit_max_examples=args.refit_max_examples,
        eval_max_examples=args.eval_max_examples,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=1e-3,
        lr_scheduler="linear",
        warmup_ratio=0.1,
        seed=0,
        checkpoint_dir=args.output_dir / "checkpoints",
    )

    def on_epoch(epoch) -> None:
        print(
            json.dumps(
                {
                    "phase": "refit",
                    "epoch": epoch.epoch,
                    "acc_norm": epoch.macro_accuracy,
                    "gold_nll": epoch.macro_gold_nll,
                }
            ),
            flush=True,
        )

    print("refitting …", flush=True)
    result = PortalAdapterRefitter(source, target, dataset, config=config).refit(
        on_epoch=on_epoch
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_artifact = args.output_dir / "refit-artifact"
    result.artifact.save_pretrained(out_artifact)

    history = [
        {
            "epoch": e.epoch,
            "acc_norm": e.macro_accuracy,
            "gold_nll": e.macro_gold_nll,
        }
        for e in result.history
    ]
    report = {
        "source": args.source,
        "target": args.target_id,
        "refit_max_examples": args.refit_max_examples,
        "epochs_cfg": args.epochs,
        "hosting_safe": args.hosting_safe,
        "device": str(device),
        "dtype": str(dtype),
        "artifact_dir": str(out_artifact),
        "best_epoch": result.best_epoch,
        "best_loss_epoch": result.best_loss_epoch,
        "history": history,
        "diagnostics": result.diagnostics,
    }
    out = args.output_dir / "refit_smoke.json"
    out.write_text(json.dumps(report, indent=2, default=str) + "\n")
    print(json.dumps(report, indent=2, default=str), flush=True)
    print(f"wrote {out}", flush=True)
    print("portallib refit smoke ok", flush=True)


if __name__ == "__main__":
    main()
