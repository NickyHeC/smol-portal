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
from dataclasses import dataclass
from pathlib import Path


DEFAULT_DATASET = "RampPublic/portallib-tasks"
DEFAULT_DATASET_REVISION = "d35f1e8a813cfae662166164fc25965a31b01ae0"
DEFAULT_SOURCE = "RampPublic/portal-qwen3-1.7b"
DEFAULT_SOURCE_REVISION = "v0.1.0"
# Small Qwen3 target so A10 can refit without H200-class VRAM.
DEFAULT_TARGET_ID = "Qwen/Qwen3-0.6B"
DEFAULT_TARGET_REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"


@dataclass(frozen=True)
class BaseRecipe:
    model_id: str
    revision: str
    layer_path: str = "model.layers"


def _apply_hosting_knobs(*, force_fp32: bool, math_sdpa: bool) -> None:
    import torch

    if math_sdpa and hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "enable_flash_sdp"):
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
        print("sdpa: math only", flush=True)
    os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
    if force_fp32:
        print("dtype: fp32 (hosting-safe)", flush=True)


def _load_base(recipe: BaseRecipe, *, device, dtype, hosting_safe: bool):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from portallib import PortalBase

    tokenizer = AutoTokenizer.from_pretrained(recipe.model_id, revision=recipe.revision)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    kwargs: dict = {"revision": recipe.revision, "torch_dtype": dtype}
    if hosting_safe and device.type == "cuda":
        kwargs["device_map"] = "cuda"
        model = AutoModelForCausalLM.from_pretrained(recipe.model_id, **kwargs)
    else:
        model = AutoModelForCausalLM.from_pretrained(recipe.model_id, **kwargs).to(device)
    return PortalBase(
        model_id=recipe.model_id,
        model=model,
        tokenizer=tokenizer,
        revision=recipe.revision,
        layer_path=recipe.layer_path,
    )


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
    parser.add_argument("--output-dir", type=Path, default=Path("/tmp/portallib-refit-smoke"))
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
            ChoiceDataset,
            PortalAdapterRefitter,
            PortalModel,
            PortalTrainingConfig,
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

    dataset = ChoiceDataset.from_hub(args.dataset, revision=args.dataset_revision)
    if args.dry_run:
        print("dry-run ok", flush=True)
        sys.exit(0)

    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"cuda_available={torch.cuda.is_available()} device={device}", flush=True)
    if args.hosting_safe:
        _apply_hosting_knobs(force_fp32=True, math_sdpa=device.type == "cuda")
        dtype = torch.float32
    else:
        dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    source = PortalModel.from_pretrained(args.source, revision=args.source_revision)
    target = _load_base(
        BaseRecipe(args.target_id, args.target_revision),
        device=device,
        dtype=dtype,
        hosting_safe=args.hosting_safe,
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
    result = PortalAdapterRefitter(source, target, dataset, config=config).refit(on_epoch=on_epoch)

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
