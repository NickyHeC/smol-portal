#!/usr/bin/env python3
"""T5c: dual-source ``PortalCoreTrainer`` under smolvm (or bare metal).

Loads two raw HF bases in one process, jointly trains shared task latents +
canonical core + one alignment per base. Defaults match portallib
``examples/train_example.py`` (paper recipe); CLI knobs shrink for smoke.

Writes epoch JSON to stdout and ``train_live.jsonl`` (line-buffered) so host
``-v`` mounts can still miss bulk artifacts but progress survives in the
smolvm console stream.
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
DEFAULT_BASES = (
    ("Qwen/Qwen3-1.7B", "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"),
    ("Qwen/Qwen3-4B", "1cfa9a7208912126459214e8b04321603b3df60c"),
)


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
    if force_fp32:
        print("dtype: fp32 (hosting-safe)", flush=True)


def _always_safe_env() -> None:
    # HF+compile is unsupported through remoting; set even for bf16 recipes.
    os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")


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


def _model_slug(model_id: str) -> str:
    return model_id.split("/")[-1].lower().replace(".", "-")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--dataset-revision", default=DEFAULT_DATASET_REVISION)
    parser.add_argument(
        "--base",
        action="append",
        metavar="ID@REV",
        help="Source base as model_id@revision (repeat). Default: paper Qwen3-1.7B+4B.",
    )
    parser.add_argument("--source-max-examples", type=int, default=2000)
    parser.add_argument("--source-steps-per-epoch", type=int, default=500)
    parser.add_argument("--eval-max-examples", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--output-dir", type=Path, default=Path("/tmp/portallib-train"))
    parser.add_argument(
        "--hosting-safe",
        action=argparse.BooleanOptionalAction,
        default=os.environ.get("PORTALLIB_HOST") == "smolvm",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    try:
        import portallib
        from portallib import ChoiceDataset, PortalCoreTrainer, PortalTrainingConfig
    except ImportError as exc:
        raise SystemExit("portallib not importable") from exc

    base_specs = args.base or [f"{mid}@{rev}" for mid, rev in DEFAULT_BASES]
    recipes: list[BaseRecipe] = []
    for spec in base_specs:
        if "@" not in spec:
            raise SystemExit(f"base must be model_id@revision, got {spec!r}")
        model_id, revision = spec.rsplit("@", 1)
        recipes.append(BaseRecipe(model_id, revision))

    banner = {
        "portallib": getattr(portallib, "__version__", None),
        "bases": [r.model_id for r in recipes],
        "source_max_examples": args.source_max_examples,
        "source_steps_per_epoch": args.source_steps_per_epoch,
        "eval_max_examples": args.eval_max_examples,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "hosting_safe": args.hosting_safe,
        "host": os.environ.get("PORTALLIB_HOST", "unknown"),
    }
    print(json.dumps(banner), flush=True)

    dataset = ChoiceDataset.from_hub(args.dataset, revision=args.dataset_revision)
    if args.dry_run:
        print("dry-run ok", flush=True)
        sys.exit(0)

    import torch

    _always_safe_env()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"cuda_available={torch.cuda.is_available()} device={device}", flush=True)
    if args.hosting_safe:
        _apply_hosting_knobs(force_fp32=True, math_sdpa=device.type == "cuda")
        dtype = torch.float32
    else:
        dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    bases = [
        _load_base(recipe, device=device, dtype=dtype, hosting_safe=args.hosting_safe) for recipe in recipes
    ]
    if device.type == "cuda":
        alloc = torch.cuda.memory_allocated() / (1024**3)
        reserved = torch.cuda.memory_reserved() / (1024**3)
        print(json.dumps({"phase": "loaded", "alloc_gib": alloc, "reserved_gib": reserved}), flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    live_path = args.output_dir / "train_live.jsonl"
    live = live_path.open("a", buffering=1)

    config = PortalTrainingConfig(
        modules=("q", "v"),
        rank=8,
        alpha=16,
        d_z=256,
        d_layer=32,
        hidden=512,
        d_core=1024,
        source_max_examples=args.source_max_examples,
        source_resample_each_epoch=False,
        source_steps_per_epoch=args.source_steps_per_epoch,
        eval_max_examples=args.eval_max_examples,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=1e-3,
        latent_learning_rate=2e-3,
        lr_scheduler="linear",
        warmup_ratio=0.1,
        seed=0,
        checkpoint_dir=args.output_dir / "checkpoints",
    )

    def on_epoch(epoch) -> None:
        row = {
            "phase": "source",
            "epoch": epoch.epoch,
            "acc_norm": epoch.macro_accuracy,
            "gold_nll": epoch.macro_gold_nll,
            "bases": {
                name: {"acc_norm": result.macro_accuracy, "gold_nll": result.macro_gold_nll}
                for name, result in epoch.evaluations.items()
            },
        }
        line = json.dumps(row, default=str)
        print(line, flush=True)
        live.write(line + "\n")
        live.flush()
        try:
            os.fsync(live.fileno())
        except OSError:
            pass

    print("training …", flush=True)
    result = PortalCoreTrainer(bases, dataset, config=config).train(on_epoch=on_epoch)

    outputs: dict[str, str] = {}
    for recipe in recipes:
        destination = args.output_dir / f"source-{_model_slug(recipe.model_id)}"
        result.artifacts[recipe.model_id].save_pretrained(destination)
        outputs[recipe.model_id] = str(destination)

    report = {
        "bases": [r.model_id for r in recipes],
        "epochs_cfg": args.epochs,
        "source_steps_per_epoch": args.source_steps_per_epoch,
        "source_max_examples": args.source_max_examples,
        "eval_max_examples": args.eval_max_examples,
        "batch_size": args.batch_size,
        "hosting_safe": args.hosting_safe,
        "device": str(device),
        "dtype": str(dtype),
        "best_epoch": result.best_epoch,
        "best_loss_epoch": result.best_loss_epoch,
        "outputs": outputs,
        "diagnostics": result.diagnostics,
    }
    out = args.output_dir / "train_smoke.json"
    out.write_text(json.dumps(report, indent=2, default=str) + "\n")
    print(json.dumps(report, indent=2, default=str), flush=True)
    print(f"wrote {out}", flush=True)
    print("portallib train smoke ok", flush=True)
    live.close()


if __name__ == "__main__":
    main()
