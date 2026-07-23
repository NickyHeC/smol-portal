#!/usr/bin/env python3
"""T2 smoke: tiny ``acc_norm`` eval of a published portallib artifact.

Designed for smolvm CUDA remoting (and bare-metal twin runs). Uses the public
v0.2 runtime API and native artifact format.

Host wrapper (after building ``portallib-cuda.tar``):

    smolvm machine run --net --cuda --mem 16384 \\
      -e HF_HOME=/tmp/hf \\
      -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False \\
      --image ./portallib-cuda.tar -- \\
      python3 /workspace/smol-portal/examples/smolvm/smoke_portallib.py \\
        --task rte --max-examples 8

Default artifact is the smaller Qwen3-1.7B published source (fits A10 better than
4B/8B for first green). Override with ``--artifact`` / ``--base-id``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


DEFAULT_DATASET = "RampPublic/portallib-tasks"
DEFAULT_DATASET_REVISION = "ffc3c0e44f529bf64a5ae62ed5db090952db97ea"
DEFAULT_ARTIFACT = "RampPublic/portal-qwen3-1.7b"
DEFAULT_ARTIFACT_REVISION = "v0.2.0"
DEFAULT_BASE_ID = "Qwen/Qwen3-1.7B"
DEFAULT_BASE_REVISION = "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
DEFAULT_TASK = "rte"


def _apply_hosting_knobs(*, force_fp32: bool, math_sdpa: bool) -> None:
    """Best-effort knobs for constrained CUDA remoting (smolvm)."""
    import torch

    if (
        math_sdpa
        and hasattr(torch.backends, "cuda")
        and hasattr(torch.backends.cuda, "enable_flash_sdp")
    ):
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
        print("sdpa: math only (flash/mem_efficient off)", flush=True)

    # Inductor / compile is hostile in slim guests; keep off for smoke.
    os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
    if force_fp32:
        print("dtype: fp32 (hosting-safe)", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--dataset-revision", default=DEFAULT_DATASET_REVISION)
    parser.add_argument("--artifact", default=DEFAULT_ARTIFACT)
    parser.add_argument("--artifact-revision", default=DEFAULT_ARTIFACT_REVISION)
    parser.add_argument("--base-id", default=DEFAULT_BASE_ID)
    parser.add_argument("--base-revision", default=DEFAULT_BASE_REVISION)
    parser.add_argument(
        "--task", default=DEFAULT_TASK, help="Focus task for lift printout."
    )
    parser.add_argument(
        "--tasks",
        default=None,
        help="Comma-separated task subset to evaluate. "
        "Default: all artifact tasks. Use 'focus' to evaluate only --task.",
    )
    parser.add_argument("--max-examples", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=1)
    parser.add_argument("--max-prompt", type=int, default=512)
    parser.add_argument("--output-dir", type=Path, default=Path("/tmp/portallib-smoke"))
    parser.add_argument(
        "--hosting-safe",
        action=argparse.BooleanOptionalAction,
        default=os.environ.get("PORTALLIB_HOST") == "smolvm",
        help="fp32 + device_map=cuda (default on when PORTALLIB_HOST=smolvm).",
    )
    parser.add_argument(
        "--math-sdpa",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Force math-only SDPA (default: on with --hosting-safe, else off/fused).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Import portallib + load dataset slice only; skip model download/eval.",
    )
    args = parser.parse_args()
    math_sdpa = args.hosting_safe if args.math_sdpa is None else args.math_sdpa

    try:
        import portallib
        from portallib import (
            BaseModelSpec,
            PortalEvaluator,
            PortalModel,
            load_base,
            load_dataset,
            runtime_device,
        )
    except ImportError as exc:
        raise SystemExit(
            "portallib is not importable. Rebuild Dockerfile.portallib-cuda with "
            "PORTALLIB_SPEC='portallib[training]==0.2.0'."
        ) from exc

    print(
        json.dumps(
            {
                "portallib": getattr(portallib, "__version__", None),
                "artifact": args.artifact,
                "artifact_revision": args.artifact_revision,
                "base": args.base_id,
                "task": args.task,
                "tasks_arg": args.tasks,
                "max_examples": args.max_examples,
                "hosting_safe": args.hosting_safe,
                "math_sdpa": math_sdpa,
                "dataset_revision": args.dataset_revision,
                "host": os.environ.get("PORTALLIB_HOST", "unknown"),
            }
        ),
        flush=True,
    )

    dataset = load_dataset(args.dataset, revision=args.dataset_revision)
    if args.task not in dataset.tasks:
        raise SystemExit(
            f"task={args.task!r} not in dataset tasks={list(dataset.tasks)}"
        )
    print(f"dataset tasks={len(dataset.tasks)}; focus={args.task!r}", flush=True)

    if args.dry_run:
        print("dry-run ok (import + dataset only)", flush=True)
        sys.exit(0)

    import torch

    device, dtype = runtime_device("auto", "float32" if args.hosting_safe else "auto")
    print(f"cuda_available={torch.cuda.is_available()} device={device}", flush=True)
    if args.hosting_safe:
        _apply_hosting_knobs(
            force_fp32=True, math_sdpa=math_sdpa and device.type == "cuda"
        )
    else:
        if math_sdpa and device.type == "cuda":
            _apply_hosting_knobs(force_fp32=False, math_sdpa=True)
        print(f"dtype: {dtype}", flush=True)
    if device.type == "cuda" and not math_sdpa:
        print("sdpa: fused/sdpa allowed (no math-only force)", flush=True)

    portal = PortalModel.from_pretrained(args.artifact, revision=args.artifact_revision)
    if portal.config.base_model_name_or_path != args.base_id:
        raise SystemExit(
            f"artifact expects base {portal.config.base_model_name_or_path!r}, "
            f"got --base-id {args.base_id!r}"
        )
    if args.task not in portal.config.tasks:
        raise SystemExit(
            f"task={args.task!r} not in artifact tasks={list(portal.config.tasks)}"
        )

    recipe = BaseModelSpec(
        args.base_id,
        args.base_revision,
        dtype=dtype,
        device_map="cuda" if device.type == "cuda" and args.hosting_safe else None,
    )
    print(f"loading base {recipe.model_id}@{recipe.revision} …", flush=True)
    base = load_base(
        recipe,
        device=device,
        dtype=dtype,
    )

    evaluator = PortalEvaluator(
        max_prompt=args.max_prompt, batch_size=args.eval_batch_size
    )
    if args.tasks is None:
        tasks = tuple(portal.config.tasks)
    elif args.tasks.strip().lower() == "focus":
        tasks = (args.task,)
    else:
        tasks = tuple(t.strip() for t in args.tasks.split(",") if t.strip())
        missing = [t for t in tasks if t not in portal.config.tasks]
        if missing:
            raise SystemExit(
                f"tasks not in artifact: {missing}; have {list(portal.config.tasks)}"
            )
    print(
        f"evaluating base floor over {len(tasks)} tasks {tasks!r} (max_examples={args.max_examples}) …",
        flush=True,
    )
    base_result = evaluator.evaluate(
        base, dataset, tasks=tasks, max_examples=args.max_examples
    )
    print("evaluating portal-adapted …", flush=True)
    portal_result = evaluator.evaluate(
        base,
        dataset,
        tasks=tasks,
        portal=portal,
        max_examples=args.max_examples,
    )

    focus = args.task
    report = {
        "artifact": args.artifact,
        "artifact_revision": args.artifact_revision,
        "base_id": args.base_id,
        "focus_task": focus,
        "tasks": list(tasks),
        "max_examples": args.max_examples,
        "hosting_safe": args.hosting_safe,
        "math_sdpa": math_sdpa,
        "device": str(device),
        "dtype": str(dtype),
        "base": base_result.to_dict(),
        "portal": portal_result.to_dict(),
        "macro_accuracy_lift": portal_result.macro_accuracy
        - base_result.macro_accuracy,
        "focus_accuracy_lift": (
            portal_result.tasks[focus].accuracy - base_result.tasks[focus].accuracy
            if focus in portal_result.tasks and focus in base_result.tasks
            else None
        ),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    out = args.output_dir / "smoke_eval.json"
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    print(f"wrote {out}", flush=True)

    # Finite metrics = plumbing PASS (accuracy not judged at smoke size).
    if not (
        0.0 <= base_result.macro_accuracy <= 1.0
        and 0.0 <= portal_result.macro_accuracy <= 1.0
    ):
        raise SystemExit("acc_norm out of [0,1] — eval plumbing broken")
    if portal_result.macro_gold_nll != portal_result.macro_gold_nll:  # NaN
        raise SystemExit("portal gold_nll is NaN")
    print("portallib smoke ok", flush=True)


if __name__ == "__main__":
    main()
