#!/usr/bin/env python3
"""Smoke harness for one Ramp ``portallib-tasks`` task inside smolvm.

Ready to run the moment ``portallib`` is installable (ramp-public/portallib issue #1).
Until then, ``--dry-run`` validates dataset access + import discovery without GPU.

Intended host wrapper (after building ``portallib-cuda.tar``):

    smolvm machine run --net --cuda --mem 16384 \\
      -e HF_HOME=/tmp/hf \\
      -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False \\
      --image ./portallib-cuda.tar -- \\
      python3 /path/to/smoke_portallib.py --task boolq --max-samples 8

What this script does today
---------------------------
1. Loads a single task slice from ``RampPublic/portallib-tasks``.
2. Discovers a callable entry point on the installed ``portallib`` package
   (CLI module or Python API — shapes TBD until their alpha merges).
3. Invokes a **minimal** train/eval path when discovery succeeds; otherwise
   prints what was found and exits non-zero so the connector can adapt.

What it deliberately does *not* do
----------------------------------
- Reimplement PorTAL. Engine = portallib.
- Claim accuracy. This is hosting-fidelity plumbing smoke only.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


DEFAULT_DATASET = "RampPublic/portallib-tasks"
# boolq is a standard multiple-choice row in the suite; override freely.
DEFAULT_TASK = "boolq"


def _load_task_rows(dataset_id: str, task: str, max_samples: int, split: str) -> list[dict[str, Any]]:
    from datasets import load_dataset

    ds = load_dataset(dataset_id, split=split)
    if "task" not in ds.column_names:
        raise SystemExit(
            f"{dataset_id!r} has no 'task' column; columns={ds.column_names}. "
            "Expected RampPublic/portallib-tasks schema."
        )
    filtered = ds.filter(lambda row: row["task"] == task)
    if len(filtered) == 0:
        # Help the operator pick a real task name.
        names = sorted(set(ds["task"]))
        raise SystemExit(
            f"No rows for task={task!r} in {dataset_id} ({split=}). "
            f"Available tasks ({len(names)}): {names[:20]}{'…' if len(names) > 20 else ''}"
        )
    n = min(max_samples, len(filtered))
    return [dict(filtered[i]) for i in range(n)]


def _discover_portallib() -> dict[str, Any]:
    """Probe installed portallib for CLI / API entry points once the alpha lands."""
    info: dict[str, Any] = {"importable": False, "version": None, "cli": None, "api_attrs": []}
    if importlib.util.find_spec("portallib") is None:
        return info
    info["importable"] = True
    mod = importlib.import_module("portallib")
    info["version"] = getattr(mod, "__version__", None)
    info["api_attrs"] = sorted(
        name
        for name in dir(mod)
        if not name.startswith("_") and callable(getattr(mod, name, None))
    )[:40]

    # Prefer a console script if present.
    for cmd in ("portallib", "portal"):
        try:
            proc = subprocess.run(
                [cmd, "--help"],
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
            if proc.returncode == 0 or "usage" in (proc.stdout + proc.stderr).lower():
                info["cli"] = cmd
                info["cli_help_head"] = (proc.stdout or proc.stderr)[:500]
                break
        except (FileNotFoundError, subprocess.TimeoutExpired):
            continue

    # Common module layouts to try when there's no console script yet.
    for mod_name in (
        "portallib.cli",
        "portallib.__main__",
        "portallib.train",
        "portallib.eval",
        "portallib.port",
    ):
        if importlib.util.find_spec(mod_name) is not None:
            info.setdefault("modules", []).append(mod_name)

    return info


def _try_run_engine(
    discovery: dict[str, Any],
    *,
    task: str,
    rows: list[dict[str, Any]],
    output_dir: Path,
    max_samples: int,
) -> int:
    """Best-effort invoke once APIs exist. Returns process exit code."""
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "task": task,
        "n_rows": len(rows),
        "max_samples": max_samples,
        "host": os.environ.get("PORTALLIB_HOST", "unknown"),
        "discovery": {
            k: discovery[k] for k in ("importable", "version", "cli", "modules") if k in discovery
        },
    }
    (output_dir / "smoke_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    cli = discovery.get("cli")
    if cli == "portallib":
        # Placeholder argv until we read their real CLI after #1 merges.
        # Prefer eval-on-samples if that subcommand exists; else fail soft.
        candidates = [
            [cli, "eval", "--task", task, "--max-samples", str(max_samples),
             "--output-dir", str(output_dir)],
            [cli, "--help"],
        ]
        for argv in candidates:
            print(f"trying: {' '.join(argv)}", flush=True)
            proc = subprocess.run(argv, check=False)
            if proc.returncode == 0 and argv[-1] != "--help":
                print("portallib smoke ok", flush=True)
                return 0
            if argv[-1] == "--help":
                print(
                    "portallib CLI found but train/port/eval argv not yet wired in this harness; "
                    "update smoke_portallib.py after reading the landed API.",
                    flush=True,
                )
                return 2
        return proc.returncode

    print(
        "portallib is importable but no supported CLI entry was discovered.\n"
        f"  version={discovery.get('version')!r}\n"
        f"  modules={discovery.get('modules')}\n"
        f"  api_attrs(sample)={discovery.get('api_attrs')}\n"
        "Update this harness against the landed API (see ROADMAP Phase A3).",
        flush=True,
    )
    return 3


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=DEFAULT_DATASET, help="HF dataset id.")
    parser.add_argument("--task", default=DEFAULT_TASK, help="Task name inside the dataset.")
    parser.add_argument("--split", default="validation", help="HF split to sample.")
    parser.add_argument("--max-samples", type=int, default=8, help="Rows to load (smoke size).")
    parser.add_argument("--output-dir", type=Path, default=Path("/tmp/portallib-smoke"))
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Load task rows + discover portallib; do not invoke the engine.",
    )
    args = parser.parse_args()

    print(f"loading {args.dataset} task={args.task!r} split={args.split} n≤{args.max_samples}")
    rows = _load_task_rows(args.dataset, args.task, args.max_samples, args.split)
    sample_keys = sorted(rows[0].keys())
    print(f"loaded {len(rows)} rows; keys={sample_keys}")
    # Don't dump full prompts (can be long); show gold index only.
    print(f"first gold_idx={rows[0].get('gold_idx')!r}")

    discovery = _discover_portallib()
    print("portallib discovery:", json.dumps(
        {k: discovery[k] for k in ("importable", "version", "cli") if k in discovery},
        indent=2,
    ))

    if args.dry_run:
        print("dry-run ok (dataset + discovery only)")
        if not discovery["importable"]:
            print(
                "note: portallib not installed — rebuild with INSTALL_PORTALLIB=1 "
                "once ramp-public/portallib#1 merges, or pip install inside the VM.",
                flush=True,
            )
        sys.exit(0)

    if not discovery["importable"]:
        raise SystemExit(
            "portallib is not importable in this environment. "
            "Build examples/smolvm/Dockerfile.portallib-cuda with INSTALL_PORTALLIB=1 "
            "(requires the alpha package on GitHub/PyPI)."
        )

    raise SystemExit(
        _try_run_engine(
            discovery,
            task=args.task,
            rows=rows,
            output_dir=args.output_dir,
            max_samples=args.max_samples,
        )
    )


if __name__ == "__main__":
    main()
