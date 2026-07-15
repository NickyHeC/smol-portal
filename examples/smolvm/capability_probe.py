#!/usr/bin/env python3
"""Probe CUDA ops that portallib may need, through smolvm remoting.

Run *inside* a ``portal-cuda`` / ``portallib-cuda`` guest (not on the Lambda host).
Prints a JSON summary of pass/fail — paste into the private session log and use
to harden Ben-feedback Tier 2 / worker-image force-offs.

Example (host wrapper)::

    smolvm machine run --net --cuda --mem 16384 \\
      -e HF_HOME=/tmp/hf \\
      -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False \\
      --image ./portal-cuda.tar -- \\
      python3 examples/smolvm/capability_probe.py

Probes (in order):
  1. fp32 + device_map=cuda load/forward/backward  (known baseline)
  2. bf16 load + one train step
  3. fused (flash/mem-efficient) SDPA forward+backward
  4. torch.compile on a tiny model (one step)
     Caveats (2026-07-15): needs a C compiler in the guest; Inductor/Triton also
     link ``-lcuda`` and fail if only ``libcuda.so.1`` is staged — add
     ``ln -s libcuda.so.1 …/libcuda.so``. Simple ``nn.Module`` then PASSes through
     remoting; HF CausalLM+compile may still fail for unrelated dynamo reasons.
  5. multi-GPU visibility + optional NCCL init (skipped if device_count < 2)
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from typing import Any, Callable


TINY = "hf-internal-testing/tiny-random-LlamaForCausalLM"


def _run(name: str, fn: Callable[[], Any]) -> dict[str, Any]:
    row: dict[str, Any] = {"name": name, "ok": False}
    try:
        detail = fn()
        row["ok"] = True
        if detail is not None:
            row["detail"] = detail
    except Exception as exc:  # noqa: BLE001 — probe must never abort the matrix
        row["ok"] = False
        row["error"] = f"{type(exc).__name__}: {exc}"
        row["traceback"] = traceback.format_exc(limit=6)
    print(f"[{'PASS' if row['ok'] else 'FAIL'}] {name}", flush=True)
    if not row["ok"]:
        print(f"       {row['error']}", flush=True)
    return row


def probe_fp32() -> dict[str, Any]:
    import torch
    from transformers import AutoModelForCausalLM

    assert torch.cuda.is_available(), "cuda not available"
    m = AutoModelForCausalLM.from_pretrained(TINY, torch_dtype=torch.float32, device_map="cuda")
    x = torch.randint(0, 100, (1, 8), device="cuda")
    loss = m(x, labels=x).loss
    loss.backward()
    return {"device": str(next(m.parameters()).device), "dtype": str(next(m.parameters()).dtype)}


def probe_bf16() -> dict[str, Any]:
    import torch
    from transformers import AutoModelForCausalLM

    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("torch.cuda.is_bf16_supported() is False on this GPU")
    m = AutoModelForCausalLM.from_pretrained(TINY, torch_dtype=torch.bfloat16, device_map="cuda")
    x = torch.randint(0, 100, (1, 8), device="cuda")
    loss = m(x, labels=x).loss
    loss.backward()
    return {"dtype": str(next(m.parameters()).dtype)}


def probe_fused_sdpa() -> dict[str, Any]:
    import torch
    from transformers import AutoModelForCausalLM

    # Force fused backends on (opposite of portal.cuda math default).
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    torch.backends.cuda.enable_math_sdp(True)
    m = AutoModelForCausalLM.from_pretrained(TINY, torch_dtype=torch.float32, device_map="cuda")
    x = torch.randint(0, 100, (1, 16), device="cuda")
    loss = m(x, labels=x).loss
    loss.backward()
    return {
        "flash": torch.backends.cuda.flash_sdp_enabled(),
        "mem_efficient": torch.backends.cuda.mem_efficient_sdp_enabled(),
        "math": torch.backends.cuda.math_sdp_enabled(),
    }


def probe_torch_compile() -> dict[str, Any]:
    import torch
    from transformers import AutoModelForCausalLM

    m = AutoModelForCausalLM.from_pretrained(TINY, torch_dtype=torch.float32, device_map="cuda")
    compiled = torch.compile(m, mode="reduce-overhead")
    x = torch.randint(0, 100, (1, 8), device="cuda")
    loss = compiled(x, labels=x).loss
    loss.backward()
    return {"compile": True}


def probe_multi_gpu() -> dict[str, Any]:
    import torch

    n = torch.cuda.device_count()
    names = [torch.cuda.get_device_name(i) for i in range(n)]
    detail: dict[str, Any] = {"device_count": n, "names": names}
    if n < 2:
        detail["nccl"] = "skipped (single GPU)"
        return detail

    # Minimal NCCL ping between cuda:0 and cuda:1.
    import torch.distributed as dist

    # Use a single-process gloo/nccl group if available — prefer nccl.
    try:
        dist.init_process_group(
            backend="nccl",
            init_method="tcp://127.0.0.1:29500",
            rank=0,
            world_size=1,
        )
        detail["nccl"] = "init_ok (world_size=1)"
        dist.destroy_process_group()
    except Exception as exc:  # noqa: BLE001
        detail["nccl"] = f"fail: {type(exc).__name__}: {exc}"
        # Still count as ok for "multi-GPU visible"; NCCL detail is informational.
    return detail


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skip",
        action="append",
        default=[],
        choices=["fp32", "bf16", "fused_sdpa", "torch_compile", "multi_gpu"],
        help="Skip a probe by name (repeatable).",
    )
    parser.add_argument(
        "--json-out",
        default="",
        help="Optional path to write the full JSON report.",
    )
    args = parser.parse_args()

    import torch

    meta = {
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "torch": torch.__version__,
        "cuda_version": getattr(torch.version, "cuda", None),
    }
    print("meta:", json.dumps(meta), flush=True)
    if not meta["cuda_available"]:
        print("cuda: False — aborting probes (fix shims / staging first)", flush=True)
        sys.exit(2)

    probes: list[tuple[str, Callable[[], Any]]] = [
        ("fp32", probe_fp32),
        ("bf16", probe_bf16),
        ("fused_sdpa", probe_fused_sdpa),
        ("torch_compile", probe_torch_compile),
        ("multi_gpu", probe_multi_gpu),
    ]
    results = []
    for name, fn in probes:
        if name in args.skip:
            print(f"[SKIP] {name}", flush=True)
            results.append({"name": name, "ok": None, "skipped": True})
            continue
        results.append(_run(name, fn))

    report = {"meta": meta, "results": results}
    text = json.dumps(report, indent=2)
    print("--- capability probe report ---")
    print(text)
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            f.write(text + "\n")
        print(f"wrote {args.json_out}", flush=True)

    # Exit 1 if any non-skipped probe failed (multi_gpu visibility never fails the
    # run solely on NCCL — only on exceptions inside the probe).
    hard_fail = [r for r in results if r.get("ok") is False]
    sys.exit(1 if hard_fail else 0)


if __name__ == "__main__":
    main()
