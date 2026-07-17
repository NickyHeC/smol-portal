# smol-portal — Agent Guide

This file helps a coding agent drive **PorTAL** (Portable Task Adapters) for a
user. The expected flow: the user describes what they want in natural language,
the agent gathers the missing details, then the agent runs PorTAL — normally
**inside a CUDA-enabled [smolvm](https://github.com/smol-machines/smolvm)
microVM** on a local NVIDIA GPU.

Read [`SPEC.md`](SPEC.md) for the full contract (CLI, config schema, artifacts,
smolvm integration). This guide is the operational playbook.

> **Three-folder product.** This work spans three local folders and agents must
> maintain all three: **smolvm** (`/Users/nicky/Documents/smolvm`, public runtime
> — catch bugs, submit upstream), **smolvm-notes**
> (`/Users/nicky/Documents/smolvm-notes`, private local-only scratchpad — raw
> logs, canonical runbook, unpublished plans), and **smol-portal** (this repo,
> public product). Flow: _private notes → distill → public smol-portal; runtime
> bugs → smolvm → upstream._ Never commit secrets/box IPs to public repos; keep
> them in notes and sanitize outward. Canonical allocation table:
> [`.cursor/rules/smolvm-interop.mdc`](.cursor/rules/smolvm-interop.mdc).

## What PorTAL does

Learn a task once on a **source** model (LoRA), compress it into a base-agnostic
**task latent**, then **port** it to a **target** model with a slim converter,
and **eval** the result. Pipeline: `train → extract → convert → eval`.

## Intake — what to establish before running

Ask only for what's missing; infer sensible defaults for the rest and state them.

| Field | Needed for | Default / note |
|-------|-----------|----------------|
| Source model (HF id) | train, port | e.g. `Qwen/Qwen3-0.6B` |
| Target model (HF id) | convert, port | e.g. `TinyLlama/TinyLlama-1.1B-Chat-v1.0`. **Gated** models (Gemma, Llama) need license + `HF_TOKEN` |
| Task name | all stages | free-form label; drives artifact dirs |
| Dataset (HF id) | train, eval, calibration | e.g. `stanfordnlp/imdb` |
| Goal | sizing | smoke test vs. real accuracy run |
| GPU spec | memory/dtype | VRAM, driver; validated on A10 22 GiB |
| smolvm version | compatibility | **≥ 1.6.4** recommended — first release that bundles CUDA shims (`--cuda` works out of the box; #601 shipped / #596 fixed). ≥1.5.2 works but stock ≤1.6.3 needs a manual shim build. Avoid 1.6.0/1.6.1 on Ubuntu 22.04 — see SPEC. |
| SDPA | perf | math (default, safe) or fused (`PORTAL_SKIP_CUDA_SMOLVM=1`, smolvm ≥ 1.5.2) |

If the user only wants to train an adapter (not port it), use `portal train`
alone. If they want the full port, use the e2e path below.

## Environment prerequisites (host)

1. **smolvm ≥ 1.6.4** — the v1.6.4 tarball bundles the CUDA shims in
   `agent-rootfs` ([#601](https://github.com/smol-machines/smolvm/pull/601) shipped,
   [#596](https://github.com/smol-machines/smolvm/issues/596) fixed), so `--cuda`
   works with no manual shim step. ≥1.5.2 also works but stock tarballs **through
   1.6.3** omit shims — build+copy from the matching tag. On Ubuntu 22.04, skip
   stock **1.6.0 / 1.6.1** (GLIBC_2.39). *(Stock ≥1.6.4 bundles shims; GPU-validated
   on v1.6.8 A10 2026-07-17 — CUDA gate + warm fork PASS.)*
2. **Worker image** `portal-cuda.tar` — a pre-baked pip-torch image so smolvm's
   CUDA staging can interpose at pull time. Build:
   `docker build -f examples/smolvm/Dockerfile.portal-cuda -t portal-cuda . && docker save portal-cuda -o portal-cuda.tar`
3. An NVIDIA GPU with a working driver (`nvidia-smi`, `libcuda.so.1`).

Verify CUDA before training:

```bash
smolvm machine run --net --cuda --image ./portal-cuda.tar -- \
  python3 -c "import torch; print('cuda:', torch.cuda.is_available())"   # -> cuda: True
```

## Canonical run recipe (inside smolvm)

`portal` must be installed **inside the VM** (Python ≥ 3.11; the slim image has
no `git`, so install from a zip URL, not `git+https://`). Set `PORTAL_ZIP` to the
archive for the branch/tag you want:

```bash
PORTAL_ZIP="https://github.com/OWNER/smol-portal/archive/refs/heads/main.zip#subdirectory=pipeline/portal"

smolvm machine run --net --cuda --mem 16384 \
  -e HF_HOME=/tmp/hf \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False \
  --image ./portal-cuda.tar -- \
  sh -c "pip install -q \"portal @ ${PORTAL_ZIP}\" \
    typer rich pydantic safetensors 'datasets>=3.0,<4' accelerate \
    'transformers>=4.45,<4.52' 'peft>=0.14,<0.18' && \
  portal train --model <SOURCE> --task <TASK> --dataset <DATASET> \
    --max-samples 64 --epochs 1 --batch-size 1 --max-seq-length 128 --rank 8 \
    --output-dir /tmp/artifacts"
```

Replace `OWNER` with the repository owner (or point at a published package /
mounted local checkout). For gated models add `-e HF_TOKEN=hf_...`.

## Full port e2e

`portal port` does not yet expose sample/epoch sizing flags. For controllable
smoke-sized end-to-end runs, use the reference script
[`examples/smolvm/port_e2e.py`](examples/smolvm/port_e2e.py):

```bash
python3 examples/smolvm/port_e2e.py \
  --source Qwen/Qwen3-0.6B \
  --target TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --task my-task --dataset stanfordnlp/imdb \
  --max-samples 64 --max-seq-length 128
```

For a default-sized run where CLI flags suffice, `portal port --from ... --to ...
--task ... --dataset ...` works directly.

## Constraints the agent must apply

- **fp32 + `device_map="cuda"`.** PorTAL loads models in fp32 and places them
  incrementally; bulk `model.to("cuda")` can fail on remoted CUDA. Handled in
  `portal.cuda`; don't override.
- **Memory.** Default guest RAM (8 GiB) is too tight; use `--mem 16384`+.
- **`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False`** — safer on the current
  shim VMM surface.
- **SDPA.** Math SDPA is the default and is safe everywhere. Fused SDPA
  (`PORTAL_SKIP_CUDA_SMOLVM=1`) needs smolvm ≥ 1.5.2.
- **Gated models → 401.** If a run fails with `GatedRepoError`/401, tell the user
  to accept the model license on HuggingFace and pass `-e HF_TOKEN=...`, or pick
  an ungated target.
- **Ephemeral runs.** `machine run` discards all state on exit. Copy artifacts out
  (`/tmp/...`) before the VM exits, or use a persistent machine.

## Sizing guidance

| Goal | max-samples | max-seq | epochs | Notes |
|------|-------------|---------|--------|-------|
| Smoke / plumbing | 8–64 | 64–128 | 1 | seconds; metrics not meaningful |
| Small real run | 256–1000 | 256 | 1–3 | minutes on A10 |
| Accuracy target | dataset-scale | 512 | 3+ | compare vs. direct-LoRA baseline |

## Sanity checks after a run

- Converter calibration loss should **decrease** across epochs (a flat loss means
  the converter isn't learning).
- Eval perplexity should be **finite and mid-range**, and **independent of**
  `PORTAL_SKIP_CUDA_SMOLVM` (eval pins math SDPA for reproducibility).
- Artifacts land under `{output_dir}/{task}/…` (see SPEC §artifacts).

## Where things live

- CLI + config contract: [`SPEC.md`](SPEC.md)
- smolvm worker image + runbook: [`examples/smolvm/`](examples/smolvm/)
- Reference e2e driver: [`examples/smolvm/port_e2e.py`](examples/smolvm/port_e2e.py)
- Pipeline code: `pipeline/portal/portal/`
