# PorTAL on smolvm (CUDA)

Development log and runbook for GPU training through smolvm's CUDA remoting stack.
**Validated end-to-end on a cloud A10 (2026-07-13), including real-model
Qwen3-0.6B → TinyLlama `portal port`.** See also `memory.md` (repo root).

**Cloud GPU quick start:** [`lambda-instructions.md`](./lambda-instructions.md) — bootstrap,
CUDA verify, `portal train` / fused SDPA / `portal port` e2e copy-paste blocks. Replace
`OWNER` with your `smol-portal` fork owner and use your own SSH key for the box.

## Summary

| What | Detail |
|------|--------|
| **Minimum smolvm** | **v1.5.2** Linux x86_64 + shims built from matching upstream git tag |
| GPU host | Lambda `gpu_1x_a10`, driver 580.x, `/dev/kvm` |
| Worker image | `portal-cuda.tar` (this Dockerfile) |
| `portal train` | ✅ smoke + **Qwen3-0.6B** real (math SDPA) |
| `portal port` e2e | ✅ **Qwen → TinyLlama** (`port e2e ok`); Gemma needs `HF_TOKEN` |
| Fused SDPA (real models) | ✅ train + **full port e2e** (`PORTAL_SKIP_CUDA_SMOLVM=1`) |
| PorTAL CUDA config | `portal.cuda.configure_cuda_for_smolvm()` — math SDPA default; `PORTAL_SKIP_CUDA_SMOLVM=1` for fused SDPA on smolvm ≥1.5.2 |

## Why a custom image?

smolvm bind-mounts CUDA shims over pip NVIDIA wheels **when the image is pulled**.
Libraries installed later (`pip install torch` inside `machine run`) are invisible to
staging. Conda `pytorch/pytorch` images put cuBLAS under `/opt/conda/lib/`, also outside
staging paths.

**Build on a machine with Docker:**

```bash
docker build -f examples/smolvm/Dockerfile.portal-cuda -t portal-cuda .
docker save portal-cuda -o portal-cuda.tar
```

## CUDA shims (release tarball gap — [#596](https://github.com/smol-machines/smolvm/issues/596))

Official smolvm Linux tarballs may omit shims from `agent-rootfs`. **Shim git tag must
match the release tarball version** (v1.5.2 shims + v1.5.2 binary).

```bash
git clone https://github.com/smol-machines/smolvm.git
cd smolvm
git fetch --tags
git checkout v1.5.2
cargo build --release -p smolvm-cudart-shim -p smolvm-cuda-shim
SHIM_DIR=/path/to/smolvm-1.5.2-linux-x86_64/agent-rootfs/usr/local/lib/smolvm-cuda
mkdir -p "$SHIM_DIR"
cp target/release/libcudart.so "$SHIM_DIR/libcudart-shim.so"
cp target/release/libcuda.so  "$SHIM_DIR/libcuda.so.1"
```

Verify: rebuild takes ~10s+ (not 0.07s); `libcudart-shim.so` ≈ 887616 bytes on v1.5.2.
Inside a VM, `libcublas.so.12` under `site-packages/nvidia/` should be ~622 KB (shim), not
~109 MB.

## Install portal inside the VM

`portal` requires Python ≥3.11 (in `portal-cuda.tar`). Install **inside** `machine run`,
not on the Lambda host (Ubuntu 3.10). Use the GitHub zip — slim image has no `git`:

```text
portal @ https://github.com/OWNER/smol-portal/archive/refs/heads/main.zip#subdirectory=pipeline/portal
```

## Smolfile

```bash
smolvm machine run --net --cuda --mem 16384 -s examples/smolvm/portal.smolfile -- \
  portal train --model ... --task ... --dataset stanfordnlp/imdb ...
```

| Field | Value | Why |
|-------|-------|-----|
| `cuda` | `true` | Enable vsock GPU remoting |
| `net` | `true` | HF model + dataset download |
| `memory` | `16384` | 8 GiB default too tight for training |
| `PYTORCH_CUDA_ALLOC_CONF` | `expandable_segments:False` | Safer on current shim VMM surface |
| SDPA | `portal.cuda` or `PORTAL_SKIP_CUDA_SMOLVM=1` | Math SDPA default; fused OK on smolvm ≥1.5.2 |

## Run training (ephemeral)

```bash
smolvm machine run --net --cuda --mem 16384 \
  -e HF_HOME=/tmp/hf \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False \
  --image ./portal-cuda.tar -- \
  sh -c 'pip install -q \
    "portal @ https://github.com/OWNER/smol-portal/archive/refs/heads/main.zip#subdirectory=pipeline/portal" \
    typer rich pydantic safetensors "datasets>=3.0,<4" accelerate \
    "transformers>=4.45,<4.52" "peft>=0.14,<0.18" && \
  portal train \
    --model hf-internal-testing/tiny-random-LlamaForCausalLM \
    --task smoke \
    --dataset stanfordnlp/imdb \
    --max-samples 8 --epochs 1 --batch-size 1 \
    --max-seq-length 64 --rank 4 \
    --output-dir /tmp/artifacts'
```

**Fused SDPA on smolvm v1.5.2+:** pass `-e PORTAL_SKIP_CUDA_SMOLVM=1` (after that env gate
lands on `main`) or use the inline `fused_only` patch from `cuda-build-plan.md`.

## Full port e2e (reference driver)

[`port_e2e.py`](port_e2e.py) runs `train → extract → convert → eval` with the
smoke-sizing knobs `portal port` does not yet expose. Install portal in the VM,
then run it (see the script header for the full `machine run` wrapper):

```bash
python3 examples/smolvm/port_e2e.py \
  --source Qwen/Qwen3-0.6B \
  --target TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --task my-task --dataset stanfordnlp/imdb \
  --max-samples 64 --max-seq-length 128
```

For agent-driven use and the full contract, see [`AGENTS.md`](../../AGENTS.md)
and [`SPEC.md`](../../SPEC.md).

## SDPA / #597 history

| smolvm | Fused SDPA backward | Workaround |
|--------|---------------------|------------|
| 1.5.0 | FAIL (`invalid argument`) | math SDPA via `configure_cuda_for_smolvm()` |
| **1.5.2** | **PASS** | optional — `PORTAL_SKIP_CUDA_SMOLVM=1` |

[#597](https://github.com/smol-machines/smolvm/issues/597) closed after Lambda re-validation
on v1.5.2 (2026-07-13).

## Upstream smolvm issues still open

1. [#596](https://github.com/smol-machines/smolvm/issues/596) — release missing bundled shims (PR [#601](https://github.com/smol-machines/smolvm/pull/601))
2. [#598](https://github.com/smol-machines/smolvm/issues/598) — image layout docs (PR [#600](https://github.com/smol-machines/smolvm/pull/600))

See [`SPEC.md`](../../SPEC.md) and [`AGENTS.md`](../../AGENTS.md) for the full contract.
