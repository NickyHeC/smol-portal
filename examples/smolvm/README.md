# PorTAL on smolvm (CUDA)

Development log and runbook for GPU training through smolvm's CUDA remoting stack.
**Validated end-to-end on Lambda Cloud A10 (2026-07-13).** See also `memory.md` (repo root)
and `~/Documents/smolvm-notes/cuda-build-plan.md` (smolvm-side validation log).

**Lambda quick start:** [`lambda-instructions.md`](./lambda-instructions.md) — PEM at
`~/Documents/PorTAL.pem`.

## Summary

| What | Detail |
|------|--------|
| **Minimum smolvm** | **v1.5.2** Linux x86_64 + shims built from matching upstream git tag |
| GPU host | Lambda `gpu_1x_a10`, driver 580.x, `/dev/kvm` |
| Worker image | `portal-cuda.tar` (this Dockerfile) |
| `portal train` | ✅ LoRA smoke (math or fused SDPA on v1.5.2) |
| `portal port` e2e | ✅ train → extract → convert → eval (fused SDPA, smoke config) |
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
portal @ https://github.com/NickyHeC/smol-portal/archive/refs/heads/main.zip#subdirectory=pipeline/portal
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
    "portal @ https://github.com/NickyHeC/smol-portal/archive/refs/heads/main.zip#subdirectory=pipeline/portal" \
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

Full notes: `~/Documents/smolvm-notes/cuda-build-plan.md`.
