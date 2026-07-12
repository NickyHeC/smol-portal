# PorTAL on smolvm (CUDA)

Development log and runbook for GPU training through smolvm's CUDA remoting stack.
Validated on Lambda Cloud A10 (2026-07-11/12). See also `memory.md` (repo root) and
`~/Documents/smolvm-notes/cuda-build-plan.md` (smolvm-side validation log).

## Summary

| What | Detail |
|------|--------|
| smolvm version | 1.5.0 Linux x86_64 release tarball |
| GPU host | Lambda `gpu_1x_a10`, driver 580.105.08, `/dev/kvm` |
| Worker image | `portal-cuda.tar` (this Dockerfile) |
| PorTAL change | `portal.cuda.configure_cuda_for_smolvm()` — **required** for Llama backward |
| Step 3 result | ✅ tiny Llama LoRA train, adapter artifact written |

## Why a custom image?

smolvm bind-mounts CUDA shims over pip NVIDIA wheels **when the image is pulled**.
Libraries installed later (`pip install torch` inside `machine run`) are invisible to
staging. Conda `pytorch/pytorch` images put cuBLAS under `/opt/conda/lib/`, also outside
staging paths.

**Build on a machine with Docker** (Lambda on-demand instances have Docker CE preinstalled):

```bash
docker build -f examples/smolvm/Dockerfile.portal-cuda -t portal-cuda .
docker save portal-cuda -o portal-cuda.tar
```

## CUDA shims (release tarball gap)

Official smolvm Linux tarballs may omit shims from `agent-rootfs`. Build once per extract:

```bash
git clone --depth 1 --branch v1.5.0 https://github.com/smol-machines/smolvm.git
cd smolvm
cargo build --release -p smolvm-cudart-shim -p smolvm-cuda-shim
SHIM_DIR=/path/to/smolvm-*/agent-rootfs/usr/local/lib/smolvm-cuda
mkdir -p "$SHIM_DIR"
cp target/release/libcudart.so "$SHIM_DIR/libcudart-shim.so"
cp target/release/libcuda.so  "$SHIM_DIR/libcuda.so.1"
```

Verify inside a VM: `libcublas.so.12` under `site-packages/nvidia/` should be ~622 KB
(shim bind-mount), not ~109 MB.

## Smolfile

```bash
smolvm machine run --net --cuda --mem 16384 -s examples/smolvm/portal.smolfile -- \
  portal train --model ... --task ... --dataset stanfordnlp/imdb ...
```

Adjust `image = "./portal-cuda.tar"` to your tar path. Install `portal` + deps in the
container command or bake into a derived image.

`portal.smolfile` fields:

| Field | Value | Why |
|-------|-------|-----|
| `cuda` | `true` | Enable vsock GPU remoting |
| `net` | `true` | HF model + dataset download |
| `memory` | `16384` | 8 GiB default too tight for training |
| `PYTORCH_CUDA_ALLOC_CONF` | `expandable_segments:False` | Safer on current shim VMM surface |
| Math SDPA | in `portal.cuda` | Flash/mem-efficient SDPA backward fails on shim |

## Run training (ephemeral)

```bash
smolvm machine run --net --cuda --mem 16384 \
  -e HF_HOME=/tmp/hf \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False \
  --image ./portal-cuda.tar -- \
  sh -c 'apt-get update -qq && apt-get install -y -qq git && \
  pip install -q "portal @ git+https://github.com/NickyHeC/smol-portal.git#subdirectory=pipeline/portal" \
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

No `LD_PRELOAD` needed with this image layout.

## Debugging notes (2026-07-12)

**Symptom:** `CUDA error: invalid argument` on `loss.backward()` during Llama training.

**Bisect:**

```text
(x @ x).sum().backward()           → OK
HF Llama + default SDPA backward    → FAIL
HF Llama + math SDPA only           → OK
portal train + math SDPA            → OK
```

**Not the fix alone:** `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False` (helpful but insufficient without math SDPA).

## Upstream smolvm issues to file

1. Release `agent-rootfs` missing bundled CUDA shims
2. Auto-staging pull-time only; conda layout unsupported
3. Fused SDPA backward incompatible with remoted CUDA
4. `--memory` alias for `--mem`
5. AGENTS.md CUDA docs stale

Full draft bodies: `~/Documents/smolvm-notes/cuda-build-plan.md`.
