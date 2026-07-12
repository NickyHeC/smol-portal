# PorTAL on smolvm (CUDA)

Validated on Lambda Cloud A10 with smolvm 1.5.0.

## Worker image

smolvm auto-stages CUDA shims over pip NVIDIA wheels **at image pull time**. The
image must already contain `torch` (and `site-packages/nvidia/*`) when smolvm
loads it — not installed at runtime inside `machine run`.

```bash
docker build -f examples/smolvm/Dockerfile.portal-cuda -t portal-cuda .
docker save portal-cuda -o portal-cuda.tar
```

Do **not** use `pytorch/pytorch` conda images unless you also `LD_PRELOAD` the
libcudart shim; conda `libcublas` paths are outside auto-staging.

## CUDA shims (release tarball)

Official Linux tarballs may omit shims from `agent-rootfs`. Build and install:

```bash
git clone --depth 1 --branch v1.5.0 https://github.com/smol-machines/smolvm.git
cd smolvm
cargo build --release -p smolvm-cudart-shim -p smolvm-cuda-shim
SHIM_DIR=/path/to/smolvm-*/agent-rootfs/usr/local/lib/smolvm-cuda
mkdir -p "$SHIM_DIR"
cp target/release/libcudart.so "$SHIM_DIR/libcudart-shim.so"
cp target/release/libcuda.so  "$SHIM_DIR/libcuda.so.1"
```

## Run training

Portal configures math SDPA and fp32 `device_map` load automatically when CUDA
is available (`portal.cuda.configure_cuda_for_smolvm`).

```bash
smolvm machine run --net --cuda --mem 16384 \
  -e HF_HOME=/tmp/hf \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False \
  --image ./portal-cuda.tar -- \
  portal train \
    --model hf-internal-testing/tiny-random-LlamaForCausalLM \
    --task smoke \
    --dataset stanfordnlp/imdb \
    --max-samples 8 --epochs 1 --batch-size 1 \
    --max-seq-length 64 --rank 4 \
    --output-dir /tmp/artifacts
```

Or use `examples/smolvm/portal.smolfile` after adjusting `image`.

## Known smolvm gaps (upstream)

- Fused SDPA backward (flash/mem-efficient) fails — portal forces math SDPA
- Release `agent-rootfs` missing bundled shims
- Runtime `pip install torch` inside ephemeral runs misses auto-staging
