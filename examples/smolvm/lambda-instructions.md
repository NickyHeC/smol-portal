# Cloud GPU — quick startup (PorTAL / smolvm CUDA)

Generic runbook for a bare cloud NVIDIA box (validated on Lambda A10). Before
using, replace `OWNER` with your `smol-portal` fork owner and point the SSH key
paths at your own key.

**Assume a bare instance every time.** Terminating a cloud box wipes the home disk —
nothing persists (`portal-cuda.tar`, clones, rustup, shims). Run the full bootstrap
below on each new instance.

**Last validated:** 2026-07-13 — real-model PorTAL + fused SDPA full `portal port` e2e on smolvm **v1.5.2** (cloud A10, ~1 h).

**Today’s target (2026-07-14 afternoon):** host on **smolvm v1.6.0** — see **§9** (shim rebuild,
CUDA gates, `portal` e2e with CLI sizing knobs, capability probe matrix). Historical §1–§5
blocks still say `1.5.2`; for this session set `VER=1.6.0` everywhere.

| Resource | Path |
|----------|------|
| SSH key | `~/.ssh/gpu-box.pem` (use your own) |
| Session log (PorTAL) | `smol-portal/memory.md` |
| Capability probe | [`capability_probe.py`](./capability_probe.py) |

---

## 0. From your Mac — SSH in

```bash
chmod 400 ~/.ssh/gpu-box.pem
ssh -i ~/.ssh/gpu-box.pem ubuntu@<INSTANCE_IP>
```

Optional `~/.ssh/config` (update IP each launch):

```
Host lambda
  HostName <INSTANCE_IP>
  User ubuntu
  IdentityFile ~/.ssh/gpu-box.pem
  IdentitiesOnly yes
```

---

## 1. Bare-instance bootstrap (copy-paste block)

Run this whole block on every fresh Lambda GPU box before any test.

```bash
set -e

# --- Preflight ---
ls -l /dev/kvm
egrep -c '(vmx|svm)' /proc/cpuinfo
nvidia-smi

# --- Groups (kvm for microVMs, docker for image build) ---
sudo usermod -aG kvm,docker ubuntu
newgrp docker <<'EOF'
# nested shell so docker group applies without logout
set -e

# --- Host packages ---
sudo apt-get update
sudo apt-get install -y e2fsprogs curl git build-essential pkg-config libssl-dev

# --- Rust (do NOT use apt install cargo) ---
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source "$HOME/.cargo/env"
rustc --version

# --- smolvm release tarball ---
# 2026-07-14 afternoon: VER=1.6.0 (shim tag MUST match). Prior validated: 1.5.2.
cd ~
VER=1.6.0
curl -L --progress-bar -o smolvm.tar.gz \
  "https://github.com/smol-machines/smolvm/releases/download/v${VER}/smolvm-${VER}-linux-x86_64.tar.gz"
tar xzf smolvm.tar.gz
~/smolvm-${VER}-linux-x86_64/smolvm --version

# --- CUDA shims (release tarball omits them — #596) ---
# **Shim version MUST match the release tarball** (v1.5.2 shims + v1.5.2 binary).
# Mismatch → cuda: False with staging OK and cuda.sock present (protocol skew).

git clone https://github.com/smol-machines/smolvm.git ~/smolvm
cd ~/smolvm
git fetch --tags

git checkout v${VER}
git log -1 --oneline   # must show the v${VER} commit, not an old branch

# A bare box has no git identity by default — for cherry-pick/merge use:
#   git -c user.email="you@local" -c user.name="you" ...

source "$HOME/.cargo/env"
cargo build --release -p smolvm-cudart-shim -p smolvm-cuda-shim
# Rebuild must take ~10s+, not 0.07s (wrong tag / no recompile)
# libcudart-shim.so ≈ 887616 bytes on v1.5.2 (v1.5.1 ≈ 795952)

SHIM_DIR=~/smolvm-${VER}-linux-x86_64/agent-rootfs/usr/local/lib/smolvm-cuda
mkdir -p "$SHIM_DIR"
cp target/release/libcudart.so "$SHIM_DIR/libcudart-shim.so"
cp target/release/libcuda.so  "$SHIM_DIR/libcuda.so.1"
ls -la "$SHIM_DIR"

# --- Worker image (portal-cuda.tar) ---
git clone https://github.com/OWNER/smol-portal.git ~/smol-portal
cd ~/smol-portal
sudo docker build -f examples/smolvm/Dockerfile.portal-cuda -t portal-cuda .
sudo docker save portal-cuda -o ~/portal-cuda.tar
sudo chown "$USER:$USER" ~/portal-cuda.tar   # save runs as root otherwise
ls -lh ~/portal-cuda.tar

echo "Bootstrap complete."
EOF
```

**If `newgrp` / nested shell is awkward**, prefix every `docker` command with `sudo`:

```bash
sudo docker build -f examples/smolvm/Dockerfile.portal-cuda -t portal-cuda .
sudo docker save portal-cuda -o ~/portal-cuda.tar
sudo chown "$USER:$USER" ~/portal-cuda.tar
```

---

## 2. Faster path — skip Docker build on Lambda

`portal-cuda.tar` is ~1–2 GB. Upload from Mac if you already built it:

```bash
# Mac:
scp -i ~/.ssh/gpu-box.pem ~/path/to/portal-cuda.tar ubuntu@<IP>:~/portal-cuda.tar

# Lambda:
ls -lh ~/portal-cuda.tar
```

Still run smolvm tarball + shim build on the bare box (quick).

---

## 3. Session checklist

| Step | Verify |
|------|--------|
| SSH | `ssh -i ~/.ssh/gpu-box.pem ubuntu@<IP>` |
| KVM + GPU | `ls -l /dev/kvm && nvidia-smi` |
| smolvm | `~/smolvm-1.5.2-linux-x86_64/smolvm --version` |
| Shims | `ls ~/smolvm-1.5.2-linux-x86_64/agent-rootfs/usr/local/lib/smolvm-cuda/` → two `.so` files |
| Shim tag | `cd ~/smolvm && git describe --tags --always` → `v1.5.2` |
| Worker image | `ls -lh ~/portal-cuda.tar` |
| Rust | `source ~/.cargo/env` in each new shell |

**Standard run flags:**

```text
--net --cuda --mem 16384
-e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False
-e HF_HOME=/tmp/hf
```

**Install portal inside the VM** (not on Lambda host — host is Python 3.10, portal needs 3.11+).
Slim image has no `git` — use GitHub zip, not `git+https://`:

```text
portal @ https://github.com/OWNER/smol-portal/archive/refs/heads/main.zip#subdirectory=pipeline/portal
```

**Fused SDPA on smolvm v1.5.2+:** pass `-e PORTAL_SKIP_CUDA_SMOLVM=1` on `machine run`
(merged to smol-portal `main`, 2026-07-13). Default `portal.cuda` still forces math SDPA (safe).

No `LD_PRELOAD` with `portal-cuda.tar`.

---

## 4. Verify CUDA

**`machine run` is ephemeral** — VM data under `~/.cache/smolvm/vms/<hash>/` is deleted when
the guest command exits. Grab `agent-startup-error.log` / `cuda.sock` **while the VM runs**.

### A — Quick smoke

```bash
cd ~/smolvm-1.5.2-linux-x86_64

./smolvm machine run --net --cuda --image ~/portal-cuda.tar -- \
  python3 -c "import torch; print('cuda:', torch.cuda.is_available())"
```

Expect `cuda: True`.

### B — Shim staging inside guest

```bash
./smolvm machine run --net --cuda --image ~/portal-cuda.tar -- \
  sh -c 'ls -la /opt/smolvm-cuda/; ls -la /usr/local/lib/python3.11/site-packages/nvidia/cuda_runtime/lib/libcudart.so.12'
```

Expect:
- `/opt/smolvm-cuda/libcuda.so.1` and `libcudart-shim.so` present
- `libcudart.so.12` ≈ **622–800 KB** (shim bind-mount), not ~109 MB

### C — Guest vsock

```bash
./smolvm machine run --net --cuda --image ~/portal-cuda.tar -- \
  python3 -c "
import socket
AF_VSOCK = getattr(socket, 'AF_VSOCK', 40)
s = socket.socket(AF_VSOCK, socket.SOCK_STREAM)
s.settimeout(5)
s.connect((2, 7000))
print('vsock ok')
"
```

### D — Host-only CUDA (no microVM)

```bash
cd ~/smolvm && git checkout v1.5.2 && source ~/.cargo/env
cargo run --release -p smolvm-cuda --example gpu_loopback
```

If this fails, fix host driver before debugging guest/shim.

### Troubleshooting `cuda: False` with staging OK

| Check | What to look for |
|-------|------------------|
| Shim tag vs tarball | `git describe --tags` must match tarball version |
| Rebuild actually ran | `cargo build` ~10s+, not 0.07s |
| Shim size | `libcudart-shim.so` ≈ 887616 bytes (v1.5.2) |
| Fork tags | `git fetch --tags` if `v1.5.2` missing on your fork |

### Persistent debug machine (logs survive)

```bash
cd ~/smolvm-1.5.2-linux-x86_64

./smolvm machine create --name cuda-debug --net --cuda --mem 16384 \
  --image ~/portal-cuda.tar
./smolvm machine start --name cuda-debug

grep -i cuda "$(./smolvm machine data-dir --name cuda-debug)/agent-startup-error.log"
ls -la "$(./smolvm machine data-dir --name cuda-debug)/cuda.sock"

./smolvm machine exec --name cuda-debug -- \
  python3 -c "import torch; print('cuda:', torch.cuda.is_available())"

./smolvm machine delete --name cuda-debug -f
```

---

## 5. PorTAL validation recipes (2026-07-13)

All commands install portal **inside** the VM. Replace `main.zip` with a branch/commit zip if needed.

### 5a — `portal train` (math SDPA default)

```bash
cd ~/smolvm-1.5.2-linux-x86_64

./smolvm machine run --net --cuda --mem 16384 \
  -e HF_HOME=/tmp/hf \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False \
  --image ~/portal-cuda.tar -- \
  sh -c 'pip install -q \
    "portal @ https://github.com/OWNER/smol-portal/archive/refs/heads/main.zip#subdirectory=pipeline/portal" \
    typer rich pydantic safetensors "datasets>=3.0,<4" accelerate \
    "transformers>=4.45,<4.52" "peft>=0.14,<0.18" && \
  portal train \
    --model hf-internal-testing/tiny-random-LlamaForCausalLM \
    --task smoke597 \
    --dataset stanfordnlp/imdb \
    --max-samples 8 --epochs 1 --batch-size 1 \
    --max-seq-length 64 --rank 4 \
    --output-dir /tmp/artifacts'
```

**Success:** adapter saved, 8/8 steps, ~2–3 s.

### 5b — Fused SDPA Trainer (no math workaround)

After merge to `main` (2026-07-13), add `-e PORTAL_SKIP_CUDA_SMOLVM=1` to `machine run`.
Until a fresh Lambda box, inline patch below still works with older zip installs.

```bash
cd ~/smolvm-1.5.2-linux-x86_64

./smolvm machine run --net --cuda --mem 16384 \
  -e HF_HOME=/tmp/hf \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False \
  --image ~/portal-cuda.tar -- \
  sh -c 'pip install -q \
    "portal @ https://github.com/OWNER/smol-portal/archive/refs/heads/main.zip#subdirectory=pipeline/portal" \
    typer rich pydantic safetensors "datasets>=3.0,<4" accelerate \
    "transformers>=4.45,<4.52" "peft>=0.14,<0.18" && \
  python3 - <<'"'"'PY'"'"'
import portal.train as pt
import portal.hypernetwork as ph
import portal.converter as pc
import portal.eval as pe

def fused_only():
    import os, torch
    if not torch.cuda.is_available():
        return
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:False")
    torch.backends.cudnn.enabled = False

for mod in (pt, ph, pc, pe):
    mod.configure_cuda_for_smolvm = fused_only

from pathlib import Path
from portal.config import TrainConfig, LoraConfig

pt.train_source_lora(
    TrainConfig(
        source_model="hf-internal-testing/tiny-random-LlamaForCausalLM",
        task_name="smoke597-fused",
        dataset_name="stanfordnlp/imdb",
        max_samples=8, num_epochs=1, batch_size=1, max_seq_length=64,
        lora=LoraConfig(rank=4, alpha=8),
    ),
    Path("/tmp/artifacts-fused"),
)
print("fused SDPA train ok")
PY'
```

**Success:** `fused SDPA train ok`.

### 5c — `portal port` e2e (train → extract → convert → eval)

`portal port` CLI lacks smoke-sized epoch flags — use explicit steps:

```bash
cd ~/smolvm-1.5.2-linux-x86_64

./smolvm machine run --net --cuda --mem 16384 \
  -e HF_HOME=/tmp/hf \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False \
  --image ~/portal-cuda.tar -- \
  sh -c 'pip install -q \
    "portal @ https://github.com/OWNER/smol-portal/archive/refs/heads/main.zip#subdirectory=pipeline/portal" \
    typer rich pydantic safetensors "datasets>=3.0,<4" accelerate \
    "transformers>=4.45,<4.52" "peft>=0.14,<0.18" && \
  python3 - <<'"'"'PY'"'"'
import portal.train as pt
import portal.hypernetwork as ph
import portal.converter as pconv
import portal.eval as pe

def fused_only():
    import os, torch
    if not torch.cuda.is_available():
        return
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:False")
    torch.backends.cudnn.enabled = False

for mod in (pt, ph, pconv, pe):
    mod.configure_cuda_for_smolvm = fused_only

from pathlib import Path
from portal.artifacts import load_adapter_path
from portal.config import TrainConfig, HypernetConfig, ConverterConfig, EvalConfig, LoraConfig

TINY = "hf-internal-testing/tiny-random-LlamaForCausalLM"
TASK = "port-smoke597"
DS = "stanfordnlp/imdb"
OUT = Path("/tmp/port-artifacts")

print("Step 1/4: train")
adapter_artifact = pt.train_source_lora(
    TrainConfig(
        source_model=TINY, task_name=TASK, dataset_name=DS,
        max_samples=8, num_epochs=1, batch_size=1, max_seq_length=64,
        lora=LoraConfig(rank=4, alpha=8),
    ),
    OUT,
)
print("  ->", adapter_artifact)

print("Step 2/4: extract")
latent_dir = ph.extract_task_latent(
    load_adapter_path(adapter_artifact), TINY, TASK,
    HypernetConfig(num_epochs=10, latent_dim=64, hidden_dim=128, num_layers=2),
    OUT,
)
print("  ->", latent_dir)

print("Step 3/4: convert")
target_adapter = pconv.convert_latent_to_adapter(
    latent_dir, TASK,
    ConverterConfig(target_model=TINY, calibration_dataset=DS,
                    calibration_samples=8, num_epochs=5, hidden_dim=128),
    OUT, lora_rank=4,
)
print("  ->", target_adapter)

print("Step 4/4: eval")
eval_dir = pe.evaluate_adapter(
    target_adapter,
    EvalConfig(model_name=TINY, task_name=TASK, dataset_name=DS,
               dataset_split="test", max_samples=8, batch_size=1, max_seq_length=64),
    OUT,
)
print("  ->", eval_dir)
print("port e2e ok")
PY'
```

**Success:** `port e2e ok` with four artifact paths. High perplexity on tiny-random model is expected.

### 5d — Fused SDPA backward smoke (no portal)

```bash
cd ~/smolvm-1.5.2-linux-x86_64

./smolvm machine run --net --cuda --mem 16384 \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False \
  --image ~/portal-cuda.tar -- \
  sh -c 'pip install -q transformers accelerate && python3 -c "
import torch
from transformers import AutoModelForCausalLM
m = AutoModelForCausalLM.from_pretrained(
    \"hf-internal-testing/tiny-random-LlamaForCausalLM\", device_map=\"cuda\")
x = torch.randint(0, 100, (1, 8), device=\"cuda\")
m(x, labels=x).loss.backward()
print(\"backward ok\")
"'
```

**Note:** [#597](https://github.com/smol-machines/smolvm/issues/597) **passes on v1.5.2** (closed 2026-07-13). Failed on v1.5.0.

### 5e — Real-model `portal train` (Qwen3-0.6B, math SDPA)

**Validated 2026-07-13 (~10 s, 64 steps @ ~7 it/s).**

```bash
cd ~/smolvm-1.5.2-linux-x86_64

./smolvm machine run --net --cuda --mem 16384 \
  -e HF_HOME=/tmp/hf \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False \
  --image ~/portal-cuda.tar -- \
  sh -c 'pip install -q \
    "portal @ https://github.com/OWNER/smol-portal/archive/refs/heads/main.zip#subdirectory=pipeline/portal" \
    typer rich pydantic safetensors "datasets>=3.0,<4" accelerate \
    "transformers>=4.45,<4.52" "peft>=0.14,<0.18" && \
  portal train \
    --model Qwen/Qwen3-0.6B \
    --task imdb-qwen-real \
    --dataset stanfordnlp/imdb \
    --max-samples 64 --epochs 1 --batch-size 1 \
    --max-seq-length 128 --rank 8 \
    --output-dir /tmp/artifacts'
```

### 5f — Real-model `portal port` e2e (Qwen → TinyLlama)

**Validated 2026-07-13 (`port e2e ok`).** Use TinyLlama (ungated). For Gemma-3, add
`-e HF_TOKEN=...` and accept the HF license first.

```bash
cd ~/smolvm-1.5.2-linux-x86_64

./smolvm machine run --net --cuda --mem 16384 \
  -e HF_HOME=/tmp/hf \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False \
  --image ~/portal-cuda.tar -- \
  sh -c 'pip install -q \
    "portal @ https://github.com/OWNER/smol-portal/archive/refs/heads/main.zip#subdirectory=pipeline/portal" \
    typer rich pydantic safetensors "datasets>=3.0,<4" accelerate \
    "transformers>=4.45,<4.52" "peft>=0.14,<0.18" && \
  python3 - <<'"'"'PY'"'"'
import portal.train as pt
import portal.hypernetwork as ph
import portal.converter as pconv
import portal.eval as pe
from pathlib import Path
from portal.artifacts import load_adapter_path
from portal.config import TrainConfig, HypernetConfig, ConverterConfig, EvalConfig, LoraConfig

SOURCE = "Qwen/Qwen3-0.6B"
TARGET = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
TASK = "imdb-port-real"
DS = "stanfordnlp/imdb"
OUT = Path("/tmp/port-artifacts")

adapter_artifact = pt.train_source_lora(
    TrainConfig(source_model=SOURCE, task_name=TASK, dataset_name=DS,
                max_samples=64, num_epochs=1, batch_size=1, max_seq_length=128,
                lora=LoraConfig(rank=8, alpha=16)), OUT)
latent_dir = ph.extract_task_latent(
    load_adapter_path(adapter_artifact), SOURCE, TASK,
    HypernetConfig(num_epochs=10, latent_dim=64, hidden_dim=128, num_layers=2), OUT)
target_adapter = pconv.convert_latent_to_adapter(
    latent_dir, TASK,
    ConverterConfig(target_model=TARGET, calibration_dataset=DS,
                    calibration_samples=64, num_epochs=5, hidden_dim=128),
    OUT, lora_rank=8)
pe.evaluate_adapter(target_adapter,
    EvalConfig(model_name=TARGET, task_name=TASK, dataset_name=DS,
               dataset_split="test", max_samples=64, batch_size=1, max_seq_length=128), OUT)
print("port e2e ok")
PY'
```

### 5g — Fused SDPA on real models

Add `-e PORTAL_SKIP_CUDA_SMOLVM=1` (smolvm ≥1.5.2). **Train validated 2026-07-13** (Qwen3-0.6B, ~7.65 it/s). **Full port e2e validated**
(Qwen → TinyLlama, `port e2e ok (fused SDPA)`).

```bash
  -e PORTAL_SKIP_CUDA_SMOLVM=1 \
```

**Success:** train completes without `CUDA error: invalid argument` on backward.

**Full port e2e (fused SDPA):** §5f + `PORTAL_SKIP_CUDA_SMOLVM=1`, use `TASK=imdb-port-fused`.

---

## 6. Known issues & workarounds

| Issue | Symptom | Workaround |
|-------|---------|------------|
| [#596](https://github.com/smol-machines/smolvm/issues/596) release missing shims | `cuda: False` (801) on stock tarball | Manual `cargo build` + copy to `agent-rootfs` (PR [#601](https://github.com/smol-machines/smolvm/pull/601)) |
| Shim version skew | staging OK, `cuda.sock` present, `cuda: False` | Match git tag to tarball; `git fetch upstream --tags` |
| [#598](https://github.com/smol-machines/smolvm/issues/598) staging layout | conda / runtime `pip install torch` misses shims | Pre-bake pip torch in `portal-cuda.tar` |
| [#597](https://github.com/smol-machines/smolvm/issues/597) fused SDPA | FAIL on **v1.5.0** | **Fixed in v1.5.2** — use v1.5.2+; math SDPA workaround still default in portal |
| `pip install portal` on host | Python 3.10 vs portal ≥3.11 | Install inside VM via zip URL (§3) |
| Slim image no `git` | pip `git+https://` fails | Use GitHub `archive/.../main.zip#subdirectory=...` |
| Docker permission denied | `permission denied` on docker socket | `sudo docker` or `usermod -aG docker` + re-login |
| Ephemeral log grep empty | no `agent-startup-error.log` after run | VM dir deleted on exit — use `sleep` trick or persistent machine (§4D) |
| Gated HF models (Gemma-3) | 401 on `config.json` | Accept license on HF; `-e HF_TOKEN=hf_...` on `machine run`, or use ungated target (§5f) |

---

## 7. Copy files Mac ↔ Lambda

```bash
# Mac → Lambda (cached portal-cuda.tar)
scp -i ~/.ssh/gpu-box.pem ./portal-cuda.tar ubuntu@<IP>:~/

# Lambda → Mac (logs / artifacts)
scp -i ~/.ssh/gpu-box.pem ubuntu@<IP>:/tmp/port-artifacts/ ./port-artifacts/
```

---

## 8. Tear down

Terminate the Lambda instance when done. Next session = new IP, bare box, re-run §1
(or §1 minus docker if you scp `portal-cuda.tar`).

---

## 9. Session plan — 2026-07-14 afternoon (smolvm v1.6.0 hosting de-risk)

**Goal:** confirm the PorTAL hosting substrate still works after the v1.5.2 → v1.6.0 bump,
and record which CUDA ops portallib may need (feeds Ben feedback + worker force-offs).
Do **not** chase own-pipeline science (Phase A2 demoted).

**Time budget (A10, fresh box):** ~60–90 min bootstrap + gates; ~30–45 min e2e; ~20 min probes.

### Order of battle

| # | Step | Success signal | Abort if |
|---|------|----------------|----------|
| 0 | Launch box, SSH, §1 bootstrap with `VER=1.6.0` | `smolvm --version` ≈ 1.6.0; shims present; `portal-cuda.tar` built | no KVM / no GPU |
| 1 | §4 A–C CUDA gates | `cuda: True`; libcudart ≈ 622–800 KB; `vsock ok` | `cuda: False` → shim tag skew / rebuild |
| 2 | §4 D `gpu_loopback` at `v1.6.0` | host CUDA example OK | host driver broken |
| 3 | §5a tiny `portal train` (math SDPA) | 8/8 steps, adapter saved | train/backward fail |
| 4 | `portal port` smoke via **CLI sizing knobs** (below) | `port` completes; finite eval | convert/eval crash |
| 5 | Fused path: `-e PORTAL_SKIP_CUDA_SMOLVM=1` + §5a or §5d | train / `backward ok` | fused broken on 1.6.0 |
| 6 | [`capability_probe.py`](./capability_probe.py) matrix | JSON report; note each FAIL | — (record, don’t block) |
| 7 | Optional: §5e/§5f real Qwen→TinyLlama if time | `port e2e ok` | OOM / time |
| 8 | Log results → private notes; distill public `memory.md` | — | — |

### 9a — Bootstrap delta (`VER=1.6.0`)

Same as §1, but:

```bash
VER=1.6.0
# tarball + git checkout v${VER} + cargo build shims + copy into agent-rootfs
# Worker image: still Dockerfile.portal-cuda → ~/portal-cuda.tar (legacy engine for this session)
# Optional deps-only portallib image (no engine yet):
#   sudo docker build -f examples/smolvm/Dockerfile.portallib-cuda \
#     --build-arg INSTALL_PORTALLIB=0 -t portallib-cuda .
```

Checklist replacements for §3:

```text
~/smolvm-1.6.0-linux-x86_64/smolvm --version
cd ~/smolvm && git describe --tags --always   # expect v1.6.0
```

### 9b — `portal port` smoke (CLI knobs — preferred over §5c inline Python)

```bash
cd ~/smolvm-1.6.0-linux-x86_64
PORTAL_ZIP="https://github.com/OWNER/smol-portal/archive/refs/heads/main.zip#subdirectory=pipeline/portal"

./smolvm machine run --net --cuda --mem 16384 \
  -e HF_HOME=/tmp/hf \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False \
  --image ~/portal-cuda.tar -- \
  sh -c "pip install -q \"portal @ ${PORTAL_ZIP}\" \
    typer rich pydantic safetensors 'datasets>=3.0,<4' accelerate \
    'transformers>=4.45,<4.52' 'peft>=0.14,<0.18' && \
  portal port \
    --from hf-internal-testing/tiny-random-LlamaForCausalLM \
    --to hf-internal-testing/tiny-random-LlamaForCausalLM \
    --task port-smoke160 \
    --dataset stanfordnlp/imdb \
    --max-samples 8 --max-seq-length 64 --batch-size 1 --rank 4 \
    --train-epochs 1 --extract-epochs 10 --convert-epochs 5 \
    --cal-samples 8 --latent-dim 64 --hidden-dim 128 \
    --output-dir /tmp/artifacts"
```

**Success:** command exits 0; artifacts under `/tmp/artifacts/port-smoke160/`. (Guest is
ephemeral — read metrics from stdout or copy out before exit.)

Fused variant: add `-e PORTAL_SKIP_CUDA_SMOLVM=1` and change `--task port-smoke160-fused`.

### 9c — Capability probe matrix

Copy the script into the guest (ephemeral `machine run` has no repo mount unless you
add a volume). Easiest: pip-install portal (pulls nothing for this script) **or** curl
the raw file, **or** mount the checkout:

```bash
cd ~/smolvm-1.6.0-linux-x86_64

# If smol-portal is cloned on the host (`-v` = HOST:GUEST[:ro]):
./smolvm machine run --net --cuda --mem 16384 \
  -e HF_HOME=/tmp/hf \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False \
  --image ~/portal-cuda.tar \
  -v ~/smol-portal:/workspace/smol-portal:ro -- \
  sh -c 'pip install -q transformers accelerate && \
    python3 /workspace/smol-portal/examples/smolvm/capability_probe.py \
      --json-out /tmp/capability_probe.json; \
    cat /tmp/capability_probe.json'
```

No host mount? `curl -fsSL` the raw `capability_probe.py` from your fork into `/tmp` and run it.

| Probe | What it means for portallib hosting |
|-------|-------------------------------------|
| `fp32` | Baseline remoting path (must PASS) |
| `bf16` | If FAIL → force fp32 / dtype knob (feedback Tier 2) |
| `fused_sdpa` | If FAIL on 1.6.0 → keep math SDPA default; re-open/trace |
| `torch_compile` | If FAIL → force-disable compile in worker / upstream knob |
| `multi_gpu` | `device_count` + NCCL note; `gpu_1x_a10` → expect skip |

Paste the JSON into the private session card (`smolvm-notes/`).

### 9d — After the box (same day)

1. Fill pass/fail table in private notes; update `portallib-feedback.md` Tier 2 with evidence.
2. Distill public summary into `memory.md` + bump “Last validated” here if e2e PASS.
3. If v1.6.0 changes the working minimum, update `SPEC.md` / README “≥ …” line.
4. Skim `src/cuda_daemon.rs` notes only if shim install path looked different (optional).

---

## Historical — #597 trace (superseded 2026-07-13)

Repro **passes on v1.5.2** without math SDPA. PR [#603](https://github.com/smol-machines/smolvm/pull/603) closed.
Only revisit if fused SDPA regresses on a future smolvm release.

<details>
<summary>Old trace command (v1.5.0 era)</summary>

```bash
SMOLVM_CUDA_SHIM_TRACE=1 ./smolvm machine run --net --cuda --mem 16384 \
  -e CUDA_LAUNCH_BLOCKING=1 \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False \
  --image ~/portal-cuda.tar -- \
  sh -c 'pip install -q transformers accelerate && python3 -c "
from transformers import AutoModelForCausalLM
import torch
m = AutoModelForCausalLM.from_pretrained(
    \"hf-internal-testing/tiny-random-LlamaForCausalLM\", device_map=\"cuda\")
x = torch.randint(0, 100, (1, 8), device=\"cuda\")
m(x, labels=x).loss.backward()
"'
```

</details>
