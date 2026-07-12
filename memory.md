# memory.md

Session log for **smol-portal**. Reverse chronological order — newest on
top. Each entry: date, summary, key outcomes, and decisions made.

---

## 2026-07-12 — smolvm CUDA validation complete (Lambda A10)

- **Goal:** run PorTAL LoRA training inside smolvm with `--cuda` on Lambda Cloud.
- **Result:** Step 3 passed — `portal train` on `hf-internal-testing/tiny-random-LlamaForCausalLM`, 8/8 steps, adapter saved (~3 s).
- **Stack:** smolvm 1.5.0 Linux tarball, manually built CUDA shims in `agent-rootfs`, `portal-cuda.tar` worker image (pip torch cu124 pre-baked).
- **Root causes found:**
  1. Release tarball ships without CUDA shims — must `cargo build` cudart/cuda shims and copy into `agent-rootfs/usr/local/lib/smolvm-cuda/`.
  2. Auto-staging only overlays pip NVIDIA wheels at **image pull time** — conda `pytorch/pytorch` and runtime `pip install torch` miss cuBLAS interposition.
  3. **Fused SDPA backward** (flash / mem-efficient attention) fails through remoted CUDA (`invalid argument`); **math SDPA works**. Matmul-only backward always worked.
- **Code landed (PR #1, branch `feat/smolvm-cuda-backends`):**
  - `portal/cuda.py` — `configure_cuda_for_smolvm()`, `causal_lm_load_kwargs()`
  - Wired into `train.py`, `converter.py`, `eval.py`, `hypernetwork.py`
  - `examples/smolvm/Dockerfile.portal-cuda`, `portal.smolfile`, `README.md`
- **Worker recipe:** `--net --cuda --mem 16384`, `portal-cuda.tar`, env `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False`, no `LD_PRELOAD`.
- **Deferred:** Step 4 (`portal port` e2e) — same stack, validate after PR merge.
- **smolvm upstream:** [#596](https://github.com/smol-machines/smolvm/issues/596) (release shims), [#597](https://github.com/smol-machines/smolvm/issues/597) (SDPA backward), [#598](https://github.com/smol-machines/smolvm/issues/598) (image layout docs).
- **Lambda session:** closed; no further GPU host testing until Step 4 or upstream fixes.

## 2026-07-03 — Phase 0 proven: LoRA training works under gVisor nvproxy

- **Premise validated:** CUDA forward, backward, and optimizer steps all succeed
  inside gVisor's ioctl-forwarding sandbox on an NVIDIA A10 (Lambda Cloud).
- **Driver:** `580.105.08` — exactly on gVisor's supported list (no version forcing).
- **Ioctl trace captured:** ~3,500 forwarded ioctls for a 10-step LoRA run;
  `NV_ESC_RM_ALLOC_MEMORY` (19×), `UVM` ioctls (493×), frontend ioctls (3,038×).
  Saved as `docs/phase0-ioctl-trace.txt` (to be copied from Lambda).
- **Workarounds found:**
  - `nvidia-persistenced` socket must be a regular file (not a Unix socket) for
    gVisor's gofer; `sudo touch /run/nvidia-persistenced/socket`.
  - NGC image requests `video` driver cap; add `--nvproxy-allowed-driver-capabilities=compute,utility,video`.
  - gVisor DNS broken with `--network=host`; pre-install packages outside gVisor.
  - Platform must be `systrap` (not KVM) on Lambda.
- **Kill criterion passed:** ioctl-forwarding supports full ML training; building
  a Rust reimplementation is justified.

## 2026-07-03 — Architecture decided: own Rust runtime, not libkrun PR

- **Decision:** build our own Rust CUDA-first sandbox runtime ("Rust gVisor"),
  not a contribution to libkrun or `virtio-gpu-nv`.
- `virtio-gpu-nv` (nestrilabs) is **design-only** (README + ARCHITECTURE.md, zero
  code). Their focus is cloud gaming / NVENC; ours is ML compute.
- gVisor `nvproxy` (Go) is the proven reference — we port its ioctl dispatch +
  ABI tables to Rust.
- Scope: NVIDIA-only, Linux + KVM only, compute-only (no graphics/NVENC/DRM).
- Security model: KVM VM isolation for CPU/control; Rust backend as ioctl
  validation chokepoint.

## 2026-07-02 — Project created, plan drafted

- **Goal:** run Ramp Labs PorTAL (portable task adapters for LLMs) securely in a
  CUDA-capable microVM on local NVIDIA GPUs.
- Researched PorTAL (hypernetwork → task latent → converter on new base; ~94–98%
  of LoRA accuracy at ~half calibration cost).
- Researched smolvm (libkrun, virtio-gpu Venus = Vulkan only, not CUDA).
- Identified the blocker: smolvm exposes Vulkan via Venus, not CUDA. Three routes
  evaluated: VFIO passthrough, CUDA API remoting, native-context ioctl forwarding.
- **Chose route 3:** driver-level ioctl forwarding (as proven by gVisor nvproxy).
- Created `ROADMAP.md` (6-phase plan) and `reference-material.md` (curated links).
- Repo: bare — just LICENSE, .gitignore, and the two docs.

---

## Key facts (stable across sessions)

- **Target hardware:** NVIDIA GPUs, Linux x86_64, KVM.
- **Tested on:** Lambda Cloud A10, driver 580.105.08, CUDA 13.0.
- **DGX Spark:** available but user lacks sudo/docker group; use Lambda for now.
- **Lambda SSH:** key is `~/Downloads/PorTAL.pem`, user `ubuntu`, SSH config alias `lambda`.
- **gVisor install:** `runsc` at `/usr/local/bin/runsc`, runtime `runsc-gpu` with
  `--nvproxy=true --nvproxy-docker=true --platform=systrap --nvproxy-allowed-driver-capabilities=compute,utility,video`.
- **Primary references:** gVisor `pkg/sentry/devices/nvproxy` (Go handlers),
  `pkg/abi/nvgpu` (struct defs), NVIDIA open-kernel-modules (ABI source of truth).
- **Repo structure target:** `crates/` (nv-abi, portal-orchestrator, portal-agent),
  `vmm/`, `guest-driver/`, `pipeline/` (Python PorTAL), `spikes/`.
