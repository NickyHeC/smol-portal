# Reference Material

Curated learning materials for **smol-portal**: PorTAL orchestration on
CUDA-enabled smolvm microVMs, using Driver-API remoting over vsock.

Ordered roughly by when you'll need it. Each entry says *why it matters*
so you can skim strategically.

---

## 0. smolvm — the platform we're extending

smolvm is the microVM runtime we contribute CUDA support to. Understand its
architecture, crate layout, and dev workflow before writing protocol code.

- **smolvm repo** — https://github.com/smol-machines/smolvm
  Read `AGENTS.md` (full CLI reference, platform support, persistence model),
  `docs/DEVELOPMENT.md` (build with `cargo make`, agent rootfs, tests),
  and `README.md` (user-facing examples).
- **smolvm Cargo workspace** — the crate layout we extend:
  - `crates/smolvm-cuda/` — existing CUDA Driver-API remoting (proto, client,
    host, GpuBackend). **Start here.** This is the code we grow.
  - `crates/smolvm-cuda-guest/` — guest-side e2e test (vector-add over vsock).
  - `crates/smolvm-protocol/` — vsock ports, including `ports::CUDA = 7000`.
  - `crates/smolvm-smolfile/` — Smolfile parsing (`gpu`, `gpu_vram` fields;
    we add `cuda`, `cuda_device`).
  - `crates/smolvm-agent/` — in-VM agent (vsock listener, process exec).
- **smol-machines/libkrun** — smolvm's libkrun fork (submodule). The VMM layer.
  We don't modify it for Path A; vsock is already wired.

## 1. Existing smolvm-cuda code (read first)

The existing crates are the foundation. Read them before writing new code.

- **`smolvm-cuda/src/proto.rs`** — wire protocol: framing, request/response
  codec for Driver API ops (`cuInit`, `cuMemAlloc`, `cuLaunchKernel`, etc.).
- **`smolvm-cuda/src/host.rs`** — host-side dispatch: `Backend` trait,
  `serve()` loop, `CpuBackend` (emulation for testing).
- **`smolvm-cuda/src/host/gpu.rs`** — `GpuBackend`: real CUDA via `dlopen`
  of `libcuda.so`. No CUDA toolkit needed on host — just the driver.
- **`smolvm-cuda/src/client.rs`** — guest-side `Client`: marshals `cu*` calls
  over any `Read`/`Write` stream.
- **`smolvm-cuda/examples/gpu_loopback.rs`** — loopback test exercising the
  full stack (client → proto → serve → GpuBackend) in one process.
- **`smolvm-cuda-guest/src/main.rs`** — vector-add proof inside a real guest
  VM (vsock to host). Prints `SMOLVM-CUDA-OK` on success.

> Read `proto.rs` + `host/gpu.rs` before adding new RPC methods. The existing
> patterns (request encoding, Backend trait, handle passing) are the template
> for every new operation you add.

## 2. CUDA APIs (what we're remoting)

CUDA has two main API layers. smolvm-cuda currently remotes the **Driver API**;
PyTorch mostly calls the **Runtime API**. Understanding both is essential for
building the guest shim.

- **CUDA Driver API** — https://docs.nvidia.com/cuda/cuda-driver-api/
  Low-level (`cu*` prefix): `cuInit`, `cuCtxCreate`, `cuMemAlloc`,
  `cuModuleLoadData`, `cuLaunchKernel`. What `GpuBackend` calls today.
- **CUDA Runtime API** — https://docs.nvidia.com/cuda/cuda-runtime-api/
  Higher-level (`cuda*` prefix): `cudaMalloc`, `cudaMemcpy`,
  `cudaLaunchKernel`, `cudaStreamCreate`. What PyTorch calls. Built on top
  of the Driver API. The guest shim will implement these by translating to
  Driver API RPCs.
- **CUDA C Programming Guide** — https://docs.nvidia.com/cuda/cuda-c-programming-guide/
  Mental model for contexts, streams, events, memory. Read the "Driver API"
  chapter for the mapping between Runtime and Driver calls.
- **Relationship:** Runtime API wraps Driver API. `cudaMalloc` → `cuMemAlloc`.
  `cudaLaunchKernel` → `cuLaunchKernel`. The shim can either implement
  Runtime calls directly as new RPC ops, or translate to existing Driver ops
  client-side. Both work; pick per-operation based on simplicity.

## 3. Prior art in CUDA API remoting (learn from, don't copy)

These projects did the same thing (forward CUDA calls to a remote GPU).
They all hit **maintenance burden** chasing CUDA versions. Study their
architecture and memory-transfer patterns; don't adopt their codebases.

- **qCUDA** — https://github.com/coldfunction/qCUDA
  virtio-based CUDA remoting for QEMU. Closest to our approach. Study its
  guest shim (`libcudart` replacement) and host server architecture.
  ~95% bandwidth efficiency reported.
- **virtio-cuda-module** — https://github.com/juniorprincewang/virtio-cuda-module
  Academic virtio CUDA forwarding. Instructive guest driver patterns.
- **cricket** — https://github.com/RWTH-ACS/cricket
  GPU disaggregation via API interception. Good memory management patterns.
- **rCUDA** — http://www.rcuda.net/
  Closed source, historical. Got stuck at CUDA 9.0 — cautionary tale on
  maintenance cost.
- **WSL2 GPU paravirtualization** — https://learn.microsoft.com/en-us/windows/wsl/tutorials/gpu-compute
  Production proof (~30K LoC) that driver-level GPU proxying works. Different
  approach (kernel-level `/dev/dxg`) but validates the concept.
- **Fly.io GPU discussion (HN)** — https://news.ycombinator.com/item?id=43054504
  Candid engineering trade-offs of various GPU virtualization approaches.

## 4. gVisor nvproxy — background reference (not our approach)

We chose Path A (API remoting), not Path B (ioctl forwarding). nvproxy is
still useful background for understanding NVIDIA driver behavior and what
the driver stack looks like from below the CUDA API.

- **gVisor GPU user guide** — https://gvisor.dev/docs/user_guide/gpu/
- **`pkg/sentry/devices/nvproxy`** — https://github.com/google/gvisor/tree/master/pkg/sentry/devices/nvproxy
- **nvproxy design notes** — https://github.com/google/gvisor/blob/master/pkg/sentry/devices/nvproxy/nvproxy.go
  Top-of-file comments explain the ioctl forwarding mental model.
- **`pkg/abi/nvgpu`** — https://github.com/google/gvisor/tree/master/pkg/abi/nvgpu
  NVIDIA ioctl struct definitions. Not needed for Path A, but helpful if
  debugging driver behavior.

## 5. virtio-gpu-nv — design reference only

A design-only repo (no code) describing ioctl-forwarding architecture for
NVIDIA in libkrun. We chose a different approach but the design doc is a
good mental model for how GPU virtualization *could* work at the driver level.

- **Repo** — https://github.com/nestrilabs/virtio-gpu-nv
  Read `ARCHITECTURE.md` for the ioctl dispatch model and `README.md` for
  the rationale comparing Venus, VFIO, and native-context approaches.

## 6. smolvm internals (deeper reading for contributions)

- **libkrun** — https://github.com/containers/libkrun
  The upstream microVM library smolvm forks. Relevant for understanding
  vsock wiring, virtio device model, and build system. We don't modify
  libkrun for Path A — vsock already exists.
- **Enabling GPU on macOS (Sergio López)** — https://sinrega.org/2024-03-06-enabling-containers-gpu-macos/
  Explains why smolvm's GPU support uses API remoting (Venus). Our CUDA
  approach follows the same philosophy.
- **muvm** — https://github.com/AsahiLinux/muvm
  libkrun-based microVM for Apple/AMD GPUs. Same structural idea.
- **Venus (Vulkan-over-virtio)** — https://docs.mesa3d.org/drivers/venus.html
  smolvm's existing GPU acceleration (Vulkan). Our CUDA channel is the
  analog for compute workloads.

## 7. The workload: PorTAL, LoRA, hypernetworks

What actually runs inside the VMs.

- **Ramp Labs — PorTAL paper** (Geist, 2026) — https://x.com/RampLabs/status/2072381992285647280
  *Portable Task Adapters.* Learns a **base-agnostic task latent** `z_t` (dim 256)
  and a hypernetwork decoder `D_b` = a **shared base-agnostic core** + a **thin
  per-base converter**, FiLM-conditioned (`z_t` scales/shifts a trunk fed per-layer
  embeddings `e_ℓ`), trained end-to-end on gold-continuation NLL. To **port** to an
  unseen base, freeze `z_t` + core and refit only the thin converter
  (`{e_ℓ, P_in, P_out}`) on a small calibration set. Results, as *recovered lift*
  = (acc_m − acc_b)/(acc_L − acc_b): **~98%** on unseen Qwen3-8B (within-family)
  and **~94%** on Gemma-3-4B (cross-family), vs Cross-LoRA ~14%; matches
  from-scratch LoRA at ~half the calibration data. Builds on Sakana's Text-to-LoRA
  (hypernetwork LoRA generation) but adds cross-base portability. Primary-source
  PDF saved at `~/Desktop/X.pdf`.
- **⭐ Ramp — `portallib` (official open-source implementation)** —
  https://github.com/ramp-public/portallib (public 2026-07-13)
  The real reference implementation. **Use it to check our reimplementation's
  assumptions** (hypernetwork/converter design, calibration protocol, baselines)
  rather than reverse-engineering from the announcement. smol-portal is *not* a
  fork — it's our own smolvm-hosted orchestration — but portallib is now the
  ground-truth recipe.
- **Ramp — `RampPublic/portallib-tasks` (dataset)** —
  https://huggingface.co/datasets/RampPublic/portallib-tasks
  The 14-task multiple-choice suite used by portallib (129,212 train / 19,548
  val). Rows: `task`, `prompt`, `choices`, `gold_idx`. **Validation metric is
  `acc_norm`** — continuation log-prob normalized by character length — *not*
  perplexity. This is the metric our Phase A2 task-metric work should target.
- **Sakana AI — Text-to-LoRA** — the technique PorTAL builds on: a hypernetwork
  that generates task LoRAs from a text description. PorTAL's contribution is
  making the generated adapter portable across (unseen) base models.
- **Ramp Labs — Building with Tinker** — https://ramplabs.substack.com/p/building-with-tinker
  Context on their fine-tuning tooling and async reward-function workflow.
- **LoRA (Hu et al., 2021)** — https://arxiv.org/abs/2106.09685
  The base technique PorTAL makes portable.
- **HypeLoRA (hypernetwork-generated LoRA)** — https://github.com/btrojan-official/HypeLoRA
  (paper: arXiv:2603.19278) — closest public analog to PorTAL's hypernetwork.
  Generates per-layer LoRA factors from a learned embedding.
- **HuggingFace PEFT** — https://github.com/huggingface/peft
  Practical LoRA/adapter implementations to build the pipeline on.

## 8. Reference VMMs (design context, not our build target)

We are not building a VMM — smolvm is the VMM. These are useful for
understanding the design space if you want deeper context.

- **Firecracker** — https://github.com/firecracker-microvm/firecracker
- **Cloud Hypervisor** — https://github.com/cloud-hypervisor/cloud-hypervisor
- **rust-vmm** — https://github.com/rust-vmm

---

## Suggested reading order

1. smolvm `AGENTS.md` + `docs/DEVELOPMENT.md` (Section 0) — understand the tool
2. `smolvm-cuda` crate: `proto.rs`, `host/gpu.rs`, `client.rs` (Section 1) — the code you extend
3. CUDA Driver API + Runtime API docs (Section 2) — what you're remoting
4. qCUDA guest shim architecture (Section 3) — prior art for the shim pattern
5. Phase 0 hands-on: run `gpu_loopback` on Spark, trace bare-metal PyTorch
6. PorTAL / HypeLoRA / PEFT (Section 7) — the ML pipeline
