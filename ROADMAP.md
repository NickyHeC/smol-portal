# smol-portal — Roadmap

## Vision

Run Ramp Labs' **PorTAL** fine-tuning process **securely** inside
**smolvm microVMs** on one or more **local NVIDIA GPUs**.

We contribute **CUDA support to smolvm** (via a fork → upstream PRs) using
**Driver-API remoting over vsock** — smolvm's existing communication channel.
smol-portal is the **PorTAL orchestration layer** built on top of CUDA-enabled
smolvm.

**Guiding principle:** keep it simple (distributed-systems discipline). Every
component should be independently testable, stateless where possible, and
idempotent. Prefer boring, correct mechanisms over clever ones.

## Architecture — "CUDA is just data"

smolvm already has a guest↔host communication channel (vsock). CUDA operations
are **another protocol on that wire**, like networking. The guest never talks
to a GPU directly — it sends RPC requests over vsock; the host smolvm process
executes them on the real NVIDIA GPU via `libcuda.so`.

From the NVIDIA driver's perspective, each smolvm instance is just **another
host process** — like a Chrome tab using GPU for video decode. The driver
handles scheduling, memory management, and OOM (out-of-memory errors) natively.
smolvm does not need to build its own GPU scheduler or VRAM accounting.

```text
┌─ Guest VM ────────────────────────┐     ┌─ Host (smolvm process) ──────────┐
│  PyTorch / PorTAL pipeline        │     │  smolvm-cuda server              │
│    │                              │     │    │                             │
│  CUDA shim (libcudart substitute) │     │  GpuBackend (dlopen libcuda.so) │
│    │                              │     │    │                             │
│  smolvm-cuda Client               │     │  Real NVIDIA driver → GPU HW    │
│    └── vsock port 7000 ──────────────►──┘                                 │
└───────────────────────────────────┘     └──────────────────────────────────┘
                                                       ▲
                                          Other smolvm VMs, Chrome, etc.
                                          all share the GPU as host processes.
```

### Why not ioctl forwarding (nvproxy / virtio-gpu-nv)?

We initially planned to port gVisor's nvproxy (kernel-level ioctl forwarding
with a guest kernel module and virtio device). After discussion with the smolvm
author (@binsquare), we chose **API-level remoting** instead:

- **Fits smolvm's architecture.** smolvm already remotes Vulkan (Venus) and
  networking over virtio/vsock channels. CUDA is the same pattern.
- **No guest kernel module.** No `virtio_gpu_nv.ko`, no SHM BAR, no nv-abi
  ioctl struct port. The guest runs a thin userspace shim.
- **No GPU management in the VMM.** The host NVIDIA driver handles scheduling,
  OOM, and multi-process sharing. smolvm stays simple.
- **Preserves smolvm properties.** Fast boot, fork/snapshot, elastic memory,
  cross-platform host (macOS HVF + Linux KVM) all remain unchanged. GPU state
  lives host-side and doesn't interfere.

The tradeoff: the guest cannot run unmodified NVIDIA userland. It needs a
**CUDA shim** that marshals calls over vsock. This is incremental work — add
RPC methods as workloads need them — rather than a one-shot ioctl port.

## Scope decisions (locked)

- **smolvm fork → upstream PRs.** CUDA contributions target smol-machines/smolvm;
  smol-portal depends on a CUDA-enabled smolvm build.
- **Path A: Driver-API remoting over vsock.** Not ioctl forwarding (Path B).
  Confirmed with @binsquare.
- **Guest shim in-tree** (in the smolvm fork, `crates/smolvm-cuda-shim`),
  kept small and focused. Split out if it grows complex.
- **Multi-VM on one GPU: yes.** NVIDIA driver handles scheduling and OOM.
  smolvm does not serialize CUDA RPCs between VMs.
- **Device selection: TBD.** Either delegate to the NVIDIA scheduler (loose) or
  pin a specific GPU per VM via Smolfile (strict, stronger guarantees). Pin is
  preferred for PorTAL orchestration.
- **Small, concise PRs.** One concern per PR; grow the protocol incrementally.

## The two repos

- **smolvm fork** (you/smolvm, `cuda-pytorch` branch): extend `smolvm-cuda`
  protocol + host handlers, add guest CUDA shim, wire `cuda = true` in Smolfile,
  e2e tests. Submit PRs upstream to smol-machines/smolvm.
- **smol-portal** (this repo): PorTAL Python pipeline, Rust orchestrator,
  worker Smolfiles/images. Depends on CUDA-enabled smolvm; does not own VMM,
  guest drivers, or GPU management code.

## Security model

- **CPU / memory / filesystem:** full KVM VM isolation (hypervisor boundary).
- **GPU path:** process-level isolation, not hardware isolation. Guest CUDA
  calls execute as the smolvm host process — the NVIDIA driver treats them like
  any other host process. Guests sharing a GPU can see each other's impact on
  VRAM and compute scheduling (same as multiple programs on a desktop).
- **Accepted trade-off:** sufficient for running your own PorTAL jobs; not a
  multi-tenant GPU sandbox. The isolation boundary is the VM for CPU/memory/disk
  and the host process boundary for GPU.

## Repo structure (target)

```text
you/smolvm (fork)                     smol-portal/ (this repo)
├── crates/smolvm-cuda/                ├── ROADMAP.md
│   ├── src/proto.rs   (extend)        ├── reference-material.md
│   ├── src/host.rs    (extend)        ├── pipeline/
│   ├── src/host/gpu.rs               │   └── portal/          # PorTAL Python
│   └── src/client.rs  (extend)        ├── crates/
├── crates/smolvm-cuda-shim/ (NEW)     │   └── portal-orchestrator/
│   └── libcudart shim for guest       ├── images/
├── crates/smolvm-cuda-guest/          │   └── portal-worker/   # OCI rootfs
│   └── e2e test binary                ├── Smolfiles/
├── crates/smolvm-smolfile/            │   └── portal-worker.toml
│   └── cuda = true support            └── spikes/
└── ...                                    └── phase0/
```

## Phased plan

Two tracks run in parallel. Track 1 (smolvm CUDA) is prerequisite for
Track 2 Phase B, but Track 2 Phase A (bare-metal PorTAL) starts immediately.

### Track 1 — smolvm CUDA contributions (fork → PRs)

#### Phase 0 — Validate existing smolvm-cuda on Spark (days)
- Run `cargo run -p smolvm-cuda --example gpu_loopback` on DGX Spark (GB10).
- Run `smolvm-cuda-guest` inside a smolvm VM → expect `SMOLVM-CUDA-OK`.
- Bare-metal trace: run tiny LoRA finetune, capture which `cuda*` / `cu*` calls
  PyTorch actually makes → this becomes the RPC backlog.
- **DoD:** `SMOLVM-CUDA-OK` on GB10; CUDA API call list captured.
- **Kill criterion:** if GB10's driver can't `dlopen` for `GpuBackend`, debug
  before proceeding.

#### Phase 1 — Wire `cuda = true` in Smolfile (PR #1)
- `cuda = true` in Smolfile → smolvm starts vsock CUDA server on VM boot.
- Optional: `cuda_device = N` for GPU pinning.
- **DoD:** `smolvm machine run --cuda ... -- echo ok` starts with CUDA channel.

#### Phase 2 — Runtime API shim for PyTorch (PRs #2–4)
- Add Runtime API RPC ops to `proto.rs` + `host.rs`: `cudaSetDevice`,
  `cudaMalloc`, `cudaFree`, `cudaMemcpy`, `cudaLaunchKernel`, streams.
- Guest shim (`crates/smolvm-cuda-shim`): `libcudart.so` replacement exporting
  the symbols PyTorch needs, forwarding to `smolvm-cuda` Client over vsock.
- **DoD:** `python -c "import torch; print(torch.cuda.is_available())"` → `True`
  inside a smolvm guest.

#### Phase 3 — Training step in guest (PRs #5+)
- Expand protocol as LoRA test hits missing symbols (cuBLAS, cuDNN paths).
- **DoD:** 5-step LoRA finetune completes inside `smolvm machine run --cuda`.

#### Phase 4 — Harden + upstream (ongoing)
- Error handling, versioning, perf profiling.
- Multi-GPU device routing.
- Submit PRs to smol-machines/smolvm.

### Track 2 — smol-portal (PorTAL product)

#### Phase A — PorTAL pipeline, bare metal (weeks) — starts now
- Implement PorTAL: hypernetwork → base-agnostic task latent → slim converter →
  eval, single GPU, bare metal on Spark (reference: HypeLoRA + PEFT).
- Reproduce headline result: port Qwen3→Gemma-3 recovering ~94–98% of LoRA
  accuracy at ~half the calibration data on a small task.
- Freeze CLI contracts + artifact formats (latents, adapters, eval JSON).
- **DoD:** `portal port --from qwen3 --to gemma3 --task X` works end-to-end.

##### Phase A implementation plan

**Stack:** Python 3.11+, uv + pyproject.toml, typer CLI, HuggingFace
(Transformers, PEFT, Datasets, Safetensors), PyTorch. Dev models: small
variants (Qwen3-0.5B, Gemma-3-1B) for fast iteration, scale up on Spark.

**Step 1 — Project scaffolding & CLI skeleton** ✅
- `pipeline/portal/` Python package with `pyproject.toml` (uv + hatchling).
- `portal` CLI entry point (typer): subcommands `train`, `extract`, `convert`,
  `eval`, `port`.
- Argument contracts frozen:
  - `portal port --from <model> --to <model> --task <name> --dataset <hf_id>`
  - `portal train --model <model> --task <name> --dataset <hf_id> [--rank, --epochs, --lr, ...]`
  - `portal extract --adapter-dir <path> --model <model> --task <name>`
  - `portal convert --latent-dir <path> --target <model> --task <name>`
  - `portal eval --adapter-dir <path> --model <model> --task <name> --dataset <hf_id>`

**Step 2 — Artifact format specification** ✅
- Content-addressed on disk: `{output_dir}/{task_name}/{kind}_{sha256[:16]}/`
- **Task latents:** `task_latent.safetensors` + `task_latent_meta.json`
  (source_model, task_name, latent_dim, config hash, timestamp).
- **Adapters:** PEFT-compatible directory (`adapter_model.safetensors` +
  `adapter_config.json`) inside an `adapter/` subfolder.
- **Eval results:** `eval_results.json` (config, metrics dict, timestamp).
- `find_artifact()` for idempotent cache-hit lookups (same config → same dir).

**Step 3 — Source LoRA training (`portal train`)**
- Train a standard LoRA adapter on the source model (Qwen3-0.5B) for a task.
- HuggingFace PEFT + Transformers Trainer, single-GPU, bf16.
- Dataset: a HuggingFace benchmark (e.g. classification or instruction-following
  subset). Pick something small and reproducible for dev.
- Output: PEFT adapter saved as content-addressed artifact.
- Validate: adapter loads back, loss decreases over epochs.

**Step 4 — Hypernetwork & task latent extraction (`portal extract`)**
- LoRA autoencoder: flatten all LoRA weight matrices (A, B per layer) →
  encoder → compact task latent (z) → decoder → reconstruct.
- Architecture: MLP encoder/decoder (configurable depth/width), MSE
  reconstruction loss. Reference: HypeLoRA (arXiv:2603.19278).
- Train the autoencoder on the single adapter's weights (overfit is fine —
  we want a faithful compression, not generalization at this stage).
- Output: `task_latent.safetensors` — a single vector of `latent_dim` (default
  256) that captures the task-specific information base-agnostically.

**Step 5 — Slim converter (`portal convert`)**
- `LatentToLoraConverter` MLP: maps task latent → flattened target LoRA weights.
- Training loop: inject predicted weights into target PeftModel, compute
  cross-entropy loss on calibration data, backprop through the converter.
- Calibration data: small subset (~256 examples) from the task dataset,
  tokenized for the target model.
- Output: target PEFT adapter saved as content-addressed artifact.
- This is the core novel piece — the converter learns to "project" the
  base-agnostic latent into the target model's weight space.

**Step 6 — Evaluation (`portal eval`)**
- Load target base model + generated LoRA adapter.
- Compute perplexity / loss on a held-out split of the task dataset.
- Compare against baselines: (a) direct LoRA on target (upper bound),
  (b) no adapter (lower bound).
- Output: `eval_results.json` with loss, perplexity, sample count.
- Target: recover ~94–98% of direct LoRA accuracy.

**Step 7 — End-to-end `portal port` wiring**
- Orchestrate: train → extract → convert → eval in one command.
- `--skip-train` + `--source-adapter-dir` for reuse of existing source LoRA.
- Verify idempotency: same inputs → same content-addressed artifacts.
- All 4 steps print progress via rich console.

**Step 8 — Reproduce headline result & freeze**
- Full pipeline: Qwen3-0.5B → Gemma-3-1B on chosen task.
- Validate accuracy target (94–98% of direct LoRA on Gemma-3).
- If dev-scale result holds, scale to larger models on Spark.
- Freeze CLI args, artifact formats, eval JSON schema.
- Update this roadmap with results and any discovered constraints.

#### Phase B — Single-VM integration (after Track 1 Phase 3)
- Package Phase A pipeline as OCI image / Smolfile with CUDA shims preinstalled.
- Run full `converter-fit + eval` inside one CUDA-enabled smolvm VM.
- ```bash
  smolvm machine run --cuda -s Smolfiles/portal-worker.toml \
    --volume ./artifacts:/artifacts \
    -- portal port --from qwen3 --to gemma3 --task X
  ```
- **DoD:** Phase A result reproduced inside the VM within acceptable overhead.

#### Phase C — Multi-GPU orchestration (weeks)
- `portal-orchestrator` fans out N `convert+eval` jobs via
  `smolvm machine run --cuda --cuda-device $GPU_ID`.
- Stateless, idempotent, content-addressed artifacts, retryable jobs.
- One job per GPU recommended; multi-VM same GPU works (NVIDIA driver
  handles scheduling and OOM) but gives weaker perf guarantees.
- **DoD:** "new base model dropped → one command ports every task across all
  local GPUs."

## Collaboration

- **@binsquare (smolvm author):** architecture confirmed (Path A, vsock RPC).
  Reviews PRs. Consult on Smolfile `cuda` flag shape, device selection, guest
  shim placement.
- **nestrilabs/virtio-gpu-nv:** design reference only (we chose Path A, not
  their ioctl-forwarding approach). May revisit if Path A hits fundamental
  limits.
- **gVisor nvproxy:** background reference for understanding NVIDIA driver
  behavior, not a porting target.

## Design decisions log

| Decision | Chosen | Alternative considered | Rationale |
|---|---|---|---|
| GPU approach | Path A: vsock API remoting | Path B: ioctl forwarding (nvproxy) | Fits smolvm architecture; no guest kernel module; Bin's preference |
| GPU scheduling | Delegate to NVIDIA driver | Custom smolvm GPU scheduler | "NVIDIA is much better at scheduling and handling OOM" — Bin |
| Multi-VM same GPU | Yes, process-level sharing | Hardware partitioning (MIG/SR-IOV) | Natural in Path A; Chrome tab analogy |
| Guest shim | In-tree (smolvm repo) | Separate repo | Simpler if kept small; split later if needed |
| Device selection | Pin OR delegate (TBD) | Always pin | Pinning gives stronger guarantees for orchestrator |
| PR strategy | Small and concise | Large feature PRs | Bin's preference |
| Repo split | smolvm fork + smol-portal | Single monorepo | smol-portal doesn't own VMM code |

## Risks & mitigations

| Risk | Mitigation |
|---|---|
| API-remoting can't cover full PyTorch surface | Incremental: trace bare-metal calls, add RPC ops as needed |
| Shim maintenance as CUDA versions evolve | Shim wraps Driver API (stable); Runtime API surface is smaller than ioctl ABI |
| vsock latency on hot CUDA paths | Profile in Phase 3; batch small ops if needed |
| Multi-VM OOM on shared GPU | NVIDIA driver returns clean errors; orchestrator pins one job per GPU |
| PorTAL recipe unreproducible (no public code) | Phase A validates bare-metal before VM integration |
| smolvm upstream rejects PRs | Fork works independently; keep PRs small to ease review |
| GB10 / Blackwell / arm64 edge cases | Phase 0 validates on actual hardware before protocol work |

## Immediate next steps

1. **Fork smolvm** → `cuda-pytorch` branch.
2. **Phase 0 on Spark:** run `gpu_loopback`, run `smolvm-cuda-guest` in VM,
   capture bare-metal PyTorch CUDA API trace.
3. **Phase A in parallel:** start PorTAL pipeline on bare metal.
4. **First PR draft:** wire `cuda = true` in Smolfile → start vsock server.

---
_Status: planning. Architecture confirmed with smolvm author. No implementation
yet. Update this spec as prototyping reveals new constraints._
