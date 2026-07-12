# smol-portal — Roadmap

## Vision

Run Ramp Labs' **PorTAL** fine-tuning process **securely** inside
**smolvm microVMs** on one or more **local NVIDIA GPUs**.

smol-portal is the **PorTAL orchestration layer**: Python ML pipeline, Rust
multi-GPU orchestrator, and worker Smolfiles/images. It depends on a
CUDA-enabled smolvm build but does not own VMM, guest drivers, or GPU
management code — that work lives in the [smolvm](https://github.com/smol-machines/smolvm) repo.

**Guiding principle:** keep it simple (distributed-systems discipline). Every
component should be independently testable, stateless where possible, and
idempotent. Prefer boring, correct mechanisms over clever ones.

## Architecture

```text
┌─ Guest VM ────────────────────────┐     ┌─ Host (smolvm process) ──────────┐
│  PyTorch / PorTAL pipeline        │     │  smolvm-cuda server              │
│  CUDA shim (libcudart substitute) │     │  GpuBackend → NVIDIA driver      │
│  smolvm-cuda Client               │     │                                  │
│    └── vsock port 7000 ──────────────►──┘                                 │
└───────────────────────────────────┘
```

The PorTAL pipeline runs inside the guest VM. smol-portal packages it,
defines artifact formats, and (in Phase C) fans out jobs across GPUs via
`smolvm machine run --cuda`.

## Security model

- **CPU / memory / filesystem:** full KVM VM isolation (hypervisor boundary).
- **GPU path:** process-level isolation, not hardware isolation. Guest CUDA
  calls execute as the smolvm host process — the NVIDIA driver treats them like
  any other host process. Guests sharing a GPU can see each other's impact on
  VRAM and compute scheduling (same as multiple programs on a desktop).
- **Accepted trade-off:** sufficient for running your own PorTAL jobs; not a
  multi-tenant GPU sandbox.

## Repo structure (target)

```text
smol-portal/
├── ROADMAP.md
├── reference-material.md
├── pipeline/
│   └── portal/              # PorTAL Python package (CLI + ML pipeline)
├── crates/
│   └── portal-orchestrator/ # Rust multi-GPU fan-out (Phase C)
├── images/
│   └── portal-worker/       # OCI rootfs (Phase B)
├── Smolfiles/
│   └── portal-worker.toml
└── spikes/
```

## Phased plan

### Phase A — PorTAL pipeline, bare metal (weeks) — in progress

Implement PorTAL: hypernetwork → base-agnostic task latent → slim converter →
eval, single GPU, bare metal on Spark (reference: HypeLoRA + PEFT).

Reproduce headline result: port Qwen3→Gemma-3 recovering ~94–98% of LoRA
accuracy at ~half the calibration data on a small task.

Freeze CLI contracts + artifact formats (latents, adapters, eval JSON).

**DoD:** `portal port --from qwen3 --to gemma3 --task X` works end-to-end.

#### Implementation steps

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

**Step 3 — Source LoRA training (`portal train`)** ✅ (validated on smolvm CUDA)
- Train a standard LoRA adapter on the source model for a task.
- HuggingFace PEFT + Transformers Trainer, single-GPU.
- Output: PEFT adapter saved as content-addressed artifact.
- **Validated 2026-07-12 on Lambda A10 via smolvm `--cuda`:** `portal train` on
  `hf-internal-testing/tiny-random-LlamaForCausalLM`, 8/8 steps, adapter written.
  See [`examples/smolvm/README.md`](examples/smolvm/README.md) and `memory.md`.
- **smolvm constraints discovered** (see Phase B notes + smolvm #596–#598):
  - fp32 weights + `device_map="cuda"` (bulk `model.to("cuda")` fails on shim).
  - Must force **math SDPA** — fused flash/mem-efficient SDPA backward fails on
    the CUDA remoting path. Handled by `portal.cuda.configure_cuda_for_smolvm()`.

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

### Phase B — Single-VM integration (in progress — CUDA training validated)

Requires CUDA-enabled smolvm with PyTorch working inside a guest VM.

- ✅ **CUDA training smoke inside smolvm** (2026-07-12, Lambda A10): `portal train`
  LoRA on tiny Llama completes through the `--cuda` remoting path.
- ✅ **Worker image + Smolfile checked in:** [`examples/smolvm/`](examples/smolvm/)
  (`Dockerfile.portal-cuda`, `portal.smolfile`, README with full run log).
- ☐ Run full `port` (train → extract → convert → eval) inside one CUDA VM.
- ☐ Reproduce a real headline result inside the VM.

```bash
smolvm machine run --net --cuda --mem 16384 -s examples/smolvm/portal.smolfile \
  -- portal port --from qwen3 --to gemma3 --task X
```

- **DoD:** Phase A result reproduced inside the VM within acceptable overhead.

#### smolvm integration constraints (discovered 2026-07-12)

These bit us during the Lambda validation and are tracked upstream. Workarounds
live in `portal.cuda` and `examples/smolvm/`; remove them as upstream lands fixes.

| smolvm issue | Symptom that blocked us | Our workaround |
|---|---|---|
| [#596](https://github.com/smol-machines/smolvm/issues/596) release ships `agent-rootfs` without CUDA shims | `torch.cuda.is_available()==False` (err 801) on stock v1.5.0 tarball | build shims from source, copy into `agent-rootfs/usr/local/lib/smolvm-cuda/` |
| [#597](https://github.com/smol-machines/smolvm/issues/597) fused SDPA backward fails on the shim | `loss.backward()` → `CUDA error: invalid argument` | `portal.cuda.configure_cuda_for_smolvm()` forces math SDPA |
| [#598](https://github.com/smol-machines/smolvm/issues/598) auto-staging is pull-time + `site-packages/nvidia/`-only, undocumented | conda / runtime-`pip install torch` images silently loaded real 109 MB cuBLAS | pre-bake pip torch into `portal-cuda.tar` so wheels exist at pull time |

### Phase C — Multi-GPU orchestration (weeks)

- `portal-orchestrator` (Rust) fans out N `convert+eval` jobs via
  `smolvm machine run --cuda --cuda-device $GPU_ID`.
- Stateless, idempotent, content-addressed artifacts, retryable jobs.
- One job per GPU recommended; multi-VM same GPU works (NVIDIA driver
  handles scheduling and OOM) but gives weaker perf guarantees.
- **DoD:** "new base model dropped → one command ports every task across all
  local GPUs."

## Design decisions log

| Decision | Chosen | Alternative considered | Rationale |
|---|---|---|---|
| ML stack | Python (PyTorch + PEFT) | Rust (tch-rs) | No viable Rust LoRA/PEFT ecosystem |
| Orchestrator | Rust (`portal-orchestrator`) | Python subprocess | Native smolvm integration, typed VM control |
| Repo split | smolvm + smol-portal | Single monorepo | smol-portal doesn't own VMM code |
| Device selection | Pin per VM (TBD) | Delegate to NVIDIA scheduler | Pinning gives stronger guarantees for orchestrator |
| Artifact storage | Content-addressed on disk | Database / object store | Simple, idempotent, independently testable |

## Risks & mitigations

| Risk | Mitigation |
|---|---|
| PorTAL recipe unreproducible (no public code) | Phase A validates bare-metal before VM integration |
| Converter accuracy below target | Iterate on hypernetwork/converter architecture; compare against direct LoRA baseline |
| smolvm CUDA not ready for Phase B | Phase A runs on bare metal independently |
| Multi-VM OOM on shared GPU | NVIDIA driver returns clean errors; orchestrator pins one job per GPU |
| vsock latency on hot CUDA paths | Profile in Phase B; acceptable for training workloads |

## Immediate next steps

1. **Step 4–6:** Extract task latent, convert to target, eval (bare metal + smolvm).
2. **Step 7:** Wire/validate end-to-end `portal port` inside a single CUDA VM (Phase B).
3. **Step 8:** Reproduce headline result and freeze contracts.
4. **Upstream tracking:** watch smolvm #596–#598; drop the `portal.cuda` /
   pre-baked-image workarounds as each fix lands.

---
_Status: Phase A Steps 1–3 done (Step 3 validated on smolvm CUDA, 2026-07-12).
Phase B in progress — single-VM CUDA training smoke passes; `port` e2e pending.
Update this spec as prototyping reveals new constraints._
