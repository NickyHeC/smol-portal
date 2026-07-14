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
  Re-validated on **smolvm v1.5.2** (2026-07-13) with fused SDPA optional.
  See [`examples/smolvm/README.md`](examples/smolvm/README.md) and `memory.md`.
- **smolvm constraints discovered** (see Phase B notes + smolvm #596–#598):
  - fp32 weights + `device_map="cuda"` (bulk `model.to("cuda")` fails on shim).
  - **Math SDPA** required on smolvm &lt;1.5.2; on **v1.5.2+** fused SDPA works
    (`PORTAL_SKIP_CUDA_SMOLVM=1` or default workaround still safe).

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

**Step 6 — Evaluation (`portal eval` + `portal baseline`)**
- Load target base model + generated LoRA adapter.
- Compute perplexity / loss on a held-out split of the task dataset.
- Compare against baselines: (a) direct LoRA on target (upper bound) — now via
  `portal baseline`; (b) no adapter (lower bound) — eval the base model with no
  adapter. ⚠️ These comparisons are **not yet automated inside `portal port`**;
  run `portal baseline` separately and compare the eval JSONs (see Phase A2).
- Output: `eval_results.json` with loss, perplexity, sample count.
- ⚠️ Metrics are currently **loss/perplexity only** — task-specific accuracy/F1
  is Phase A2 work. The "~94–98% of direct LoRA accuracy" figure is an
  aspirational target from Ramp's writeup, **not** a measured result here.

**Step 7 — End-to-end `portal port` wiring** ✅ (smolvm CUDA smoke, 2026-07-13)
- Orchestrate: train → extract → convert → eval in one command.
- `--skip-train` + `--source-adapter-dir` for reuse of existing source LoRA.
- **Validated on Lambda A10 / smolvm v1.5.2:** full pipeline with tiny Llama,
  fused SDPA, smoke-sized hypernet/converter epochs (`port e2e ok`).

**Step 8 — Reproduce headline result & freeze**
- Full pipeline: Qwen3-0.5B → Gemma-3-1B on chosen task.
- Validate accuracy target (94–98% of direct LoRA on Gemma-3).
- If dev-scale result holds, scale to larger models on Spark.
- Freeze CLI args, artifact formats, eval JSON schema.
- Update this roadmap with results and any discovered constraints.

### Phase A2 — Scientific validation (design the experiment that can disprove the claim)

The systems path is proven: `train → extract → convert → eval` completes on real
models inside CUDA smolvm. What is **not** proven is the PorTAL *mechanism* — that
the task latent is model-independent and that porting recovers most of a direct
LoRA's accuracy at a fraction of the cost. Two structural facts make this the
priority, not more infrastructure:

- **The latent may not matter.** The autoencoder is trained on a *single*
  adapter vector (`hypernetwork.py`, `unsqueeze(0)`), and the converter sees one
  *fixed* latent `z` every step. So the converter could be learning the task
  directly from calibration data and ignoring the latent entirely.
- **There is no measured comparison.** Eval reports perplexity only; there was
  no direct-LoRA baseline and no task metric, so "94–98% of direct LoRA" has
  nothing behind it in this repo.

> **Guiding rule for this phase:** design runs that can *disprove* our own claim.
> "It completed without error" is not "the mechanism worked."

> **Ground truth now exists.** Ramp published the official implementation,
> [`portallib`](https://github.com/ramp-public/portallib), and the
> [`portallib-tasks`](https://huggingface.co/datasets/RampPublic/portallib-tasks)
> 14-task suite. Their validation metric is **`acc_norm`** (continuation log-prob
> normalized by character length), *not* perplexity. Phase A2's task-metric work
> should target `acc_norm` on that suite so our numbers are directly comparable,
> and we should check our hypernetwork/converter assumptions against portallib
> rather than the announcement alone. See `reference-material.md`.

#### Build plan

**Landed now (local, no GPU — see PR / this branch):**
- `portal/data.py` — explicit dataset text resolution + `DatasetSchemaError`;
  wired into train/convert/eval. Unknown schemas now fail loudly instead of
  training on empty strings.
- `portal convert` — `--cal-dataset` is **required** (no more silent fallback to
  the target-model id, which was never a dataset). `portal port` still sets it
  from `--dataset` automatically.
- `portal/env.py` — runtime provenance manifest (torch/transformers/peft/…
  versions, git commit, platform) embedded in every artifact's metadata,
  **excluded from the content hash** so reruns stay idempotent.
- `--latent-mode {real|zero|random|shuffled}` on `portal convert` — the ablation
  knob for the "does the latent matter?" experiment.
- `portal baseline` — trains + evals a direct LoRA on the *target* model: the
  comparison point the port result is measured against.

**Next (needs Lambda GPU — run in the next several days):**
1. **Latent-matters ablation** (highest priority). Same task/target/calibration,
   sweep `--latent-mode` over `real|zero|random|shuffled`. If post-training eval
   is ~equal across modes, the converter is ignoring the latent → the
   portability mechanism is not yet doing anything.
2. **Baseline comparison.** `portal baseline` (direct LoRA on target) vs
   `portal port` (ported adapter), same split/samples. Report the ratio.
3. **Task-specific metrics.** Add accuracy/F1 (classification) / exact-match /
   token-F1 to `portal eval` alongside perplexity; wire the labeled path. Only
   then can any "% of direct LoRA" number be stated.
4. **`portal port` sizing parity.** Fold `port_e2e.py`'s smoke knobs
   (sample/seq/epoch/rank) into `portal port` so the CLI can drive real
   experiments (SPEC §3 known gap).
5. **Cross-task converter reuse.** Save the `LatentToLoraConverter` as a reusable
   artifact; train one converter over several task latents, hold one out. This is
   the first real test of amortization/portability.
6. **Converter output-head scaling.** The final dense layer emits *all* target
   LoRA params at once (`hidden × total_lora_params`) — does not scale to large
   models. Prototype layer-wise / factorized generation before scaling up.

#### Test plan

**CPU unit tests (local, landed — `uv run python -m pytest -q`, 27 passing):**
- `test_data.py` — field precedence (`text`→`input`→…), instruction/response,
  chat `messages`, and loud failure on unknown schemas.
- `test_latent_mode.py` — `real` identity; `zero`; `random` deterministic &
  differs; `shuffled` is a permutation.
- `test_env.py` — manifest shape + JSON-serialisable + tracks key packages.
- `test_artifacts.py::test_latent_records_runtime_but_hash_excludes_it` —
  provenance recorded, content address unchanged.
- `test_converter.py` — `functional_call` keeps the converter in the autograd
  graph (regression guard for the detach bug).

**Lambda GPU validation (next session — extend `examples/smolvm/port_e2e.py`):**
- `ppl(real) < ppl(base)` — the ported adapter beats the unadapted target.
- `ppl(real) ≈ ppl(zero|random|shuffled)?` — the decisive ablation. Want `real`
  meaningfully better; if not, the latent is inert.
- `ppl(port) vs ppl(baseline)` — record the recovery ratio (no accuracy claim
  until task metrics land).
- Determinism: fixed seed + pinned inputs → same eval within tolerance.
- Loud-failure smoke: a dataset with no known text field raises
  `DatasetSchemaError`; `convert` without `--cal-dataset` raises before download.

### Phase B — Single-VM integration (in progress — CUDA training validated)

Requires CUDA-enabled smolvm with PyTorch working inside a guest VM.

- ✅ **CUDA training smoke inside smolvm** (2026-07-12/13, Lambda A10): `portal train`
  LoRA on tiny Llama completes through the `--cuda` remoting path (v1.5.2).
- ✅ **Worker image + Smolfile checked in:** [`examples/smolvm/`](examples/smolvm/)
  (`Dockerfile.portal-cuda`, `portal.smolfile`, README with full run log).
- ✅ **Full `portal port` e2e** (2026-07-13): train → extract → convert → eval on smolvm v1.5.2.
- ✅ **Real-model `portal port` inside VM** (2026-07-13, ~1 h): Qwen3-0.6B → TinyLlama-1.1B on IMDB (`port e2e ok`). Gemma-3 gated — retry with `HF_TOKEN`.

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
| [#597](https://github.com/smol-machines/smolvm/issues/597) fused SDPA backward | `loss.backward()` → `CUDA error: invalid argument` on **v1.5.0** | **Fixed in v1.5.2** (issue closed). Math SDPA workaround still default in `portal.cuda`; use `PORTAL_SKIP_CUDA_SMOLVM=1` on 1.5.2+ |
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

1. **Phase A2 code (local, done):** dataset schema guard, required `--cal-dataset`,
   runtime manifest, `--latent-mode` ablation, `portal baseline`. All CPU-unit-tested.
2. **Phase A2 on Lambda (next several days):** run the latent-matters ablation
   (`real` vs `zero|random|shuffled`) and the `portal baseline` vs `portal port`
   comparison. This is the priority — it tells us whether the mechanism works.
3. **Step 8:** Scale Qwen → target (Gemma with `HF_TOKEN` or TinyLlama) for
   accuracy, not just pipeline smoke — after task metrics land.
4. **Upstream tracking:** smolvm #596 / #598 — drop manual shim install and
   pre-baked-image docs when PRs land.

---
_Status: Systems path (Phase B) complete on smolvm v1.5.2 — real-model `portal port`
+ fused SDPA e2e (2026-07-13). **The PorTAL mechanism itself is not yet validated**
(latent may be inert; no measured baseline). Next: Phase A2 scientific validation on
Lambda, then Gemma/accuracy scaling and multi-GPU orchestration (Phase C)._
