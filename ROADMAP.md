# smol-portal — Roadmap

## Vision

Run Ramp Labs' **PorTAL** **securely** inside **smolvm microVMs** on one or more
**local NVIDIA GPUs**.

smol-portal is the **secure-VM connector for PorTAL**: it runs Ramp's official
implementation ([`portallib`](https://github.com/ramp-public/portallib)) as the
**engine** inside CUDA-enabled smolvm microVMs, and adds the packaging, CLI/UX,
artifact plumbing, and multi-GPU orchestration around it. It does **not** own the
ML method (that's Ramp's `portallib`) nor the VMM / GPU remoting (that's
[smolvm](https://github.com/smol-machines/smolvm)). Three layers, one system:

```text
portallib (ML engine — Ramp)  →  smol-portal (connector/orchestration)  →  smolvm (secure GPU infra)
```

**Guiding principle:** keep it simple (distributed-systems discipline). Own the
seams, not the internals. Every component independently testable, stateless where
possible, idempotent. Prefer boring, correct mechanisms over clever ones.

> **Direction pivot (2026-07-14).** Ramp open-sourced `portallib` and its author
> (Ben Geist) explicitly endorsed smol-portal wrapping it as the engine ("if you
> want to use the repo I built that would be ideal"); it's expected feature-complete
> by end of week and we have alpha access. So smol-portal **adopts `portallib` as
> the engine** rather than maintaining a parallel ML reimplementation. Our
> `pipeline/portal` ML code becomes a legacy fallback/reference until portallib is
> released and confirmed to run under smolvm's CUDA constraints; then we thin the
> CLI into an adapter over portallib. Our differentiator is the **secure-VM hosting
> + orchestration**, not the recipe. See the connector phase below.

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

### Phase A — PorTAL pipeline, bare metal — ⚠️ SUPERSEDED by portallib adoption

> **Superseded (2026-07-14).** This phase built our own reverse-engineered PorTAL
> pipeline (hypernetwork → task latent → converter → eval) to reproduce the paper.
> Ramp's official `portallib` makes reproducing the *method* ourselves unnecessary —
> we adopt it as the engine instead (see the Connector phase). The reusable
> **orchestration** pieces this phase produced (CLI shape, artifact/content-address
> formats, `portal.cuda` constraints, dataset schema guard, runtime manifest, the
> `portal port` sizing UX) are **retained** and carry over to the connector. The
> ML internals (`hypernetwork.py`, `converter.py`, our latent/eval science) are kept
> only as a legacy fallback until portallib is confirmed to run under smolvm CUDA.
> Original goal (kept for history): reproduce Qwen3→Gemma-3 ~94–98% of LoRA lift.

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

### Phase A2 — Own-pipeline scientific validation — ⚠️ DEMOTED (Ramp owns the science)

> **Demoted (2026-07-14).** This phase was going to prove *our* reimplementation's
> mechanism (is the latent model-independent? does porting recover direct-LoRA
> accuracy?). With `portallib` adopted as the engine, **the science is Ramp's** —
> we don't need to defend a reimplementation we're replacing. The "disprove our own
> latent" experiments (latent-matters ablation, converter reuse, output-head
> scaling) are **shelved**, not deleted; `portal baseline` and `--latent-mode`
> remain in-tree but are no longer priorities.
>
> **Retained (host-any-engine, already landed + CPU-tested, 31 passing):**
> `portal/data.py` dataset schema guard, `portal/env.py` runtime manifest,
> `portal convert --cal-dataset` requirement, and the `portal port` sizing knobs.
> These serve the connector regardless of engine.
>
> **What replaces the science track:** *hosting-fidelity* validation — run
> `portallib`'s **own** `acc_norm` eval on the `portallib-tasks` suite and confirm
> the numbers inside a smolvm VM match bare metal. That's our job (faithful secure
> hosting), not reproducing the research. See the Connector phase.

### Phase A3 — Connector: adopt portallib as the engine (ACTIVE)

Make smol-portal run Ramp's `portallib` unmodified inside a smolvm CUDA microVM.

**This week (before portallib feature-completes — mostly no GPU):**
1. **Worker image that installs `portallib`** (from the GitHub repo now; PyPI when
   published) alongside pinned torch/transformers/peft, pre-baked so smolvm CUDA
   staging interposes shims at pull time.
2. **Draft API/feature feedback for Ben** (issue-ready) — done: see private
   `smolvm-notes/portallib-feedback.md`. File as issues/PRs when his code lands.
3. **Smoke harness** around one `portallib-tasks` task, ready to run the moment
   the engine is available.

**When portallib lands (end of week, per Ben):**
4. Read it end-to-end; confirm the train/port/eval API + artifact shapes.
5. Run one task **in-VM**; capture any smolvm-hostile ops (bf16, flash-attn,
   `torch.compile`, multi-GPU) → file issues/PRs upstream (offer our `portal.cuda`
   knobs as a PR).
6. Thin `pipeline/portal`'s CLI into an **adapter over portallib**; keep our
   artifact/content-address + orchestration layer on top. Retire our ML internals
   once the hosted engine passes hosting-fidelity.

**Hosting-fidelity DoD:** `portallib` runs a `portallib-tasks` task inside smolvm
and reproduces its bare-metal `acc_norm` within tolerance.

#### GPU test plan (Lambda, runnable now — de-risks hosting ahead of the drop)

Since portallib isn't out yet, today's GPU value is **re-validating our hosting
substrate on the new smolvm and probing the ops portallib will need**:

- **Rebuild CUDA shims to match smolvm (v1.6.0)** and re-run the CUDA gates +
  `portal train` + `portal port` e2e (with the new CLI sizing knobs). Confirms the
  version bump didn't regress hosting.
- **Capability probe matrix** (what portallib may require, tested through the
  remoted CUDA path): fp32 ✓ (known), **bf16**, **flash/mem-efficient SDPA**
  (fused), **`torch.compile`**, **multi-GPU/NCCL**. Record pass/fail per op — this
  directly seeds the "suggestions for Ben" and tells us what to force-off in the
  worker image.
- **Investigate `src/cuda_daemon.rs`** + the new upstream CUDA branches
  (`cuda-independent-serving`, `cuda-shim-hygiene`, `cuda-vmresources`) — do they
  change how we install/interpose shims? Note in `smolvm-notes/`.

Detailed runbook lives in `smolvm-notes/portal-reference-plan.md` +
`examples/smolvm/lambda-instructions.md`.

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
| ~~PorTAL recipe unreproducible~~ → **portallib released** | Adopt it as the engine; we own hosting, not the recipe |
| portallib uses smolvm-hostile ops (bf16 / flash-attn / `torch.compile` / multi-GPU) | Probe on Lambda ahead of the drop; offer `portal.cuda` force-off knobs as an upstream PR; force-off in worker image |
| portallib API/artifacts churn before v1 | We're an alpha user with author contact — track the API, keep the connector a thin adapter |
| smolvm CUDA surface in flux (v1.6.0 + `cuda_daemon.rs` + rework branches) | Pin to a tag for each Lambda run; don't over-fit shim workarounds; re-validate per bump |
| Multi-VM OOM on shared GPU | NVIDIA driver returns clean errors; orchestrator pins one job per GPU |
| vsock latency on hot CUDA paths | Profile during hosting validation; acceptable for training workloads |

## Near-term direction (2026-07-14) — connector-first, author-endorsed

Ramp's PorTAL author (Ben Geist) confirmed the direction directly: he open-sourced
[`portallib`](https://github.com/ramp-public/portallib), endorsed smol-portal
wrapping it ("if you want to use the repo I built that would be ideal"), expects it
feature-complete **by end of week**, and gave us alpha access + an open invitation to
suggest improvements. So:

- **Adopt `portallib` as the engine.** smol-portal = the **secure-VM connector**:
  packaging, CLI/UX, artifacts, `portal.cuda` constraints, multi-GPU. Don't
  reimplement the ML.
- **Own hosting fidelity, not the science.** Our validation is "portallib's own
  `acc_norm` on `portallib-tasks` matches inside a smolvm VM vs. bare metal," not
  reproducing the paper.
- **Contribute upstream.** Ben asked for feedback — file issue-ready API/hosting
  suggestions (drafted in `smolvm-notes/portallib-feedback.md`) and offer our
  constrained-backend `portal.cuda` knobs as a PR.
- **Keep our `pipeline/portal` as legacy fallback** until portallib is confirmed to
  run under smolvm CUDA; then thin the CLI into an adapter over it.

## Immediate next steps

1. **This week — connector prep (local, no GPU):** worker image that installs
   `portallib` (pinned deps, pre-baked); a `portallib-tasks` smoke harness;
   finalize the feedback list for Ben.
2. **Today — Lambda GPU (de-risk hosting on new smolvm):** rebuild shims for
   **v1.6.0**, re-run CUDA gates + `portal port` e2e, and run the **capability
   probe matrix** (bf16 / flash-fused SDPA / `torch.compile` / multi-GPU) — results
   feed the Ben feedback + worker-image force-off flags. (Phase A3 GPU test plan.)
3. **When portallib lands (end of week):** read it, run one task in-VM, capture
   smolvm-hostile ops, file issues/PRs, and start the CLI-adapter-over-portallib.
4. **Retained/landed (host-any-engine, CPU-tested):** `portal port` sizing knobs,
   dataset schema guard, runtime manifest, `--cal-dataset` requirement.
5. **smolvm tracking:** upstream at **v1.6.0-11** with active CUDA rework
   (`cuda-independent-serving`, `cuda-shim-hygiene`, `cuda-vmresources`); local
   `main` diverged by our `.cursor` commit — rebase or build from the `v1.6.0` tag
   before the Lambda run. Don't over-fit shim workarounds while the surface churns.

---
_Status (2026-07-14): pivoted to **connector-first** — adopt Ramp's `portallib` as
the engine (author-endorsed), smol-portal owns secure-VM hosting + orchestration.
Systems/hosting path validated on smolvm v1.5.2 (real-model `portal port` e2e,
2026-07-13); smolvm now v1.6.0 (re-validate). Own-pipeline science (Phase A/A2)
superseded/demoted. Next: connector prep + Lambda hosting de-risk, then adopt
portallib when it drops (end of week)._
