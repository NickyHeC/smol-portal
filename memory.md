# memory.md

Session log for **smol-portal**. Reverse chronological order — newest on
top. Each entry: date, summary, key outcomes, and decisions made.

---

## 2026-07-15 (late) — smolvm #601 validated from main on A10; stock v1.6.3 still ships no CUDA shims

- **smolvm #601** (bundle CUDA shims + `smolvm-cuda-run` in agent-rootfs) **merged**
  on `main` (`1d0b795`). Tag **v1.6.3** was cut earlier the same day from a
  separate `release-1.6.3` branch and **does not include #601**. Empirically:
  stock `smolvm-1.6.3-linux-x86_64.tar.gz` still has empty
  `agent-rootfs/usr/local/lib/` and no `smolvm-cuda-run` (same #596 gap).
- **Do not** treat “smolvm ≥ 1.6.3” as “no manual shim copy.” Keep recommended
  floor **≥ 1.6.2** (glibc) + manual shim build until a *released* tarball
  ships #601 (likely next point release). No Bin ping — they’ll pick it up.
- **A10 validation** (Lambda, Ubuntu 22.04): built from `main`, packaged rootfs
  via #601 env-var path only (`CUDART_SHIM_*` + `--no-build-agent`), no hand
  `cp` into `smolvm-cuda/`. `portal-cuda` gate → `cuda: True` / `NVIDIA A10`.
  Host binary must match shim commit (v1.6.2 host + main shims → err 801).
  Conda `pytorch/pytorch` images still need #602-style path staging; use
  pip-torch `portal-cuda` / `portallib-cuda` for packaging proofs.
- **portallib smoke** on that same remoting stack: `portal-qwen3-1.7b@v0.1.0`,
  14×8 ex, fp32+math → macro **0.607 → 0.741** (matches prior T2). No Gemma.
- Local-tar `-v` host mounts remain flaky for injecting scripts; base64 inject
  still works.

## 2026-07-15 (evening) — portallib v0.1.0 landed; connector T0–T5b PASS; upstream issues filed

- **portallib v0.1.0 shipped** (PyPI `portallib==0.1.0`, tag/release, HF artifacts
  `RampPublic/portal-qwen3-{1.7b,4b,8b}` + `portal-gemma-3-4b`, dataset
  `RampPublic/portallib-tasks`). Public API is library-shaped:
  `PortalCoreTrainer` / `PortalAdapterRefitter` / `PortalEvaluator` / `PortalModel`
  (+ Hub `from_pretrained` / `export_peft`). No CLI.
- **Connector test plan run** (T0–T4 PASS; **T5a + T5b PASS** on 1× H100):
  - **T0** (Mac CPU): `pip install 'portallib[training]==0.1.0'`; 39 pytest pass.
  - **T1** (Mac CPU): `PortalModel.from_pretrained("…portal-qwen3-4b", "v0.1.0")` →
    `export_peft("rte")` → ordinary PEFT dir (no base LLM load needed).
  - **T2** (Lambda A10, smolvm **v1.6.2** + matching shims): tiny `acc_norm` eval
    inside the microVM, fp32 + math SDPA + `device_map="cuda"`. macro acc
    0.607 → 0.741 on a 14-task × 8-example smoke slice.
  - **T3** (same box, bare `docker --gpus` twin): **identical** macro acc / lift to
    T2 (Δacc = 0; portal NLL within ~1e-7). Hosting-fidelity DoD met at smoke size.
  - **T4** (in smolvm): tiny `PortalAdapterRefitter` `portal-qwen3-1.7b` →
    Qwen3-0.6B, 1 epoch / 8 ex — plumbing PASS (grad flowing, gold_nll 3.42→2.52).
  - **T5a** (Lambda **1× H100 80GB**, smolvm v1.6.2): published
    `portal-qwen3-8b@v0.1.0` + Qwen3-8B, 14×64 ex, fp32 hosting-safe.
    smolvm macro 0.682 → 0.776 (lift +0.095); bare twin **identical** (Δacc = 0,
    NLL Δ ≈ 1e-8). Real-scale hosting fidelity **PASS**.
  - **T5b** (same H100): `portal-qwen3-4b` → Qwen3-8B refit, **1000 ex/task**,
    1 epoch, batch 4, **bf16** (fp32 hung at 0% util). acc_norm 0.680 → 0.785;
    gold_nll 3.704 → 1.321; 250 steps × 14 tasks; ~15 min. Train path at paper
    scale under remoting **PASS**. T5c dual-source train still optional.
- **Connector assets updated** (`examples/smolvm/`): `Dockerfile.portallib-cuda`
  bumped to `portallib[training]==0.1.0` (transformers ≥ 4.52, dropping the legacy
  `<4.52` pin); `smoke_portallib.py` rewritten to the real `PortalEvaluator` API
  (+ hosting-safe knobs); new `smoke_refit_portallib.py` for the refit path.
- **Upstream issues filed** on `ramp-public/portallib` (Tier-1, concise + evidence):
  [#6](https://github.com/ramp-public/portallib/issues/6) subset eval (+ PR offer),
  [#7](https://github.com/ramp-public/portallib/issues/7) examples hardcode
  bf16/`.to(device)` (library itself is placement-agnostic via caller-owned
  `PortalBase`), [#8](https://github.com/ramp-public/portallib/issues/8) scope
  question on the intended automation entrypoint. DM'd Ben with the summary.
- **Notes:** `PortalEvaluator.evaluate(..., portal=)` currently requires the full
  task tuple (→ #6). Under remoting, a benign `cuBLAS: no current CUDA context`
  warning appears then recovers. smolvm local-tar `-v` mounts of guest *outputs*
  were flaky (stdout-only); HF-cache `-v` and bare docker `-v` worked.

## 2026-07-15 (later) — #636 closed; stock min → v1.6.2 on 22.04

- Upstream merged [#644](https://github.com/smol-machines/smolvm/pull/644) (rebuild
  libkrun floor 2.34 + CI gate) and cut **[v1.6.2](https://github.com/smol-machines/smolvm/releases/tag/v1.6.2)**.
- Closed our duplicate [#639](https://github.com/smol-machines/smolvm/pull/639)
  (superseded) and [#636](https://github.com/smol-machines/smolvm/issues/636).
- **Guidance:** on Ubuntu 22.04 use **smolvm ≥ 1.6.2** (or stay on 1.5.2). Stock
  1.6.0/1.6.1 still need the §9e host rebuild if you must use them. SPEC /
  AGENTS / runbook updated; no more default workaround for new boxes on 1.6.2+.

## 2026-07-15 — Track A Lambda: smolvm v1.6.0 hosting de-risk (A10)

- **Box:** Lambda Ubuntu 22.04 / glibc 2.35 / A10. Bootstrap `VER=1.6.0` + matching CUDA
  shims + `portal-cuda.tar` (NickyHeC/smol-portal `main`).
- **Blocker → workaround:** stock v1.6.0 `lib/libkrun.so` requires **GLIBC_2.39**
  (`machine run` fails to boot). Same failure class as in-tree `build-libkrun.yml`
  (v1.2.0 note). Rebuilt on-box with `SKIP_LIBKRUNFW=1 GPU=1 ./scripts/build-libkrun-linux.sh`
  → floor **2.34**; replaced release lib. **Upstream:**
  [smol-machines/smolvm#636](https://github.com/smol-machines/smolvm/issues/636).
- **Hosting gates (§4):** PASS — `cuda: True`, libcudart shim ~901 KB, vsock,
  `gpu_loopback` (`GPU-VERIFY-OK` A10).
- **PorTAL smoke:** PASS tiny `portal train`; PASS `portal port` CLI sizing knobs
  (tiny→tiny); PASS fused SDPA `loss.backward()`.
- **Capability probe (through remoted CUDA):**
  - fp32 / bf16 / fused SDPA — **PASS**
  - multi-GPU — skip (1× A10)
  - `torch.compile`:
    - FAIL in stock slim image (no `CC`)
    - FAIL with gcc only — linker `cannot find -lcuda` (staging has `libcuda.so.1`,
      not unversioned `libcuda.so` that Triton `-lcuda` needs)
    - **PASS** simple `nn.Module` after `ln -s libcuda.so.1 → libcuda.so` + gcc
    - HF CausalLM + compile still **FAIL** (`NameError: torch` inside dynamo /
      transformers wrapping) — treat HF+compile as unsupported for now
- **Implications:** worker images that may run `torch.compile` need **gcc** and an
  unversioned `libcuda.so` link (or staging should install it). Keep compile
  force-off for portallib until HF path is green. (Follow-up same day: #636 fixed
  in upstream v1.6.2 — see entry above.)
- **Skipped:** real Qwen→TinyLlama (time / Track A goal was hosting + probe).

## 2026-07-14 (afternoon pickup) — Connector prep: portallib worker image + smoke

- **Daily-startup:** smolvm `main` == `upstream/main` at **v1.6.0-11** (0/0, pristine).
  `portallib` still README-only; issue #1 open (no code drop). Private watch logs updated.
- **ROADMAP connector-first edits** were already on `main` (`9818a19`); corrected the stale
  “main diverged” note.
- **A.1 Worker image (local, no Docker on this Mac):**
  `examples/smolvm/Dockerfile.portallib-cuda` pre-bakes torch cu124 + pinned
  transformers/peft/datasets/accelerate; installs `portallib` from GitHub via
  `PORTALLIB_SPEC` (`INSTALL_PORTALLIB=0` for deps-only until #1 merges). Plus
  `portallib.smolfile`.
- **A.2 Smoke harness:** `examples/smolvm/smoke_portallib.py` — one
  `RampPublic/portallib-tasks` task (`boolq`), discovery + `--dry-run`; argv wires
  when their CLI lands.
- **A.3 Ben feedback:** refreshed in private `portallib-feedback.md` (still hold filing).
- **Next:** Lambda shim rebuild @ v1.6.0 + capability matrix; full image build when
  portallib is installable.

## 2026-07-14 (later) — Direction: orchestration UX; portallib reactive; smolvm v1.6.0

- **Strategy set (user):** focus smol-portal on the **orchestration layer** —
  seamless to run PorTAL inside smolvm — not on the ML recipe (Ramp owns that via
  `portallib`). Interim while portallib is README-only: test + document. When it
  ships code: compare, adopt as engine, file issues/PRs upstream to portallib.
- **UX landed (local, no GPU):** folded `port_e2e.py` sizing knobs into
  `portal port` (`--max-samples/--max-seq-length/--batch-size/--rank/
  --train-epochs/--extract-epochs/--convert-epochs/--cal-samples/--latent-dim/
  --hidden-dim/--latent-mode`, `--cal-dataset`). Threaded via new `PortConfig`
  knobs + `build_hypernet_config()`; defaults equal stage defaults so artifact
  hashes don't drift. Closed the SPEC "known gap"; `port_e2e.py` kept as
  programmatic/debug reference. +`test_port_config.py` (31 tests pass, ruff clean).
- **smolvm bumped to v1.6.0** (from v1.5.2) via the new daily-startup routine —
  new `src/cuda_daemon.rs` + `cuda-fork-raw-handles` upstream; shims must be
  rebuilt for v1.6.0 before the next Lambda run. Logged in
  `smolvm-notes/smolvm-version-watch.md`.
- **Committed + pushed** the prior Phase A2 + 3-folder-docs work (`bdef823`).
- **Daily watches added:** `daily-startup.mdc` rule (sync smolvm + check portallib)
  + private `portallib-watch.md` / `smolvm-version-watch.md` logs.

## 2026-07-14 — Phase A2 prep: scientific-validation code + plans (local, no GPU)

- **Context:** reviewed an external audit of the repo vs. Ramp's PorTAL writeup.
  Verdict confirmed against code: systems path works, but the *mechanism* is
  unproven — autoencoder trains on one adapter vector, converter sees one fixed
  latent `z`, no baseline, perplexity-only eval. Risk = mistaking "pipeline ran"
  for "PorTAL worked."
- **Landed now (CPU-only, `pytest` 11→27 green, ruff clean):**
  - `portal/data.py` — explicit dataset text resolution + `DatasetSchemaError`;
    wired into train/convert/eval (no more silent empty-string training). Keeps
    `text`→`input` precedence so IMDB runs are unchanged.
  - `portal convert` — `--cal-dataset` now **required** (killed the silent
    fallback to target-model id). `portal port` still auto-sets it from `--dataset`.
  - `portal/env.py` — `runtime_manifest()` (lib versions, git commit, platform)
    embedded in every artifact's metadata, **excluded from the content hash**.
  - `--latent-mode {real|zero|random|shuffled}` on `convert` (`LatentMode` enum +
    `apply_latent_mode`) — the "does the latent matter?" ablation knob.
  - `portal baseline` (`portal/baseline.py`) — direct target-LoRA train+eval; the
    comparison point for `port`. Uses `{task}__baseline` artifact namespace.
  - New tests: `test_data.py`, `test_latent_mode.py`, `test_env.py`,
    `test_artifacts.py::test_latent_records_runtime_but_hash_excludes_it`.
- **Docs:** ROADMAP Phase A2 (build + test plan, local-done vs Lambda-next);
  SPEC convert/baseline/metrics/§7 determinism softened + manifest; README scope
  note + softened reproducibility claim.
- **Next Lambda session (priority order):** (1) latent-matters ablation — sweep
  `--latent-mode`, want `real` ≪ `zero|random|shuffled`; (2) `portal baseline` vs
  `portal port` recovery ratio; (3) task-specific metrics (accuracy/F1) before any
  "% of direct LoRA" claim; (4) fold `port_e2e.py` smoke knobs into `portal port`.
- **Not done (needs GPU / bigger design):** task metrics, cross-task converter
  reuse (save converter as artifact), converter output-head scaling (dense
  `hidden × total_lora_params` won't scale), wiring `find_artifact` caching
  (blocked on latent hash depending on post-extraction `input_dim`).

## 2026-07-13 (Lambda, ~1 h) — Real-model PorTAL on smolvm v1.5.2

- **Stack:** smolvm **v1.5.2** tarball + upstream `v1.5.2` shims, `portal-cuda.tar` (built on Lambda), A10.
- **Runbook:** `lambda-instructions.md` bootstrap + §4/§5 — session much faster than prior days.
- **CUDA gates (§4):** all PASS — `cuda: True`, `libcudart.so.12` shim 887616 B, vsock, `gpu_loopback` A10.
- **`portal train` smoke:** PASS tiny-random Llama 8/8 @ ~3.2 it/s (math SDPA).
- **`portal train` real:** PASS **Qwen/Qwen3-0.6B** on IMDB (64 samples, 1 epoch) @ ~6.6–7.5 it/s.
- **`portal port` e2e real:** PASS **Qwen3-0.6B → TinyLlama/TinyLlama-1.1B-Chat-v1.0** (`port e2e ok`, math SDPA). `google/gemma-3-1b-it` blocked (gated repo, 401).
- **Fused SDPA real:** PASS Qwen train @ ~7.65 it/s; PASS full **`portal port` e2e** (`port e2e ok (fused SDPA)`).
- **Metrics anomaly → root-caused:** fused eval `loss=0.0 ppl=1.0` / math `ppl=89385` traced to (1) converter never trained — predicted LoRA weights injected via `param.data.copy_(detach())` broke autograd (flat calib loss `12.3736`), and (2) eval scored pad tokens + toggled SDPA backend. Confirmed on Lambda: `converter.grad after injected-path backward: None`.
- **Fix (branch `fix/converter-autograd-eval-metrics`, pushed):** converter uses `torch.func.functional_call` (grads reach the MLP); eval masks pads (`labels=-100`), pins math SDPA, weights loss by scored tokens; +2 regression tests; ruff/pytest green (11 pass).
- **Fix validated on Lambda (v1.5.2, A10):** converter loss now **descends** `11.82→9.39→8.37→7.79`; eval **`ppl=4169`** (finite, sane). `port e2e ok (fixed)`.
- **Docs added (same branch):** `SPEC.md`, `AGENTS.md`, de-anonymized `examples/smolvm/port_e2e.py` reference driver; personal Lambda copy in `smolvm-notes/port_e2e_lambda.py`.
- **Next:** open PR for the fix branch; de-anonymize existing `examples/smolvm/` personal refs; Gemma target with `HF_TOKEN`; scale samples/epochs for real accuracy; land smolvm PRs #600–#602.

## 2026-07-13 (evening) — Lambda session complete: full PorTAL e2e on smolvm v1.5.2

- **Stack:** smolvm **v1.5.2** tarball + upstream `v1.5.2` shims, `portal-cuda.tar`, Lambda A10.
- **`portal train`:** PASS (math SDPA default workaround).
- **Fused SDPA Trainer:** PASS without math-SDPA override (`fused SDPA train ok`).
- **`portal port` e2e:** PASS — train → extract → convert → eval (`port e2e ok`), fused SDPA, smoke-sized config.
- **#597:** commented + closed on upstream; PR [#603](https://github.com/smol-machines/smolvm/pull/603) closed as superseded.
- **Fork:** `NickyHeC/smolvm` synced to upstream; `v1.5.2` tag pushed.
- **Code:** `PORTAL_SKIP_CUDA_SMOLVM` merged to `main` via PR [#2](https://github.com/NickyHeC/smol-portal/pull/2) (2026-07-13).
- **Next session:** real-model `portal port` on Lambda (qwen → gemma or similar).
- **Docs:** `examples/smolvm/lambda-instructions.md` synced (full §5 recipes); `memory.md`, `README.md`, `ROADMAP.md`, `examples/smolvm/README.md` updated.

## 2026-07-13 — Lambda v1.5.2: #597 repro PASS; GitHub todos filed

- **#597 repro:** tiny Llama fused-SDPA `loss.backward()` **PASS** on smolvm **v1.5.2** + matching upstream shims (`backward ok`, with and without `SMOLVM_CUDA_RING=0`). Failed on v1.5.0 last session.
- **False negatives today:** fork missing `v1.5.2` tag → accidental v1.5.1 shims until `git fetch smol-machines v1.5.2`.
- **`portal train`:** not completed — `pip install portal` on host (Py3.10) fails; must install inside VM (`portal` needs Py3.11+).
- **GitHub action items:** `~/Documents/smolvm-notes/github-action-items.md` (comment/close #597, PR #603 disposition, smol-portal math-SDPA workaround).
- **Still testing:** `portal train` without math SDPA on v1.5.2.

## 2026-07-12 (afternoon) — upstream v1.5.2; Lambda runbook + cross-repo agent rules

- **Upstream** advanced to **v1.5.2** (`402f6c1a`): sparse storage-template copy, llama.cpp/CUDA 13, cuBLAS Level-1/2/3, graph coldstart + RTT modeling. **Does not fix #597** (fused SDPA backward) — launch-path overlap is memoization/graph only.
- **No blockers** for today's Lambda plan: PR [#603](https://github.com/smol-machines/smolvm/pull/603) CI green + MERGEABLE; still need GPU trace.
- **Added** `examples/smolvm/lambda-instructions.md` (+ canonical copy in `~/Documents/smolvm-notes/lambda-instructions.md`). PEM path: `~/Documents/PorTAL.pem`.
- **Agent rules:** smolvm `.cursor/rules/smol-portal-interop.mdc` and smol-portal `.cursor/rules/smolvm-interop.mdc` — agents must cross-edit both local repos before pushing when the change spans runtime ↔ PorTAL.

## 2026-07-12 (evening) — smolvm upstream PRs filed; next Lambda = PR #597 trace

- **Fork synced** to upstream v1.5.1; opened PRs [#600](https://github.com/smol-machines/smolvm/pull/600)–[#603](https://github.com/smol-machines/smolvm/pull/603) on `smol-machines/smolvm`.
- **PR #597 fix is NOT done** — [#603](https://github.com/smol-machines/smolvm/pull/603) is diagnostics only (`SMOLVM_CUDA_SHIM_TRACE` on launch path). Real fix follows after a GPU trace run.
- **Next Lambda session (first task):** build from `nickyhec/cuda-launch-diagnostics-597`, run fused-SDPA backward repro with `SMOLVM_CUDA_SHIM_TRACE=1` + `CUDA_LAUNCH_BLOCKING=1`, read which `[shim]` line fires. Full script in `~/Documents/smolvm-notes/cuda-build-plan.md` § "Next Lambda session" and `examples/smolvm/lambda-instructions.md` §5.
- **Rust:** installed on Mac (can compile/test smolvm PRs locally). Lambda rustup — revisit next session.

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
- **Lambda session:** Step 3 done; **next session starts with PR #597 GPU trace** (see cuda-build-plan.md).

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
- **Runbook:** `examples/smolvm/lambda-instructions.md` (synced from `~/Documents/smolvm-notes/`)
- **gVisor install:** `runsc` at `/usr/local/bin/runsc`, runtime `runsc-gpu` with
  `--nvproxy=true --nvproxy-docker=true --platform=systrap --nvproxy-allowed-driver-capabilities=compute,utility,video`.
- **Primary references:** gVisor `pkg/sentry/devices/nvproxy` (Go handlers),
  `pkg/abi/nvgpu` (struct defs), NVIDIA open-kernel-modules (ABI source of truth).
- **Repo structure target:** `crates/` (nv-abi, portal-orchestrator, portal-agent),
  `vmm/`, `guest-driver/`, `pipeline/` (Python PorTAL), `spikes/`.
