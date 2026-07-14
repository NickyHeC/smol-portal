# smol-portal

Run [PorTAL](https://x.com/RampLabs/status/2072381992285647280) (Portable Task Adapters) fine-tuning **securely** inside [smolvm](https://github.com/smol-machines/smolvm) microVMs on local NVIDIA GPUs.

Learn a task once on a source model, extract a base-agnostic task latent, then port it to new base models with a slim converter — at roughly half the calibration cost of full LoRA, targeting ~94–98% of direct LoRA accuracy.

> **Scope note.** The end-to-end *systems* path (train → extract → convert → eval
> inside CUDA smolvm) works on real models. The PorTAL *research claim* — a
> model-independent task latent that recovers ~94–98% of direct-LoRA accuracy at
> ~half the cost — is **not yet reproduced or measured here** (that figure is
> Ramp's target, not a result of this repo). See [ROADMAP.md](ROADMAP.md) Phase A2
> for the validation plan.

## Status

**smolvm CUDA validation (Lambda A10, 2026-07-13):** LoRA training and full `portal port`
e2e on **real models** (Qwen3-0.6B → TinyLlama) inside smolvm **v1.5.2** with the
[`examples/smolvm/`](examples/smolvm/) worker image. See [memory.md](memory.md).

| Step | Description | Status |
|------|-------------|--------|
| 1 | Project scaffolding & CLI | Done |
| 2 | Artifact formats (content-addressed) | Done |
| 3–8 | Train → extract → convert → eval → end-to-end | In progress |
| B1 | smolvm CUDA LoRA smoke (tiny Llama) | **Done** (Lambda, v1.5.2) |
| B2 | Smolfile + `portal-cuda` image | Done (PR #1) |
| B3 | `portal port` e2e on smolvm | **Done** (Lambda, v1.5.2; real models 2026-07-13) |

VM integration details: [`examples/smolvm/README.md`](examples/smolvm/README.md). Multi-GPU
(Phase C) not started. See [ROADMAP.md](ROADMAP.md).

## Architecture

smol-portal is the **orchestration layer** on top of CUDA-enabled smolvm. It does not own VMM, guest drivers, or GPU management code — those live in the [smolvm](https://github.com/smol-machines/smolvm) repo.

```text
┌─ Guest VM ────────────────────────┐     ┌─ Host (smolvm process) ──────────┐
│  PyTorch / PorTAL pipeline        │     │  smolvm-cuda server              │
│  CUDA shim (libcudart substitute) │     │  GpuBackend → NVIDIA driver      │
│  smolvm-cuda Client               │     │                                  │
│    └── vsock port 7000 ──────────────►──┘                                 │
└───────────────────────────────────┘
```

**Dependency:** requires a CUDA-enabled [smolvm](https://github.com/smol-machines/smolvm) build for Phase B+.

## PorTAL pipeline

The core workflow:

1. **Train** — LoRA fine-tune a source model (e.g. Qwen3) on a task
2. **Extract** — hypernetwork compresses adapter weights into a base-agnostic task latent
3. **Convert** — slim MLP projects the latent into target model LoRA weights
4. **Eval** — benchmark the ported adapter against direct LoRA baselines

```bash
portal port --from qwen3 --to gemma3 --task my-task --dataset <hf_dataset_id>
```

Individual steps are also available: `portal train`, `portal extract`, `portal convert`, `portal eval`. `portal baseline` trains + evals a direct LoRA on the target model — the comparison point a ported adapter is measured against.

## Quick start

Requires Python 3.11+, [uv](https://docs.astral.sh/uv/), and a CUDA-capable GPU for training.

```bash
cd pipeline/portal
uv sync --extra dev

# CLI help
uv run portal --help
uv run portal port --help

# Run tests
uv run python -m pytest -q
```

## Repo layout

```text
smol-portal/
├── pipeline/portal/          # PorTAL Python package (CLI + ML pipeline)
│   ├── portal/
│   │   ├── cli.py            # typer entry point
│   │   ├── train.py          # source LoRA training
│   │   ├── hypernetwork.py   # task latent extraction
│   │   ├── converter.py      # latent → target adapter
│   │   ├── eval.py           # benchmark evaluation
│   │   └── port.py           # end-to-end orchestration
│   └── tests/
├── crates/                   # (planned) portal-orchestrator — Rust multi-GPU fan-out
├── Smolfiles/                # (planned) portal-worker Smolfile
├── ROADMAP.md
└── reference-material.md
```

## Artifacts

All outputs are content-addressed on disk:

```text
artifacts/{task_name}/
├── source_lora_{hash}/adapter/     # PEFT LoRA checkpoint
├── task_latent_{hash}/             # .safetensors + metadata JSON
├── target_lora_{hash}/adapter/     # ported PEFT adapter
└── eval_{hash}/eval_results.json   # metrics
```

Same **normalized config → same hash → same artifact directory** (path identity,
so reruns can cache-hit). This is config identity, not bit-for-bit output
reproducibility: the hash does not cover library/model/dataset revisions or
hardware, and GPU kernels aren't guaranteed deterministic. Each artifact embeds a
`runtime` manifest (library versions, git commit, platform) — recorded for
provenance and **excluded from the hash**. See [SPEC.md](SPEC.md) §7.

## Security model

- **CPU / memory / filesystem:** full KVM VM isolation
- **GPU path:** process-level isolation (guest CUDA calls execute as the smolvm host process). Sufficient for running your own PorTAL jobs; not a multi-tenant GPU sandbox.

## References

- [SPEC.md](SPEC.md) — human-readable contract: pipeline, CLI, config, artifacts, smolvm integration
- [AGENTS.md](AGENTS.md) — playbook for a coding agent driving PorTAL for a user
- [ROADMAP.md](ROADMAP.md) — phased plan, architecture decisions, risks
- [reference-material.md](reference-material.md) — smolvm-cuda, CUDA APIs, HypeLoRA, PEFT reading list
- [smolvm](https://github.com/smol-machines/smolvm) — microVM runtime
- [HypeLoRA](https://github.com/btrojan-official/HypeLoRA) — closest public analog to PorTAL's hypernetwork approach

## License

MIT — see [LICENSE](LICENSE).
