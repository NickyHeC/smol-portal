# PorTAL Specification

PorTAL (**Por**table **T**ask **A**dapters) learns a task once on a source model
and ports it to other base models. This document is the human-readable contract:
concepts, pipeline, CLI, config, artifacts, and the smolvm integration.

For the agent-driven operational playbook see [`AGENTS.md`](AGENTS.md).

---

## 1. Concepts

- **Source model** — the model you first fine-tune on a task (LoRA).
- **Task latent** — a compact, base-agnostic vector distilled from the source
  adapter by a hypernetwork autoencoder.
- **Target model** — the model you want the task on, without retraining from
  scratch.
- **Converter** — a slim MLP that projects the task latent into target-model
  LoRA weights, trained with a little calibration data.
- **Adapter** — a PEFT-compatible LoRA checkpoint.

Goal: recover most of a direct-LoRA's accuracy on the target at a fraction of the
calibration cost.

---

## 2. Pipeline

| Stage | Command | Input | Output |
|-------|---------|-------|--------|
| Train | `portal train` | source model + dataset | source LoRA adapter |
| Extract | `portal extract` | source adapter | task latent (`.safetensors`) |
| Convert | `portal convert` | task latent + target model | target LoRA adapter |
| Eval | `portal eval` | target adapter + dataset | metrics JSON |
| End-to-end | `portal port` | source + target + dataset | all of the above |

Individual stages are composable; `portal port` wires them together.

---

## 3. CLI contract

All options are named. Common: `--output-dir/-o` (default `artifacts/`),
`--seed` (default 42).

### `portal train`
| Option | Default | Meaning |
|--------|---------|---------|
| `--model/-m` | — | source model HF id (required) |
| `--task/-t` | — | task name (required) |
| `--dataset/-d` | — | HF dataset id (required) |
| `--rank` | 16 | LoRA rank |
| `--epochs` | 3 | training epochs |
| `--lr` | 2e-4 | learning rate |
| `--batch-size` | 4 | batch size |
| `--max-seq-length` | 512 | token length |
| `--max-samples` | none | limit dataset rows |

### `portal extract`
`--adapter-dir/-a` (required), `--model/-m` (source, required), `--task/-t`
(required), `--latent-dim` (256), `--epochs` (50).

### `portal convert`
`--latent-dir/-l` (required), `--target/-t` (required), `--task` (required),
`--cal-dataset` (defaults to target model id if unset), `--cal-samples` (256),
`--epochs` (30).

### `portal eval`
`--adapter-dir/-a` (required), `--model/-m` (required), `--task/-t` (required),
`--dataset/-d` (required), `--split` (`test`), `--batch-size` (8),
`--max-samples` (none).

### `portal port`
`--from` (required), `--to` (required), `--task/-t` (required), `--dataset/-d`
(required), `--skip-train`, `--source-adapter-dir`.

> **Known gap:** `portal port` does not currently expose per-stage sample/epoch
> sizing. For controllable smoke-sized e2e runs use
> [`examples/smolvm/port_e2e.py`](examples/smolvm/port_e2e.py), which drives the
> same pipeline via the Python API.

---

## 4. Config schema

Pydantic models in `portal/config.py`. Defaults shown.

```
LoraConfig       rank=16, alpha=32, dropout=0.05,
                 target_modules=[q_proj, v_proj, k_proj, o_proj]

TrainConfig      source_model, task_name, dataset_name,
                 dataset_split="train", max_samples=None, lora=LoraConfig,
                 learning_rate=2e-4, num_epochs=3, batch_size=4,
                 max_seq_length=512, seed=42

HypernetConfig   latent_dim=256, hidden_dim=512, num_layers=3,
                 learning_rate=1e-3, num_epochs=50, seed=42

ConverterConfig  target_model, calibration_dataset=None,
                 calibration_split="train", calibration_samples=256,
                 hidden_dim=512, learning_rate=1e-3, num_epochs=30, seed=42

EvalConfig       model_name, task_name, dataset_name,
                 dataset_split="test", max_samples=None, batch_size=8,
                 max_seq_length=512

PortConfig       source_model, target_model, task_name, dataset_name,
                 output_dir, train?, hypernet, converter?, eval_split="test",
                 skip_train=False
```

---

## 5. Artifacts

Content-addressed on disk — same config produces the same directory (idempotent
reruns). The directory suffix is the first 16 hex chars of a SHA-256 over the
config.

```
{output_dir}/{task_name}/
├── source_lora_{hash}/adapter/     # PEFT LoRA (adapter_model.safetensors + config)
├── task_latent_{hash}/             # task_latent.safetensors + task_latent_meta.json
├── target_lora_{hash}/adapter/     # ported PEFT adapter
└── eval_{hash}/eval_results.json   # {config, metrics, created_at}
```

Eval metrics: `loss`, `perplexity`, `num_samples`, `num_batches`.

---

## 6. smolvm integration

PorTAL is the orchestration layer; the microVM + CUDA remoting live in
[smolvm](https://github.com/smol-machines/smolvm). Guest CUDA calls execute as
the smolvm host process (process-level GPU isolation).

**Requirements**
- **smolvm ≥ 1.5.2**, with CUDA shims present in `agent-rootfs/usr/local/lib/smolvm-cuda/`
  (`libcuda.so.1`, `libcudart-shim.so`). Shim build must match the smolvm version.
- **Worker image** `portal-cuda.tar` — pre-bakes pip `torch` (CUDA build) so
  smolvm's staging interposes its shims at image-pull time. Libraries installed
  *after* launch are invisible to staging.

**Run contract**
```bash
smolvm machine run --net --cuda --mem 16384 \
  -e HF_HOME=/tmp/hf \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False \
  --image ./portal-cuda.tar -- \
  sh -c 'pip install "portal @ <zip-url>" ... && portal ...'
```

- `portal` installs **inside** the VM (Python ≥ 3.11). The slim image has no
  `git`; install from a GitHub `archive/....zip#subdirectory=pipeline/portal` URL.
- **Networking** (`--net`) is required for HF model/dataset downloads.
- **Gated models** need `-e HF_TOKEN=...` and an accepted license.

**CUDA backend behavior** (`portal.cuda`)
- Models load fp32 with `device_map="cuda"` (incremental placement).
- `configure_cuda_for_smolvm()` disables cuDNN and pins `expandable_segments:False`.
- **SDPA:** math SDPA by default (safe on all supported smolvm). Set
  `PORTAL_SKIP_CUDA_SMOLVM=1` to allow fused (flash/mem-efficient) SDPA on
  smolvm ≥ 1.5.2.
- **Eval pins math SDPA** regardless of that flag, so perplexity is reproducible
  and independent of the training configuration.

---

## 7. Determinism

- All stages call `set_seed`/`manual_seed` (default 42) and are content-addressed,
  so identical configs yield identical artifacts and let reruns cache-hit.
- Training results are independent of the SDPA backend (math vs. fused produce
  the same curves); only performance differs.
