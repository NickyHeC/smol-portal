#!/usr/bin/env bash
# Host-vs-bare fidelity check for the portallib connector.
#
# Runs the SAME portallib evaluation recipe (smoke_portallib.py) twice:
#   A) inside a smolvm CUDA microVM  (remoted GPU)
#   B) bare on the host with docker --gpus all  (direct GPU)
# then asserts the macro acc_norm produced through smolvm matches bare metal
# within a tolerance. This is the hosting-fidelity gate: remoting must not change
# the numbers (our T3 "Δacc = 0").
#
# Both runs print their report JSON after a sentinel line so we parse stdout
# directly and do NOT depend on virtiofs persisting the guest output file
# (guest -v writes to the host mount have been flaky; see README / lambda-instructions).
#
# Prerequisites:
#   - portallib-cuda image built + saved: see Dockerfile.portallib-cuda header
#     (PORTALLIB_SPEC='portallib[training]==0.2.0').
#   - smolvm >= 1.6.4 on an NVIDIA GPU host (shims bundled); docker with the
#     NVIDIA runtime for the bare twin.
#   - Run from the smol-portal repo root.
#
# Usage:
#   examples/smolvm/fidelity_check.sh                 # defaults: Qwen3-1.7B @ v0.2.0, 3 tasks x 8
#   examples/smolvm/fidelity_check.sh --tasks rte,boolq,winogrande --max-examples 8 --tol 0.02
#
# NOTE: validated-on-A10 pending — this recipe is drafted against the released
# portallib 0.2.0 API and Ben's smol-portal 0.2 connector; confirm on a GPU box
# before offering the pattern upstream.
set -euo pipefail

IMAGE_TAG="portallib-cuda"
IMAGE_TAR="./portallib-cuda.tar"
ARTIFACT="RampPublic/portal-qwen3-1.7b"
ARTIFACT_REVISION="v0.2.0"
BASE_ID="Qwen/Qwen3-1.7B"
BASE_REVISION="70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
TASKS="rte,boolq,winogrande"
MAX_EXAMPLES="8"
TOL="0.02"
MEM="16384"
SENTINEL="===FIDELITY_JSON==="

while [ $# -gt 0 ]; do
  case "$1" in
    --image-tag) IMAGE_TAG="$2"; shift 2 ;;
    --image-tar) IMAGE_TAR="$2"; shift 2 ;;
    --artifact) ARTIFACT="$2"; shift 2 ;;
    --artifact-revision) ARTIFACT_REVISION="$2"; shift 2 ;;
    --base-id) BASE_ID="$2"; shift 2 ;;
    --base-revision) BASE_REVISION="$2"; shift 2 ;;
    --tasks) TASKS="$2"; shift 2 ;;
    --max-examples) MAX_EXAMPLES="$2"; shift 2 ;;
    --tol) TOL="$2"; shift 2 ;;
    --mem) MEM="$2"; shift 2 ;;
    -h|--help) sed -n '2,33p' "$0"; exit 0 ;;
    *) echo "unknown flag: $1" >&2; exit 2 ;;
  esac
done

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
SCRIPT="examples/smolvm/smoke_portallib.py"
OUTDIR="$(mktemp -d)"
trap 'rm -rf "$OUTDIR"' EXIT

# The evaluation recipe, identical for both runs. --hosting-safe pins fp32 +
# device_map=cuda + math SDPA, matching the connector's remoting-safe defaults.
SMOKE_ARGS="--artifact ${ARTIFACT} --artifact-revision ${ARTIFACT_REVISION} \
--base-id ${BASE_ID} --base-revision ${BASE_REVISION} \
--tasks ${TASKS} --max-examples ${MAX_EXAMPLES} --hosting-safe \
--output-dir /tmp/fidelity"

# Guest command: run the recipe, then print the report JSON after a sentinel so
# the host can parse it from stdout without relying on a mounted output file.
GUEST_CMD="python3 /workspace/smol-portal/${SCRIPT} ${SMOKE_ARGS} \
&& echo '${SENTINEL}' && cat /tmp/fidelity/smoke_eval.json"

extract_json() { # strip everything up to and including the sentinel line
  awk -v s="${SENTINEL}" 'f{print} $0==s{f=1}'
}

echo ">> [A] smolvm remoted-CUDA run ..." >&2
smolvm machine run --net --cuda --mem "${MEM}" \
  -e HF_HOME=/tmp/hf \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False \
  -e PORTALLIB_HOST=smolvm \
  -v "${REPO}:/workspace/smol-portal" \
  --image "${IMAGE_TAR}" -- \
  sh -c "${GUEST_CMD}" | tee "${OUTDIR}/smolvm.log" | extract_json > "${OUTDIR}/smolvm.json"

echo ">> [B] bare docker --gpus run ..." >&2
if [ -f "${IMAGE_TAR}" ]; then docker load -i "${IMAGE_TAR}" >/dev/null; fi
docker run --rm --gpus all \
  -e HF_HOME=/tmp/hf \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False \
  -e PORTALLIB_HOST=bare \
  -v "${REPO}:/workspace/smol-portal" -w /workspace/smol-portal \
  "${IMAGE_TAG}" \
  sh -c "${GUEST_CMD}" | tee "${OUTDIR}/bare.log" | extract_json > "${OUTDIR}/bare.json"

echo ">> comparing macro acc_norm (tolerance ${TOL}) ..." >&2
SMOLVM_JSON="${OUTDIR}/smolvm.json" BARE_JSON="${OUTDIR}/bare.json" TOL="${TOL}" python3 - <<'PY'
import json, os, sys

tol = float(os.environ["TOL"])
with open(os.environ["SMOLVM_JSON"]) as fh:
    a = json.load(fh)
with open(os.environ["BARE_JSON"]) as fh:
    b = json.load(fh)

rows = []
ok = True
for label in ("base", "portal"):
    va = a[label]["macro_accuracy"]
    vb = b[label]["macro_accuracy"]
    delta = abs(va - vb)
    rows.append((label, va, vb, delta))
    ok = ok and delta <= tol

print(f"{'metric':<8} {'smolvm':>10} {'bare':>10} {'|Δ|':>10}")
for label, va, vb, delta in rows:
    print(f"{label:<8} {va:>10.4f} {vb:>10.4f} {delta:>10.4f}")
lift_a = a["macro_accuracy_lift"]
lift_b = b["macro_accuracy_lift"]
print(f"{'lift':<8} {lift_a:>10.4f} {lift_b:>10.4f} {abs(lift_a-lift_b):>10.4f}")

if not ok:
    print(f"FAIL: macro acc_norm differs by more than {tol}", file=sys.stderr)
    sys.exit(1)
print("PASS: smolvm remoting matches bare metal within tolerance")
PY
