#!/usr/bin/env bash
set -euo pipefail

_CORE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${REPO:-$_CORE_DIR/../..}" && pwd)"
RUNS="${RUNS:-$REPO/runs}"; RUNS="${RUNS%/}"
if [[ "$(basename "$RUNS")" != runs ]]; then
  echo "env.sh: RUNS must end in /runs (got $RUNS)" >&2; exit 1
fi
mkdir -p "$RUNS"; RUNS="$(cd "$RUNS" && pwd)"
PY="${PY:-python}"
SRC="$REPO/src"
CORE="$REPO/scripts/core"
CONF="$REPO/configs"
BR_WORK="${BR_WORK:-$REPO/third_party}"
OMNITOK_DIR="${OMNITOK_DIR:-$BR_WORK/OmniTokenizer}"
export REPO RUNS PY SRC CORE CONF BR_WORK OMNITOK_DIR
export BR_RUNS="$RUNS"
[[ -n "${OMNITOK_CKPT:-}" ]] && export OMNITOK_CKPT
[[ -n "${OMAG_CKPT:-}" ]] && export OMAG_CKPT OM2_CKPT="$OMAG_CKPT"
[[ -n "${COSMOS_CKPT:-}" ]] && export COSMOS_CKPT COSMOS_DV_DIR="$COSMOS_CKPT"
[[ -n "${SEED_VOKEN_DIR:-}" ]] && export SEED_VOKEN_DIR
[[ -n "${DATA_ROOT:-}" ]] && export DATA_ROOT K400_VAL_LIST="$DATA_ROOT/k400_val.txt"
export PYTHONPATH="$SRC${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES="${GPU:-${CUDA_VISIBLE_DEVICES:-0}}"

F=17
KEEP=0.60
NGEN=2000
TOKENS=500000000
SAMPLER=(--temperature 0.9 --top-k 256)
PEN="$RUNS/omni_oracle_gen_fvd"
OMAG_PEN="$RUNS/omag_gen_fvd"
DV_SWEEP="$RUNS/anchoring_jobs23_dv"
REAL="$RUNS/real_decode_F17_val"
REAL_SSV2="$RUNS/real_ssv2_F17_val"
RAWTR="$RUNS/omni_decode_extract_F17_train"
RAWVA="$RUNS/omni_decode_extract_F17_val"

mkdir -p "$RUNS" "$PEN"
cd "$(dirname "$RUNS")"
