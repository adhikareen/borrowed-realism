#!/usr/bin/env bash
set -euo pipefail
: "${REPO:?set REPO to the repository root}"
: "${DATA_ROOT:?set DATA_ROOT to the directory with the k400_/ssv2_ clip lists}"
: "${RUNS:?set RUNS to the run directory}"
: "${OMNITOK_CKPT:?set OMNITOK_CKPT to the OmniTokenizer imagenet_k600.ckpt}"
: "${PY:?set PY to the python interpreter}"
mkdir -p "$RUNS"
RUNS="$(cd "$RUNS" && pwd)"; REPO="$(cd "$REPO" && pwd)"; DATA_ROOT="$(cd "$DATA_ROOT" && pwd)"
SRC="$REPO/src"
export OMNITOK_CKPT
export PYTHONPATH="$SRC${PYTHONPATH:+:$PYTHONPATH}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES="${GPU:-${CUDA_VISIBLE_DEVICES:-0}}"
cd "$(dirname "$RUNS")"
NGEN=2000
KEEP=0.60

log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*"; }
nclips(){ { ls "$1"/gen_*.npy 2>/dev/null || true; } | wc -l; }
fvdval(){ "$PY" -c "import json,sys;print(round(json.load(open(sys.argv[1]))['fvd'],2))" "$1"; }

build_cache(){
  local RAW=$1 OUT=$2 SEL=$3 N=$4
  [[ -f "$OUT/manifest.json" ]] && { log "SKIP cache $OUT"; return 0; }
  log "cache sel=$SEL n=$N -> $OUT"
  "$PY" "$SRC/build_stream_cache_omni_long.py" --raw-dir "$RAW" --out-dir "$OUT" \
    --sel "$SEL" --keep-frac $KEEP --num-videos "$N"
}

train_ar(){
  local OUT=$1 TR=$2 VA=$3 NAME=$4 SEED=$5 BS=$6 GA=${7:-1} TOK=${8:-500000000} EV=${9:-500} LE=${10:-100}
  [[ -f "$OUT/done.json" ]] && { log "SKIP train $OUT"; return 0; }
  log "train $NAME (seed $SEED, bs $BS x accum $GA, target $TOK tokens)"
  "$PY" "$SRC/train.py" --train-dir "$TR" --val-dir "$VA" --out-dir "$OUT" --run-name "$NAME" \
    --batch-size "$BS" --grad-accum "$GA" --target-train-tokens "$TOK" \
    --num-workers 2 --lr 3e-4 --min-lr 3e-5 --warmup-steps 500 \
    --eval-every "$EV" --val-max-batches 20 --log-every "$LE" \
    --seed "$SEED" --amp-bf16 --early-stop-patience 8 \
    --dim 512 --layers 8 --heads 8 > "$OUT.trainlog" 2>&1
  [[ -f "$OUT/done.json" ]] || { log "!! FAIL train $OUT (see $OUT.trainlog)"; return 1; }
}

gen_cached(){
  local TAG=$1 CK=$2 MODE=$3 REF=$4 GBS=$5 OUT="$RUNS/gen_$1"
  [[ -f "$RUNS/fvd_$TAG.json" ]] && return 0
  [[ "$(nclips "$OUT")" -eq $NGEN ]] && { log "reuse generation $TAG"; return 0; }
  rm -rf "$OUT"; rm -f "$RUNS/_features/gen_${TAG}_gen_penultimate.pt"
  log "gen $TAG ($MODE, $(basename "$REF"))"
  "$PY" "$SRC/ar_generate.py" --ckpt "$CK" --mode "$MODE" --omnitok-ckpt "$OMNITOK_CKPT" \
    --out-dir "$OUT" --packed-ref-cache "$REF" --num-samples $NGEN --batch-size "$GBS" \
    --temperature 0.9 --top-k 256 --seed 42 --device cuda:0 > "$RUNS/gen_$TAG.log" 2>&1
  [[ "$(nclips "$OUT")" -eq $NGEN ]] || { log "!! FAIL gen $TAG"; return 1; }
}

gen_online(){
  local TAG=$1 CK=$2 MODE=$3 OUT="$RUNS/gen_$1"
  [[ -f "$RUNS/fvd_$TAG.json" ]] && return 0
  [[ "$(nclips "$OUT")" -eq $NGEN ]] && { log "reuse generation $TAG"; return 0; }
  rm -rf "$OUT"; rm -f "$RUNS/_features/gen_${TAG}_gen_penultimate.pt"
  log "gen $TAG (online, mode=$MODE)"
  "$PY" "$SRC/gen_online_select.py" --ckpt "$CK" --mode "$MODE" \
    --head "$RUNS/coarse_only_selector.pt" --out-dir "$OUT" --omnitok-ckpt "$OMNITOK_CKPT" \
    --num-samples $NGEN --batch-size 96 --temperature 0.9 --top-k 256 --keep-frac $KEEP \
    --seed 42 > "$RUNS/gen_$TAG.log" 2>&1
  [[ "$(nclips "$OUT")" -eq $NGEN ]] || { log "!! FAIL gen $TAG"; return 1; }
}

fvd(){
  local TAG=$1 REAL=$2 NF=$3 FBS=$4 OUT="$RUNS/gen_$1" FJ="$RUNS/fvd_$1.json"
  [[ -f "$FJ" ]] && { log "SKIP fvd $TAG"; return 0; }
  "$PY" "$SRC/compute_fvd.py" --gen-dir "$OUT" --real-dir "$REAL" --out-json "$FJ" \
    --num-samples $NGEN --batch-size "$FBS" --seed 42 --device cuda:0 --num-frames "$NF" \
    --use-penultimate --resume-features > "$RUNS/fvd_$TAG.log" 2>&1
  [[ -f "$FJ" ]] || { log "!! FAIL fvd $TAG (see $RUNS/fvd_$TAG.log)"; return 1; }
  log "$TAG = $(fvdval "$FJ") FVD"
  [[ "${KEEP_GEN:-0}" == 1 ]] || rm -f "$OUT"/gen_*.npy
}

dump_real_frames(){
  local RAW=$1 OUT=$2
  [[ "$(ls "$OUT"/*.npy 2>/dev/null | wc -l)" -ge $NGEN ]] && { log "SKIP real ref $OUT"; return 0; }
  log "real ref $RAW -> $OUT"
  RAW_SHARDS="$RAW/shards" REAL_OUT="$OUT" "$PY" - <<'PYEOF'
import numpy as np, glob, os
from pathlib import Path
out = Path(os.environ["REAL_OUT"]); out.mkdir(exist_ok=True, parents=True)
i = 0
for sh in sorted(glob.glob(os.path.join(os.environ["RAW_SHARDS"], "*.npz"))):
    with np.load(sh) as d:
        if "frames" not in d.files: continue
        fr = d["frames"]
        for k in range(fr.shape[0]):
            np.save(out / f"real_{i:05d}.npy", fr[k].astype(np.float16)); i += 1
            if i >= 2000: break
    if i >= 2000: break
print("real ref clips:", i)
PYEOF
}
