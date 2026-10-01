#!/usr/bin/env bash
_LIB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$_LIB_DIR/env.sh"

log(){ echo "[core][$(date '+%F %T')] $*"; }
die(){ log "FATAL $*"; exit 1; }
need(){ local v; for v in "$@"; do [[ -n "${!v:-}" ]] || die "set $v (see scripts/core/env.sh)"; done; }
fvdval(){ "$PY" -c "import json,sys;print(round(json.load(open(sys.argv[1]))['fvd'],3))" "$1" 2>/dev/null || echo NA; }
nclips(){ ls "$1"/gen_*.npy 2>/dev/null | wc -l; }

gen_bs(){
  local rel="${1#$RUNS/}" b
  b=$(awk -F'\t' -v g="$rel" '$1==g {print $3}' "$CONF/generations.tsv")
  echo "${b:-$2}"
}

build_cache(){
  local RAW=$1 OUT=$2 SEL=$3 KP=$4 N=$5; shift 5
  [[ -f "$OUT/manifest.json" ]] && { log "SKIP cache $OUT"; return 0; }
  log "CACHE sel=$SEL keep=$KP N=$N -> $OUT"
  "$PY" "$SRC/build_stream_cache_omni_long.py" --raw-dir "$RAW" --out-dir "$OUT" \
    --sel "$SEL" --keep-frac "$KP" --num-videos "$N" "$@"
  [[ -f "$OUT/manifest.json" ]] || die "cache $OUT"
}

check_K(){
  local fb; fb=$("$PY" -c "import json,sys;print(json.load(open(sys.argv[1]+'/manifest.json'))['packed_fine_budget'])" "$1")
  [[ "$fb" == "$2" ]] || die "$1 fine_budget=$fb != $2"
}

train_run(){
  local RUN=$1; shift
  local OUT="$RUNS/$RUN" CFG="$CONF/runs.json"
  [[ -f "$CFG" ]] || die "no recorded config $CFG"
  [[ -f "$OUT/done.json" ]] && { log "SKIP train $RUN (done.json)"; return 0; }
  local -a ARGS
  mapfile -t ARGS < <("$PY" "$CORE/py/train_cmd.py" --config "$CFG" --run "$RUN" --runs "$RUNS" --repo "$REPO" "$@")
  [[ ${#ARGS[@]} -gt 0 ]] || die "no config for $RUN in $CFG"
  mkdir -p "$OUT"
  log "TRAIN $RUN -> $RUN.trainlog"
  "$PY" "$SRC/train.py" "${ARGS[@]}" >> "$RUNS/$RUN.trainlog" 2>&1
  [[ -f "$OUT/done.json" && -f "$OUT/best.pt" ]] || die "train $RUN (see $RUNS/$RUN.trainlog)"
}

gen(){
  local TOK=$1 OUT=$2 CK=$3 MODE=$4 REF=$5 DBS=$6; shift 6
  [[ -f "$OUT/ar_generate_done.json" ]] && { log "SKIP gen $OUT"; return 0; }
  [[ -f "$CK" ]] || die "missing checkpoint $CK"
  local -a TK REFA=()
  case $TOK in
    omni)   need OMNITOK_CKPT; TK=(--omnitok-ckpt "$OMNITOK_CKPT") ;;
    om2)    need OMAG_CKPT;    TK=(--om2-ckpt "$OMAG_CKPT") ;;
    cosmos) need COSMOS_CKPT;  TK=(--cosmos-ckpt "$COSMOS_CKPT") ;;
    *) die "gen: unknown tokenizer $TOK" ;;
  esac
  [[ "$REF" != "-" ]] && REFA=(--packed-ref-cache "$REF")
  local BS; BS=$(gen_bs "$OUT" "$DBS")
  log "GEN $OUT (mode=$MODE bs=$BS $*)"
  "$PY" "$SRC/ar_generate.py" --ckpt "$CK" --mode "$MODE" "${TK[@]}" "${REFA[@]}" --out-dir "$OUT" \
    --num-samples $NGEN --batch-size "$BS" --seed 42 --device cuda:0 "$@"
  [[ -f "$OUT/ar_generate_done.json" ]] || die "gen $OUT"
}

fvd(){
  local G=$1 R=$2 OUT=$3 FBS=$4
  [[ -f "$OUT" ]] && { log "SKIP fvd $OUT = $(fvdval "$OUT")"; return 0; }
  mkdir -p "$(dirname "$OUT")"
  "$PY" "$SRC/compute_fvd.py" --gen-dir "$G" --real-dir "$R" --out-json "$OUT" \
    --num-samples $NGEN --batch-size "$FBS" --seed 42 --device cuda:0 --num-frames $F \
    --use-penultimate --resume-features
  [[ -f "$OUT" ]] || die "fvd $OUT"
  log "FVD $(basename "$OUT") = $(fvdval "$OUT")"
}

blank(){
  [[ -f "$2" ]] && return 0
  "$PY" "$SRC/blank_collapse.py" --gen-dir "$1" --num-samples $NGEN > "$2" 2>/dev/null || log "blank_collapse $1 failed"
}

blank_nonframe(){
  [[ -f "$2" ]] && return 0
  "$PY" - "$1" "$2" <<'PYEOF'
import sys,glob,json,numpy as np
gd,out=sys.argv[1],sys.argv[2]
fs=[x for x in sorted(glob.glob(gd+"/*.npy")) if 'done' not in x]
st=[float(np.asarray(np.load(x),dtype=np.float32).std()) for x in fs]
nb=sum(1 for s in st if s<0.02)
json.dump({"n":len(fs),"n_blank":nb,"blank_pct":round(100*nb/max(len(fs),1),2),
          "mean_std":float(np.mean(st)) if st else 0.0},open(out,"w"),indent=2)
PYEOF
}

omni_arm(){
  local TAG=$1 RUN=$2 REFC=$3 MODE=$4 DBS=$5 FBS=$6; shift 6
  local -a S=("${SAMPLER[@]}"); [[ $# -gt 0 ]] && S=("$@")
  gen omni "$RUNS/gen_$TAG" "$RUNS/$RUN/best.pt" "$MODE" "$REFC" "$DBS" "${S[@]}"
  fvd "$RUNS/gen_$TAG" "$REAL" "$PEN/fvd_$TAG.json" "$FBS"
}
