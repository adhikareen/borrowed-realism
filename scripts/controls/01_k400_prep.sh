#!/usr/bin/env bash
source "$(dirname "$0")/common.sh"

for SPL in train val; do
  N=8000; SF=""; [[ $SPL == val ]] && { N=2000; SF="--store-frames"; }
  OUT=$RUNS/omni_decode_extract_F17_$SPL
  [[ -f "$OUT/.extract_done" ]] && { log "SKIP extract $SPL"; continue; }
  log "extract K400 $SPL (n=$N)"
  "$PY" "$SRC/extract_omnitok_decode.py" --ckpt "$OMNITOK_CKPT" --filelist "$DATA_ROOT/k400_$SPL.txt" \
    --out-dir "$OUT" --num $N --batch-size 32 --shard-size 1000 --num-frames 17 --num-workers 2 $SF \
    > "$RUNS/rebuild_extract_$SPL.log" 2>&1 && touch "$OUT/.extract_done"
done

[[ -f "$RUNS/real_decode_F17_val/real_0000000.npy" ]] || \
  "$PY" "$SRC/dump_real_ref_long.py" --raw-dir "$RUNS/omni_decode_extract_F17_val" \
    --out-dir "$RUNS/real_decode_F17_val" --limit $NGEN

build_cache "$RUNS/omni_decode_extract_F17_train" "$RUNS/cache_omnidec_F17_k60_decode_train" decode 8000
build_cache "$RUNS/omni_decode_extract_F17_val"   "$RUNS/cache_omnidec_F17_k60_decode_val"   decode 2000
build_cache "$RUNS/omni_decode_extract_F17_train" "$RUNS/cache_omnidec_F17_k60_random_train" random 8000
build_cache "$RUNS/omni_decode_extract_F17_val"   "$RUNS/cache_omnidec_F17_k60_random_val"   random 2000

if [[ ! -f "$RUNS/coarse_only_selector.pt" ]]; then
  log "train coarse-only head"
  "$PY" "$SRC/train_coarse_only_selector.py" --omni-ckpt "$OMNITOK_CKPT" \
    --train-raw "$RUNS/omni_decode_extract_F17_train" --val-raw "$RUNS/omni_decode_extract_F17_val" \
    --out "$RUNS/coarse_only_selector.pt" --metrics-json "$RUNS/coarse_only_selector_metrics.json" \
    --out-train "$RUNS/omni_coarseonly_extract_F17_train" --out-val "$RUNS/omni_coarseonly_extract_F17_val" \
    --train-num 8000 --keep-frac $KEEP
fi
log "K400_PREP_DONE"
