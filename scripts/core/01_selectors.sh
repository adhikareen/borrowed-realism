#!/usr/bin/env bash
source "$(dirname "$0")/lib.sh"
PHASE=${1:-all}
T=omnidec_F${F}_k60

phaseA(){
  need OMNITOK_CKPT
  build_cache "$RAWTR" "$RUNS/cache_${T}_decode_train" decode $KEEP 8000
  build_cache "$RAWVA" "$RUNS/cache_${T}_decode_val"   decode $KEEP 2000
  build_cache "$RAWTR" "$RUNS/cache_${T}_random_train" random $KEEP 8000
  build_cache "$RAWVA" "$RUNS/cache_${T}_random_val"   random $KEEP 2000
  build_cache "$RAWTR" "$RUNS/cache_${T}_full_train"   full   1.0   8000
  build_cache "$RAWVA" "$RUNS/cache_${T}_full_val"     full   1.0   2000
  build_cache "$RAWTR" "$RUNS/cache_${T}_l2gap_train" l2gap $KEEP 8000
  build_cache "$RAWVA" "$RUNS/cache_${T}_l2gap_val"   l2gap $KEEP 2000
  local PREDTR=$RUNS/omni_learned_extract_F${F}_train PREDVA=$RUNS/omni_learned_extract_F${F}_val
  local SELMETRICS=$RUNS/p2_learned_selector_metrics.json
  if [[ -f "$SELMETRICS" && -f "$PREDVA/shards/shard_00000.npz" ]]; then
    log "SKIP selector-train (metrics + pred-shards present)"
  else
    log "SELECTOR-TRAIN (mlp on decode-free codebook features)"
    "$PY" "$SRC/train_learned_selector.py" --omni-ckpt "$OMNITOK_CKPT" \
      --train-raw "$RAWTR" --val-raw "$RAWVA" \
      --out-train "$PREDTR" --out-val "$PREDVA" \
      --metrics-json "$SELMETRICS" \
      --keep-frac $KEEP --epochs 6 --train-num 8000 --seed 1337
  fi
  [[ -f "$PREDVA/shards/shard_00000.npz" ]] || die "no pred shards"
  build_cache "$PREDTR" "$RUNS/cache_${T}_learned_train" learned $KEEP 8000
  build_cache "$PREDVA" "$RUNS/cache_${T}_learned_val"   learned $KEEP 2000
  local SEL
  for SEL in contigmax l2gap_smooth; do
    build_cache "$RAWTR" "$RUNS/cache_${T}_${SEL}_train" $SEL $KEEP 8000
    build_cache "$RAWVA" "$RUNS/cache_${T}_${SEL}_val"   $SEL $KEEP 2000
  done
  for SEL in decode random l2gap learned contigmax l2gap_smooth; do
    check_K "$RUNS/cache_${T}_${SEL}_train" 768; check_K "$RUNS/cache_${T}_${SEL}_val" 768
  done
  local KK KP
  for KK in k40:0.40 k25:0.25 k80:0.80; do
    KP=${KK#*:}; KK=${KK%%:*}
    for SEL in decode random; do
      build_cache "$RAWTR" "$RUNS/cache_omnidec_F${F}_${KK}_${SEL}_train" $SEL $KP 8000
      build_cache "$RAWVA" "$RUNS/cache_omnidec_F${F}_${KK}_${SEL}_val"   $SEL $KP 2000
    done
  done
}

extract_scored(){
  local SCRIPT=$1 CACHE=$2 RAW=$3 OUT=$4 EBS=$5
  [[ -f "$OUT/summary.json" ]] && { log "SKIP $SCRIPT $OUT (summary exists)"; return 0; }
  log "$SCRIPT $CACHE -> $OUT (bs=$EBS)"
  "$PY" "$SRC/$SCRIPT.py" --ckpt "$RUNS/train_${T}_full/best.pt" \
    --cache-dir "$CACHE" --raw-dir "$RAW" --out-dir "$OUT" \
    --batch-size "$EBS" --device cuda:0
  [[ -f "$OUT/summary.json" ]] || die "$SCRIPT extraction failed for $OUT"
}

phaseB(){
  [[ -f "$RUNS/train_${T}_full/best.pt" ]] || die "phase B needs $RUNS/train_${T}_full/best.pt (02_train.sh train_${T}_full)"
  local SEL SPL N RAW
  extract_scored extract_cell_nll "$RUNS/cache_${T}_full_train" "$RAWTR" "$RUNS/cell_nll_F${F}_train" 32
  extract_scored extract_cell_nll "$RUNS/cache_${T}_full_val"   "$RAWVA" "$RUNS/cell_nll_F${F}_val"   32
  for SEL in genscore genscore05; do
    build_cache "$RAWTR" "$RUNS/cache_${T}_${SEL}_train" $SEL $KEEP 8000 --nll-dir "$RUNS/cell_nll_F${F}_train"
    build_cache "$RAWVA" "$RUNS/cache_${T}_${SEL}_val"   $SEL $KEEP 2000 --nll-dir "$RUNS/cell_nll_F${F}_val"
  done
  extract_scored extract_gensal  "$RUNS/cache_${T}_full_train" "$RAWTR" "$RUNS/gensal_extract_F${F}_train"  16
  extract_scored extract_gensal  "$RUNS/cache_${T}_full_val"   "$RAWVA" "$RUNS/gensal_extract_F${F}_val"    16
  extract_scored extract_attninf "$RUNS/cache_${T}_full_train" "$RAWTR" "$RUNS/attninf_extract_F${F}_train" 8
  extract_scored extract_attninf "$RUNS/cache_${T}_full_val"   "$RAWVA" "$RUNS/attninf_extract_F${F}_val"   8
  for SEL in gensal attninf; do
    build_cache "$RAWTR" "$RUNS/cache_${T}_${SEL}_train" $SEL $KEEP 8000 --sal-dir "$RUNS/${SEL}_extract_F${F}_train"
    build_cache "$RAWVA" "$RUNS/cache_${T}_${SEL}_val"   $SEL $KEEP 2000 --sal-dir "$RUNS/${SEL}_extract_F${F}_val"
  done
  for SEL in genscore genscore05 gensal attninf; do
    check_K "$RUNS/cache_${T}_${SEL}_train" 768; check_K "$RUNS/cache_${T}_${SEL}_val" 768
  done
}

case $PHASE in
  A) phaseA ;; B) phaseB ;;
  all) phaseA; phaseB ;;
  *) die "usage: $0 [A|B|all]" ;;
esac
log "01_selectors $PHASE DONE"
