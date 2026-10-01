#!/usr/bin/env bash
source "$(dirname "$0")/lib.sh"
need OMAG_CKPT
TAG=omag_F${F}_k60
RAWTR_OM=$RUNS/omag_F17_extract_train; RAWVA_OM=$RUNS/omag_F17_extract_val
[[ -f "$RAWTR_OM/.extract_done" && -f "$RAWVA_OM/.extract_done" ]] || die "run 00_extract.sh omag first"
mkdir -p "$OMAG_PEN"

cache_om(){
  [[ -f "$5/manifest.json" ]] && { log "SKIP cache $5"; return 0; }
  log "CACHE(om2) sel=$2 keep=$3 -> $5"
  "$PY" "$SRC/build_stream_cache_omag.py" --raw-dir "$1" --out-dir "$5" \
    --sel "$2" --keep-frac "$3" --num-videos "$4" --seed 1337
}
for SEL in decode random l2gap contigmax; do
  cache_om "$RAWTR_OM" $SEL $KEEP 8000 "$RUNS/cache_${TAG}_${SEL}_train"
  cache_om "$RAWVA_OM" $SEL $KEEP 2000 "$RUNS/cache_${TAG}_${SEL}_val"
done
cache_om "$RAWTR_OM" full 1.0 8000 "$RUNS/cache_omag_F${F}_full_train"
cache_om "$RAWVA_OM" full 1.0 2000 "$RUNS/cache_omag_F${F}_full_val"

PREDTR=$RUNS/omag_learned_extract_F17_train; PREDVA=$RUNS/omag_learned_extract_F17_val
SELMETRICS=$RUNS/omag_selector_metrics.json
if [[ -f "$SELMETRICS" && -f "$PREDVA/shards/shard_00000.npz" ]]; then
  log "SKIP selector-train (metrics + pred shards present)"
else
  log "SELECTOR-TRAIN (54-d LFQ sign features)"
  "$PY" "$SRC/train_learned_selector_omag.py" \
    --train-raw "$RAWTR_OM" --val-raw "$RAWVA_OM" \
    --out-train "$PREDTR" --out-val "$PREDVA" \
    --metrics-json "$SELMETRICS" \
    --keep-frac $KEEP --epochs 6 --train-num 8000 --seed 1337
fi
cache_om "$PREDTR" learned $KEEP 8000 "$RUNS/cache_${TAG}_learned_train"
cache_om "$PREDVA" learned $KEEP 2000 "$RUNS/cache_${TAG}_learned_val"
for SEL in decode random l2gap learned contigmax; do
  check_K "$RUNS/cache_${TAG}_${SEL}_train" 768; check_K "$RUNS/cache_${TAG}_${SEL}_val" 768
done

for SEL in decode random l2gap learned; do train_run "train_${TAG}_$SEL"; done
train_run "train_omag_F${F}_full"
train_run "train_${TAG}_contigmax"

for SEL in decode random l2gap learned contigmax; do
  gen om2 "$RUNS/gen_${TAG}_$SEL" "$RUNS/train_${TAG}_$SEL/best.pt" packed "$RUNS/cache_${TAG}_${SEL}_val" 16 "${SAMPLER[@]}"
  fvd "$RUNS/gen_${TAG}_$SEL" "$REAL" "$OMAG_PEN/fvd_${TAG}_$SEL.json" 16
done
gen om2 "$RUNS/gen_omag_F${F}_full" "$RUNS/train_omag_F${F}_full/best.pt" packed "$RUNS/cache_omag_F${F}_full_val" 16 "${SAMPLER[@]}"
fvd "$RUNS/gen_omag_F${F}_full" "$REAL" "$OMAG_PEN/fvd_omag_F${F}_full.json" 16

for SEL in decode random contigmax; do
  gen om2 "$RUNS/gen_${TAG}_${SEL}_gc" "$RUNS/train_${TAG}_$SEL/best.pt" packed_gencoarse_ref "$RUNS/cache_${TAG}_${SEL}_val" 16 "${SAMPLER[@]}"
  fvd "$RUNS/gen_${TAG}_${SEL}_gc" "$REAL" "$OMAG_PEN/fvd_${TAG}_${SEL}_gc.json" 16
  if [[ $SEL != contigmax ]]; then blank_nonframe "$RUNS/gen_${TAG}_${SEL}_gc" "$OMAG_PEN/fvd_${TAG}_${SEL}_gc_blank.json"; fi
done

for t in decode random contigmax decode_gc random_gc contigmax_gc; do
  fvd "$RUNS/gen_${TAG}_$t" "$REAL" "$PEN/fvd_${TAG}_$t.json" 8
done

for SEL in decode random l2gap learned contigmax; do
  log "OM2 $SEL  RS $(fvdval "$OMAG_PEN/fvd_${TAG}_$SEL.json")  SG $(fvdval "$OMAG_PEN/fvd_${TAG}_${SEL}_gc.json")"
done
log "OM2 full RS $(fvdval "$OMAG_PEN/fvd_omag_F${F}_full.json")"
log "18_openmagvit2 DONE"
