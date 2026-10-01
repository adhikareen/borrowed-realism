#!/usr/bin/env bash
source "$(dirname "$0")/lib.sh"
need OMNITOK_CKPT
T=omnidec_F${F}_k60

for A in contigmax l2gap; do
  train_run "train_${T}_$A"
  omni_arm "${T}_$A"    "train_${T}_$A" "$RUNS/cache_${T}_${A}_val" packed               96 16
  omni_arm "${T}_${A}_gc" "train_${T}_$A" "$RUNS/cache_${T}_${A}_val" packed_gencoarse_ref 96 16
  blank "$RUNS/gen_${T}_${A}_gc" "$RUNS/gc_blank_k60_$A.json"
done
for A in contigmax l2gap learned; do
  for S in 42 123; do
    RUN="train_${T}_${A}_s$S"; REF="$RUNS/cache_${T}_${A}_val"
    train_run "$RUN"
    omni_arm "${T}_${A}_s$S" "$RUN" "$REF" packed 96 16
    omni_arm "${T}_${A}_s${S}_gc" "$RUN" "$REF" packed_gencoarse_ref 64 8
    blank "$RUNS/gen_${T}_${A}_s${S}_gc" "$RUNS/gc_blank_k60_${A}_s$S.json"
  done
done
for A in contigmax l2gap learned; do
  for S in 1337 42 123; do
    SUF=""; [[ $S != 1337 ]] && SUF="_s$S"
    log "$A s$S  RS $(fvdval "$PEN/fvd_${T}_${A}${SUF}.json")  SG $(fvdval "$PEN/fvd_${T}_${A}${SUF}_gc.json")"
  done
done
log "12_three_seed_controls DONE"
