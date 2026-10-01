#!/usr/bin/env bash
source "$(dirname "$0")/lib.sh"
need OMNITOK_CKPT

for KK in k40 k25 k80; do
  for A in decode random; do
    TAG=omnidec_F${F}_${KK}_${A}; REF="$RUNS/cache_${TAG}_val"
    train_run "train_$TAG"
    omni_arm "$TAG"    "train_$TAG" "$REF" packed               96 16
    omni_arm "${TAG}_gc" "train_$TAG" "$REF" packed_gencoarse_ref 16 16
    blank_nonframe "$RUNS/gen_${TAG}_gc" "$PEN/fvd_${TAG}_gc_blank.json"
  done
done
for KK in k25 k40 k60 k80; do
  for A in decode random; do
    TAG=omnidec_F${F}_${KK}_${A}
    log "$KK $A  RS $(fvdval "$PEN/fvd_$TAG.json")  SG $(fvdval "$PEN/fvd_${TAG}_gc.json")"
  done
done
log "13_budget_sweep DONE"
