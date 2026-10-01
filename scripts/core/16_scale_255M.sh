#!/usr/bin/env bash
source "$(dirname "$0")/lib.sh"
need OMNITOK_CKPT
T0=omnidec_F${F}_k60
ARMS=("$@"); [[ ${#ARMS[@]} -eq 0 ]] && ARMS=(decode random)

for A in "${ARMS[@]}"; do
  T=${T0}_$A; REF="$RUNS/cache_${T}_val"
  train_run "train_${T}_255M"
  omni_arm "${T}_255M"    "train_${T}_255M" "$REF" packed               64 8
  omni_arm "${T}_255M_gc" "train_${T}_255M" "$REF" packed_gencoarse_ref 64 8
  EXT="train_${T}_255M_ext"
  if [[ ! -f "$RUNS/$EXT/done.json" ]]; then
    if [[ -f "$RUNS/$EXT/latest.pt" ]]; then
      train_run "$EXT"
    else
      [[ -f "$RUNS/train_${T}_255M/latest.pt" ]] || die "missing $RUNS/train_${T}_255M/latest.pt"
      train_run "$EXT" --set "resume_from=$RUNS/train_${T}_255M/latest.pt"
    fi
  fi
  omni_arm "${T}_255M_ext"    "$EXT" "$REF" packed               48 8
  omni_arm "${T}_255M_ext_gc" "$EXT" "$REF" packed_gencoarse_ref 48 8
done
for A in "${ARMS[@]}"; do
  for X in 255M 255M_ext; do
    log "$A $X  RS $(fvdval "$PEN/fvd_${T0}_${A}_$X.json")  SG $(fvdval "$PEN/fvd_${T0}_${A}_${X}_gc.json")"
  done
done
log "16_scale_255M DONE"
