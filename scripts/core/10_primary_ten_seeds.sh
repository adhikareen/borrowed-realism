#!/usr/bin/env bash
source "$(dirname "$0")/lib.sh"
need OMNITOK_CKPT
T=omnidec_F${F}_k60
ARMS=("$@"); [[ ${#ARMS[@]} -eq 0 ]] && ARMS=(decode random learned)
SEEDS=(1337 42 101 123 202 303 404 456 505 789)

for A in "${ARMS[@]}"; do
  for S in "${SEEDS[@]}"; do
    SUF=""; [[ $S != 1337 ]] && SUF="_s$S"
    RUN="train_${T}_${A}${SUF}"; REF="$RUNS/cache_${T}_${A}_val"
    train_run "$RUN"
    case $S in
      1337)            RSBS=64; SGBS=96; FRS=16; FSG=16 ;;
      42|123|456|789)  RSBS=96; SGBS=96; FRS=16; FSG=16 ;;
      *)               RSBS=64; SGBS=64; FRS=8;  FSG=8 ;;
    esac
    if [[ $A == learned ]]; then
      if [[ $S == 1337 ]]; then RSBS=96; else SGBS=64; FSG=8; fi
    fi
    if [[ $A != learned || $S == 1337 || $S == 42 || $S == 123 ]]; then
      omni_arm "${T}_${A}${SUF}" "$RUN" "$REF" packed $RSBS $FRS
    fi
    omni_arm "${T}_${A}${SUF}_gc" "$RUN" "$REF" packed_gencoarse_ref $SGBS $FSG
    blank "$RUNS/gen_${T}_${A}${SUF}_gc" "$RUNS/gc_blank_k60_${A}${SUF}.json"
  done
done

for A in "${ARMS[@]}"; do
  for S in "${SEEDS[@]}"; do
    SUF=""; [[ $S != 1337 ]] && SUF="_s$S"
    log "$A s$S  RS $(fvdval "$PEN/fvd_${T}_${A}${SUF}.json")  SG $(fvdval "$PEN/fvd_${T}_${A}${SUF}_gc.json")"
  done
done
log "10_primary_ten_seeds DONE"
