#!/usr/bin/env bash
source "$(dirname "$0")/common.sh"
REAL=$RUNS/real_decode_F17_val
C=$RUNS/cache_omnidec_F17_k60
[[ -f $RUNS/coarse_only_selector.pt ]] || { log "run 01_k400_prep.sh first (coarse-only head)"; exit 1; }

build_cache "$RUNS/omni_coarseonly_extract_F17_train" "${C}_coarseonly_train" learned 8000
build_cache "$RUNS/omni_coarseonly_extract_F17_val"   "${C}_coarseonly_val"   learned 2000
build_cache "$RUNS/omni_decode_extract_F17_train"     "${C}_randomNEW_train"  random  8000
build_cache "$RUNS/omni_decode_extract_F17_val"       "${C}_randomNEW_val"    random  2000

DESIGN="42 123 456 789"
PROSP="7 99 555 1001 1002 1003 1004 1005 1006 1007 1008"
for S in $DESIGN $PROSP; do
  for A in coarseonly randomNEW; do
    OUT=$RUNS/train_omnidec_F17_k60_${A}_s$S
    train_ar "$OUT" "${C}_${A}_train" "${C}_${A}_val" "$(basename "$OUT")" "$S" 64
  done
done

ck(){ echo "$RUNS/train_omnidec_F17_k60_$1_s$2/best.pt"; }

for S in $DESIGN; do
  gen_online "online_coarseonly_s$S"                   "$(ck coarseonly $S)" coarse_only
  fvd        "online_coarseonly_s$S"                   "$REAL" 17 16
  gen_online "online_randomNEW_s$S"                    "$(ck randomNEW $S)"  random
  fvd        "online_randomNEW_s$S"                    "$REAL" 17 16
  gen_online "cross_train-coarseonly_mask-random_s$S"  "$(ck coarseonly $S)" random
  fvd        "cross_train-coarseonly_mask-random_s$S"  "$REAL" 17 16
  gen_online "cross_train-randomNEW_mask-coarse_only_s$S" "$(ck randomNEW $S)" coarse_only
  fvd        "cross_train-randomNEW_mask-coarse_only_s$S" "$REAL" 17 16
done

for S in $PROSP; do
  for A in coarseonly randomNEW; do
    gen_online "prosp_${A}_s$S" "$(ck $A $S)" random
    fvd        "prosp_${A}_s$S" "$REAL" 17 16
  done
done
log "CURATION_DONE"
