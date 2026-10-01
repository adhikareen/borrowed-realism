#!/usr/bin/env bash
source "$(dirname "$0")/common.sh"
REAL=$RUNS/real_decode_F17_val
C=$RUNS/cache_omnidec_F17_k60

build_cache "$RUNS/omni_decode_extract_F17_train" "${C}_taildrop_train" taildrop 8000
build_cache "$RUNS/omni_decode_extract_F17_val"   "${C}_taildrop_val"   taildrop 2000
build_cache "$RUNS/omni_decode_extract_F17_val"   "${C}_randomNEW_val"  random   2000

for S in 42 123; do
  train_ar "$RUNS/train_omnidec_F17_k60_taildrop_s$S" "${C}_taildrop_train" "${C}_taildrop_val" "taildrop_s$S" $S 64
  train_ar "$RUNS/train_omnidec_F17_k60_random_s$S"   "${C}_random_train"   "${C}_random_val"   "k60_random_s$S" $S 64
done

for S in 42 123; do
  for R in rs sg; do
    MODE=packed; [[ $R == sg ]] && MODE=packed_gencoarse_ref
    gen_cached "taildrop_${R}_s$S" "$RUNS/train_omnidec_F17_k60_taildrop_s$S/best.pt" $MODE "${C}_taildrop_val" 96
    fvd        "taildrop_${R}_s$S" "$REAL" 17 16
    gen_cached "fact_${R}_random_s$S" "$RUNS/train_omnidec_F17_k60_random_s$S/best.pt" $MODE "${C}_randomNEW_val" 96
    fvd        "fact_${R}_random_s$S" "$REAL" 17 16
  done
done
log "PREFIX_TAILDROP_DONE"
