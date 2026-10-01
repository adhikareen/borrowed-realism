#!/usr/bin/env bash
source "$(dirname "$0")/common.sh"
SEEDS="42 101 123 202 303 404 456 789"
REAL=$RUNS/real_decode_F17_val
C=$RUNS/cache_omnidec_F17_k60

for S in $SEEDS; do
  case $S in 42|123) BS=64; NAME=k60_decode_s$S ;;
             456|789) BS=48; NAME=omnidec_F17_k60_decode_s$S ;;
             *)       BS=32; NAME=omnidec_F17_k60_decode_s$S ;;
  esac
  train_ar "$RUNS/train_omnidec_F17_k60_decode_s$S" "${C}_decode_train" "${C}_decode_val" "$NAME" "$S" "$BS"
done

RAWV=$RUNS/omni_decode_extract_F17_val
build_cache "$RAWV" "${C}_alignedNEW_val"   decode          2000
build_cache "$RAWV" "${C}_derangedNEW_val"  decode_deranged 2000
build_cache "$RAWV" "${C}_contigmaxNEW_val" contigmax       2000
build_cache "$RAWV" "${C}_randomNEW_val"    random          2000

for S in $SEEDS; do
  CK=$RUNS/train_omnidec_F17_k60_decode_s$S/best.pt
  for R in rs sg; do
    MODE=packed; [[ $R == sg ]] && MODE=packed_gencoarse_ref
    for M in aligned deranged fixed; do
      case $M in aligned) CA=alignedNEW ;; deranged) CA=derangedNEW ;; fixed) CA=contigmaxNEW ;; esac
      gen_cached "fact_${R}_${M}_s$S" "$CK" $MODE "${C}_${CA}_val" 96
      fvd        "fact_${R}_${M}_s$S" "$REAL" 17 16
    done
    FBS=8; [[ $R == rs && ( $S == 101 || $S == 456 ) ]] && FBS=16
    gen_cached "fact2_${R}_random_s$S" "$CK" $MODE "${C}_randomNEW_val" 96
    fvd        "fact2_${R}_random_s$S" "$REAL" 17 $FBS
  done
  gen_online "fact2_sg_online_s$S" "$CK" coarse_only
  fvd        "fact2_sg_online_s$S" "$REAL" 17 16
done
log "MASK_FACTORIAL_K400_DONE"
