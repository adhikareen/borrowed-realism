#!/usr/bin/env bash
source "$(dirname "$0")/lib.sh"
need OMNITOK_CKPT
[[ -d "$REAL_SSV2" ]] || die "run 00_extract.sh ssv2 first"

for SEL in decode random; do
  for SPL in train val; do
    N=8000; [[ $SPL == val ]] && N=2000
    build_cache "$RUNS/ssv2_decode_extract_F17_${SPL}" "$RUNS/cache_ssv2_F17_k60_${SEL}_${SPL}" $SEL $KEEP $N
  done
done
for SEL in decode random; do train_run "train_ssv2_F17_k60_$SEL"; done
for SEL in decode random; do
  T=ssv2_F17_k60_$SEL
  for M in packed packed_gencoarse_ref; do
    SUF=""; [[ $M == packed_gencoarse_ref ]] && SUF="_gc"
    gen omni "$RUNS/gen_${T}${SUF}" "$RUNS/train_$T/best.pt" $M "$RUNS/cache_${T}_val" 64 "${SAMPLER[@]}"
    fvd "$RUNS/gen_${T}${SUF}" "$REAL_SSV2" "$PEN/fvd_${T}${SUF}.json" 8
  done
done
for SEL in decode random; do
  log "SSv2 $SEL  RS $(fvdval "$PEN/fvd_ssv2_F17_k60_$SEL.json")  SG $(fvdval "$PEN/fvd_ssv2_F17_k60_${SEL}_gc.json")"
done
log "17_ssv2 DONE"
