#!/usr/bin/env bash
source "$(dirname "$0")/lib.sh"
need OMNITOK_CKPT
T=omnidec_F${F}_k60
for A in decode random; do [[ -f "$RUNS/train_${T}_$A/best.pt" ]] || die "run 11_ladder_seed1337.sh first ($A)"; done

for AL in 0.5 0.75 0.25; do
  for A in decode random; do
    omni_arm "r3_${A}_a${AL/0./}" "train_${T}_$A" "$RUNS/cache_${T}_${A}_val" packed_gencoarse_ref 64 8 \
      --temperature 0.9 --top-k 256 --coarse-mix-alpha $AL
  done
done
for S in "0.7 64" "1.0 256"; do read -r TT KK <<< "$S"
  for A in decode random; do
    omni_arm "r3b_${A}_T${TT}_k${KK}" "train_${T}_$A" "$RUNS/cache_${T}_${A}_val" packed_gencoarse_ref 64 8 \
      --temperature $TT --top-k $KK
  done
done
for t in a25 a5 a75 T0.7_k64 T1.0_k256; do
  p=r3; [[ $t == T* ]] && p=r3b
  log "$p $t  decode $(fvdval "$PEN/fvd_${p}_decode_$t.json")  random $(fvdval "$PEN/fvd_${p}_random_$t.json")"
done
log "14_mixing_and_samplers DONE"
