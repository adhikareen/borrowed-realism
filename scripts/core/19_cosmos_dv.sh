#!/usr/bin/env bash
source "$(dirname "$0")/lib.sh"
need COSMOS_CKPT
WHICH=${1:-core}
DV_REAL=$RUNS/real_npy_from_stage0_val
CKPT_BLEND=$RUNS/train_67M_packed_tpf50_seed1337/best.pt
CKPT_DENSE=$RUNS/train_67M_dense_seed1337/best.pt
REF_TPF50=$RUNS/cache_packed_tpf50_val
declare -A REF_FOR_K=( [190]=$RUNS/cache_packed_tpf30_val [530]=$RUNS/cache_packed_tpf50_val [955]=$RUNS/cache_packed_tpf75_val )
declare -A CKPT_FOR_K=( [190]=$RUNS/train_67M_packed_tpf30_seed1337/best.pt [530]=$CKPT_BLEND [955]=$RUNS/train_67M_packed_tpf75_seed1337/best.pt )
mkdir -p "$DV_SWEEP"

ensure_ckpt(){
  [[ -f "$RUNS/$1/best.pt" ]] && return 0
  [[ -f "$RUNS/$2/manifest.json" ]] || die "missing $RUNS/$1/best.pt and its cache $RUNS/$2 (Cosmos-DV caches are not built by this repository)"
  train_run "$1"
}
run_metrics(){
  local tag=$1 g="$DV_SWEEP/gen_$1"
  if [[ ! -f "$DV_SWEEP/blank_$tag.json" ]]; then
    "$PY" "$SRC/blank_collapse.py" --gen-dir "$g" --num-samples $NGEN --thresh 0.02 --out-json "$DV_SWEEP/blank_$tag.json"
  fi
  fvd "$g" "$DV_REAL" "$DV_SWEEP/fvd_$tag.json" 16
}

job2a(){
  ensure_ckpt train_67M_packed_tpf50_seed1337 cache_packed_tpf50_train
  local arm mode T K tag
  for arm in tf gen; do
    if [[ $arm == tf ]]; then mode=packed; else mode=packed_gencoarse_ref; fi
    for T in 0.85 1.0; do for K in 64 256; do
      tag="j2a_${arm}_T${T}_k${K}"
      gen cosmos "$DV_SWEEP/gen_$tag" "$CKPT_BLEND" $mode "$REF_TPF50" 48 --temperature $T --top-k $K
      run_metrics "$tag"
    done; done
  done
}
job3(){
  ensure_ckpt train_67M_dense_seed1337 cache_dense_train
  local TP RP T tag
  for TP in 0.9 0.95; do for RP in 1.0 1.1; do for T in 0.85 1.0; do
    tag="j3_dense_tp${TP}_rp${RP}_T${T}"
    gen cosmos "$DV_SWEEP/gen_$tag" "$CKPT_DENSE" dense - 48 \
      --temperature $T --top-k 0 --top-p $TP --repetition-penalty $RP
    run_metrics "$tag"
  done; done; done
}
job2b(){
  local Kf
  for Kf in 190 530 955; do
    gen cosmos "$DV_SWEEP/gen_j2b_anch_K$Kf" "${CKPT_FOR_K[$Kf]}" packed "${REF_FOR_K[$Kf]}" 48 --temperature 1.0 --top-k 256
    run_metrics "j2b_anch_K$Kf"
    gen cosmos "$DV_SWEEP/gen_j2b_denstrunc_N$Kf" "$CKPT_DENSE" dense_trunc - 48 --dense-trunc-n $Kf --temperature 1.0 --top-k 256
    run_metrics "j2b_denstrunc_N$Kf"
  done
}
case $WHICH in
  2a) job2a ;; 3) job3 ;; 2b) job2b ;;
  core) job2a; job3 ;;
  *) die "usage: $0 [2a|3|2b|core]" ;;
esac
for f in "$DV_SWEEP"/fvd_j2a_*.json "$DV_SWEEP"/fvd_j3_*.json; do
  [[ -f $f ]] && log "$(basename "$f" .json)  FVD $(fvdval "$f")"
done
log "19_cosmos_dv $WHICH DONE"
