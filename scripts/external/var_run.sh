#!/usr/bin/env bash
set -euo pipefail
REPO=${REPO:-$(cd "$(dirname "$0")/../.." && pwd)}
: "${EXT_ROOT:?set EXT_ROOT to the directory holding the upstream checkouts}"
: "${IMAGENET_TRAIN:?ImageNet train folder (class subfolders): source of the real-prior images}"
: "${IMAGENET_VAL:?ImageNet val folder (class subfolders): FID reference}"
PY=${PY:-python}
GPU=${GPU:-0}
VAR=$EXT_ROOT/VAR
RUNS=runs
COMMON=(--depth 16 --var_ckpt ckpt/var_d16.pth --vae_ckpt ckpt/vae_ch160v4096z32.pth --prune_from_si 6
        --cfg 4.0 --top_k 900 --top_p 0.95 --num_classes 1000 --imgs_per_class 10 --classes_per_batch 10
        --real_img_dir "$IMAGENET_TRAIN")
export CUDA_VISIBLE_DEVICES=$GPU PYTHONUNBUFFERED=1
cd "$VAR"
STAGES=("$@"); [ ${#STAGES[@]} -eq 0 ] && STAGES=(ref table3 sweep collect)

sel(){
  local out=$1 seed=$2; shift 2
  "$PY" var_select_run.py --out_dir "$RUNS/$out" --ref_stats "$RUNS/leak_full/ref_stats.npz" --seed "$seed" \
        --keep_ratios "$@" "${COMMON[@]}" 2>&1 | tee "$RUNS/$out.log"
}

for st in "${STAGES[@]}"; do case $st in
  ref)
           "$PY" var_leak_run.py --out_dir "$RUNS/leak_full" --ref_img_dir "$IMAGENET_VAL" --ref_seed 7 \
                 --keep_ratios 1.0 0.5 0.25 "${COMMON[@]}" 2>&1 | tee "$RUNS/leak_full.log" ;;
  table3)
           sel leak_select 0 0.25 0.5 ;;
  sweep)
           sel leak_law_d16             0 0.1 0.4 0.75 0.9
           sel leak_law_d16_seed1       1 0.25
           sel leak_law_d16_seed1_extra 1 0.1 0.75 0.9
           sel leak_law_d16_seed1_mid   1 0.4 0.5
           sel leak_law_d16_seed2       2 0.25 0.4 0.5
           sel leak_law_d16_seed2_extra 2 0.1 0.75 0.9 ;;
  collect) for d in leak_full leak_select leak_law_d16 leak_law_d16_seed1 leak_law_d16_seed1_extra \
                    leak_law_d16_seed1_mid leak_law_d16_seed2 leak_law_d16_seed2_extra; do
             mkdir -p "$REPO/results/external/var/$d"; cp "$RUNS/$d/results.json" "$REPO/results/external/var/$d/"
           done
           cp "$RUNS/leak_select.log" "$REPO/results/external/var/leak_select.log" ;;
  *) echo "unknown stage $st" >&2; exit 2 ;;
esac; done
