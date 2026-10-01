#!/usr/bin/env bash
set -euo pipefail
REPO=${REPO:-$(cd "$(dirname "$0")/../.." && pwd)}
: "${EXT_ROOT:?set EXT_ROOT to the directory holding the upstream checkouts}"
: "${IMAGENET_TRAIN:?ImageNet train folder: source of the real residual prefix}"
: "${IMAGENET_VAL:?ImageNet val folder: FID reference}"
PY=${PY:-python}
GPU=${GPU:-0}
RQ=$EXT_ROOT/rqvae
export CUDA_VISIBLE_DEVICES=$GPU PYTHONUNBUFFERED=1
cd "$RQ"
if [ "${1:-run}" != "collect" ]; then
  "$PY" rq_select_run.py \
      --model-ar-path pretrained/imagenet_480M/stage2/model.pt \
      --model-vqvae-path pretrained/imagenet_480M/stage1/model.pt \
      --out_dir runs/rq_leak_select --real_img_dir "$IMAGENET_TRAIN" --ref_img_dir "$IMAGENET_VAL" \
      --num_classes 1000 --imgs_per_class 10 --classes_per_batch 10 \
      --prune_from_d 2 --keep_ratio 0.25 --temp 1.0 --top_k 256 --top_p 0.95 \
      --seed 0 --leak_seed 0 --ref_seed 7 2>&1 | tee runs/rqvae_leak_full.log
fi
mkdir -p "$REPO/results/external/rqvae"
cp runs/rqvae_leak_full.log "$REPO/results/external/rqvae/"
[ -f runs/rqvae_leak_results.json ] && cp runs/rqvae_leak_results.json "$REPO/results/external/rqvae/" || \
  echo "write runs/rqvae_leak_results.json (results_fid_10k: smart/random x self/real) from runs/rq_leak_select/results.json"
