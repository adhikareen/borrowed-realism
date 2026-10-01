#!/usr/bin/env bash
source "$(dirname "$0")/common.sh"
for R in rs sg; do for S in 42 101 123 202 303 404 456 789; do
  for T in "fact_${R}_aligned_s$S" "fact2_${R}_random_s$S"; do
    [[ -f "$RUNS/_features/gen_${T}_gen_penultimate.pt" ]] || { log "!! missing features for $T (run 10 first)"; exit 1; }
  done
done; done
CUDA_VISIBLE_DEVICES="" "$PY" "$(dirname "$0")/paired_extra_metrics.py" \
  --features-dir "$RUNS/_features" --src "$SRC" --out-json "$RUNS/paired_extra_metrics.json" \
  | tee "$RUNS/paired_metrics.log"
