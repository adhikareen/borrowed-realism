#!/usr/bin/env bash
source "$(dirname "$0")/lib.sh"
need OMNITOK_CKPT
T=omnidec_F${F}_k60
for A in decode random; do [[ -f "$RUNS/cache_${T}_${A}_val/manifest.json" ]] || die "missing $RUNS/cache_${T}_${A}_val"; done
if [[ ! -f "$RUNS/cache_${T}_decode_val_xclip/manifest.json" || ! -f "$RUNS/cache_${T}_random_val_xclip/manifest.json" ]]; then
  "$PY" "$CORE/py/build_xclip_cache.py"
fi
for A in decode random; do
  omni_arm "${T}_${A}_xclip" "train_${T}_$A" "$RUNS/cache_${T}_${A}_val_xclip" packed 64 8
done
log "xclip  decode $(fvdval "$PEN/fvd_${T}_decode_xclip.json")  random $(fvdval "$PEN/fvd_${T}_random_xclip.json")"
log "15_scaffold_derangement DONE"
