#!/usr/bin/env bash
source "$(dirname "$0")/lib.sh"
need DATA_ROOT
WHICH=${1:-all}

k400(){
  need OMNITOK_CKPT
  local SPL N SF OUT
  for SPL in train val; do
    N=8000; SF=""; [[ $SPL == val ]] && { N=2000; SF="--store-frames"; }
    OUT=$RUNS/omni_decode_extract_F${F}_$SPL
    if [[ -f "$OUT/.extract_done" ]]; then log "SKIP extract $OUT"; continue; fi
    log "EXTRACT K400 F=$F N=$N split=$SPL -> $OUT"
    "$PY" "$SRC/extract_omnitok_decode.py" --filelist "$DATA_ROOT/k400_$SPL.txt" --out-dir "$OUT" \
      --num $N --batch-size 32 --shard-size 1000 --num-frames $F \
      --num-workers 2 $SF || die "extract $OUT"
    touch "$OUT/.extract_done"
  done
  if [[ ! -f "$REAL/real_0000000.npy" ]]; then
    log "DUMP real ref -> $REAL"
    "$PY" "$SRC/dump_real_ref_long.py" --raw-dir "$RAWVA" --out-dir "$REAL" --limit $NGEN
  fi
}

ssv2(){
  need OMNITOK_CKPT
  if [[ ! -d $RUNS/ssv2_decode_extract_F17_train/shards ]]; then
    log "extract SSv2 train"
    "$PY" "$SRC/extract_omnitok_decode.py" --filelist "$DATA_ROOT/ssv2_train.txt" \
      --out-dir "$RUNS/ssv2_decode_extract_F17_train" --num 8000 --batch-size 16
  fi
  if [[ ! -d $RUNS/ssv2_decode_extract_F17_val/shards ]]; then
    log "extract SSv2 val"
    "$PY" "$SRC/extract_omnitok_decode.py" --filelist "$DATA_ROOT/ssv2_val.txt" \
      --out-dir "$RUNS/ssv2_decode_extract_F17_val" --num 2000 --batch-size 16 --store-frames
  fi
  if [[ ! -d "$REAL_SSV2" ]]; then
    log "build SSv2 real ref"
    "$PY" - <<'PYEOF'
import numpy as np, glob
from pathlib import Path
out=Path("runs/real_ssv2_F17_val"); out.mkdir(exist_ok=True)
i=0
for sh in sorted(glob.glob("runs/ssv2_decode_extract_F17_val/shards/*.npz")):
    d=np.load(sh)
    fr=d["frames"]
    for k in range(fr.shape[0]):
        np.save(out/f"real_{i:05d}.npy", fr[k].astype(np.float16)); i+=1
        if i>=2000: break
    if i>=2000: break
print("real ref clips:", i)
PYEOF
  fi
}

omag(){
  need OMAG_CKPT
  local SPL N OUT
  for SPL in train val; do
    N=8000; [[ $SPL == val ]] && N=2000
    OUT=$RUNS/omag_F17_extract_$SPL
    if [[ -f "$OUT/.extract_done" ]]; then log "SKIP extract $OUT"; continue; fi
    log "EXTRACT OM2 N=$N split=$SPL -> $OUT (RECONSTRUCTED flags)"
    "$PY" "$SRC/extract_omag_decode.py" --filelist "$DATA_ROOT/k400_$SPL.txt" --out-dir "$OUT" \
      --num $N || die "extract $OUT"
    touch "$OUT/.extract_done"
  done
}

case $WHICH in
  k400) k400 ;; ssv2) ssv2 ;; omag) omag ;;
  all) k400; ssv2; omag ;;
  *) die "usage: $0 [k400|ssv2|omag|all]" ;;
esac
log "00_extract $WHICH DONE"
