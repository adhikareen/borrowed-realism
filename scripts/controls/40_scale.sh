#!/usr/bin/env bash
source "$(dirname "$0")/common.sh"

RAWB=$RUNS/omni_decode_extract_F65
REALB=$RUNS/real_omni_F65_val
for SPL in val train; do
  N=2000; SF="--store-frames"; [[ $SPL == train ]] && { N=8000; SF=""; }
  [[ "$(ls "$RAWB/$SPL"/shards/*.npz 2>/dev/null | wc -l)" -gt 0 ]] && continue
  log "extract F65 $SPL"
  "$PY" "$SRC/extract_omnitok_decode.py" --ckpt "$OMNITOK_CKPT" --filelist "$DATA_ROOT/k400_$SPL.txt" \
    --out-dir "$RAWB/$SPL" --num $N --batch-size 8 --image-size 128 --num-frames 65 \
    --num-workers 2 $SF > "$RUNS/axisB_extract_$SPL.log" 2>&1
done
dump_real_frames "$RAWB/val" "$REALB"
for SEL in decode random; do
  build_cache "$RAWB/train" "$RUNS/cache_omniF65_k60_${SEL}_train" $SEL 8000
  build_cache "$RAWB/val"   "$RUNS/cache_omniF65_k60_${SEL}_val"   $SEL 2000
done
for S in 42 123; do
  for SEL in decode random; do
    NAME=omniF65_k60_${SEL}_s$S; [[ $S == 42 ]] && NAME=omniF65_$SEL
    train_ar "$RUNS/train_omniF65_k60_${SEL}_s$S" "$RUNS/cache_omniF65_k60_${SEL}_train" \
      "$RUNS/cache_omniF65_k60_${SEL}_val" "$NAME" $S 16 4 1700000000 147 50
  done
done
for S in 42 123; do
  SUF=""; [[ $S == 123 ]] && SUF=_s123
  for SEL in decode random; do
    for R in rs sg; do
      MODE=packed; [[ $R == sg ]] && MODE=packed_gencoarse_ref
      gen_cached "axisB65_${R}_${SEL}$SUF" "$RUNS/train_omniF65_k60_${SEL}_s$S/best.pt" $MODE \
        "$RUNS/cache_omniF65_k60_${SEL}_val" 32
      fvd "axisB65_${R}_${SEL}$SUF" "$REALB" 65 4
    done
  done
done

RAWA=$RUNS/omni_decode_extract_F17_256
REALA=$RUNS/real_omni_F17_256_val
for SPL in val train; do
  N=2000; SF="--store-frames"; [[ $SPL == train ]] && { N=8000; SF=""; }
  [[ "$(ls "$RAWA/$SPL"/shards/*.npz 2>/dev/null | wc -l)" -gt 0 ]] && continue
  log "extract 256px $SPL"
  "$PY" "$SRC/extract_omnitok_decode.py" --ckpt "$OMNITOK_CKPT" --filelist "$DATA_ROOT/k400_$SPL.txt" \
    --out-dir "$RAWA/$SPL" --num $N --batch-size 8 --image-size 256 --num-frames 17 \
    --num-workers 2 $SF > "$RUNS/axisA_extract_$SPL.log" 2>&1
done
dump_real_frames "$RAWA/val" "$REALA"
for SEL in decode random; do
  build_cache "$RAWA/train" "$RUNS/cache_omni256_F17_k60_${SEL}_train" $SEL 8000
  build_cache "$RAWA/val"   "$RUNS/cache_omni256_F17_k60_${SEL}_val"   $SEL 2000
done
for S in 42 123; do
  for SEL in decode random; do
    NAME=omni256_F17_k60_${SEL}_s$S; [[ $S == 42 ]] && NAME=omni256_$SEL
    train_ar "$RUNS/train_omni256_F17_k60_${SEL}_s$S" "$RUNS/cache_omni256_F17_k60_${SEL}_train" \
      "$RUNS/cache_omni256_F17_k60_${SEL}_val" "$NAME" $S 8 8 2000000000 125 50
  done
done
for S in 42 123; do
  SUF=""; [[ $S == 123 ]] && SUF=_s123
  for SEL in decode random; do
    for R in rs sg; do
      MODE=packed; [[ $R == sg ]] && MODE=packed_gencoarse_ref
      gen_cached "axisA256_${R}_${SEL}$SUF" "$RUNS/train_omni256_F17_k60_${SEL}_s$S/best.pt" $MODE \
        "$RUNS/cache_omni256_F17_k60_${SEL}_val" 24
      fvd "axisA256_${R}_${SEL}$SUF" "$REALA" 17 8
    done
  done
done

build_cache "$RAWA/val" "$RUNS/cache_omni256_F17_k60_decode_deranged_val" decode_deranged 2000
build_cache "$RAWA/val" "$RUNS/cache_omni256_F17_k60_contigmax_val"       contigmax       2000
for S in 42 123; do
  P=ctl256; [[ $S == 123 ]] && P=ctl256b
  for SEL in decode_deranged contigmax random; do
    gen_cached "${P}_rs_${SEL}" "$RUNS/train_omni256_F17_k60_decode_s$S/best.pt" packed \
      "$RUNS/cache_omni256_F17_k60_${SEL}_val" 24
    fvd "${P}_rs_${SEL}" "$REALA" 17 8
  done
done
log "SCALE_DONE"
