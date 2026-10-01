#!/usr/bin/env bash
source "$(dirname "$0")/common.sh"
REAL=$RUNS/real_ssv2_F17_val
C=$RUNS/cache_ssv2_F17_k60

[[ -d $RUNS/ssv2_decode_extract_F17_val/shards ]] || \
  "$PY" "$SRC/extract_omnitok_decode.py" --ckpt "$OMNITOK_CKPT" --filelist "$DATA_ROOT/ssv2_val.txt" \
    --out-dir "$RUNS/ssv2_decode_extract_F17_val" --num 2000 --batch-size 16 --store-frames \
    > "$RUNS/ssv2_extract_val.log" 2>&1
[[ -d $RUNS/ssv2_decode_extract_F17_train/shards ]] || \
  "$PY" "$SRC/extract_omnitok_decode.py" --ckpt "$OMNITOK_CKPT" --filelist "$DATA_ROOT/ssv2_train.txt" \
    --out-dir "$RUNS/ssv2_decode_extract_F17_train" --num 8000 --batch-size 16 \
    > "$RUNS/ssv2_extract_train.log" 2>&1

dump_real_frames "$RUNS/ssv2_decode_extract_F17_val" "$REAL"

build_cache "$RUNS/ssv2_decode_extract_F17_train" "${C}_decode_train"          decode          8000
build_cache "$RUNS/ssv2_decode_extract_F17_val"   "${C}_decode_val"            decode          2000
build_cache "$RUNS/ssv2_decode_extract_F17_val"   "${C}_decode_deranged_val"   decode_deranged 2000
build_cache "$RUNS/ssv2_decode_extract_F17_val"   "${C}_contigmax_val"         contigmax       2000
build_cache "$RUNS/ssv2_decode_extract_F17_val"   "${C}_random_val"            random          2000

train_ar "$RUNS/train_ssv2_F17_k60_decode" "${C}_decode_train" "${C}_decode_val" ssv2_F17_k60_decode 1337 32

CK=$RUNS/train_ssv2_F17_k60_decode/best.pt
for R in rs sg; do
  MODE=packed; [[ $R == sg ]] && MODE=packed_gencoarse_ref
  for MASK in decode decode_deranged contigmax; do
    gen_cached "ssv2fact_${R}_${MASK}" "$CK" $MODE "${C}_${MASK}_val" 64
    fvd        "ssv2fact_${R}_${MASK}" "$REAL" 17 8
  done
  gen_cached "ssv2fact2_${R}_random" "$CK" $MODE "${C}_random_val" 64
  fvd        "ssv2fact2_${R}_random" "$REAL" 17 8
done
log "MASK_SSV2_DONE"
