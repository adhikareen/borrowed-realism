#!/usr/bin/env bash
set -euo pipefail
REPO=${REPO:-$(cd "$(dirname "$0")/../.." && pwd)}
: "${EXT_ROOT:?set EXT_ROOT to the directory holding the upstream checkouts}"
PY=${PY:-python}
TORCHRUN=${TORCHRUN:-torchrun}
GPU=${GPU:-0}
AD=$EXT_ROOT/AdapTok
ANN_DIR=${ANN_DIR:-$AD/runs/annot}
AR_DIR=${AR_DIR:-$AD/runs/ar}
EVAL_DIR=${EVAL_DIR:-$AD/runs/eval}
SUBSETS=(42 1337 7 2024 555 99 314)
export CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH=$AD PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$AD"
mkdir -p "$ANN_DIR" "$AR_DIR" "$EVAL_DIR"
STAGES=("$@"); [ ${#STAGES[@]} -eq 0 ] && STAGES=(annot train eval harvest)

CK_FP_A=$AR_DIR/fp_adaptive/adaptok_fp_ar/FP_lr0.0006_llama-abs-XS__fp_adaptive/epoch-last.pth
CK_FP_U=$AR_DIR/fp_uniform/adaptok_fp_ar/FP_lr0.0006_llama-abs-XS__fp_uniform/epoch-last.pth
CK_FR_A=$AR_DIR/fr_adaptive/adaptok_ar_k400/lr0.0006_wd0.05_llama-abs-XS__tierB_full/epoch-last.pth
CK_FR_U=$AR_DIR/fr_uniform/adaptok_ar_k400/lr0.0006_wd0.05_llama-abs-XS__uniform212/epoch-last.pth

for st in "${STAGES[@]}"; do case $st in
  annot)
    "$PY" -u build_k400_ar_annotations.py --scorer ckpts/scorer --csv_file k400_train_12k.csv \
        --out "$ANN_DIR/k400_12k_ilp212.pt" --token_select_num 212 --num_frames 16 --input_size 128 \
        --batch_size 32 --num_workers 4 --use_all_frames_step 64 --frame_rate native --save_iter 200 --device cuda:0
    "$PY" -u build_ar_csv_from_annotations.py --annot "$ANN_DIR/k400_12k_ilp212.pt" \
        --src_csv data/metadata/k400_train_12k.csv --out_csv data/metadata/k400_ar_train_12k.csv
    "$PY" make_uniform_annotation.py --src "$ANN_DIR/k400_12k_ilp212.pt" --out "$ANN_DIR/k400_12k_uniform212.pt"
    "$PY" -u build_k400_fp_annotations.py --scorer ckpts/scorer --csv_file k400_train_12k.csv \
        --reuse "$ANN_DIR/k400_12k_ilp212.pt" --out "$ANN_DIR/k400_12k_fp_ilp212.pt" \
        --token_select_num 212 --num_cond_frames 5 --num_frames 16 --input_size 128 --use_all_frames_step 64 ;;
  train)
    for ARM in adaptive uniform; do
      FIXED=-1; [ $ARM = uniform ] && FIXED=212
      "$TORCHRUN" --nproc_per_node=1 --standalone train.py --cfg cfgs/adaptok_ar_fp.yaml --manualSeed 42 \
        --tag fp_$ARM --csv_file k400_ar_train_12k.csv --out_path "$AR_DIR/fp_$ARM" --name adaptok_fp_ar \
        -b 32 -j 6 --frame_num 16 --input_size 128 --replace --opts \
        test_dataset.csv_paths.k600_val k400_val_500.csv \
        train_dataset.args.ar_ann_path "$ANN_DIR/k400_12k_fp_ilp212.pt" test_dataset.args.ar_ann_path "$ANN_DIR/k400_12k_fp_ilp212.pt" \
        model.name llama-abs-XS model.args.num_classes 400 model.args.start_ar_block_idx 0 \
        vae.checkpoint ckpts/tokenizer vae.fixed_per_block $FIXED \
        use_amp true amp_dtype bfloat16 compile false \
        max_epoch 49 eval_epoch 100000000 vis_epoch 100000000 latest_interval 5 save_epoch 100000000 \
        stepwise_logging true stepwise_logging_interval 50 \
        optimizer.args.lr 0.0006 optimizer.warmup_epoch 1 clip_grad_max_norm 1.0
    done
    for ARM in adaptive uniform; do
      ANN=$ANN_DIR/k400_12k_ilp212.pt; TAG=tierB_full
      [ $ARM = uniform ] && { ANN=$ANN_DIR/k400_12k_uniform212.pt; TAG=uniform212; }
      "$TORCHRUN" --nproc_per_node=1 --standalone train.py --cfg cfgs/adaptok_ar_k400_67m.yaml --manualSeed 42 \
        --tag $TAG --csv_file k400_ar_train_12k.csv --out_path "$AR_DIR/fr_$ARM" --name adaptok_ar_k400 \
        -b 64 -j 4 --frame_num 16 --input_size 128 --opts \
        test_dataset.csv_paths.k400_val k400_val_500.csv \
        train_dataset.args.ar_ann_path "$ANN" test_dataset.args.ar_ann_path "$ANN" vae.checkpoint ckpts/tokenizer
    done ;;
  eval)
    COMMON=(--tokenizer ckpts/tokenizer --model_type adaptok --dataset_csv k400_val_500.csv --num_samples 500
            --sample_batch_size 50 --cfg_scale 1.0 --temperature 1.0 --top_k 0 --top_p 1.0 --dtype bfloat16
            --stats_only --replace)
    for S in "${SUBSETS[@]}"; do
      for ARM in fp_adaptive fp_uniform fr_adaptive fr_uniform; do
        case $ARM in
          fp_adaptive) CK=$CK_FP_A; EXTRA=(--frame_prediction --num_cond_frames 5 --force_fp16_enc) ;;
          fp_uniform)  CK=$CK_FP_U; EXTRA=(--frame_prediction --num_cond_frames 5 --force_fp16_enc --cap_per_block 212) ;;
          fr_adaptive) CK=$CK_FR_A; EXTRA=(--fvd_resolution 64 --tokenizer_decode_fp32) ;;
          fr_uniform)  CK=$CK_FR_U; EXTRA=(--fvd_resolution 64 --tokenizer_decode_fp32 --cap_per_block 212) ;;
        esac
        O=$EVAL_DIR/fp_robust_${ARM}_s$S
        mkdir -p "$O"
        "$PY" -u sample.py --ar_model "$CK" --dataset_split_seed $S --output_dir "$O" "${COMMON[@]}" "${EXTRA[@]}" \
            > "$O.log" 2>&1
        echo "[subset $S] $ARM $(grep -aoE 'FVD: [0-9.]+' "$O.log" | tail -1)"
      done
    done ;;
  harvest)
    CUDA_VISIBLE_DEVICES="" "$PY" "$REPO/scripts/external/adaptok_fvd_from_stats.py" --runs_dir "$EVAL_DIR" \
        --out "$REPO/results/external/adaptok/adaptok_7subset_fvd.json" ;;
  *) echo "unknown stage $st" >&2; exit 2 ;;
esac; done
