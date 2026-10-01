#!/usr/bin/env bash
set -uo pipefail
REPO=${REPO:-$(cd "$(dirname "$0")/../.." && pwd)}
: "${EXT_ROOT:?set EXT_ROOT to the directory holding the upstream checkouts}"
PY=${PY:-python}
GPU=${GPU:-0}
FV=$EXT_ROOT/FastVAR/Infinity
OUT=${OUT:-$FV/runs/out}
VAE=weights/infinity_vae_d32reg.pth
IMGS=weights/MJHQ30K
META=weights/MJHQ30K/meta_data.json
export CUDA_VISIBLE_DEVICES=$GPU PYTHONUNBUFFERED=1
mkdir -p "$OUT/jobq"
cd "$FV"
STAGES=("$@"); [ ${#STAGES[@]} -eq 0 ] && STAGES=(run1 runs23 dial collect)
log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" >> "$1"; }
gen(){
  "$PY" infinity_leak_exp.py --mode gen --select "$2" --coarse "$3" --gt_leak "$4" --num_samples "$5" \
      --out_dir "$1" --vae_path $VAE --mjhq_imgs $IMGS --mjhq_meta $META --seed "$6" > "$7" 2>&1
}
fid(){ "$PY" tools/fid_score.py "$1" "$2" 2>/dev/null | tail -1; }

for st in "${STAGES[@]}"; do case $st in
  run1)
    L=$OUT/jobq/fastvar2.log
    for SEL in smart random; do for CO in self real; do
      GT=0; [ "$CO" = real ] && GT=9
      log "$L" "gen $SEL/$CO (n=1000, gt_leak=$GT)"
      gen runs/leak_${SEL}_${CO} $SEL $CO $GT 1000 0 "$OUT/fastvar2_${SEL}_${CO}.log"
    done; done
    for SEL in smart random; do for CO in self real; do
      N=$(ls runs/leak_${SEL}_${CO}/pred/*.png | wc -l)
      log "$L" "FID $SEL/$CO = $(fid runs/leak_${SEL}_${CO}/pred runs/real_ref)  (n=$N)"
    done; done ;;
  runs23)
    L=$OUT/jobq/fastvar_seeds.log
    for SEED in 1 2; do
      PAR=runs/seed$SEED
      for SEL in smart random; do for CO in real self; do
        GT=0; [ "$CO" = real ] && GT=9
        log "$L" "gen s$SEED $SEL/$CO (n=1000, gt_leak=$GT)"
        gen $PAR/leak_${SEL}_${CO} $SEL $CO $GT 1000 $SEED "$OUT/fastvar_s${SEED}_${SEL}_${CO}.log"
      done; done
      for SEL in smart random; do for CO in real self; do
        N=$(ls $PAR/leak_${SEL}_${CO}/pred/*.png | wc -l)
        log "$L" "FID s$SEED $SEL/$CO = $(fid $PAR/leak_${SEL}_${CO}/pred $PAR/real_ref)  (n=$N)"
      done; done
    done ;;
  dial)
    L=$OUT/jobq/fastvar_dial.log
    for GL in 0 3 5 7 9; do for SEL in smart random; do
      CO=real; [ "$GL" -eq 0 ] && CO=self
      D=runs/dial/leak_${SEL}_gl${GL}
      gen $D $SEL $CO $GL 500 0 "$OUT/dial_${SEL}_gl${GL}.log"
      N=$(ls $D/pred/*.png | wc -l)
      F=$(fid $D/pred runs/dial/real_ref | grep -oE "[0-9]+\.[0-9]+" | tail -1)
      "$PY" -c "import json;json.dump({'selector':'$SEL','gt_leak':$GL,'fid':float('$F'),'n':$N},open('$OUT/fastvar_dial_${SEL}_gl${GL}.json','w'))"
      log "$L" "FID $SEL gl=$GL = $F  (n=$N)"
    done; done ;;
  collect)
    R=$REPO/results/external/fastvar; mkdir -p "$R/dial" "$R/runs1k"
    cp "$OUT"/fastvar_dial_*.json "$OUT/jobq/fastvar_dial.log" "$R/dial/"
    cp "$OUT/jobq/fastvar2.log" "$OUT/jobq/fastvar_seeds.log" "$R/runs1k/" ;;
  *) echo "unknown stage $st" >&2; exit 2 ;;
esac; done
