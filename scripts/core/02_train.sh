#!/usr/bin/env bash
source "$(dirname "$0")/lib.sh"
[[ $# -ge 1 ]] || die "usage: $0 <run_name> | <arm> <seed> [keep]"
if [[ $1 == train_* ]]; then
  RUN=$1
else
  ARM=$1; SEED=${2:?seed}; KK=${3:-60}
  RUN=train_omnidec_F${F}_k${KK}_${ARM}; [[ $SEED != 1337 ]] && RUN=${RUN}_s${SEED}
fi
[[ $RUN == *_255M_ext ]] && die "$RUN is a resumed continuation; run 16_scale_255M.sh"
train_run "$RUN"
log "$RUN done: $("$PY" -c "import json,sys;d=json.load(open(sys.argv[1]));print('steps',d['total_optim_steps'],'best_val',round(d['best_val_loss'],4))" "$RUNS/$RUN/done.json")"
