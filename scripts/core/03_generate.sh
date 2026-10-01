#!/usr/bin/env bash
source "$(dirname "$0")/lib.sh"
ARM=${1:?arm}; SEED=${2:?seed}; REG=${3:?RS|SG}; KK=${4:-60}
TAG=omnidec_F${F}_k${KK}_${ARM}; [[ $SEED != 1337 ]] && TAG=${TAG}_s${SEED}
CK=$RUNS/train_$TAG/best.pt; REF=$RUNS/cache_omnidec_F${F}_k${KK}_${ARM}_val
case $REG in
  RS) gen omni "$RUNS/gen_$TAG" "$CK" packed "$REF" 64 "${SAMPLER[@]}" ;;
  SG) if [[ $ARM == full ]]; then gen omni "$RUNS/gen_${TAG}_gc" "$CK" packed_gencoarse - 64 "${SAMPLER[@]}"
      else gen omni "$RUNS/gen_${TAG}_gc" "$CK" packed_gencoarse_ref "$REF" 64 "${SAMPLER[@]}"; fi ;;
  *) die "regime must be RS or SG" ;;
esac
