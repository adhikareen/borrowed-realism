#!/usr/bin/env bash
source "$(dirname "$0")/lib.sh"
TAG=${1:?tag}; R=${2:-$REAL}; OUT=${3:-$PEN/fvd_$TAG.json}; FBS=${4:-16}
fvd "$RUNS/gen_$TAG" "$R" "$OUT" "$FBS"
if [[ $TAG =~ ^omnidec_F17_k60_(.+)_gc$ ]]; then
  blank "$RUNS/gen_$TAG" "$RUNS/gc_blank_k60_${BASH_REMATCH[1]}.json"
fi
