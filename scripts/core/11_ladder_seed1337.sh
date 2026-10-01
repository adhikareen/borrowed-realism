#!/usr/bin/env bash
source "$(dirname "$0")/lib.sh"
need OMNITOK_CKPT
T=omnidec_F${F}_k60
ARMS=(full decode contigmax l2gap attninf l2gap_smooth learned genscore05 genscore gensal random)

bash "$CORE/01_selectors.sh" A
for A in full decode random l2gap learned contigmax l2gap_smooth; do train_run "train_${T}_$A"; done
bash "$CORE/01_selectors.sh" B
for A in genscore genscore05 gensal attninf; do train_run "train_${T}_$A"; done

for A in "${ARMS[@]}"; do
  DBS=96; [[ $A == decode || $A == random ]] && DBS=64; [[ $A == full ]] && DBS=48
  omni_arm "${T}_$A" "train_${T}_$A" "$RUNS/cache_${T}_${A}_val" packed $DBS 16
done
for A in "${ARMS[@]}"; do
  if [[ $A == full ]]; then
    omni_arm "${T}_${A}_gc" "train_${T}_$A" - packed_gencoarse 96 16
  else
    omni_arm "${T}_${A}_gc" "train_${T}_$A" "$RUNS/cache_${T}_${A}_val" packed_gencoarse_ref 96 16
  fi
  blank "$RUNS/gen_${T}_${A}_gc" "$RUNS/gc_blank_k60_$A.json"
done

"$PY" "$SRC/keepset_diag.py" --runs "$RUNS" --nll-dir "$RUNS/cell_nll_F17_val" --raw-val "$RAWVA" \
  --arms decode l2gap learned random genscore genscore05 \
  --out-json "$RUNS/keepset_diagnostics.json" || log "keep-set diagnostics failed (non-fatal)"
"$PY" "$SRC/keepset_diag.py" --runs "$RUNS" --nll-dir "$RUNS/cell_nll_F17_val" --raw-val "$RAWVA" \
  --arms decode l2gap learned random genscore gensal attninf \
  --out-json "$RUNS/keepset_diagnostics_scored.json" || log "keep-set diagnostics (gensal/attninf) failed (non-fatal)"

"$PY" - "$PEN" "$T" "$RUNS" <<'PYAGG'
import json, sys, os
pen, tag, runs = sys.argv[1], sys.argv[2], sys.argv[3]
out = {}
for sel in ["decode","random","full"]:
    fj=os.path.join(pen,f"fvd_{tag}_{sel}.json"); tj=os.path.join(runs,f"train_{tag}_{sel}","done.json")
    mj=os.path.join(runs,f"cache_{tag}_{sel}_val","manifest.json")
    rec={}
    if os.path.exists(fj):
        fd=json.load(open(fj)); rec["fvd"]=fd.get("fvd"); rec["n_gen"]=fd.get("n_gen"); rec["n_real"]=fd.get("n_real")
    if os.path.exists(tj):
        d=json.load(open(tj)); rec["best_val_loss"]=round(d["best_val_loss"],4)
        rec["tokens_seen"]=int(d["total_tokens_seen"]); rec["steps"]=d["total_optim_steps"]
    if os.path.exists(mj):
        m=json.load(open(mj)); rec["seq_len"]=m["seq_len"]; rec["K"]=m["packed_fine_budget"]
        rec["keep_frac"]=m["keep_frac"]; rec["tpf"]=round(m["tpf"],2)
    out[sel]=rec
res={"phase":"1b_generation","tokenizer":"OmniTokenizer_VQGAN_K600_8K","dataset":"K400",
     "F":17,"latent":[5,16,16],"keep_frac":0.60,"n_gen":2000,"target_tokens":5e8,
     "recipe":"dim512/8L/8H seed1337 bf16 early-stop-8","selectors":out}
d=out.get("decode",{}); r=out.get("random",{}); f=out.get("full",{})
if "fvd" in d and "fvd" in r: res["delta_fvd_decode_minus_random"]=round(d["fvd"]-r["fvd"],3)
if "fvd" in d and "fvd" in f: res["delta_fvd_decode_minus_full"]=round(d["fvd"]-f["fvd"],3)
if "fvd" in r and "fvd" in f: res["delta_fvd_random_minus_full"]=round(r["fvd"]-f["fvd"],3)
json.dump(res, open(os.path.join(runs,"omnitok_oracle_gen.json"),"w"), indent=2)
print("WROTE", os.path.join(runs,"omnitok_oracle_gen.json"))
print(json.dumps(res, indent=2))
PYAGG

for A in "${ARMS[@]}"; do
  log "$A  RS $(fvdval "$PEN/fvd_${T}_$A.json")  SG $(fvdval "$PEN/fvd_${T}_${A}_gc.json")"
done
log "11_ladder_seed1337 DONE"
