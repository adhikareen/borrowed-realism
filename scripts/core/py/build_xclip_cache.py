#!/usr/bin/env python3
import numpy as np, shutil, json
from pathlib import Path
rng=np.random.default_rng(7)
perm=rng.permutation(2000)
fix=np.where(perm==np.arange(2000))[0]
for i in fix: j=(i+1)%2000; perm[i],perm[j]=perm[j],perm[i]
assert (perm==np.arange(2000)).sum()==0
for arm in ["decode","random"]:
    src=Path(f"runs/cache_omnidec_F17_k60_{arm}_val"); dst=Path(f"runs/cache_omnidec_F17_k60_{arm}_val_xclip")
    dst.mkdir(exist_ok=True)
    ii=np.load(src/"input_ids.npy"); C=320
    ii2=ii.copy(); ii2[:,1:1+C]=ii[perm,1:1+C]
    np.save(dst/"input_ids.npy",ii2)
    for f in ["type_ids.npy","source_pos.npy"]: shutil.copy(src/f,dst/f)
    man=json.load(open(src/"manifest.json")); man["note"]="coarse permuted across clips (derangement seed 7) for decoupled-RS"
    json.dump(man,open(dst/"manifest.json","w"),indent=1)
    print("built",dst)
