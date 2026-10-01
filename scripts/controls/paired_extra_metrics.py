import argparse, json, sys
from pathlib import Path

import numpy as np
from scipy import stats

ap = argparse.ArgumentParser()
ap.add_argument("--features-dir", required=True, type=Path, help="$RUNS/_features (written by compute_fvd.py)")
ap.add_argument("--src", required=True, type=Path, help="$REPO/src")
ap.add_argument("--out-json", required=True, type=Path)
args = ap.parse_args()
sys.path.insert(0, str(args.src))
from extra_metrics_from_features import kid, prdc, load_feats  # noqa: E402

F = args.features_dir
real = load_feats(F / "real_decode_F17_val_real_penultimate.pt")
SEEDS = [42, 101, 123, 202, 303, 404, 456, 789]


def tag(reg, mask, s):
    pre = "fact2" if mask == "random" else "fact"
    return F / f"gen_{pre}_{reg}_{mask}_s{s}_gen_penultimate.pt"


out = {}
for reg in ("rs", "sg"):
    per = {}
    for mask in ("aligned", "random"):
        rows = []
        for s in SEEDS:
            p = tag(reg, mask, s)
            if not p.exists(): rows.append(None); continue
            f = load_feats(p)
            k, _ = kid(real, f)
            pr, rc, de, co = prdc(real, f, k=5)
            div = float(np.mean(np.linalg.norm(f[:500, None] - f[None, :500], axis=-1)))
            rows.append(dict(kid=k, precision=pr, recall=rc, density=de, coverage=co, diversity=div))
        per[mask] = rows
    out[reg] = per

print("PAIRED decode(aligned)-vs-random effects, 8 seeds, I3D-R50 2048-d features")
print("positive = decode better (KID lower / precision higher)\n")
for m in ("kid", "precision", "recall", "density", "coverage", "diversity"):
    line = f"{m:10s}"
    for reg in ("rs", "sg"):
        a = np.array([r[m] for r in out[reg]["aligned"] if r])
        b = np.array([r[m] for r in out[reg]["random"] if r])
        d = (b - a) if m in ("kid",) else (a - b)
        t = stats.ttest_rel(a, b)
        lo, hi = stats.t.interval(0.95, len(d) - 1, loc=d.mean(), scale=stats.sem(d))
        line += f" | {reg.upper()} {d.mean():+8.4f} CI[{lo:+.4f},{hi:+.4f}] p={t.pvalue:.4f}"
    print(line)
json.dump({k: {m: [r for r in v[m]] for m in v} for k, v in out.items()},
          open(args.out_json, "w"), indent=1)
print(f"\nwrote {args.out_json}")
