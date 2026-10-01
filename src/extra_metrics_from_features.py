from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import torch


def load_feats(p: Path) -> np.ndarray:
    x = torch.load(p, map_location="cpu")
    if isinstance(x, dict):
        x = x.get("feats", x.get("features", next(iter(x.values()))))
    return np.asarray(x, dtype=np.float64)


def kid(a, b, subsets=100, subset_size=1000, seed=0):
    rng = np.random.default_rng(seed); d = a.shape[1]; vals = []
    m = min(subset_size, len(a), len(b))
    for _ in range(subsets):
        x = a[rng.choice(len(a), m, replace=False)]
        y = b[rng.choice(len(b), m, replace=False)]
        kxx = (x @ x.T / d + 1) ** 3; kyy = (y @ y.T / d + 1) ** 3; kxy = (x @ y.T / d + 1) ** 3
        np.fill_diagonal(kxx, 0); np.fill_diagonal(kyy, 0)
        vals.append(kxx.sum()/(m*(m-1)) + kyy.sum()/(m*(m-1)) - 2*kxy.mean())
    return float(np.mean(vals)), float(np.std(vals))


def _knn_radii(x, k):
    d = torch.cdist(torch.from_numpy(x), torch.from_numpy(x)).numpy()
    np.fill_diagonal(d, np.inf)
    return np.sort(d, axis=1)[:, k - 1]


def prdc(real, fake, k=5):
    rr = _knn_radii(real, k); fr = _knn_radii(fake, k)
    d_rf = torch.cdist(torch.from_numpy(real), torch.from_numpy(fake)).numpy()
    precision = float((d_rf < rr[:, None]).any(axis=0).mean())
    recall    = float((d_rf < fr[None, :]).any(axis=1).mean())
    density   = float((d_rf < rr[:, None]).sum(axis=0).mean() / k)
    coverage  = float((d_rf.min(axis=1) < rr).mean())
    return precision, recall, density, coverage


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen-feats", required=True, type=Path)
    ap.add_argument("--real-feats", required=True, type=Path)
    ap.add_argument("--out-json", required=True, type=Path)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--prdc-n", type=int, default=2000)
    args = ap.parse_args()

    g = load_feats(args.gen_feats); r = load_feats(args.real_feats)
    n = min(args.prdc_n, len(g), len(r))
    gs, rs = g[:n], r[:n]
    km, ks = kid(g, r)
    p, rc, dn, cv = prdc(rs, gs, k=args.k)
    div = float(np.mean(torch.cdist(torch.from_numpy(gs), torch.from_numpy(gs)).numpy()))
    out = {"n_gen": int(len(g)), "n_real": int(len(r)), "n_prdc": int(n), "k": args.k,
           "kid": km, "kid_std": ks, "precision": p, "recall": rc,
           "density": dn, "coverage": cv, "diversity": div}
    args.out_json.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
