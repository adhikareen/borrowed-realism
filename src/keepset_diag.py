from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np

FINE_OFF = 323
COARSE_N = 320
T, H, W = 5, 16, 16
FINE_N = T * H * W


def load_grids(shard_dir: Path, key: str, n: int) -> np.ndarray:
    out = []
    for sh in sorted(Path(shard_dir).glob("shard_*.npz")):
        with np.load(sh) as z:
            out.append(z[key].astype(np.float32))
    arr = np.concatenate(out, 0)[:n]
    assert arr.shape == (n, T, H, W), arr.shape
    return arr.reshape(n, FINE_N)


def keep_sets_from_cache(cache_dir: Path) -> np.ndarray:
    man = json.loads((Path(cache_dir) / "manifest.json").read_text())
    K = int(man["packed_fine_budget"])
    sp = np.load(Path(cache_dir) / "source_pos.npy", mmap_mode="r")
    kept = np.asarray(sp[:, 1 + COARSE_N:1 + COARSE_N + K]).astype(np.int64) - FINE_OFF
    assert kept.min() >= 0 and kept.max() < FINE_N
    return kept


def contiguity(kept_rows: np.ndarray) -> float:
    N, K = kept_rows.shape
    fracs = np.empty(N, dtype=np.float64)
    for i in range(N):
        m = np.zeros(FINE_N, dtype=bool)
        m[kept_rows[i]] = True
        g = m.reshape(T, H, W)
        nb = np.zeros_like(g)
        nb[:, 1:, :] |= g[:, :-1, :]
        nb[:, :-1, :] |= g[:, 1:, :]
        nb[:, :, 1:] |= g[:, :, :-1]
        nb[:, :, :-1] |= g[:, :, 1:]
        fracs[i] = float((nb & g).sum()) / max(1, int(g.sum()))
    return float(fracs.mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=Path, default=Path("runs"))
    ap.add_argument("--arms", nargs="+",
                    default=["decode", "l2gap", "learned", "random", "genscore"])
    ap.add_argument("--nll-dir", type=Path, default=Path("runs/cell_nll_F17_val"))
    ap.add_argument("--raw-val", type=Path, default=Path("runs/omni_decode_extract_F17_val"))
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--out-json", type=Path, default=Path("runs/keepset_diagnostics.json"))
    args = ap.parse_args()

    nll = load_grids(args.nll_dir / "shards", "nll", args.n)
    imp = load_grids(args.raw_val / "shards", "decode_imp", args.n)
    imp_tot = imp.sum(1)

    kept = {}
    res = {"keep_frac": 0.60, "K": 768, "n_clips": args.n,
           "global": {"mean_cell_nll_all": float(nll.mean())}, "arms": {}}
    for arm in args.arms:
        cdir = args.runs / f"cache_omnidec_F17_k60_{arm}_val"
        if not (cdir / "manifest.json").exists():
            print(f"[keepset] SKIP {arm}: no cache at {cdir}")
            continue
        ks = keep_sets_from_cache(cdir)
        kept[arm] = ks
        rows = np.arange(ks.shape[0])[:, None]
        kept_nll = nll[rows, ks]
        kept_imp = imp[rows, ks]
        res["arms"][arm] = {
            "mean_kept_cell_nll": float(kept_nll.mean()),
            "p90_kept_cell_nll": float(np.percentile(kept_nll, 90)),
            "decode_imp_captured_frac": float((kept_imp.sum(1) / np.maximum(imp_tot, 1e-12)).mean()),
            "contiguity_4nb": contiguity(ks),
        }
        print(f"[keepset] {arm}: {res['arms'][arm]}")

    names = list(kept.keys())
    jac = {}
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = kept[names[i]], kept[names[j]]
            inter = np.empty(a.shape[0])
            for r in range(a.shape[0]):
                inter[r] = np.intersect1d(a[r], b[r], assume_unique=True).size
            K = a.shape[1]
            jac[f"{names[i]}|{names[j]}"] = float((inter / (2 * K - inter)).mean())
    res["pairwise_jaccard"] = jac
    args.out_json.write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
