from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np


def clip_is_blank(arr: np.ndarray, thresh: float = 0.02) -> tuple[bool, float]:
    s = float(arr.astype(np.float64).std())
    return (s < thresh), s


def scan(gen_dir: Path, num_samples: int | None, thresh: float, seed: int = 42):
    paths = sorted(gen_dir.glob("gen_*.npy"))
    if not paths:
        paths = sorted(gen_dir.glob("*.npy"))
    if num_samples is not None and len(paths) > num_samples:
        paths = paths[:num_samples]
    stds = np.empty(len(paths), dtype=np.float64)
    blanks = np.zeros(len(paths), dtype=bool)
    for i, p in enumerate(paths):
        arr = np.load(p)
        b, s = clip_is_blank(arr, thresh)
        stds[i] = s
        blanks[i] = b
    n = len(paths)
    n_blank = int(blanks.sum())
    out = {
        "gen_dir": str(gen_dir),
        "n": n,
        "n_blank": n_blank,
        "blank_frac": (n_blank / n) if n else float("nan"),
        "blank_pct": (100.0 * n_blank / n) if n else float("nan"),
        "thresh": thresh,
        "mean_std": float(stds.mean()) if n else float("nan"),
        "median_std": float(np.median(stds)) if n else float("nan"),
        "min_std": float(stds.min()) if n else float("nan"),
        "max_std": float(stds.max()) if n else float("nan"),
    }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen-dir", required=True, type=Path)
    ap.add_argument("--num-samples", type=int, default=None)
    ap.add_argument("--thresh", type=float, default=0.02)
    ap.add_argument("--out-json", type=Path, default=None)
    args = ap.parse_args()
    res = scan(args.gen_dir, args.num_samples, args.thresh)
    print(json.dumps(res, indent=2))
    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
