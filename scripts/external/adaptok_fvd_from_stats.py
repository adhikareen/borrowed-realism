#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
import torch

SUBSETS = [42, 1337, 7, 2024, 555, 99, 314]
ARMS = ["fp_adaptive", "fp_uniform", "fr_adaptive", "fr_uniform"]
SEED42_DIRS = {"fp_adaptive": "adaptok_fp_gen_adaptive", "fp_uniform": "adaptok_fp_gen_uniform",
               "fr_adaptive": "fp_deconf_adaptive", "fr_uniform": "fp_deconf_uniform"}


def arm_dir(runs: Path, arm: str, subset: int) -> Path:
    d = runs / f"fp_robust_{arm}_s{subset}"
    if subset == 42 and not d.exists():
        d = runs / SEED42_DIRS[arm]
    return d


def mean_cov(pkl: Path):
    with open(pkl, "rb") as f:
        d = pickle.load(f)
    assert d["capture_mean_cov"] and not d["only_stats_mode"], pkl
    n = d["num_items"]
    mean = d["raw_mean"] / n
    cov = d["raw_cov"] / n - np.outer(mean, mean)
    return mean, cov, int(n)


def _sqrtm_sym(mat: torch.Tensor, eps: float = 1e-10) -> torch.Tensor:
    u, s, v = torch.svd(mat)
    si = torch.where(s < eps, s, torch.sqrt(s))
    return u @ torch.diag(si) @ v.t()


def fvd(gen_pkl: Path, gt_pkl: Path):
    mg, cg, ng = mean_cov(gen_pkl)
    mr, cr, nr = mean_cov(gt_pkl)
    mg, cg, mr, cr = (torch.from_numpy(x) for x in (mg, cg, mr, cr))
    s = _sqrtm_sym(cg)
    trace_sqrt = torch.trace(_sqrtm_sym(s @ cr @ s))
    val = torch.sum((mg - mr) ** 2) + torch.trace(cg + cr) - 2.0 * trace_sqrt
    return float(val), ng, nr


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_dir", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()
    torch.set_num_threads(4)
    per = {}
    for s in SUBSETS:
        row = {}
        for arm in ARMS:
            d = arm_dir(a.runs_dir, arm, s)
            g, t = d / "generated_fvd_stats_0_500.pkl", d / "gt_fvd_stats_0_500.pkl"
            v, ng, nr = fvd(g, t)
            row[arm] = {"fvd": v, "n_generated": ng, "n_gt": nr, "stats_dir": d.name,
                        "sha256_generated_stats": sha256(g), "sha256_gt_stats": sha256(t)}
        row["fp_advantage_uniform_minus_adaptive"] = row["fp_uniform"]["fvd"] - row["fp_adaptive"]["fvd"]
        row["fr_advantage_uniform_minus_adaptive"] = row["fr_uniform"]["fvd"] - row["fr_adaptive"]["fvd"]
        per[str(s)] = row
        print(f"subset {s:>5d}: FP adv {row['fp_advantage_uniform_minus_adaptive']:+8.3f}   "
              f"class-cond adv {row['fr_advantage_uniform_minus_adaptive']:+8.3f}")
    out = {
        "what": "AdapTok adaptive vs uniform allocation, FVD recomputed from stored I3D feature statistics",
        "fvd_formula": "AdapTok utils/fvd/fvd.py calculate_fvd (64px resize + I3D final features, as logged)",
        "subsets": SUBSETS,
        "advantage_sign": "uniform FVD minus adaptive FVD (positive = adaptive better)",
        "per_subset": per,
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=1))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
