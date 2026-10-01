from __future__ import annotations
import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import paths
import argparse, json, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from scipy.stats import spearmanr
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from extract_omnitok_decode import load_model, coarse_upsampled_idx  # noqa: E402
from train_learned_selector import SelectorMLP  # noqa: E402

FEAT_DIM = 26


def coarse_only_features(coarse_idx: np.ndarray, emb_np: np.ndarray) -> np.ndarray:
    B, T, h, w = coarse_idx.shape
    ce = emb_np[coarse_idx]
    pad = np.pad(ce, ((0, 0), (0, 0), (1, 1), (1, 1), (0, 0)), mode="edge")
    nb = np.zeros_like(ce)
    for di in range(3):
        for dj in range(3):
            nb += pad[:, :, di:di + h, dj:dj + w, :]
    nb /= 9.0
    dev = ce - nb
    up = lambda a: np.repeat(np.repeat(a, 2, axis=2), 2, axis=3)
    P, N, D = up(ce), up(nb), up(dev)
    H, W = 2 * h, 2 * w
    py = np.broadcast_to((np.arange(H) % 2).reshape(1, 1, H, 1), (B, T, H, W))[..., None]
    px = np.broadcast_to((np.arange(W) % 2).reshape(1, 1, 1, W), (B, T, H, W))[..., None]
    return np.concatenate([P, N, D, py.astype(np.float32), px.astype(np.float32)],
                          axis=-1).astype(np.float32)


def coarse_grid_from_fine(idx_np: np.ndarray, pool: int = 2) -> np.ndarray:
    return coarse_upsampled_idx(idx_np.astype(np.int64), pool)[:, :, ::pool, ::pool]


def load_split(raw_dir: Path, emb_np, num=None, pool=2):
    feats_l, tgt_l, gap_l, imp_l = [], [], [], []
    seen = 0
    for shard in sorted(Path(raw_dir).glob("shards/*.npz")):
        d = np.load(shard)
        idx = d["indices"].astype(np.int64)
        imp = d["decode_imp"].astype(np.float32)
        gap = d["l2gap"].astype(np.float32)
        if num is not None and seen + idx.shape[0] > num:
            k = num - seen
            idx, imp, gap = idx[:k], imp[:k], gap[:k]
        cg = coarse_grid_from_fine(idx, pool)
        f = coarse_only_features(cg, emb_np)
        feats_l.append(f.reshape(-1, FEAT_DIM))
        tgt_l.append(imp.reshape(-1))
        gap_l.append(gap.reshape(-1))
        imp_l.append(imp)
        seen += idx.shape[0]
        if num is not None and seen >= num:
            break
    return (np.concatenate(feats_l), np.concatenate(tgt_l),
            np.concatenate(gap_l), np.concatenate(imp_l, axis=0))


def emit_pred_shards(raw_dir: Path, out_dir: Path, net, emb_np, mu, sd, pool, device):
    (out_dir / "shards").mkdir(parents=True, exist_ok=True)
    for sh in sorted(Path(raw_dir).glob("shards/*.npz")):
        d = np.load(sh)
        data = {k: d[k] for k in d.files if k != "frames"}
        idx = d["indices"].astype(np.int64)
        N, T, H, W = idx.shape
        cg = coarse_grid_from_fine(idx, pool)
        X = coarse_only_features(cg, emb_np).reshape(-1, FEAT_DIM)
        Xn = torch.from_numpy((X - mu) / sd)
        outs = []
        with torch.no_grad():
            for i in range(0, Xn.shape[0], 262144):
                outs.append(net(Xn[i:i + 262144].to(device)).float().cpu().numpy())
        data["pred"] = np.concatenate(outs).reshape(N, T, H, W).astype(np.float16)
        np.savez_compressed(out_dir / "shards" / sh.name, **data)
        print(f"[emit] {sh.name} -> {out_dir}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--omni-ckpt", default=paths.OMNITOK_CKPT)
    ap.add_argument("--train-raw", type=Path, default=Path("runs/omni_decode_extract_F17_train"))
    ap.add_argument("--val-raw", type=Path, default=Path("runs/omni_decode_extract_F17_val"))
    ap.add_argument("--out", type=Path, default=Path("runs/coarse_only_selector.pt"))
    ap.add_argument("--metrics-json", type=Path, default=Path("runs/coarse_only_selector_metrics.json"))
    ap.add_argument("--train-num", type=int, default=8000)
    ap.add_argument("--keep-frac", type=float, default=0.6)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch", type=int, default=131072)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--pool", type=int, default=2)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--out-train", type=Path, default=Path("runs/omni_coarseonly_extract_F17_train"))
    ap.add_argument("--out-val", type=Path, default=Path("runs/omni_coarseonly_extract_F17_val"))
    args = ap.parse_args()

    t0 = time.time()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    model_tok = load_model(args.omni_ckpt, dev)
    emb_np = model_tok.codebook.embeddings.detach().cpu().numpy().astype(np.float32)
    del model_tok; torch.cuda.empty_cache()
    print(f"[coarse-head] codebook {emb_np.shape}", flush=True)

    Xtr, ytr, _, _ = load_split(args.train_raw, emb_np, args.train_num, args.pool)
    Xva, yva, gva, impva = load_split(args.val_raw, emb_np, None, args.pool)
    print(f"[coarse-head] train {Xtr.shape} val {Xva.shape}", flush=True)

    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    ltr, lva = np.log1p(ytr), np.log1p(yva)
    ymu, ysd = ltr.mean(), ltr.std() + 1e-6

    net = SelectorMLP(in_dim=FEAT_DIM, hidden=args.hidden, layers=args.layers).to(dev)
    n_params = sum(p.numel() for p in net.parameters())
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    Xt = torch.from_numpy((Xtr - mu) / sd); yt = torch.from_numpy((ltr - ymu) / ysd)
    M = Xt.shape[0]
    for ep in range(args.epochs):
        perm = torch.randperm(M)
        tot = 0.0; nb = 0
        for i in range(0, M, args.batch):
            b = perm[i:i + args.batch]
            xb, yb = Xt[b].to(dev, non_blocking=True), yt[b].to(dev, non_blocking=True)
            loss = nn.functional.mse_loss(net(xb), yb)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item(); nb += 1
        print(f"[coarse-head] epoch {ep+1}/{args.epochs} loss {tot/max(nb,1):.4f}", flush=True)

    net.eval()
    with torch.no_grad():
        pv = []
        Xv = torch.from_numpy((Xva - mu) / sd)
        for i in range(0, Xv.shape[0], 262144):
            pv.append(net(Xv[i:i + 262144].to(dev)).cpu().numpy())
        pred = np.concatenate(pv)

    n_cells = impva.shape[1] * impva.shape[2] * impva.shape[3]
    K = int(round(args.keep_frac * n_cells))
    P = pred.reshape(impva.shape[0], -1); I = impva.reshape(impva.shape[0], -1)
    G = gva.reshape(impva.shape[0], -1)
    def jac(a, b):
        ta = np.argpartition(a, -K, axis=1)[:, -K:]
        tb = np.argpartition(b, -K, axis=1)[:, -K:]
        return float(np.mean([len(np.intersect1d(x, y)) / K for x, y in zip(ta, tb)]))
    rng = np.random.default_rng(0)
    R = rng.random(P.shape)
    metrics = {
        "model": f"mlp_h{args.hidden}_L{args.layers}_coarse_only",
        "n_params": n_params,
        "feature": "parent(8)+neigh(8)+dev(8)+inblock(2)=26, coarse-only (generation-causal)",
        "keep_frac": args.keep_frac, "K_top": K, "fine_tokens": n_cells,
        "val_spearman_coarse_only_vs_decode": float(spearmanr(pred, lva).correlation),
        "val_spearman_l2gap_vs_decode": float(spearmanr(gva, lva).correlation),
        "topk_jaccard_coarse_only_vs_decode": jac(P, I),
        "topk_jaccard_l2gap_vs_decode": jac(G, I),
        "topk_jaccard_random_vs_decode": jac(R, I),
        "n_clips_val": int(impva.shape[0]),
        "elapsed_sec": time.time() - t0,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": net.state_dict(), "mu": mu, "sd": sd, "ymu": ymu, "ysd": ysd,
                "feat_dim": FEAT_DIM, "args": vars(args), "metrics": metrics}, args.out)
    args.metrics_json.write_text(json.dumps(metrics, indent=2, default=str))
    emit_pred_shards(args.train_raw, args.out_train, net, emb_np, mu, sd, args.pool, dev)
    emit_pred_shards(args.val_raw, args.out_val, net, emb_np, mu, sd, args.pool, dev)
    print(json.dumps(metrics, indent=2, default=str), flush=True)


if __name__ == "__main__":
    main()
