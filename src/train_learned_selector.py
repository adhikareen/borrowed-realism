from __future__ import annotations
import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import paths
import argparse, json, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import mode as _scipy_mode, spearmanr

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from extract_omnitok_decode import load_model, coarse_upsampled_idx  # noqa: E402


def cell_features(idx_np, emb_np, pool=2):
    B, T, H, W = idx_np.shape
    fine = emb_np[idx_np]
    cup = coarse_upsampled_idx(idx_np.astype(np.int64), pool)
    coarse = emb_np[cup]
    diff = fine - coarse
    return np.concatenate([fine, coarse, diff], axis=-1).astype(np.float32)


class SelectorMLP(nn.Module):
    def __init__(self, in_dim=24, hidden=64, layers=2, linear=False):
        super().__init__()
        if linear:
            self.net = nn.Linear(in_dim, 1)
        else:
            mods = [nn.Linear(in_dim, hidden), nn.GELU()]
            for _ in range(layers - 1):
                mods += [nn.Linear(hidden, hidden), nn.GELU()]
            mods += [nn.Linear(hidden, 1)]
            self.net = nn.Sequential(*mods)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def load_shards(raw_dir: Path, emb_np, num=None, pool=2, target_field="decode_imp"):
    shards = sorted((raw_dir / "shards").glob("shard_*.npz"))
    feats_l, tgt_l, gap_l = [], [], []
    clip_meta = []
    taken = 0
    for sh in shards:
        with np.load(sh) as d:
            idx = d["indices"].astype(np.int64)
            dec = d[target_field].astype(np.float32)
            gap = d["l2gap"].astype(np.float32)
        if num is not None and taken + idx.shape[0] > num:
            keep = num - taken
            idx, dec, gap = idx[:keep], dec[:keep], gap[:keep]
        N, T, H, W = idx.shape
        f = cell_features(idx, emb_np, pool)
        feats_l.append(f.reshape(-1, 24))
        tgt_l.append(dec.reshape(-1))
        gap_l.append(gap.reshape(-1))
        clip_meta.append((sh, N, T, H, W))
        taken += N
        if num is not None and taken >= num:
            break
    return (np.concatenate(feats_l), np.concatenate(tgt_l),
            np.concatenate(gap_l), clip_meta)


def topk_jaccard(score_a, score_b, K):
    a = set(np.argpartition(score_a, -K)[-K:].tolist())
    b = set(np.argpartition(score_b, -K)[-K:].tolist())
    return len(a & b) / len(a | b)


def per_clip_jaccard(pred_flat, ref_flat, cells_per_clip, K, nclip=200):
    js = []
    n = min(nclip, len(pred_flat) // cells_per_clip)
    for i in range(n):
        s = slice(i * cells_per_clip, (i + 1) * cells_per_clip)
        js.append(topk_jaccard(pred_flat[s], ref_flat[s], K))
    return float(np.mean(js)), n


def emit_pred_shards(raw_dir: Path, out_dir: Path, model, emb_np, mu, sd,
                     device, pool=2, num=None):
    (out_dir / "shards").mkdir(parents=True, exist_ok=True)
    shards = sorted((raw_dir / "shards").glob("shard_*.npz"))
    model.eval()
    keep_fields = ("indices", "l2gap", "decode_imp")
    for sh in shards:
        with np.load(sh) as d:
            data = {k: d[k] for k in d.files if k in keep_fields}
        idx = data["indices"].astype(np.int64)
        N, T, H, W = idx.shape
        f = cell_features(idx, emb_np, pool).reshape(-1, 24)
        f = (f - mu) / sd
        with torch.no_grad():
            preds = []
            for s in range(0, f.shape[0], 1_000_000):
                xb = torch.from_numpy(f[s:s + 1_000_000]).to(device)
                preds.append(model(xb).float().cpu().numpy())
            pred = np.concatenate(preds).reshape(N, T, H, W).astype(np.float16)
        data["pred"] = pred
        out_path = out_dir / "shards" / sh.name
        np.savez_compressed(out_path, **data)
        print(f"[emit] {sh.name} pred{pred.shape} (dropped frames) -> {out_path}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--omni-ckpt", default=paths.OMNITOK_CKPT)
    ap.add_argument("--train-raw", required=True, type=Path)
    ap.add_argument("--val-raw", required=True, type=Path)
    ap.add_argument("--out-train", required=True, type=Path, help="new shards with `pred` (train)")
    ap.add_argument("--out-val", required=True, type=Path, help="new shards with `pred` (val)")
    ap.add_argument("--metrics-json", required=True, type=Path)
    ap.add_argument("--keep-frac", type=float, default=0.60)
    ap.add_argument("--linear", action="store_true")
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch", type=int, default=131072)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--train-num", type=int, default=8000)
    ap.add_argument("--pool", type=int, default=2)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--target-field", type=str, default="decode_imp",
                    choices=["decode_imp", "sg_imp"],
                    help="decode_imp | sg_imp")
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t0 = time.time()

    print(f"[sel] loading codebook from {args.omni_ckpt}", flush=True)
    m = load_model(args.omni_ckpt, device)
    emb_np = m.codebook.embeddings.detach().cpu().numpy().astype(np.float32)
    del m
    torch.cuda.empty_cache()
    print(f"[sel] codebook emb {emb_np.shape}", flush=True)

    print(f"[sel] loading TRAIN shards (target={args.target_field}) ...", flush=True)
    Xtr, Ytr, Gtr, _ = load_shards(args.train_raw, emb_np, num=args.train_num, pool=args.pool,
                                   target_field=args.target_field)
    print("[sel] loading VAL shards ...", flush=True)
    Xva, Yva, Gva, va_meta = load_shards(args.val_raw, emb_np, num=None, pool=args.pool,
                                         target_field=args.target_field)
    cells_per_clip = va_meta[0][2] * va_meta[0][3] * va_meta[0][4]
    fine_tokens = cells_per_clip
    K = int(round(args.keep_frac * fine_tokens))
    print(f"[sel] Xtr{Xtr.shape} Xva{Xva.shape} cells/clip={cells_per_clip} K={K}", flush=True)

    mu = Xtr.mean(0, keepdims=True); sd = Xtr.std(0, keepdims=True) + 1e-6
    Xtr_n = (Xtr - mu) / sd
    Xva_n = (Xva - mu) / sd
    if args.target_field == "sg_imp":
        Ytr_t = np.sign(Ytr) * np.log1p(np.abs(Ytr))
    else:
        Ytr_t = np.log1p(np.clip(Ytr, 0, None))
    ymu, ysd = Ytr_t.mean(), Ytr_t.std() + 1e-6
    Ytr_z = (Ytr_t - ymu) / ysd

    model = SelectorMLP(24, args.hidden, args.layers, args.linear).to(device)
    nparam = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    Xtr_g = torch.from_numpy(Xtr_n).to(device)
    Ytr_g = torch.from_numpy(Ytr_z.astype(np.float32)).to(device)
    Ntr = Xtr_g.shape[0]
    print(f"[sel] model params={nparam} (linear={args.linear}) train_cells={Ntr}", flush=True)

    for ep in range(args.epochs):
        perm = torch.randperm(Ntr, device=device)
        tot = 0.0; nb = 0
        model.train()
        for s in range(0, Ntr, args.batch):
            b = perm[s:s + args.batch]
            xb, yb = Xtr_g[b], Ytr_g[b]
            opt.zero_grad(set_to_none=True)
            pr = model(xb)
            loss = F.mse_loss(pr, yb)
            loss.backward(); opt.step()
            tot += loss.item(); nb += 1
        model.eval()
        with torch.no_grad():
            sub = np.random.default_rng(0).choice(Xva_n.shape[0], min(500000, Xva_n.shape[0]), replace=False)
            pv = model(torch.from_numpy(Xva_n[sub]).to(device)).cpu().numpy()
        rho = spearmanr(pv, Yva[sub]).correlation
        print(f"[sel] epoch {ep+1}/{args.epochs} train_mse={tot/nb:.4f} val_spearman(pred,decode)={rho:.4f}", flush=True)

    model.eval()
    with torch.no_grad():
        pred_va = []
        for s in range(0, Xva_n.shape[0], 1_000_000):
            pred_va.append(model(torch.from_numpy(Xva_n[s:s + 1_000_000]).to(device)).cpu().numpy())
        pred_va = np.concatenate(pred_va)

    rho_full = spearmanr(pred_va, Yva).correlation
    rho_gap = spearmanr(Gva, Yva).correlation
    jac_learned, njc = per_clip_jaccard(pred_va, Yva, cells_per_clip, K, nclip=200)
    jac_gap, _ = per_clip_jaccard(Gva, Yva, cells_per_clip, K, nclip=200)
    rng = np.random.default_rng(0)
    jr = []
    for i in range(min(200, Xva_n.shape[0] // cells_per_clip)):
        rr = rng.random(cells_per_clip)
        s = slice(i * cells_per_clip, (i + 1) * cells_per_clip)
        jr.append(topk_jaccard(rr, Yva[s], K))
    jac_random = float(np.mean(jr))

    metrics = {
        "model": "linear" if args.linear else f"mlp_h{args.hidden}_L{args.layers}",
        "n_params": int(nparam),
        "feature": "fine_emb(8)+coarse_emb(8)+diff(8)=24, decode-free",
        "keep_frac": args.keep_frac, "K_top": int(K), "fine_tokens": int(fine_tokens),
        "val_spearman_learned_vs_decode": float(rho_full),
        "val_spearman_l2gap_vs_decode": float(rho_gap),
        "topk_jaccard_learned_vs_decode": float(jac_learned),
        "topk_jaccard_l2gap_vs_decode": float(jac_gap),
        "topk_jaccard_random_vs_decode": float(jac_random),
        "n_clips_jaccard": int(njc),
        "interpretation": "higher spearman/jaccard = closer to the decode ranking",
        "elapsed_sec": time.time() - t0,
    }
    args.metrics_json.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_json.write_text(json.dumps(metrics, indent=2))
    print("[sel] METRICS:\n" + json.dumps(metrics, indent=2), flush=True)

    print(f"[sel] emitting TRAIN pred shards -> {args.out_train}", flush=True)
    emit_pred_shards(args.train_raw, args.out_train, model, emb_np, mu, sd, device,
                     pool=args.pool, num=None)
    print(f"[sel] emitting VAL pred shards -> {args.out_val}", flush=True)
    emit_pred_shards(args.val_raw, args.out_val, model, emb_np, mu, sd, device,
                     pool=args.pool, num=None)
    torch.save({"state_dict": model.state_dict(), "mu": mu, "sd": sd,
                "ymu": ymu, "ysd": ysd, "args": vars(args), "metrics": metrics},
               args.metrics_json.with_suffix(".selector.pt"))
    print(f"[sel] ALL DONE in {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
