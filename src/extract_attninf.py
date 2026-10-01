from __future__ import annotations
import argparse, json, time, math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from model import CompactVideoLM, CompactVideoLMConfig, apply_rope


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, type=Path)
    ap.add_argument("--cache-dir", required=True, type=Path)
    ap.add_argument("--raw-dir", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--align-checks", type=int, default=24)
    args = ap.parse_args()

    device = torch.device(args.device)
    man = json.loads((args.cache_dir / "manifest.json").read_text())
    assert man["selector_name"] == "full", f"cache is {man['selector_name']}, need full"
    assert float(man["keep_frac"]) == 1.0
    T, H, W = int(man["latent_t"]), int(man["latent_h"]), int(man["latent_w"])
    coarse_n, fine_n = int(man["coarse_tokens"]), int(man["fine_tokens"])
    assert fine_n == T * H * W

    ids = np.load(args.cache_dir / "input_ids.npy", mmap_mode="r")
    typ = np.load(args.cache_dir / "type_ids.npy", mmap_mode="r")
    pos = np.load(args.cache_dir / "source_pos.npy", mmap_mode="r")
    N, L = ids.shape
    assert L == 1 + coarse_n + fine_n + 1, (L, coarse_n, fine_n)
    assert N == int(man["num_examples"])

    shards = sorted((args.raw_dir / "shards").glob("shard_*.npz"))
    sizes = []
    for sh in shards:
        with np.load(sh) as z:
            sizes.append(int(z["indices"].shape[0]))
    assert sum(sizes) == N, f"raw total {sum(sizes)} != cache N {N}"
    offs = np.concatenate([[0], np.cumsum(sizes)])
    rng = np.random.default_rng(0)
    for gi in rng.choice(N, size=min(args.align_checks, N), replace=False):
        si = int(np.searchsorted(offs, gi, side="right") - 1)
        with np.load(shards[si]) as z:
            fine = z["indices"][gi - offs[si]].reshape(-1).astype(np.int32)
        if not np.array_equal(np.asarray(ids[gi, 1 + coarse_n:-1]), fine):
            raise AssertionError(f"ALIGNMENT FAIL cache row {gi} vs raw shard {si}")
    print(f"[attninf] alignment verified on {min(args.align_checks, N)} random rows", flush=True)

    ck = torch.load(args.ckpt, map_location="cpu")
    cfg = CompactVideoLMConfig(**ck["model_config"])
    model = CompactVideoLM(cfg)
    model.load_state_dict(ck["model"])
    model.to(device).eval()
    nh, hd = cfg.n_heads, cfg.dim // cfg.n_heads
    scale = 1.0 / math.sqrt(hd)
    print(f"[attninf] loaded {args.ckpt} (step={ck.get('step')}, best_val={ck.get('best_val_loss')})",
          flush=True)

    all_attn = np.empty((N, fine_n), dtype=np.float32)
    bs = args.batch_size
    t0 = time.time()
    with torch.no_grad():
        for s in range(0, N, bs):
            e = min(N, s + bs)
            b = e - s
            ii = torch.from_numpy(np.asarray(ids[s:e])).long().to(device)
            tt = torch.from_numpy(np.asarray(typ[s:e])).long().to(device)
            pp = torch.from_numpy(np.asarray(pos[s:e])).long().to(device)
            x = model.token_emb(ii) + model.source_pos_emb(pp) + model.type_emb(tt)
            x = model.drop(x)
            recv = torch.zeros(b, L, device=device, dtype=torch.float32)
            causal = torch.ones(L, L, device=device, dtype=torch.bool).tril()
            for block in model.blocks:
                attn_mod = block.attn
                h = block.attn_norm(x)
                qkv = attn_mod.qkv(h).view(b, L, 3, nh, hd).permute(2, 0, 3, 1, 4)
                q, k, v = qkv[0], qkv[1], qkv[2]
                cos, sin = attn_mod.rope.get(0, L, x.device, x.dtype)
                q = apply_rope(q, cos, sin); k = apply_rope(k, cos, sin)
                scores = torch.matmul(q.float(), k.float().transpose(-2, -1)) * scale
                scores = scores.masked_fill(~causal, float("-inf"))
                attn = F.softmax(scores, dim=-1)
                col = attn.sum(dim=2)
                diag = torch.diagonal(attn, dim1=2, dim2=3)
                recv += (col - diag).mean(dim=1)
                y = torch.matmul(attn.to(v.dtype), v)
                y = y.transpose(1, 2).contiguous().view(b, L, cfg.dim)
                x = x + attn_mod.out(y)
                x = x + block.ffn(block.ffn_norm(x))
            fine_attn = recv[:, 1 + coarse_n:1 + coarse_n + fine_n]
            all_attn[s:e] = fine_attn.cpu().numpy()
            if (s // bs) % 20 == 0:
                print(f"[attninf] {e}/{N} ({time.time()-t0:.0f}s)", flush=True)

    assert np.isfinite(all_attn).all(), "non-finite attention influence"
    mean_a = float(all_attn.mean())
    print(f"[attninf] mean cell attention-influence = {mean_a:.6f}", flush=True)

    out_sh = args.out_dir / "shards"
    out_sh.mkdir(parents=True, exist_ok=True)
    cur = 0
    for sh, sz in zip(shards, sizes):
        np.savez(out_sh / sh.name,
                 score=all_attn[cur:cur + sz].reshape(sz, T, H, W).astype(np.float16))
        cur += sz
    summary = {
        "signal": "attention_influence = sum_{q>p,heads,layers} A[q,p], p=cell position",
        "ckpt": str(args.ckpt), "cache_dir": str(args.cache_dir), "raw_dir": str(args.raw_dir),
        "num_examples": N, "latent": [T, H, W], "coarse_n": coarse_n, "fine_n": fine_n,
        "mean_cell_attn_influence": mean_a,
        "attn_p10": float(np.percentile(all_attn, 10)),
        "attn_p50": float(np.percentile(all_attn, 50)),
        "attn_p90": float(np.percentile(all_attn, 90)),
        "shard_sizes": sizes, "alignment_checks": int(min(args.align_checks, N)),
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
