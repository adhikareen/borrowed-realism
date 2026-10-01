from __future__ import annotations
import argparse, json, time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from model import CompactVideoLM, CompactVideoLMConfig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, type=Path)
    ap.add_argument("--cache-dir", required=True, type=Path,
                    help="full packed cache")
    ap.add_argument("--raw-dir", required=True, type=Path,
                    help="extraction dir")
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--batch-size", type=int, default=16)
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
    print(f"[gensal] alignment verified on {min(args.align_checks, N)} random rows", flush=True)

    ck = torch.load(args.ckpt, map_location="cpu")
    cfg = CompactVideoLMConfig(**ck["model_config"])
    model = CompactVideoLM(cfg)
    model.load_state_dict(ck["model"])
    model.to(device).eval()
    print(f"[gensal] loaded {args.ckpt} (step={ck.get('step')}, best_val={ck.get('best_val_loss')})",
          flush=True)

    all_sal = np.empty((N, fine_n), dtype=np.float32)
    bs = args.batch_size
    t0 = time.time()
    for s in range(0, N, bs):
        e = min(N, s + bs)
        ii = torch.from_numpy(np.asarray(ids[s:e])).long().to(device)
        tt = torch.from_numpy(np.asarray(typ[s:e])).long().to(device)
        pp = torch.from_numpy(np.asarray(pos[s:e])).long().to(device)

        emb = (model.token_emb(ii) + model.source_pos_emb(pp) + model.type_emb(tt)).detach()
        emb.requires_grad_(True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                            enabled=device.type == "cuda"):
            x = model.drop(emb)
            for block in model.blocks:
                x = block(x, kv_cache=None, start_pos=0, type_ids=tt,
                          clamp_fine_to_coarse=False)
            logits = model.lm_head(model.norm(x))
            lg = logits[:, coarse_n:coarse_n + fine_n, :].float()
        tgt = ii[:, 1 + coarse_n:1 + coarse_n + fine_n]
        nll = F.cross_entropy(lg.transpose(1, 2), tgt, reduction="none")
        total = nll.sum()
        model.zero_grad(set_to_none=True)
        if emb.grad is not None:
            emb.grad = None
        total.backward()
        gnorm = emb.grad.detach().float().norm(dim=-1)
        fine_sal = gnorm[:, 1 + coarse_n:1 + coarse_n + fine_n]
        all_sal[s:e] = fine_sal.cpu().numpy()
        if (s // bs) % 20 == 0:
            print(f"[gensal] {e}/{N} ({time.time()-t0:.0f}s)", flush=True)

    assert np.isfinite(all_sal).all(), "non-finite saliency"
    mean_sal = float(all_sal.mean())
    print(f"[gensal] mean cell saliency = {mean_sal:.6f}", flush=True)

    out_sh = args.out_dir / "shards"
    out_sh.mkdir(parents=True, exist_ok=True)
    cur = 0
    for sh, sz in zip(shards, sizes):
        np.savez(out_sh / sh.name,
                 score=all_sal[cur:cur + sz].reshape(sz, T, H, W).astype(np.float16))
        cur += sz
    summary = {
        "signal": "generation_saliency = ||d(sum fine NLL)/d(input embedding of cell)||_2",
        "ckpt": str(args.ckpt), "cache_dir": str(args.cache_dir), "raw_dir": str(args.raw_dir),
        "num_examples": N, "latent": [T, H, W], "coarse_n": coarse_n, "fine_n": fine_n,
        "mean_cell_saliency": mean_sal,
        "sal_p10": float(np.percentile(all_sal, 10)),
        "sal_p50": float(np.percentile(all_sal, 50)),
        "sal_p90": float(np.percentile(all_sal, 90)),
        "shard_sizes": sizes, "alignment_checks": int(min(args.align_checks, N)),
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
