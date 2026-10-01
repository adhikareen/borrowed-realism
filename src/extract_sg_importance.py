from __future__ import annotations
import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import paths
import argparse, json, time
from pathlib import Path

import numpy as np
import torch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import CompactVideoLM, CompactVideoLMConfig
from extract_omnitok_decode import (load_model as load_omni, decode_grid,
                                     coarse_upsampled_idx, decode_importance)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ar-ckpt", required=True, type=Path,
                    help="full-cache AR checkpoint")
    ap.add_argument("--omni-ckpt",
                    default=paths.OMNITOK_CKPT)
    ap.add_argument("--cache-dir", required=True, type=Path,
                    help="full packed cache")
    ap.add_argument("--raw-dir", required=True, type=Path,
                    help="decode-extract dir")
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--num", type=int, default=0, help="0 = all examples")
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--align-checks", type=int, default=16)
    ap.add_argument("--sample", action="store_true",
                    help="sample instead of argmax")
    args = ap.parse_args()

    device = torch.device(args.device)
    man = json.loads((args.cache_dir / "manifest.json").read_text())
    assert man["selector_name"] == "full" and float(man["keep_frac"]) == 1.0, man["selector_name"]
    T, H, W = int(man["latent_t"]), int(man["latent_h"]), int(man["latent_w"])
    coarse_n, fine_n = int(man["coarse_tokens"]), int(man["fine_tokens"])
    assert fine_n == T * H * W

    ids = np.load(args.cache_dir / "input_ids.npy", mmap_mode="r")
    typ = np.load(args.cache_dir / "type_ids.npy", mmap_mode="r")
    pos = np.load(args.cache_dir / "source_pos.npy", mmap_mode="r")
    N, L = ids.shape
    assert L == 1 + coarse_n + fine_n + 1, (L, coarse_n, fine_n)

    shards = sorted((args.raw_dir / "shards").glob("shard_*.npz"))
    sizes = [int(np.load(sh)["indices"].shape[0]) for sh in shards]
    assert sum(sizes) == N, f"raw total {sum(sizes)} != cache N {N}"
    if args.num and args.num < N:
        N = args.num

    offs = np.concatenate([[0], np.cumsum(sizes)])
    rng = np.random.default_rng(0)
    for gi in rng.choice(N, size=min(args.align_checks, N), replace=False):
        si = int(np.searchsorted(offs, gi, side="right") - 1)
        with np.load(shards[si]) as z:
            fine = z["indices"][gi - offs[si]].reshape(-1).astype(np.int32)
        if not np.array_equal(np.asarray(ids[gi, 1 + coarse_n:-1]), fine):
            raise AssertionError(f"ALIGNMENT FAIL row {gi}")
    print(f"[sgimp] alignment verified on {min(args.align_checks, N)} rows", flush=True)

    ck = torch.load(args.ar_ckpt, map_location="cpu")
    cfg = CompactVideoLMConfig(**ck["model_config"])
    ar = CompactVideoLM(cfg); ar.load_state_dict(ck["model"]); ar.to(device).eval()
    print(f"[sgimp] AR loaded {args.ar_ckpt} step={ck.get('step')} best_val={ck.get('best_val_loss')}", flush=True)
    omni = load_omni(args.omni_ckpt, device)
    print(f"[sgimp] OmniTok loaded n_codes={omni.n_codes}", flush=True)

    all_sg = np.empty((N, fine_n), dtype=np.float32)
    all_ep = np.empty((N, fine_n), dtype=np.float32)
    all_di = np.empty((N, fine_n), dtype=np.float32)
    bs = args.batch_size
    t0 = time.time()
    with torch.no_grad():
        for s in range(0, N, bs):
            e = min(N, s + bs)
            ii = torch.from_numpy(np.asarray(ids[s:e])).long().to(device)
            tt = torch.from_numpy(np.asarray(typ[s:e])).long().to(device)
            pp = torch.from_numpy(np.asarray(pos[s:e])).long().to(device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                logits, _ = ar(input_ids=ii, source_pos=pp, type_ids=tt)
            lg = logits[:, coarse_n:coarse_n + fine_n, :8192].float()
            if args.sample:
                probs = torch.softmax(lg / 0.9, dim=-1)
                topv, topi = probs.topk(256, dim=-1)
                topv = topv / topv.sum(-1, keepdim=True)
                choice = torch.multinomial(topv.reshape(-1, 256), 1).reshape(lg.shape[0], fine_n, 1)
                pred = topi.gather(-1, choice).squeeze(-1)
            else:
                pred = lg.argmax(-1)
            b = e - s
            true_fine = ii[:, 1 + coarse_n:1 + coarse_n + fine_n].reshape(b, T, H, W)
            pred_grid = pred.reshape(b, T, H, W)
            coarse_up = torch.from_numpy(
                coarse_upsampled_idx(true_fine.cpu().numpy().astype(np.int64), pool=2)).to(device)

            rec_fGT = decode_grid(omni, true_fine, device)
            rec_c = decode_grid(omni, coarse_up, device)
            rec_fp = decode_grid(omni, pred_grid, device)
            Tpix = rec_fGT.shape[2]
            di = decode_importance(rec_fGT[:, :, :Tpix], rec_c[:, :, :Tpix], T)
            ep = decode_importance(rec_fp[:, :, :Tpix], rec_fGT[:, :, :Tpix], T)
            sg = di - ep
            all_di[s:e] = di.reshape(b, -1).cpu().numpy()
            all_ep[s:e] = ep.reshape(b, -1).cpu().numpy()
            all_sg[s:e] = sg.reshape(b, -1).cpu().numpy()
            if (s // bs) % 10 == 0:
                dt = time.time() - t0
                print(f"[sgimp] {e}/{N} ({dt:.0f}s, {e/max(dt,1e-6):.1f} clips/s)", flush=True)

    print(f"[sgimp] decode_imp mean={all_di.mean():.4f}  e_pred mean={all_ep.mean():.4f}  "
          f"sg_imp mean={all_sg.mean():.4f}  frac(sg<0)={float((all_sg<0).mean()):.3f}", flush=True)

    out_sh = args.out_dir / "shards"; out_sh.mkdir(parents=True, exist_ok=True)
    cur = 0
    for sh, sz in zip(shards, sizes):
        if cur >= N:
            break
        take = min(sz, N - cur)
        with np.load(sh) as z:
            data = {k: z[k][:take] for k in ("indices", "l2gap", "decode_imp") if k in z.files}
        data["sg_imp"] = all_sg[cur:cur + take].reshape(take, T, H, W).astype(np.float16)
        data["e_pred"] = all_ep[cur:cur + take].reshape(take, T, H, W).astype(np.float16)
        np.savez_compressed(out_sh / sh.name, **data)
        print(f"[sgimp] wrote {sh.name} ({take})", flush=True)
        cur += take

    summary = {
        "ar_ckpt": str(args.ar_ckpt), "raw_dir": str(args.raw_dir), "num_examples": int(N),
        "latent": [T, H, W], "coarse_n": coarse_n, "fine_n": fine_n,
        "decode_imp_mean": float(all_di.mean()), "e_pred_mean": float(all_ep.mean()),
        "sg_imp_mean": float(all_sg.mean()), "sg_neg_frac": float((all_sg < 0).mean()),
        "corr_sgimp_decodeimp": float(np.corrcoef(all_sg.reshape(-1), all_di.reshape(-1))[0, 1]),
        "sampled": bool(args.sample), "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "elapsed_min": (time.time() - t0) / 60.0,
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print("[sgimp] SUMMARY:\n" + json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
