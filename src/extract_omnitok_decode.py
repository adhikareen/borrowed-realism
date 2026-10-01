from __future__ import annotations
import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import paths
import argparse, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from scipy.stats import mode as _scipy_mode

sys.path.insert(0, paths.WORK)
sys.path.insert(0, paths.OMNITOK_DIR)
from OmniTokenizer.omnitokenizer import VQGAN
from baseline_magvit2.datasets import VideoDecodeConfig, VideoFileListDataset


def load_model(ckpt, device):
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    hp = ck.get("hyper_parameters", ck.get("args"))
    if hasattr(hp, "args"):
        margs = hp["args"]
    elif isinstance(hp, dict) and "args" in hp:
        margs = hp["args"]
    else:
        margs = hp
    for fix in ("causal_in_temporal_transformer", "causal_in_peg"):
        bad = fix.replace("causal", "casual")
        if not hasattr(margs, fix) and hasattr(margs, bad):
            setattr(margs, fix, getattr(margs, bad))
    m = VQGAN(margs).eval()
    m.load_state_dict(ck["state_dict"], strict=False)
    return m.to(device).to(torch.float32)


@torch.no_grad()
def encode_batch(m, x01, device):
    xpm1 = (x01.to(device, non_blocking=True) * 2 - 1).to(torch.float32)
    e = m.encode(xpm1, is_image=False, include_embeddings=False)
    if isinstance(e, dict):
        e = e.get("encodings", e.get("indices"))
    elif isinstance(e, tuple):
        e = e[0]
    return e


def coarse_upsampled_idx(idx_np, pool=2):
    B, T, H, W = idx_np.shape
    ch, cw = H // pool, W // pool
    win = (idx_np.reshape(B, T, ch, pool, cw, pool)
                 .transpose(0, 1, 2, 4, 3, 5).reshape(B, T, ch, cw, pool * pool))
    cmode = _scipy_mode(win, axis=-1, keepdims=False).mode.astype(np.int64)
    return np.repeat(np.repeat(cmode, pool, axis=-2), pool, axis=-1)


@torch.no_grad()
def compute_l2gap(idx, emb, pool=2):
    B, T, H, W = idx.shape
    fine_emb = F.embedding(idx, emb)
    cup = coarse_upsampled_idx(idx.cpu().numpy(), pool)
    coarse_emb = F.embedding(torch.from_numpy(cup).to(idx.device), emb)
    return (fine_emb - coarse_emb).pow(2).sum(-1)


@torch.no_grad()
def decode_grid(m, grid_long, device):
    rec = m.decode(grid_long.to(device).to(torch.long), is_image=False).float().clamp(-1, 1)
    return (rec + 1) / 2


def latent_to_pixel_frame_map(T_lat, T_pix):
    if T_lat == 1:
        return np.zeros(T_pix, dtype=np.int64)
    t = np.round(np.arange(T_pix) * (T_lat - 1) / (T_pix - 1)).astype(np.int64)
    return np.clip(t, 0, T_lat - 1)


@torch.no_grad()
def decode_importance(rec_f, rec_c, T_lat, cell=8):
    B, C, T_pix, Hp, Wp = rec_f.shape
    gh, gw = Hp // cell, Wp // cell
    se = (rec_f - rec_c).pow(2).mean(1)
    se = se.view(B, T_pix, gh, cell, gw, cell).mean(dim=(3, 5))
    p2l = latent_to_pixel_frame_map(T_lat, T_pix)
    imp = torch.zeros(B, T_lat, gh, gw, device=rec_f.device)
    cnt = torch.zeros(T_lat, device=rec_f.device)
    for p in range(T_pix):
        imp[:, p2l[p]] += se[:, p]
        cnt[p2l[p]] += 1
    return imp / cnt.clamp(min=1).view(1, T_lat, 1, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=paths.OMNITOK_CKPT)
    ap.add_argument("--filelist", required=True)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--num", type=int, default=2000)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--shard-size", type=int, default=1000)
    ap.add_argument("--image-size", type=int, default=128)
    ap.add_argument("--num-frames", type=int, default=17,
                    help="frames; (F-1)%%4 == 0")
    ap.add_argument("--pool", type=int, default=2)
    ap.add_argument("--store-frames", action="store_true")
    ap.add_argument("--num-workers", type=int, default=2)
    args = ap.parse_args()

    assert (args.num_frames - 1) % 4 == 0, \
        f"num_frames-1 must be divisible by 4 (got {args.num_frames})"

    out = Path(args.out_dir)
    (out / "shards").mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    print(f"[decode-extract] loading {args.ckpt}", flush=True)
    m = load_model(args.ckpt, device)
    emb = m.codebook.embeddings.detach().to(device)
    print(f"[decode-extract] n_codes={m.n_codes} emb={tuple(emb.shape)} F={args.num_frames} pool={args.pool}", flush=True)

    ds = VideoFileListDataset(
        filelist=args.filelist,
        config=VideoDecodeConfig(image_size=args.image_size, num_frames=args.num_frames,
                                 random_time_crop=False, force_num_frames=True),
    )
    N = min(args.num, len(ds))
    loader = DataLoader(Subset(ds, list(range(N))), batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True,
                        prefetch_factor=2 if args.num_workers > 0 else None,
                        persistent_workers=args.num_workers > 0)
    print(f"[decode-extract] {N} videos -> {out}", flush=True)

    t0 = time.time()
    buf_idx, buf_gap, buf_dec, buf_frames = [], [], [], []
    shard_num = 0

    def flush():
        nonlocal buf_idx, buf_gap, buf_dec, buf_frames, shard_num
        if not buf_idx:
            return
        arr_idx = np.stack(buf_idx).astype(np.int32)
        arr_gap = np.stack(buf_gap).astype(np.float16)
        arr_dec = np.stack(buf_dec).astype(np.float16)
        path = out / "shards" / f"shard_{shard_num:05d}.npz"
        if buf_frames:
            arr_f = np.stack(buf_frames).astype(np.float16)
            np.savez_compressed(path, indices=arr_idx, l2gap=arr_gap, decode_imp=arr_dec, frames=arr_f)
        else:
            np.savez_compressed(path, indices=arr_idx, l2gap=arr_gap, decode_imp=arr_dec)
        shard_num += 1
        buf_idx, buf_gap, buf_dec, buf_frames = [], [], [], []
        print(f"[decode-extract] saved shard {shard_num} idx{arr_idx.shape} dec{arr_dec.shape}", flush=True)

    for videos_01 in tqdm(loader, desc="decode-extract"):
        videos_01 = videos_01.float()
        idx = encode_batch(m, videos_01, device)
        Tlat = idx.shape[1]
        gap = compute_l2gap(idx, emb, args.pool)
        coarse_up_np = coarse_upsampled_idx(idx.cpu().numpy().astype(np.int64), args.pool)
        rec_f = decode_grid(m, idx, device)
        rec_c = decode_grid(m, torch.from_numpy(coarse_up_np), device)
        Tpix = min(rec_f.shape[2], videos_01.shape[2])
        dec = decode_importance(rec_f[:, :, :Tpix], rec_c[:, :, :Tpix], Tlat)

        idx_np = idx.to(torch.int32).cpu().numpy()
        gap_np = gap.to(torch.float16).cpu().numpy()
        dec_np = dec.to(torch.float16).cpu().numpy()
        f_np = videos_01.to(torch.float16).cpu().numpy() if args.store_frames else None
        for k in range(idx_np.shape[0]):
            buf_idx.append(idx_np[k]); buf_gap.append(gap_np[k]); buf_dec.append(dec_np[k])
            if args.store_frames:
                buf_frames.append(f_np[k])
            if len(buf_idx) >= args.shard_size:
                flush()
    flush()
    print(f"[decode-extract] done {N} in {(time.time()-t0)/60:.1f} min, {shard_num} shards", flush=True)


if __name__ == "__main__":
    main()
