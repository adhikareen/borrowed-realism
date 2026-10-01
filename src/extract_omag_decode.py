from __future__ import annotations
import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import paths
import argparse, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

sys.path.insert(0, paths.WORK)
sys.path.insert(0, str(Path(__file__).resolve().parent))
from om2_tok import (load_om2, om2_encode_indices, om2_indices_to_features,
                     om2_sign_codebook, OM2_VOCAB)
from baseline_magvit2.datasets import VideoDecodeConfig, VideoFileListDataset
from extract_omnitok_decode import (coarse_upsampled_idx, compute_l2gap,
                                    decode_importance)


@torch.no_grad()
def decode_grid_om2(m, grid_long, device):
    quant = om2_indices_to_features(grid_long.to(device).to(torch.long))
    rec = m.decode(quant).float().clamp(-1, 1)
    return (rec + 1) / 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--filelist", required=True)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--num", type=int, default=2000)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--shard-size", type=int, default=1000)
    ap.add_argument("--image-size", type=int, default=128)
    ap.add_argument("--num-frames", type=int, default=17)
    ap.add_argument("--pool", type=int, default=2)
    ap.add_argument("--num-workers", type=int, default=2)
    args = ap.parse_args()

    out = Path(args.out_dir)
    (out / "shards").mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    print("[omag-extract] loading OpenMAGVIT2", flush=True)
    m = load_om2(device=device)
    emb = om2_sign_codebook(device=device)
    print(f"[omag-extract] vocab={OM2_VOCAB} emb={tuple(emb.shape)} "
          f"F={args.num_frames} pool={args.pool}", flush=True)

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
    print(f"[omag-extract] {N} videos -> {out}", flush=True)

    t0 = time.time()
    buf_idx, buf_gap, buf_dec = [], [], []
    shard_num = 0

    def flush():
        nonlocal buf_idx, buf_gap, buf_dec, shard_num
        if not buf_idx:
            return
        arr_idx = np.stack(buf_idx).astype(np.int32)
        arr_gap = np.stack(buf_gap).astype(np.float16)
        arr_dec = np.stack(buf_dec).astype(np.float16)
        path = out / "shards" / f"shard_{shard_num:05d}.npz"
        np.savez_compressed(path, indices=arr_idx, l2gap=arr_gap, decode_imp=arr_dec)
        shard_num += 1
        buf_idx, buf_gap, buf_dec = [], [], []
        print(f"[omag-extract] saved shard {shard_num} idx{arr_idx.shape}", flush=True)

    for videos_01 in tqdm(loader, desc="omag-extract"):
        videos_01 = videos_01.float()
        xpm1 = (videos_01.to(device) * 2 - 1)
        idx = om2_encode_indices(m, xpm1).to(torch.long)
        Tlat = idx.shape[1]
        gap = compute_l2gap(idx, emb, args.pool)
        coarse_up_np = coarse_upsampled_idx(idx.cpu().numpy().astype(np.int64), args.pool)
        rec_f = decode_grid_om2(m, idx, device)
        rec_c = decode_grid_om2(m, torch.from_numpy(coarse_up_np), device)
        Tpix = min(rec_f.shape[2], videos_01.shape[2])
        dec = decode_importance(rec_f[:, :, :Tpix], rec_c[:, :, :Tpix], Tlat)

        idx_np = idx.to(torch.int32).cpu().numpy()
        gap_np = gap.to(torch.float16).cpu().numpy()
        dec_np = dec.to(torch.float16).cpu().numpy()
        for k in range(idx_np.shape[0]):
            buf_idx.append(idx_np[k]); buf_gap.append(gap_np[k]); buf_dec.append(dec_np[k])
            if len(buf_idx) >= args.shard_size:
                flush()
    flush()
    print(f"[omag-extract] done {N} in {(time.time()-t0)/60:.1f} min, {shard_num} shards", flush=True)


if __name__ == "__main__":
    main()
