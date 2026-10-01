from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import scipy.linalg
import torch
import torch.nn.functional as F
from tqdm import tqdm


def load_i3d(device, penultimate: bool = False):
    model = torch.hub.load(
        "facebookresearch/pytorchvideo",
        "i3d_r50",
        pretrained=True,
        verbose=False,
    )
    if penultimate:
        head = model.blocks[-1]
        if not hasattr(head, "proj"):
            raise RuntimeError("I3D head has no .proj attribute; cannot extract penultimate features")
        head.proj = torch.nn.Identity()
    else:
        pass
    model = model.to(device).eval()
    return model


def preprocess_for_i3d(vids: torch.Tensor) -> torch.Tensor:
    B, C, T, H, W = vids.shape
    if H != 224 or W != 224:
        frames = vids.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
        frames = F.interpolate(frames, size=(224, 224), mode="bilinear", align_corners=False)
        vids = frames.reshape(B, T, C, 224, 224).permute(0, 2, 1, 3, 4)
    return vids * 2.0 - 1.0


def frechet_distance(feats_gen: np.ndarray, feats_real: np.ndarray) -> float:
    mu_g = feats_gen.mean(axis=0)
    mu_r = feats_real.mean(axis=0)
    sigma_g = np.cov(feats_gen, rowvar=False)
    sigma_r = np.cov(feats_real, rowvar=False)
    diff = mu_g - mu_r
    covmean, _ = scipy.linalg.sqrtm(sigma_g @ sigma_r, disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(diff @ diff + np.trace(sigma_g + sigma_r - 2 * covmean))


def load_npy_video(path: Path, num_frames: int = 17) -> torch.Tensor:
    arr = np.load(path).astype(np.float32)
    if arr.shape[1] != num_frames:
        arr = arr[:, :num_frames] if arr.shape[1] > num_frames else \
              np.pad(arr, ((0, 0), (0, num_frames - arr.shape[1]), (0, 0), (0, 0)), mode="edge")
    return torch.from_numpy(arr)


def extract_features(paths, i3d, device, batch_size, num_frames, log):
    feats = []
    pbar = tqdm(range(0, len(paths), batch_size), desc="i3d")
    for bstart in pbar:
        batch_paths = paths[bstart:bstart + batch_size]
        vids = torch.stack([load_npy_video(p, num_frames) for p in batch_paths], 0)
        vids = preprocess_for_i3d(vids).to(device)
        with torch.no_grad():
            f = i3d(vids)
        if f.dim() > 2:
            f = f.flatten(start_dim=1)
        feats.append(f.cpu())
        del vids, f
        torch.cuda.empty_cache()
    feats = torch.cat(feats, dim=0)
    log(f"features: {feats.shape}")
    return feats


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen-dir", required=True, type=Path)
    ap.add_argument("--real-dir", required=True, type=Path)
    ap.add_argument("--out-json", required=True, type=Path)
    ap.add_argument("--num-samples", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--num-frames", type=int, default=17)
    ap.add_argument("--resume-features", action="store_true")
    ap.add_argument("--use-penultimate", action="store_true",
                    help="use 2048-d penultimate I3D features")
    return ap.parse_args()


def main():
    args = parse_args()
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    log_path = args.out_json.with_suffix(".log")

    def log(msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        with log_path.open("a") as f:
            f.write(line + "\n")

    allow_missing = os.environ.get("ALLOW_MISSING_FVD", "0") == "1"

    random.seed(args.seed)
    device = torch.device(args.device)

    gen_paths = sorted(args.gen_dir.glob("*.npy"))
    random.shuffle(gen_paths)
    gen_paths = gen_paths[:args.num_samples]
    log(f"gen: {len(gen_paths)} videos from {args.gen_dir}")

    real_paths = []
    for sub in ["", "train", "val"]:
        d = args.real_dir / sub if sub else args.real_dir
        if d.exists():
            real_paths.extend(d.glob("*.npy"))
    random.shuffle(real_paths)
    real_paths = real_paths[:args.num_samples]
    log(f"real: {len(real_paths)} videos from {args.real_dir}")

    MIN_N = 16
    if len(gen_paths) < MIN_N or len(real_paths) < MIN_N:
        reason = f"insufficient samples: n_gen={len(gen_paths)} n_real={len(real_paths)} (need >={MIN_N})"
        if allow_missing:
            args.out_json.write_text(json.dumps({"skipped": True, "reason": reason}, indent=2))
            log(f"skipping FVD: {reason}")
            return
        raise RuntimeError(reason)

    feats_dir = args.out_json.parent / "_features"
    feats_dir.mkdir(exist_ok=True)
    feat_suffix = "_penultimate" if args.use_penultimate else ""
    gen_feats_path = feats_dir / f"{args.gen_dir.name}_gen{feat_suffix}.pt"
    real_feats_path = feats_dir / f"{args.real_dir.name}_real{feat_suffix}.pt"

    log("loading I3D-R50 (PyTorchVideo) ...")
    i3d = load_i3d(device, penultimate=args.use_penultimate)
    feat_type = "penultimate_2048d" if args.use_penultimate else "logits_400d"
    log(f"I3D loaded (feat_type={feat_type})")

    if args.resume_features and gen_feats_path.exists():
        log(f"resume gen features from {gen_feats_path}")
        gen_feats = torch.load(gen_feats_path, map_location="cpu")
    else:
        t0 = time.time()
        gen_feats = extract_features(gen_paths, i3d, device, args.batch_size, args.num_frames, log)
        torch.save(gen_feats, gen_feats_path)
        log(f"gen features saved in {time.time()-t0:.1f}s")
    if gen_feats.shape[0] > args.num_samples:
        log(f"slicing cached gen features {gen_feats.shape[0]} -> {args.num_samples}")
        gen_feats = gen_feats[:args.num_samples]

    if args.resume_features and real_feats_path.exists():
        log(f"resume real features from {real_feats_path}")
        real_feats = torch.load(real_feats_path, map_location="cpu")
    else:
        t0 = time.time()
        real_feats = extract_features(real_paths, i3d, device, args.batch_size, args.num_frames, log)
        torch.save(real_feats, real_feats_path)
        log(f"real features saved in {time.time()-t0:.1f}s")
    if real_feats.shape[0] > args.num_samples:
        log(f"slicing cached real features {real_feats.shape[0]} -> {args.num_samples}")
        real_feats = real_feats[:args.num_samples]

    del i3d
    torch.cuda.empty_cache()

    log("computing Frechet distance ...")
    t0 = time.time()
    try:
        fvd = frechet_distance(gen_feats.numpy(), real_feats.numpy())
        dt = time.time() - t0
        log(f"FVD = {fvd:.3f}  ({dt:.1f}s)")
    except Exception as e:
        log(f"ERROR computing FVD: {e}")
        if allow_missing:
            args.out_json.write_text(json.dumps({"skipped": True, "reason": str(e)}, indent=2))
            return
        raise

    result = {
        "fvd": fvd,
        "n_gen": int(gen_feats.shape[0]),
        "n_real": int(real_feats.shape[0]),
        "gen_dir": str(args.gen_dir),
        "real_dir": str(args.real_dir),
        "elapsed_sec": dt,
        "feat_dim": int(gen_feats.shape[1]),
        "feat_type": feat_type,
        "backbone": "i3d_r50_pytorchvideo_kinetics400",
    }
    args.out_json.write_text(json.dumps(result, indent=2))
    log(f"saved {args.out_json}")


if __name__ == "__main__":
    main()
