import argparse
import os
import time
from collections import defaultdict

import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

import datasets
from models.adaptok import AdapTok


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scorer", default="ckpts/scorer", help="HF dir or id for scorer")
    ap.add_argument("--csv_file", default="k400_train.csv")
    ap.add_argument("--out", required=True, help="output .pt path")
    ap.add_argument("--token_select_num", type=int, default=212)
    ap.add_argument("--num_frames", type=int, default=16)
    ap.add_argument("--input_size", type=int, default=128)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--max_clips", type=int, default=None,
                    help="max clips")
    ap.add_argument("--use_all_frames_step", type=int, default=16)
    ap.add_argument("--frame_rate", type=str, default="native")
    ap.add_argument("--amp_dtype", type=str, default="float16")
    ap.add_argument("--save_iter", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str, default="cuda:0")
    return ap.parse_args()


@torch.inference_mode()
def main():
    args = parse_args()
    device = torch.device(args.device)
    amp_dtype = getattr(torch, args.amp_dtype)

    print(f"[load] scorer from {args.scorer}", flush=True)
    if os.path.isfile(args.scorer):
        model = AdapTok.from_checkpoint(args.scorer)
    else:
        model = AdapTok.from_pretrained(args.scorer)
    model = model.to(device).eval()
    if hasattr(model, "set_vq_eval_deterministic"):
        model.set_vq_eval_deterministic(True)
    model.mode = "get_ar_annotations"
    model.token_select_mode = "ilp"
    model.token_select_num = args.token_select_num
    tot = model.mask_generator.total_toks * model.mask_generator.tot_groups
    print(f"[cfg] mode={model.mode} token_select_mode={model.token_select_mode} "
          f"token_select_num={model.token_select_num} total_toks={model.mask_generator.total_toks} "
          f"tot_groups={model.mask_generator.tot_groups} (table key ilp_t{tot}_b{model.mask_generator.tot_groups})",
          flush=True)

    dataset_cfg = {
        "name": "video_dataset",
        "args": {
            "root_path": "data/metadata",
            "split": "train",
            "frame_num": args.num_frames,
            "cls_vid_num": "-1_-1",
            "crop_size": args.input_size,
            "csv_file": args.csv_file,
            "frame_rate": args.frame_rate,
            "use_all_frames": True,
            "use_all_frames_step": args.use_all_frames_step,
            "pre_load": False,
            "with_scores": False,
        },
    }
    ds = datasets.make(dataset_cfg)
    print(f"[data] {args.csv_file}: {len(ds)} (video,window) clips", flush=True)
    if args.max_clips is not None and args.max_clips < len(ds):
        idx = torch.randperm(len(ds), generator=torch.Generator().manual_seed(args.seed))[:args.max_clips].tolist()
        ds = Subset(ds, idx)
        print(f"[data] capped to {len(ds)} clips (max_clips={args.max_clips})", flush=True)

    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, drop_last=False,
                        num_workers=args.num_workers, pin_memory=True)

    all_ids, all_latent_nums, all_bottleneck_rep = [], [], []
    budget_sums = []
    t0 = time.time()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    for i, inputs in enumerate(tqdm(loader, desc="annot")):
        data = inputs["gt"].to(device)
        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=True):
            out = model.encode_eval(data)
        ln = out["latent_num"].cpu()
        all_latent_nums.extend(ln.tolist())
        all_bottleneck_rep.extend(out["bottleneck_rep"].cpu().tolist())
        all_ids.extend([f"{s}_{e}_frames__{os.path.basename(p)}"
                        for p, s, e in zip(inputs["path"], inputs["frame_start"], inputs["frame_end"])])
        budget_sums.extend(ln.sum(dim=-1).tolist())
        if (i + 1) % args.save_iter == 0:
            mean_b = sum(budget_sums) / max(1, len(budget_sums))
            print(f"[annot] {len(all_ids)} clips | mean tok/clip={mean_b:.1f} | "
                  f"{len(all_ids)/max(time.time()-t0,1e-6):.1f} clips/s", flush=True)

    save_dict = {
        "id": all_ids,
        "latent_nums": torch.tensor(all_latent_nums, dtype=torch.int16),
        "bottleneck_rep": torch.tensor(all_bottleneck_rep, dtype=torch.int16),
    }
    torch.save(save_dict, args.out)
    mean_b = sum(budget_sums) / max(1, len(budget_sums))
    import numpy as np
    bs = np.array(budget_sums)
    print(f"[done] wrote {args.out} | {len(all_ids)} clips | "
          f"mean tok/clip={mean_b:.1f} +/- {bs.std():.1f} (min {bs.min()}, max {bs.max()}) | "
          f"{time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
