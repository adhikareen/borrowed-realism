import argparse
import os
import time

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import datasets
import utils
from models.adaptok import AdapTok


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scorer", default="ckpts/scorer")
    ap.add_argument("--csv_file", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--reuse", default=None,
                    help="existing AR annotation to match")
    ap.add_argument("--token_select_num", type=int, default=212)
    ap.add_argument("--num_cond_frames", type=int, default=5)
    ap.add_argument("--num_frames", type=int, default=16)
    ap.add_argument("--input_size", type=int, default=128)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--use_all_frames_step", type=int, default=16)
    ap.add_argument("--frame_rate", type=str, default="native")
    ap.add_argument("--amp_dtype", type=str, default="float16")
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
    btn = model.bottleneck_token_num
    print(f"[cfg] mode={model.mode} ilp token_select_num={model.token_select_num} "
          f"bottleneck_token_num={btn} num_cond_frames={args.num_cond_frames}", flush=True)

    reuse = args.reuse is not None
    if reuse:
        base = torch.load(args.reuse, map_location="cpu")
        ref_ids = list(base["id"])
        N = len(ref_ids)
        assert len(set(ref_ids)) == N, "reuse ids are not unique -> id-match unsafe"
        id2row = {k: i for i, k in enumerate(ref_ids)}
        conditions = torch.full((N, btn), -1, dtype=torch.int16)
        filled = torch.zeros(N, dtype=torch.bool)
        print(f"[reuse] {args.reuse}: N={N} clips; will fill conditions by id-match", flush=True)

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
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, drop_last=False,
                        num_workers=args.num_workers, pin_memory=True)

    fresh_ids, fresh_latent_nums, fresh_bottleneck, fresh_conditions = [], [], [], []
    n_seen = 0
    t0 = time.time()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    for i, inputs in enumerate(tqdm(loader, desc="fp-annot")):
        data = inputs["gt"].to(device)
        x_cond = utils.repeat_to_m_frames(data[:, :, :args.num_cond_frames], m=data.shape[2])
        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=True):
            cond_out = model.encode(x_cond, with_latent_mask=False)
            if not reuse:
                enc_out = model.encode_eval(data)
        cond_rep = cond_out["bottleneck_rep"].to(torch.int16).cpu()

        ids = [f"{s}_{e}_frames__{os.path.basename(p)}"
               for p, s, e in zip(inputs["path"], inputs["frame_start"], inputs["frame_end"])]
        n_seen += len(ids)

        if reuse:
            for k, cid in enumerate(ids):
                row = id2row.get(cid, None)
                if row is None:
                    continue
                conditions[row] = cond_rep[k]
                filled[row] = True
        else:
            fresh_ids.extend(ids)
            fresh_latent_nums.extend(enc_out["latent_num"].cpu().tolist())
            fresh_bottleneck.extend(enc_out["bottleneck_rep"].cpu().tolist())
            fresh_conditions.extend(cond_rep.tolist())

    if reuse:
        n_missing = int((~filled).sum())
        assert n_missing == 0, f"{n_missing} clips never filled (dataloader did not cover all reuse ids)"
        save_dict = {
            "id": ref_ids,
            "latent_nums": base["latent_nums"],
            "bottleneck_rep": base["bottleneck_rep"],
            "conditions": conditions,
        }
        assert torch.equal(save_dict["bottleneck_rep"], base["bottleneck_rep"])
        assert torch.equal(save_dict["latent_nums"], base["latent_nums"])
        print(f"[reuse] all {N} conditions filled; targets byte-identical to {args.reuse}", flush=True)
    else:
        save_dict = {
            "id": fresh_ids,
            "latent_nums": torch.tensor(fresh_latent_nums, dtype=torch.int16),
            "bottleneck_rep": torch.tensor(fresh_bottleneck, dtype=torch.int16),
            "conditions": torch.tensor(fresh_conditions, dtype=torch.int16),
        }

    torch.save(save_dict, args.out)
    ln = save_dict["latent_nums"].float()
    print(f"[done] wrote {args.out} | {save_dict['latent_nums'].shape[0]} clips | seen={n_seen} | "
          f"mean per-block={ln.mean(0).tolist()} sum={ln.mean(0).sum().item():.1f} | "
          f"conditions shape={tuple(save_dict['conditions'].shape)} | {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
