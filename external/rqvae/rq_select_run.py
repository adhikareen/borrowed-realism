import argparse
import json
import math
import os
import os.path as osp
import random
import time

import numpy as np
import torch
import torchvision
import PIL.Image as PImage
from torchvision.transforms import transforms

from pytorch_fid.inception import InceptionV3
from pytorch_fid.fid_score import calculate_frechet_distance

from rqvae.utils.utils import set_seed
from rq_leak_exp import (
    build_models, build_val_transform, _class_folders,
    load_real_images, sample_codes, curate_and_decode, save_grid,
)

IMG_EXTS = ('.jpeg', '.jpg', '.png')


def _list_images(cdir):
    return sorted(fn for fn in os.listdir(cdir)
                  if fn.lower().endswith(IMG_EXTS) and osp.isfile(osp.join(cdir, fn)))


def build_ref_transform():
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])


def load_class_images(root, folders, class_ids, num_per_class, transform, seed=0):
    imgs = []
    for cid in class_ids:
        cdir = osp.join(root, folders[cid])
        files = _list_images(cdir)
        random.Random(seed * 100003 + cid).shuffle(files)
        picked = 0
        for fn in files:
            if picked >= num_per_class:
                break
            try:
                im = PImage.open(osp.join(cdir, fn)).convert('RGB')
            except Exception:
                continue
            imgs.append(transform(im))
            picked += 1
        if picked < num_per_class:
            raise RuntimeError(f'class {cid} ({folders[cid]}): only {picked}/{num_per_class} in {cdir}')
    return torch.stack(imgs)


class InceptionActivations:
    def __init__(self, device, dims=2048, batch_size=50):
        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
        self.model = InceptionV3([block_idx]).to(device).eval()
        self.device = device
        self.batch_size = batch_size

    @torch.no_grad()
    def features(self, img_N3HW_01):
        out = []
        N = img_N3HW_01.shape[0]
        for i in range(0, N, self.batch_size):
            b = img_N3HW_01[i:i + self.batch_size].to(self.device, dtype=torch.float32).clamp(0, 1)
            f = self.model(b)[0]
            out.append(f.squeeze(-1).squeeze(-1).cpu())
        return torch.cat(out, 0).numpy().astype(np.float64)


def mu_sigma(feats):
    return feats.mean(0), np.cov(feats, rowvar=False)


def build_reference(args, incept, device):
    cache = osp.join(args.out_dir, 'ref_stats.npz')
    if osp.exists(cache) and not args.force_ref:
        z = np.load(cache)
        print(f'[ref] loaded cached {cache} (n={int(z["n"])})', flush=True)
        return z['mu'], z['sigma']
    folders = _class_folders(args.ref_img_dir)
    ref_tr = build_ref_transform()
    feats_all = []
    t0 = time.time()
    grid_saved = False
    for start in range(0, args.num_classes, args.classes_per_batch):
        cids = list(range(start, min(start + args.classes_per_batch, args.num_classes)))
        imgs = load_class_images(args.ref_img_dir, folders, cids, args.imgs_per_class, ref_tr, seed=args.ref_seed)
        feats_all.append(incept.features(imgs))
        if not grid_saved:
            save_grid(imgs, osp.join(args.out_dir, 'grids', 'reference.png')); grid_saved = True
        if start % (args.classes_per_batch * 20) == 0:
            print(f'[ref] {min(start+args.classes_per_batch,args.num_classes)}/{args.num_classes} ({time.time()-t0:.0f}s)', flush=True)
    feats = np.concatenate(feats_all, 0)
    mu, sigma = mu_sigma(feats)
    np.savez(cache, mu=mu, sigma=sigma, n=feats.shape[0])
    print(f'[ref] built {feats.shape[0]} imgs -> {cache} ({time.time()-t0:.0f}s)', flush=True)
    return mu, sigma


def make_batches(args):
    return [list(range(s, min(s + args.classes_per_batch, args.num_classes)))
            for s in range(0, args.num_classes, args.classes_per_batch)]


def generate_codes_for_coarse(args, model_ar, model_vqvae, device, coarse, batches, gt_folders):
    codes = []
    leak_until_d = args.prune_from_d if coarse == 'real' else 0
    ref_tr = build_val_transform()
    t0 = time.time()
    for bi, cids in enumerate(batches):
        B = len(cids) * args.imgs_per_class
        cond = torch.tensor([c for c in cids for _ in range(args.imgs_per_class)],
                            device=device, dtype=torch.long)
        gt_code = None
        if coarse == 'real':
            real_imgs = load_class_images(args.real_img_dir, gt_folders, cids,
                                          args.imgs_per_class, ref_tr, seed=args.leak_seed).to(device)
            gt_code = model_vqvae.get_codes(real_imgs)
        set_seed(args.seed + bi)
        with torch.autocast('cuda', enabled=True, dtype=torch.float16):
            code = sample_codes(model_ar, model_vqvae, B=B, cond=cond,
                                temperature=args.temp, top_k=args.top_k, top_p=args.top_p, amp=True,
                                gt_code=gt_code, leak_until_d=leak_until_d)
        codes.append(code.cpu())
        if bi % 20 == 0 or bi == len(batches) - 1:
            done = min((bi + 1) * args.classes_per_batch, args.num_classes)
            rate = done / max(time.time() - t0, 1e-6)
            print(f'[gen {coarse}] {done}/{args.num_classes} classes ({time.time()-t0:.0f}s, '
                  f'ETA {(args.num_classes-done)/max(rate,1e-6):.0f}s)', flush=True)
    return codes


def fid_for_condition(args, model_vqvae, incept, device, codes, select, coarse,
                      ref_mu, ref_sigma):
    tag = f'{select}_{coarse}_keep{args.keep_ratio:g}'
    feats_all = []
    kept_fracs, overlaps = [], []
    grid_saved = False
    t0 = time.time()
    for bi, code in enumerate(codes):
        want_dbg = bi < args.diag_batches
        out = curate_and_decode(model_vqvae, code.to(device),
                                prune_from_d=args.prune_from_d, keep_ratio=args.keep_ratio,
                                select=select, select_seed=args.seed + bi, return_debug=want_dbg)
        img = out[0] if want_dbg else out
        if want_dbg:
            kept_fracs.append(out[1]['kept_frac'])
            if out[1]['overlap_with_smart'] is not None:
                overlaps.append(out[1]['overlap_with_smart'])
        feats_all.append(incept.features(img.float()))
        if not grid_saved:
            save_grid(img, osp.join(args.out_dir, 'grids', f'{tag}.png')); grid_saved = True
    feats = np.concatenate(feats_all, 0)
    mu, sigma = mu_sigma(feats)
    fid = calculate_frechet_distance(mu, sigma, ref_mu, ref_sigma)
    np.savez(osp.join(args.out_dir, f'stats_{tag}.npz'), mu=mu, sigma=sigma, n=feats.shape[0])
    diag = dict(kept_frac_mean=float(np.mean(kept_fracs)) if kept_fracs else None,
                overlap_with_smart_mean=float(np.mean(overlaps)) if overlaps else None)
    print(f'[{tag}] DONE n={feats.shape[0]} FID={fid:.4f} kept_frac={diag["kept_frac_mean"]} '
          f'overlap={diag["overlap_with_smart_mean"]} ({time.time()-t0:.0f}s)', flush=True)
    return fid, feats.shape[0], diag


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model-ar-path', required=True)
    ap.add_argument('--model-vqvae-path', required=True)
    ap.add_argument('--ema', action='store_true')
    ap.add_argument('--out_dir', default='runs/rq_leak_select')
    ap.add_argument('--real_img_dir', default='data/imagenet/train')
    ap.add_argument('--ref_img_dir', default='data/imagenet/val')
    ap.add_argument('--num_classes', type=int, default=1000)
    ap.add_argument('--imgs_per_class', type=int, default=10)
    ap.add_argument('--classes_per_batch', type=int, default=5)
    ap.add_argument('--prune_from_d', type=int, default=2)
    ap.add_argument('--keep_ratio', type=float, default=0.25)
    ap.add_argument('--selects', nargs='+', default=['smart', 'random'])
    ap.add_argument('--coarses', nargs='+', default=['self', 'real'])
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--leak_seed', type=int, default=0)
    ap.add_argument('--ref_seed', type=int, default=7)
    ap.add_argument('--temp', type=float, default=1.0)
    ap.add_argument('--top_k', type=int, default=16384)
    ap.add_argument('--top_p', type=float, default=0.92)
    ap.add_argument('--fid_batch', type=int, default=100)
    ap.add_argument('--diag_batches', type=int, default=3)
    ap.add_argument('--force_ref', action='store_true')
    args = ap.parse_args()

    os.makedirs(osp.join(args.out_dir, 'grids'), exist_ok=True)
    torch.backends.cudnn.benchmark = True
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'device={device}; building RQ-Transformer + RQ-VAE ...', flush=True)
    model_ar, model_vqvae, _, _ = build_models(args.model_ar_path, args.model_vqvae_path, device, ema=args.ema)
    incept = InceptionActivations(device, dims=2048, batch_size=args.fid_batch)
    (H, W, D) = model_ar.block_size
    assert args.prune_from_d < D, f'prune_from_d={args.prune_from_d} must be < D={D}'
    print(f'models ready. block_size=({H},{W},{D})', flush=True)

    ref_mu, ref_sigma = build_reference(args, incept, device)
    batches = make_batches(args)
    gt_folders = _class_folders(args.real_img_dir)

    results = {}
    for coarse in args.coarses:
        codes = generate_codes_for_coarse(args, model_ar, model_vqvae, device, coarse, batches, gt_folders)
        for select in args.selects:
            fid, n, diag = fid_for_condition(args, model_vqvae, incept, device, codes,
                                             select, coarse, ref_mu, ref_sigma)
            results[f'{select}_{coarse}_keep{args.keep_ratio:g}'] = dict(
                select=select, coarse=coarse, keep_ratio=args.keep_ratio, fid=fid, n=n, **diag)
            with open(osp.join(args.out_dir, 'results.json'), 'w') as f:
                json.dump(results, f, indent=2)
        del codes

    def fid(s, c):
        return results[f'{s}_{c}_keep{args.keep_ratio:g}']['fid']

    print('\n' + '=' * 78)
    print(f'RQ selector-ranking inflation -- FID (keep={args.keep_ratio:g}, prune_from_d={args.prune_from_d})')
    print('=' * 78)
    kr = args.keep_ratio
    if 'smart' in args.selects and 'random' in args.selects:
        summary = {}
        for c in args.coarses:
            adv = fid('random', c) - fid('smart', c)
            summary[c] = adv
            print(f'  coarse={c:>4s}: smart={fid("smart",c):8.4f}  random={fid("random",c):8.4f}  '
                  f'smart_adv(rand-smart)={adv:+.4f}')
        if 'self' in args.coarses and 'real' in args.coarses:
            infl = summary['real'] - summary['self']
            confirmed = bool(infl > 0)
            print('\n--- smart-edge inflation ---')
            print(f'  smart_advantage(self) = {summary["self"]:+.4f}')
            print(f'  smart_advantage(real) = {summary["real"]:+.4f}')
            print(f'  ranking_inflation     = {infl:+.4f}')
            print(f'  inflation > 0: {confirmed}')
            results['_summary'] = dict(keep_ratio=kr, prune_from_d=args.prune_from_d,
                                       smart_adv_self=summary['self'], smart_adv_real=summary['real'],
                                       ranking_inflation=infl, confirmed=confirmed)
            with open(osp.join(args.out_dir, 'results.json'), 'w') as f:
                json.dump(results, f, indent=2)
    print('=' * 78)
    print(f'results -> {osp.join(args.out_dir, "results.json")}')


if __name__ == '__main__':
    main()
