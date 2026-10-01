import argparse
import json
import math
import os
import os.path as osp
import random
import time
from typing import List

import numpy as np
import torch
import torchvision
import PIL.Image as PImage
from torchvision.transforms import InterpolationMode, transforms

from pytorch_fid.inception import InceptionV3
from pytorch_fid.fid_score import calculate_frechet_distance

from var_leak_exp import (
    build_model, sample_curated, build_val_transform,
    PATCH_NUMS, NUM_SCALES, _class_folders,
)

IMG_EXTS = ('.jpeg', '.jpg', '.png')


def _list_images(cdir):
    fs = []
    for fn in os.listdir(cdir):
        if fn.lower().endswith(IMG_EXTS) and osp.isfile(osp.join(cdir, fn)):
            fs.append(fn)
    return sorted(fs)


def build_ref_transform(final_reso=256, mid_reso=1.125):
    mid = round(mid_reso * final_reso)
    return transforms.Compose([
        transforms.Resize(mid, interpolation=InterpolationMode.LANCZOS),
        transforms.CenterCrop((final_reso, final_reso)),
        transforms.ToTensor(),
    ])


def load_class_images(root, folders, class_ids, num_per_class, transform, seed=0,
                      exclude=None):
    imgs = []
    used = {}
    for cid in class_ids:
        cdir = osp.join(root, folders[cid])
        files = _list_images(cdir)
        random.Random(seed * 100003 + cid).shuffle(files)
        ex = (exclude or {}).get(cid, set())
        picked = []
        for fn in files:
            if len(picked) >= num_per_class:
                break
            if fn in ex:
                continue
            try:
                im = PImage.open(osp.join(cdir, fn)).convert('RGB')
            except Exception:
                continue
            imgs.append(transform(im))
            picked.append(fn)
        if len(picked) < num_per_class:
            raise RuntimeError(f'class {cid} ({folders[cid]}): only {len(picked)}/{num_per_class} readable in {cdir}')
        used[cid] = picked
    return torch.stack(imgs), used


class InceptionActivations:
    def __init__(self, device, dims=2048, batch_size=50):
        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
        self.model = InceptionV3([block_idx]).to(device).eval()
        self.device = device
        self.dims = dims
        self.batch_size = batch_size

    @torch.no_grad()
    def features(self, img_N3HW_01):
        out = []
        N = img_N3HW_01.shape[0]
        for i in range(0, N, self.batch_size):
            b = img_N3HW_01[i:i + self.batch_size].to(self.device, dtype=torch.float32)
            b = b.clamp(0, 1)
            f = self.model(b)[0]
            out.append(f.squeeze(-1).squeeze(-1).cpu())
        return torch.cat(out, 0).numpy().astype(np.float64)


def mu_sigma(feats):
    mu = feats.mean(0)
    sigma = np.cov(feats, rowvar=False)
    return mu, sigma


def save_grid(img_N3HW_01, path, nrow=8, n=64):
    os.makedirs(osp.dirname(path), exist_ok=True)
    x = img_N3HW_01[:n].clamp(0, 1).float().cpu()
    torchvision.utils.save_image(x, path, nrow=nrow, padding=2)


def build_reference(args, incept, device):
    cache = osp.join(args.out_dir, 'ref_stats.npz')
    if osp.exists(cache) and not args.force_ref:
        z = np.load(cache)
        print(f'[ref] loaded cached stats from {cache} (n={int(z["n"])})')
        return z['mu'], z['sigma']
    folders = _class_folders(args.ref_img_dir)
    ref_transform = build_ref_transform()
    all_feats = []
    grid_saved = False
    t0 = time.time()
    n_classes = args.num_classes
    for start in range(0, n_classes, args.classes_per_batch):
        cids = list(range(start, min(start + args.classes_per_batch, n_classes)))
        imgs, _ = load_class_images(args.ref_img_dir, folders, cids,
                                    args.imgs_per_class, ref_transform, seed=args.ref_seed)
        feats = incept.features(imgs)
        all_feats.append(feats)
        if not grid_saved:
            save_grid(imgs, osp.join(args.out_dir, 'grids', 'reference.png'))
            grid_saved = True
        if start % (args.classes_per_batch * 20) == 0:
            done = min(start + args.classes_per_batch, n_classes)
            print(f'[ref] {done}/{n_classes} classes  ({time.time()-t0:.0f}s)')
    feats = np.concatenate(all_feats, 0)
    mu, sigma = mu_sigma(feats)
    np.savez(cache, mu=mu, sigma=sigma, n=feats.shape[0])
    print(f'[ref] built {feats.shape[0]} real imgs, saved {cache}  ({time.time()-t0:.0f}s)')
    return mu, sigma


def make_batches(args):
    batches = []
    for start in range(0, args.num_classes, args.classes_per_batch):
        cids = list(range(start, min(start + args.classes_per_batch, args.num_classes)))
        batches.append(cids)
    return batches


def precompute_gt(args, vae, batches, folders, device):
    leak_transform = build_val_transform()
    gt_cache = []
    t0 = time.time()
    for bi, cids in enumerate(batches):
        real_imgs, _ = load_class_images(
            args.real_img_dir, folders, cids, args.imgs_per_class,
            leak_transform, seed=args.leak_seed)
        real_imgs = real_imgs.to(device)
        with torch.inference_mode():
            gt = vae.img_to_idxBl(real_imgs)
        gt_cache.append([t.cpu() for t in gt])
        if bi % 20 == 0 or bi == len(batches) - 1:
            print(f'[leak-precompute] batch {bi+1}/{len(batches)}  ({time.time()-t0:.0f}s)', flush=True)
    print(f'[leak-precompute] done {len(batches)} batches  ({time.time()-t0:.0f}s)', flush=True)
    return gt_cache


def run_condition(args, vae, var, incept, device, mode, keep_ratio,
                  ref_mu, ref_sigma, batches, gt_cache):
    tag = f'{mode}_keep{keep_ratio:g}'
    leak_until_si = args.prune_from_si if mode == 'real' else 0

    all_feats = []
    grid_saved = False
    n_classes = args.num_classes
    t0 = time.time()
    n_batches = len(batches)
    for bi, cids in enumerate(batches):
        B = len(cids) * args.imgs_per_class
        label_B = torch.tensor(
            [c for c in cids for _ in range(args.imgs_per_class)],
            device=device, dtype=torch.long)

        gt_idxBl = None
        if mode == 'real':
            gt_idxBl = [t.to(device) for t in gt_cache[bi]]

        g_seed = args.seed + bi
        with torch.inference_mode(), torch.autocast('cuda', enabled=True, dtype=torch.float16):
            img = sample_curated(
                var, vae, B=B, label_B=label_B,
                cfg=args.cfg, top_k=args.top_k, top_p=args.top_p, g_seed=g_seed,
                gt_idxBl=gt_idxBl, leak_until_si=leak_until_si,
                prune_from_si=args.prune_from_si, keep_ratio=keep_ratio,
            )
        img = img.float()
        feats = incept.features(img)
        all_feats.append(feats)
        if not grid_saved:
            save_grid(img, osp.join(args.out_dir, 'grids', f'{tag}.png'))
            grid_saved = True
        if bi % 20 == 0 or bi == n_batches - 1:
            done = min((bi + 1) * args.classes_per_batch, n_classes)
            rate = done / max(time.time() - t0, 1e-6)
            eta = (n_classes - done) / max(rate, 1e-6)
            print(f'[{tag}] {done}/{n_classes} classes  ({time.time()-t0:.0f}s, ETA {eta:.0f}s)',
                  flush=True)

    feats = np.concatenate(all_feats, 0)
    mu, sigma = mu_sigma(feats)
    fid = calculate_frechet_distance(mu, sigma, ref_mu, ref_sigma)
    np.savez(osp.join(args.out_dir, f'stats_{tag}.npz'), mu=mu, sigma=sigma, n=feats.shape[0])
    print(f'[{tag}] DONE n={feats.shape[0]} FID={fid:.4f}  ({time.time()-t0:.0f}s)', flush=True)
    return fid, feats.shape[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out_dir', default='runs/leak_full')
    ap.add_argument('--real_img_dir', default='data/imagenet/train')
    ap.add_argument('--ref_img_dir', default='data/imagenet/val')
    ap.add_argument('--num_classes', type=int, default=1000)
    ap.add_argument('--imgs_per_class', type=int, default=10)
    ap.add_argument('--classes_per_batch', type=int, default=10)
    ap.add_argument('--prune_from_si', type=int, default=6)
    ap.add_argument('--keep_ratios', type=float, nargs='+', default=[1.0, 0.5, 0.25])
    ap.add_argument('--modes', nargs='+', default=['self', 'real'])
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--leak_seed', type=int, default=0)
    ap.add_argument('--ref_seed', type=int, default=7)
    ap.add_argument('--cfg', type=float, default=4.0)
    ap.add_argument('--top_k', type=int, default=900)
    ap.add_argument('--top_p', type=float, default=0.95)
    ap.add_argument('--depth', type=int, default=16)
    ap.add_argument('--vae_ckpt', default='ckpt/vae_ch160v4096z32.pth')
    ap.add_argument('--var_ckpt', default='ckpt/var_d16.pth')
    ap.add_argument('--fid_batch', type=int, default=100)
    ap.add_argument('--force_ref', action='store_true')
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    torch.manual_seed(args.seed); random.seed(args.seed); np.random.seed(args.seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision('high')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f'device={device}; building VAR-d{args.depth} + VQVAE ...', flush=True)
    vae, var = build_model(device, args.vae_ckpt, args.var_ckpt, depth=args.depth)
    incept = InceptionActivations(device, dims=2048, batch_size=args.fid_batch)
    print('models ready.', flush=True)

    ref_mu, ref_sigma = build_reference(args, incept, device)

    batches = make_batches(args)
    gt_cache = None
    if 'real' in args.modes:
        leak_source_folders = _class_folders(args.real_img_dir)
        gt_cache = precompute_gt(args, vae, batches, leak_source_folders, device)

    results = {}
    order = [(m, kr) for m in args.modes for kr in args.keep_ratios]
    for mode, kr in order:
        fid, n = run_condition(args, vae, var, incept, device, mode, kr,
                               ref_mu, ref_sigma, batches, gt_cache)
        results[f'{mode}_keep{kr:g}'] = dict(mode=mode, keep_ratio=kr, fid=fid, n=n)
        with open(osp.join(args.out_dir, 'results.json'), 'w') as f:
            json.dump(results, f, indent=2)

    print('\n' + '=' * 72)
    print('VAR real-coarse-leak x curation  --  FID table (prune_from_si=%d)' % args.prune_from_si)
    print('=' * 72)
    krs = args.keep_ratios
    header = 'keep_ratio | ' + ' | '.join(f'{m:>10s}' for m in args.modes)
    print(header)
    print('-' * len(header))
    for kr in krs:
        row = [f'{results[f"{m}_keep{kr:g}"]["fid"]:10.4f}' for m in args.modes]
        print(f'{kr:>10g} | ' + ' | '.join(row))

    def fid(m, kr):
        return results[f'{m}_keep{kr:g}']['fid']

    dense = max(krs)
    pruned = [kr for kr in krs if kr < dense]
    print('\n--- KEY QUANTITIES ---')
    if 'self' in args.modes:
        print(f'Curation penalty  Delta(self) = FID(keep={min(krs):g},self) - FID(dense,self) '
              f'= {fid("self",min(krs)):.4f} - {fid("self",dense):.4f} = {fid("self",min(krs))-fid("self",dense):+.4f}')
    if 'real' in args.modes:
        print(f'Curation penalty  Delta(real) = FID(keep={min(krs):g},real) - FID(dense,real) '
              f'= {fid("real",min(krs)):.4f} - {fid("real",dense):.4f} = {fid("real",min(krs))-fid("real",dense):+.4f}')

    inflations = {}
    if 'self' in args.modes and 'real' in args.modes:
        print('\nInflation = FID(keep,self) - FID(keep,real)   (>0 => real coarse looks better)')
        for kr in krs:
            inf = fid('self', kr) - fid('real', kr)
            inflations[kr] = inf
            print(f'  keep={kr:>4g}: {fid("self",kr):.4f} - {fid("real",kr):.4f} = {inf:+.4f}')

        print('\n--- checks ---')
        cond_A = all(inflations[kr] > 0 for kr in pruned)
        widen = all(inflations[a] > inflations[b]
                    for a, b in zip(sorted(pruned), sorted(pruned)[1:])) if len(pruned) >= 2 else None
        widen_pair = (inflations[min(pruned)] > inflations[max(pruned)]) if len(pruned) >= 2 else None
        print(f'  (A) real-coarse FID < self-coarse FID at pruned budgets {pruned}: {cond_A}')
        if widen_pair is not None:
            print(f'  (B) inflation grows as keep drops '
                  f'(keep={min(pruned):g}:{inflations[min(pruned)]:+.4f} > keep={max(pruned):g}:{inflations[max(pruned)]:+.4f}): {widen_pair}')
        confirmed = bool(cond_A) and (widen_pair is None or bool(widen_pair))
        print(f'\n  all checks hold: {confirmed}')
        results['_summary'] = dict(inflations={f'{k:g}': v for k, v in inflations.items()},
                                   cond_A_real_better_at_pruned=bool(cond_A),
                                   cond_B_inflation_widens=(None if widen_pair is None else bool(widen_pair)),
                                   confirmed=confirmed)
        with open(osp.join(args.out_dir, 'results.json'), 'w') as f:
            json.dump(results, f, indent=2)
    print('=' * 72)


if __name__ == '__main__':
    main()
