import argparse
import json
import os
import os.path as osp
import random
import time

import numpy as np
import torch
import torchvision

from var_leak_exp import (
    build_model, sample_curated, build_val_transform, _class_folders,
    PATCH_NUMS, NUM_SCALES,
)
from var_leak_run import (
    InceptionActivations, mu_sigma, save_grid, load_class_images,
    make_batches, precompute_gt,
)
from pytorch_fid.fid_score import calculate_frechet_distance


def load_reference(ref_stats_path):
    z = np.load(ref_stats_path)
    print(f'[ref] loaded SHARED reference stats {ref_stats_path} (n={int(z["n"])})', flush=True)
    return z['mu'], z['sigma'], int(z['n'])


def run_condition(args, vae, var, incept, device, select, coarse, keep_ratio,
                  ref_mu, ref_sigma, batches, gt_cache):
    tag = f'{select}_{coarse}_keep{keep_ratio:g}'
    leak_until_si = args.prune_from_si if coarse == 'real' else 0

    all_feats = []
    grid_saved = False
    kept_fracs, overlaps = [], []
    n_classes = args.num_classes
    n_batches = len(batches)
    t0 = time.time()
    for bi, cids in enumerate(batches):
        B = len(cids) * args.imgs_per_class
        label_B = torch.tensor(
            [c for c in cids for _ in range(args.imgs_per_class)],
            device=device, dtype=torch.long)

        gt_idxBl = None
        if coarse == 'real':
            gt_idxBl = [t.to(device) for t in gt_cache[bi]]

        g_seed = args.seed + bi
        want_dbg = (bi < args.diag_batches)
        with torch.inference_mode(), torch.autocast('cuda', enabled=True, dtype=torch.float16):
            out = sample_curated(
                var, vae, B=B, label_B=label_B,
                cfg=args.cfg, top_k=args.top_k, top_p=args.top_p, g_seed=g_seed,
                gt_idxBl=gt_idxBl, leak_until_si=leak_until_si,
                prune_from_si=args.prune_from_si, keep_ratio=keep_ratio,
                select=select, select_seed=g_seed,
                return_debug=want_dbg,
            )
        img = (out[0] if want_dbg else out).float()
        if want_dbg:
            for d in out[1]:
                if d['pruned']:
                    if d['kept_frac'] is not None:
                        kept_fracs.append(d['kept_frac'])
                    if d['overlap_with_smart'] is not None:
                        overlaps.append(d['overlap_with_smart'])
        feats = incept.features(img)
        all_feats.append(feats)
        if not grid_saved:
            save_grid(img, osp.join(args.out_dir, 'grids', f'{tag}.png'))
            grid_saved = True
        if bi % 25 == 0 or bi == n_batches - 1:
            done = min((bi + 1) * args.classes_per_batch, n_classes)
            rate = done / max(time.time() - t0, 1e-6)
            eta = (n_classes - done) / max(rate, 1e-6)
            print(f'[{tag}] {done}/{n_classes} classes  ({time.time()-t0:.0f}s, ETA {eta:.0f}s)', flush=True)

    feats = np.concatenate(all_feats, 0)
    mu, sigma = mu_sigma(feats)
    fid = calculate_frechet_distance(mu, sigma, ref_mu, ref_sigma)
    np.savez(osp.join(args.out_dir, f'stats_{tag}.npz'), mu=mu, sigma=sigma, n=feats.shape[0])

    diag = dict(
        kept_frac_mean=float(np.mean(kept_fracs)) if kept_fracs else None,
        overlap_with_smart_mean=(float(np.mean(overlaps)) if overlaps else None),
    )
    print(f'[{tag}] DONE n={feats.shape[0]} FID={fid:.4f} '
          f'kept_frac={diag["kept_frac_mean"]} '
          f'overlap_vs_smart={diag["overlap_with_smart_mean"]}  ({time.time()-t0:.0f}s)', flush=True)
    return fid, feats.shape[0], diag


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out_dir', default='runs/leak_select')
    ap.add_argument('--ref_stats', default='runs/leak_full/ref_stats.npz',
                    help='reference stats file')
    ap.add_argument('--real_img_dir', default='data/imagenet/train')
    ap.add_argument('--num_classes', type=int, default=1000)
    ap.add_argument('--imgs_per_class', type=int, default=10)
    ap.add_argument('--classes_per_batch', type=int, default=10)
    ap.add_argument('--prune_from_si', type=int, default=6)
    ap.add_argument('--keep_ratios', type=float, nargs='+', default=[0.25])
    ap.add_argument('--selects', nargs='+', default=['smart', 'random'])
    ap.add_argument('--coarses', nargs='+', default=['self', 'real'])
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--leak_seed', type=int, default=0)
    ap.add_argument('--cfg', type=float, default=4.0)
    ap.add_argument('--top_k', type=int, default=900)
    ap.add_argument('--top_p', type=float, default=0.95)
    ap.add_argument('--depth', type=int, default=16)
    ap.add_argument('--vae_ckpt', default='ckpt/vae_ch160v4096z32.pth')
    ap.add_argument('--var_ckpt', default='ckpt/var_d16.pth')
    ap.add_argument('--fid_batch', type=int, default=100)
    ap.add_argument('--diag_batches', type=int, default=3)
    args = ap.parse_args()

    os.makedirs(osp.join(args.out_dir, 'grids'), exist_ok=True)
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

    ref_mu, ref_sigma, ref_n = load_reference(args.ref_stats)

    batches = make_batches(args)
    gt_cache = None
    if 'real' in args.coarses:
        leak_source_folders = _class_folders(args.real_img_dir)
        gt_cache = precompute_gt(args, vae, batches, leak_source_folders, device)

    results = {}
    order = [(s, c, kr) for kr in args.keep_ratios for c in args.coarses for s in args.selects]
    for select, coarse, kr in order:
        fid, n, diag = run_condition(args, vae, var, incept, device, select, coarse, kr,
                                     ref_mu, ref_sigma, batches, gt_cache)
        results[f'{select}_{coarse}_keep{kr:g}'] = dict(
            select=select, coarse=coarse, keep_ratio=kr, fid=fid, n=n, **diag)
        with open(osp.join(args.out_dir, 'results.json'), 'w') as f:
            json.dump(results, f, indent=2)

    def fid(s, c, kr):
        return results[f'{s}_{c}_keep{kr:g}']['fid']

    print('\n' + '=' * 78)
    print('VAR selector-ranking inflation  --  FID table (prune_from_si=%d, ref n=%d)'
          % (args.prune_from_si, ref_n))
    print('=' * 78)
    header = f'{"keep":>6s} {"coarse":>6s} | {"smart":>10s} {"random":>10s} | {"smart_adv(rand-smart)":>22s}'
    print(header)
    print('-' * len(header))
    summary = {}
    for kr in args.keep_ratios:
        for c in args.coarses:
            s_fid = fid('smart', c, kr)
            r_fid = fid('random', c, kr)
            adv = r_fid - s_fid
            summary[f'{c}_keep{kr:g}_smart_adv'] = adv
            print(f'{kr:>6g} {c:>6s} | {s_fid:10.4f} {r_fid:10.4f} | {adv:>+22.4f}')

    print('\n--- smart-edge inflation ---')
    inflation = {}
    for kr in args.keep_ratios:
        if 'self' in args.coarses and 'real' in args.coarses:
            adv_self = summary[f'self_keep{kr:g}_smart_adv']
            adv_real = summary[f'real_keep{kr:g}_smart_adv']
            infl = adv_real - adv_self
            confirmed = bool(infl > 0)
            inflation[f'keep{kr:g}'] = dict(
                smart_adv_self=adv_self, smart_adv_real=adv_real,
                ranking_inflation=infl, confirmed=confirmed)
            print(f'  keep={kr:g}:')
            print(f'    smart_advantage(self) = FID(random,self) - FID(smart,self) = {adv_self:+.4f}')
            print(f'    smart_advantage(real) = FID(random,real) - FID(smart,real) = {adv_real:+.4f}')
            print(f'    ranking_inflation     = adv(real) - adv(self)             = {infl:+.4f}')
            print(f'    inflation > 0: {confirmed}')

    results['_summary'] = dict(
        ref_stats=args.ref_stats, ref_n=ref_n,
        prune_from_si=args.prune_from_si,
        smart_advantages=summary, inflation=inflation,
    )
    with open(osp.join(args.out_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    print('=' * 78)
    print(f'results -> {osp.join(args.out_dir, "results.json")}')


if __name__ == '__main__':
    main()
