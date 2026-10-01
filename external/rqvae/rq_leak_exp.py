import argparse
import math
import os
import os.path as osp
import random
from itertools import product
from typing import List, Optional

import numpy as np
import torch
import torchvision
import PIL.Image as PImage
from torchvision import transforms

from rqvae.models import create_model
from rqvae.utils.config import load_config, augment_arch_defaults
from rqvae.utils.utils import set_seed, sample_from_logits


def load_one_model(path, ema=False):
    model_config = osp.join(osp.dirname(path), 'config.yaml')
    config = load_config(model_config)
    config.arch = augment_arch_defaults(config.arch)
    model, _ = create_model(config.arch, ema=False)
    key = 'state_dict_ema' if ema else 'state_dict'
    ckpt = torch.load(path, map_location='cpu')[key]
    model.load_state_dict(ckpt)
    return model, config


def build_models(ar_path, vae_path, device, ema=False):
    model_ar, config_ar = load_one_model(ar_path, ema=ema)
    model_vqvae, config_vae = load_one_model(vae_path, ema=False)
    model_ar = model_ar.to(device).eval()
    model_vqvae = model_vqvae.to(device).eval()
    for p in model_ar.parameters():
        p.requires_grad_(False)
    for p in model_vqvae.parameters():
        p.requires_grad_(False)
    return model_ar, model_vqvae, config_ar, config_vae


def build_val_transform():
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])


def _class_folders(train_root):
    fs = [f for f in os.listdir(train_root) if osp.isdir(osp.join(train_root, f))]
    return sorted(fs)


def classid_to_folder(train_root, class_id):
    return _class_folders(train_root)[class_id]


def load_real_images(real_img_dir, class_id, num, transform, device, seed=0):
    folder = classid_to_folder(real_img_dir, class_id)
    cdir = osp.join(real_img_dir, folder)
    files = sorted(os.listdir(cdir))
    random.Random(seed).shuffle(files)
    imgs = []
    for fn in files:
        if len(imgs) >= num:
            break
        try:
            im = PImage.open(osp.join(cdir, fn)).convert('RGB')
        except Exception:
            continue
        imgs.append(transform(im))
    if len(imgs) < num:
        raise RuntimeError(f'only found {len(imgs)}/{num} readable images in {cdir}')
    return torch.stack(imgs).to(device), folder


@torch.no_grad()
def sample_codes(
    model_ar, model_vqvae, B: int, cond,
    temperature: float = 1.0, top_k=16384, top_p=0.92, amp: bool = True,
    gt_code: Optional[torch.Tensor] = None, leak_until_d: int = 0,
    return_debug: bool = False,
):
    device = model_ar.pos_emb_hw.device
    (H, W, D) = model_ar.block_size

    if top_k is None:
        top_k_list = [model_ar.vocab_size[i] for i in range(D)]
    elif isinstance(top_k, int):
        top_k_list = [min(top_k, model_ar.vocab_size[i]) for i in range(D)]
    else:
        top_k_list = [min(top_k[i] if len(top_k) > 1 else top_k[0], model_ar.vocab_size[i]) for i in range(D)]
    if top_p is None:
        top_p_list = [1.0 for _ in range(D)]
    elif isinstance(top_p, float):
        top_p_list = [min(top_p, 1.0) for _ in range(D)]
    else:
        top_p_list = [min(top_p[i] if len(top_p) > 1 else top_p[0], 1.0) for i in range(D)]

    if cond is not None and not torch.is_tensor(cond):
        cond = torch.as_tensor(cond, device=device)

    xs = torch.zeros(B, H, W, D, dtype=torch.long, device=device)
    dbg = {d: dict(leaked=False, agree_pre_sum=0.0, match_post_sum=0.0, n=0) for d in range(D)}

    model_ar.init_cache()
    for (h, w, d) in product(range(H), range(W), range(D)):
        xs_partial = xs[:, :h + 1]
        logits_hwd = model_ar.cached_forward(
            xs_partial, model_vqvae, cond=cond, amp=amp, sample_loc=(h, w, d))
        samples_hwd = sample_from_logits(
            logits_hwd, temperature=temperature,
            top_k=top_k_list[d], top_p=top_p_list[d])

        if gt_code is not None and d < leak_until_d:
            gt_hwd = gt_code[:, h, w, d].to(samples_hwd.device)
            own = samples_hwd
            agree_pre = (own == gt_hwd).float().mean().item()
            samples_hwd = gt_hwd
            match_post = (samples_hwd == gt_hwd).float().mean().item()
            dd = dbg[d]
            dd['leaked'] = True
            dd['agree_pre_sum'] += agree_pre
            dd['match_post_sum'] += match_post
            dd['n'] += 1

        xs[:, h, w, d] = samples_hwd
    model_ar.init_cache()

    if return_debug:
        for d in range(D):
            n = max(dbg[d]['n'], 1)
            dbg[d]['agree_pre'] = dbg[d]['agree_pre_sum'] / n
            dbg[d]['match_post'] = dbg[d]['match_post_sum'] / n
        return xs, dbg
    return xs


@torch.no_grad()
def curate_and_decode(
    model_vqvae, code: torch.Tensor,
    prune_from_d: int, keep_ratio: float,
    select: str = 'smart', select_seed: Optional[int] = None,
    return_debug: bool = False,
):
    assert select in ('smart', 'random'), select
    quant = model_vqvae.quantizer
    B, h, w, D = code.shape
    device = code.device
    n_cells = h * w

    embeds_depth, _ = quant.embed_code_with_depth(code, to_latent_shape=False)

    coarse_sum = embeds_depth[..., :prune_from_d, :].sum(dim=-2)
    full_sum = embeds_depth.sum(dim=-2)
    fine_res = embeds_depth[..., prune_from_d:, :].sum(dim=-2)
    energy = fine_res.pow(2).sum(dim=-1).reshape(B, n_cells)

    keep = max(1, math.ceil(keep_ratio * n_cells))
    keep = min(keep, n_cells)

    smart_idx = energy.topk(keep, dim=1).indices
    overlap_with_smart = None
    if select == 'random':
        gen = torch.Generator(device=device)
        base = 0 if select_seed is None else int(select_seed)
        gen.manual_seed((base * 1_000_003 + 7919) % (2 ** 63 - 1))
        rand_score = torch.rand(B, n_cells, generator=gen, device=device)
        sel_idx = rand_score.topk(keep, dim=1).indices
        if return_debug:
            sm = torch.zeros(B, n_cells, dtype=torch.bool, device=device)
            sm.scatter_(1, smart_idx, True)
            rm = torch.zeros(B, n_cells, dtype=torch.bool, device=device)
            rm.scatter_(1, sel_idx, True)
            overlap_with_smart = ((sm & rm).float().sum(1) / keep).mean().item()
    else:
        sel_idx = smart_idx

    mask = torch.zeros(B, n_cells, dtype=torch.bool, device=device)
    mask.scatter_(1, sel_idx, True)
    kept_frac = mask.float().sum(1).mean().item() / n_cells
    mask = mask.reshape(B, h, w, 1)

    z_code = torch.where(mask, full_sum, coarse_sum)
    z_q = quant.to_latent_shape(z_code)
    decoded = model_vqvae.decode(z_q)
    img = (decoded * 0.5 + 0.5).clamp(0, 1)

    if return_debug:
        return img, dict(select=select, keep=keep, kept_frac=kept_frac,
                         n_cells=n_cells, overlap_with_smart=overlap_with_smart)
    return img


def save_grid(img_N3HW_01, path, nrow=8, n=64):
    os.makedirs(osp.dirname(path), exist_ok=True)
    x = img_N3HW_01[:n].clamp(0, 1).float().cpu()
    torchvision.utils.save_image(x, path, nrow=nrow, padding=2)


def smoke_test(args, model_ar, model_vqvae, device):
    print('\n' + '=' * 72)
    print('RQ real-coarse-leak x curation  --  SMOKE / CORRECTNESS TEST')
    print('=' * 72)
    (H, W, D) = model_ar.block_size
    print(f'block_size (H,W,D)=({H},{W},{D})  code positions H*W={H*W}')
    cls = 980 if (args.class_id is None or str(args.class_id).lower() == 'random') else int(args.class_id)
    tr = build_val_transform()
    ok = True

    print('\n[CHECK 0] pure self-generation -> valid images (no hooks)')
    set_seed(args.seed)
    cond = torch.full((args.n,), cls, device=device, dtype=torch.long)
    code0 = sample_codes(model_ar, model_vqvae, B=args.n, cond=cond,
                         temperature=args.temp, top_k=args.top_k, top_p=args.top_p, amp=True)
    img0 = curate_and_decode(model_vqvae, code0, prune_from_d=D, keep_ratio=1.0, select='smart')
    finite = torch.isfinite(img0).all().item()
    print(f'  codes {tuple(code0.shape)} range=[{code0.min().item()},{code0.max().item()}]  '
          f'img {tuple(img0.shape)} finite={finite} min={img0.min():.3f} max={img0.max():.3f} std={img0.std():.4f}')
    valid0 = finite and img0.std().item() > 1e-3 and img0.min() >= 0 and img0.max() <= 1 + 1e-4
    save_grid(img0, osp.join(args.out_dir, 'smoke_check0_self.png'))
    ok &= valid0; print(f'  -> valid={valid0}')

    print(f'\n[CHECK i] real-coarse (depth) leak: swap depths d<{args.leak_until_d}')
    real_imgs, folder = load_real_images(args.real_img_dir, cls, args.n, tr, device, seed=args.seed)
    gt_code = model_vqvae.get_codes(real_imgs)
    print(f'  real imgs {tuple(real_imgs.shape)} folder={folder}  gt_code {tuple(gt_code.shape)}')
    shape_ok = tuple(gt_code.shape) == (args.n, H, W, D)
    set_seed(args.seed)
    code_leak, dbg = sample_codes(model_ar, model_vqvae, B=args.n, cond=cond,
                                  temperature=args.temp, top_k=args.top_k, top_p=args.top_p, amp=True,
                                  gt_code=gt_code, leak_until_d=args.leak_until_d, return_debug=True)
    valid_i = shape_ok
    n_leaked = 0
    for d in range(D):
        if dbg[d]['leaked']:
            n_leaked += 1
            match_post = dbg[d]['match_post']
            agree_pre = dbg[d]['agree_pre']
            post_ok = abs(match_post - 1.0) < 1e-9
            leak_ok = agree_pre < 0.98
            print(f'  depth d={d}: match_post={match_post:.4f} (must=1.0)  '
                  f'model-vs-real agree_pre={agree_pre:.4f} (leak genuine if <<1) -> '
                  f'{"OK" if (post_ok and leak_ok) else "BAD"}')
            valid_i &= post_ok and leak_ok
    used_match = (code_leak[..., :args.leak_until_d] == gt_code[..., :args.leak_until_d]).float().mean().item()
    if args.leak_until_d < D:
        fine_match = (code_leak[..., args.leak_until_d:] == gt_code[..., args.leak_until_d:]).float().mean().item()
    else:
        fine_match = float('nan')
    print(f'  used-code vs real @swapped depths = {used_match:.4f} (must=1.0)  |  '
          f'@fine depths = {fine_match:.4f} (should be low, self-generated)')
    valid_i &= (n_leaked == args.leak_until_d) and abs(used_match - 1.0) < 1e-9
    ok &= valid_i; print(f'  depths leaked={n_leaked} (expected {args.leak_until_d}) -> valid={valid_i}')

    print(f'\n[CHECK ii] curation keep_ratio={args.keep_ratio}, prune_from_d={args.prune_from_d}')
    imgS, dS = curate_and_decode(model_vqvae, code0, prune_from_d=args.prune_from_d,
                                 keep_ratio=args.keep_ratio, select='smart',
                                 select_seed=args.seed, return_debug=True)
    imgR, dR = curate_and_decode(model_vqvae, code0, prune_from_d=args.prune_from_d,
                                 keep_ratio=args.keep_ratio, select='random',
                                 select_seed=args.seed, return_debug=True)
    exp_frac = math.ceil(args.keep_ratio * dS['n_cells']) / dS['n_cells']
    frac_ok = abs(dS['kept_frac'] - exp_frac) < 1e-6 and abs(dR['kept_frac'] - exp_frac) < 1e-6
    count_ok = dS['keep'] == dR['keep']
    overlap = dR['overlap_with_smart']
    overlap_ok = abs(overlap - args.keep_ratio) < 0.10
    print(f'  smart kept_frac={dS["kept_frac"]:.4f} keep={dS["keep"]} | '
          f'random kept_frac={dR["kept_frac"]:.4f} keep={dR["keep"]} | expected_frac={exp_frac:.4f}')
    print(f'  random-vs-smart overlap={overlap:.4f} (~chance {args.keep_ratio}) -> '
          f'{"OK" if overlap_ok else "BAD"};  iso-count={count_ok}; frac={frac_ok}')
    valid_ii = frac_ok and count_ok and overlap_ok
    ok &= valid_ii; print(f'  -> valid={valid_ii}')

    print(f'\n[CHECK iii] 4 conditions (smart/random x self/real) valid images')
    valid_iii = True
    grids = {}
    for coarse in ('self', 'real'):
        set_seed(args.seed)
        gc = gt_code if coarse == 'real' else None
        lud = args.leak_until_d if coarse == 'real' else 0
        code_c = sample_codes(model_ar, model_vqvae, B=args.n, cond=cond,
                              temperature=args.temp, top_k=args.top_k, top_p=args.top_p, amp=True,
                              gt_code=gc, leak_until_d=lud)
        for select in ('smart', 'random'):
            img = curate_and_decode(model_vqvae, code_c, prune_from_d=args.prune_from_d,
                                    keep_ratio=args.keep_ratio, select=select, select_seed=args.seed)
            good = torch.isfinite(img).all().item() and img.std().item() > 1e-3
            tag = f'{select}_{coarse}'
            grids[tag] = img
            save_grid(img, osp.join(args.out_dir, f'smoke_check3_{tag}.png'))
            print(f'  {tag:14s}: img {tuple(img.shape)} finite&nondegenerate={good} std={img.std():.4f}')
            valid_iii &= good
    ok &= valid_iii; print(f'  -> valid={valid_iii}')

    print('\n' + '=' * 72)
    print(f'SMOKE TEST {"PASSED" if ok else "FAILED"}   (grids in {args.out_dir})')
    print('=' * 72)
    return ok


def main():
    ap = argparse.ArgumentParser(description='RQ real-coarse leak / curation smoke')
    ap.add_argument('--model-ar-path', required=True)
    ap.add_argument('--model-vqvae-path', required=True)
    ap.add_argument('--ema', action='store_true')
    ap.add_argument('--n', type=int, default=16, help='images for smoke')
    ap.add_argument('--class_id', default='980')
    ap.add_argument('--prune_from_d', type=int, default=2)
    ap.add_argument('--leak_until_d', type=int, default=2)
    ap.add_argument('--keep_ratio', type=float, default=0.25)
    ap.add_argument('--temp', type=float, default=1.0)
    ap.add_argument('--top_k', type=int, default=16384)
    ap.add_argument('--top_p', type=float, default=0.92)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out_dir', default='runs/rq_leak_smoke')
    ap.add_argument('--real_img_dir', default='data/imagenet/train')
    ap.add_argument('--smoke', action='store_true')
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    torch.backends.cudnn.benchmark = True
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'device={device}; building RQ-Transformer + RQ-VAE ...', flush=True)
    model_ar, model_vqvae, cfg_ar, cfg_vae = build_models(
        args.model_ar_path, args.model_vqvae_path, device, ema=args.ema)
    print(f'models ready. block_size={tuple(model_ar.block_size)} '
          f'vocab_size_cond={model_ar.vocab_size_cond}', flush=True)

    if args.smoke:
        smoke_test(args, model_ar, model_vqvae, device)
    else:
        print('nothing to do (use --smoke). Full 4-condition run: rq_select_run.py')


if __name__ == '__main__':
    main()
