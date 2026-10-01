import argparse
import math
import os
import os.path as osp
import random
from typing import List, Optional

import numpy as np
import torch
import torchvision
import PIL.Image as PImage

setattr(torch.nn.Linear, 'reset_parameters', lambda self: None)
setattr(torch.nn.LayerNorm, 'reset_parameters', lambda self: None)

from torchvision.transforms import InterpolationMode, transforms

from models import build_vae_var
from models.helpers import sample_with_top_k_top_p_, gumbel_softmax_with_rng

PATCH_NUMS = (1, 2, 3, 4, 5, 6, 8, 10, 13, 16)
NUM_SCALES = len(PATCH_NUMS)


def build_model(device, vae_ckpt, var_ckpt, depth=16):
    vae, var = build_vae_var(
        V=4096, Cvae=32, ch=160, share_quant_resi=4,
        device=device, patch_nums=PATCH_NUMS,
        num_classes=1000, depth=depth, shared_aln=False,
    )
    vae.load_state_dict(torch.load(vae_ckpt, map_location='cpu'), strict=True)
    var.load_state_dict(torch.load(var_ckpt, map_location='cpu'), strict=True)
    vae.eval(); var.eval()
    for p in vae.parameters(): p.requires_grad_(False)
    for p in var.parameters(): p.requires_grad_(False)
    return vae, var


def normalize_01_into_pm1(x):
    return x.add(x).add_(-1)


def build_val_transform(final_reso=256, mid_reso=1.125):
    mid = round(mid_reso * final_reso)
    return transforms.Compose([
        transforms.Resize(mid, interpolation=InterpolationMode.LANCZOS),
        transforms.CenterCrop((final_reso, final_reso)),
        transforms.ToTensor(),
        normalize_01_into_pm1,
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
def sample_curated(
    var, vae, B: int, label_B,
    cfg: float = 4.0, top_k: int = 900, top_p: float = 0.95,
    g_seed: Optional[int] = None, more_smooth: bool = False,
    gt_idxBl: Optional[List[torch.Tensor]] = None, leak_until_si: int = 0,
    prune_from_si: int = 10 ** 9, keep_ratio: float = 1.0,
    select: str = 'smart', select_seed: Optional[int] = None,
    return_debug: bool = False,
):
    assert select in ('smart', 'random'), select
    device = var.lvl_1L.device
    if g_seed is None:
        rng = None
    else:
        var.rng.manual_seed(g_seed); rng = var.rng

    if label_B is None:
        label_B = torch.multinomial(var.uniform_prob, num_samples=B, replacement=True, generator=rng).reshape(B)
    elif isinstance(label_B, int):
        label_B = torch.full((B,), fill_value=var.num_classes if label_B < 0 else label_B, device=device)

    sos = cond_BD = var.class_emb(torch.cat((label_B, torch.full_like(label_B, fill_value=var.num_classes)), dim=0))

    lvl_pos = var.lvl_embed(var.lvl_1L) + var.pos_1LC
    next_token_map = sos.unsqueeze(1).expand(2 * B, var.first_l, -1) + var.pos_start.expand(2 * B, var.first_l, -1) + lvl_pos[:, :var.first_l]

    cur_L = 0
    f_hat = sos.new_zeros(B, var.Cvae, var.patch_nums[-1], var.patch_nums[-1])

    debug = []
    for b in var.blocks:
        b.attn.kv_caching(True)
    for si, pn in enumerate(var.patch_nums):
        ratio = si / var.num_stages_minus_1
        cur_L += pn * pn
        cond_BD_or_gss = var.shared_ada_lin(cond_BD)
        x = next_token_map
        for b in var.blocks:
            x = b(x=x, cond_BD=cond_BD_or_gss, attn_bias=None)
        logits_BlV = var.get_logits(x, cond_BD)

        t = cfg * ratio
        logits_BlV = (1 + t) * logits_BlV[:B] - t * logits_BlV[B:]

        idx_Bl = sample_with_top_k_top_p_(logits_BlV, rng=rng, top_k=top_k, top_p=top_p, num_samples=1)[:, :, 0]

        leaked = False
        agree_pre = None
        match_post = None
        if gt_idxBl is not None and si < leak_until_si:
            gt = gt_idxBl[si].to(idx_Bl.device)
            assert gt.shape == idx_Bl.shape, f'si={si}: gt {tuple(gt.shape)} vs idx {tuple(idx_Bl.shape)}'
            agree_pre = (idx_Bl == gt).float().mean().item()
            idx_Bl = gt
            match_post = (idx_Bl == gt).float().mean().item()
            leaked = True

        if not more_smooth:
            h_BChw = var.vae_quant_proxy[0].embedding(idx_Bl)
        else:
            gum_t = max(0.27 * (1 - ratio * 0.95), 0.005)
            h_BChw = gumbel_softmax_with_rng(logits_BlV.mul(1 + ratio), tau=gum_t, hard=False, dim=-1, rng=rng) @ var.vae_quant_proxy[0].embedding.weight.unsqueeze(0)

        h_BChw = h_BChw.transpose_(1, 2).reshape(B, var.Cvae, pn, pn)

        pruned = False
        zeroed_frac = None
        kept_frac = None
        overlap_with_smart = None
        if si >= prune_from_si and keep_ratio < 1.0:
            n_cells = pn * pn
            keep = max(1, math.ceil(keep_ratio * n_cells))
            if keep < n_cells:
                score = h_BChw.pow(2).sum(dim=1).reshape(B, n_cells)
                smart_idx = score.topk(keep, dim=1).indices
                if select == 'random':
                    gen = torch.Generator(device=h_BChw.device)
                    base = 0 if select_seed is None else int(select_seed)
                    gen.manual_seed((base * 1_000_003 + si * 7919 + 12345) % (2 ** 63 - 1))
                    rand_score = torch.rand(B, n_cells, generator=gen, device=h_BChw.device)
                    sel_idx = rand_score.topk(keep, dim=1).indices
                    if return_debug:
                        sm = torch.zeros(B, n_cells, dtype=torch.bool, device=h_BChw.device)
                        sm.scatter_(1, smart_idx, True)
                        rm = torch.zeros(B, n_cells, dtype=torch.bool, device=h_BChw.device)
                        rm.scatter_(1, sel_idx, True)
                        overlap_with_smart = ((sm & rm).float().sum(dim=1) / keep).mean().item()
                else:
                    sel_idx = smart_idx
                mask = torch.zeros(B, n_cells, dtype=torch.bool, device=h_BChw.device)
                mask.scatter_(1, sel_idx, True)
                h_BChw = h_BChw * mask.reshape(B, 1, pn, pn)
                kept_frac = mask.float().sum(dim=1).mean().item() / n_cells
                zeroed_frac = 1.0 - kept_frac
                pruned = True

        f_hat, next_token_map = var.vae_quant_proxy[0].get_next_autoregressive_input(si, len(var.patch_nums), f_hat, h_BChw)
        if si != var.num_stages_minus_1:
            next_token_map = next_token_map.view(B, var.Cvae, -1).transpose(1, 2)
            next_token_map = var.word_embed(next_token_map) + lvl_pos[:, cur_L:cur_L + var.patch_nums[si + 1] ** 2]
            next_token_map = next_token_map.repeat(2, 1, 1)

        debug.append(dict(si=si, pn=pn, leaked=leaked, agree_pre=agree_pre,
                          match_post=match_post, pruned=pruned, zeroed_frac=zeroed_frac,
                          select=select, kept_frac=kept_frac,
                          overlap_with_smart=overlap_with_smart))

    for b in var.blocks:
        b.attn.kv_caching(False)
    img = vae.fhat_to_img(f_hat).add_(1).mul_(0.5)
    if return_debug:
        return img, debug
    return img


def save_images(img_B3HW, out_dir, prefix):
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    imgs = img_B3HW.clamp(0, 1).mul(255).round().to(torch.uint8).cpu()
    for i in range(imgs.shape[0]):
        p = osp.join(out_dir, f'{prefix}_{i:03d}.png')
        torchvision.utils.save_image(imgs[i].float() / 255.0, p)
        paths.append(p)
    return paths


def run_mode(args, vae, var, device):
    class_id = pick_class_id(args)
    real_imgs = None
    gt_idxBl = None
    leak_until_si = 0
    folder = None
    if args.mode == 'real':
        transform = build_val_transform()
        real_imgs, folder = load_real_images(args.real_img_dir, class_id, args.num_samples, transform, device, seed=args.seed)
        with torch.inference_mode():
            gt_idxBl = vae.img_to_idxBl(real_imgs)
        leak_until_si = args.leak_until_si

    label_B = torch.full((args.num_samples,), class_id, device=device, dtype=torch.long)
    with torch.inference_mode():
        with torch.autocast('cuda', enabled=True, dtype=torch.float16, cache_enabled=True):
            img, dbg = sample_curated(
                var, vae, B=args.num_samples, label_B=label_B,
                cfg=args.cfg, top_k=args.top_k, top_p=args.top_p, g_seed=args.seed,
                gt_idxBl=gt_idxBl, leak_until_si=leak_until_si,
                prune_from_si=args.prune_from_si, keep_ratio=args.keep_ratio,
                return_debug=True,
            )
    prefix = f'{args.mode}_cls{class_id}_keep{args.keep_ratio}_prune{args.prune_from_si}_leak{leak_until_si}'
    paths = save_images(img, args.out_dir, prefix)
    print(f'[{args.mode}] class_id={class_id} folder={folder} saved {len(paths)} imgs to {args.out_dir}')
    return img, dbg, paths


def pick_class_id(args):
    if args.class_id is None or str(args.class_id).lower() == 'random':
        return random.Random(args.seed).randint(0, 999)
    return int(args.class_id)


def smoke_test(args, vae, var, device):
    print('\n' + '=' * 70)
    print('SMOKE TEST')
    print('=' * 70)
    seed = args.seed
    cls = 980 if (args.class_id is None or str(args.class_id).lower() == 'random') else int(args.class_id)
    ok = True

    print('\n[CHECK 1] model builds & generates valid images (self, no hooks)')
    label_B = torch.full((4,), cls, device=device, dtype=torch.long)
    with torch.inference_mode(), torch.autocast('cuda', enabled=True, dtype=torch.float16):
        img1 = sample_curated(var, vae, B=4, label_B=label_B, cfg=args.cfg,
                              top_k=args.top_k, top_p=args.top_p, g_seed=seed)
    finite = torch.isfinite(img1).all().item()
    pmin, pmax, pstd = img1.min().item(), img1.max().item(), img1.std().item()
    valid1 = finite and (pstd > 1e-3) and (0.0 <= pmin) and (pmax <= 1.0 + 1e-4)
    print(f'  shape={tuple(img1.shape)} finite={finite} min={pmin:.4f} max={pmax:.4f} std={pstd:.4f} -> valid={valid1}')
    save_images(img1, args.out_dir, 'smoke_check1_self')
    ok &= valid1

    print('\n[CHECK 2] curation zeroes ~ (1-keep_ratio) of cells at pruned scales')
    print('          prune_from_si=7 (last 3 scales: pn=10,13,16), keep_ratio=0.25')
    with torch.inference_mode(), torch.autocast('cuda', enabled=True, dtype=torch.float16):
        _, dbg2 = sample_curated(var, vae, B=4, label_B=label_B, cfg=args.cfg,
                                top_k=args.top_k, top_p=args.top_p, g_seed=seed,
                                prune_from_si=7, keep_ratio=0.25, return_debug=True)
    valid2 = True
    for d in dbg2:
        if d['pruned']:
            expected = 1.0 - math.ceil(0.25 * d['pn'] ** 2) / (d['pn'] ** 2)
            good = abs(d['zeroed_frac'] - expected) < 1e-6
            print(f"  si={d['si']} pn={d['pn']} zeroed_frac={d['zeroed_frac']:.4f} "
                  f"(expected {expected:.4f}, ceil-rounded) -> {'OK' if good else 'BAD'}")
            valid2 &= good
    n_pruned = sum(1 for d in dbg2 if d['pruned'])
    valid2 &= (n_pruned == 3)
    print(f'  scales pruned = {n_pruned} (expected 3) -> valid={valid2}')
    ok &= valid2

    print('\n[CHECK 3] coarse-swap sets idx_Bl == gt_idxBl at leaked scales')
    print(f'          loading real images of class_id={cls}, leak_until_si=5')
    transform = build_val_transform()
    real_imgs, folder = load_real_images(args.real_img_dir, cls, 4, transform, device, seed=seed)
    print(f'  real imgs shape={tuple(real_imgs.shape)} range=[{real_imgs.min():.3f},{real_imgs.max():.3f}] folder={folder}')
    with torch.inference_mode():
        gt_idxBl = vae.img_to_idxBl(real_imgs)
    len_ok = (len(gt_idxBl) == NUM_SCALES)
    shapes_ok = all(gt_idxBl[si].shape == (4, PATCH_NUMS[si] ** 2) for si in range(NUM_SCALES))
    print(f'  img_to_idxBl -> len={len(gt_idxBl)} (expected {NUM_SCALES}), shapes_ok={shapes_ok}')
    with torch.inference_mode(), torch.autocast('cuda', enabled=True, dtype=torch.float16):
        _, dbg3 = sample_curated(var, vae, B=4, label_B=torch.full((4,), cls, device=device, dtype=torch.long),
                                cfg=args.cfg, top_k=args.top_k, top_p=args.top_p, g_seed=seed,
                                gt_idxBl=gt_idxBl, leak_until_si=5, return_debug=True)
    valid3 = len_ok and shapes_ok
    n_leaked = 0
    for d in dbg3:
        if d['leaked']:
            n_leaked += 1
            post_ok = (d['match_post'] is not None) and abs(d['match_post'] - 1.0) < 1e-9
            print(f"  si={d['si']} pn={d['pn']} match_post={d['match_post']:.4f} (must=1.0) "
                  f"| model-vs-real agree_pre={d['agree_pre']:.4f} -> {'OK' if post_ok else 'BAD'}")
            valid3 &= post_ok
    valid3 &= (n_leaked == 5)
    print(f'  scales leaked = {n_leaked} (expected 5) -> valid={valid3}')
    ok &= valid3

    print('\n[CHECK 4] generate 8 imgs each: (A) self-coarse vs (B) real-coarse')
    print('          keep_ratio=0.25, prune_from_si=6; (B) leak_until_si=6')
    label8 = torch.full((8,), cls, device=device, dtype=torch.long)
    valid4 = True
    try:
        with torch.inference_mode(), torch.autocast('cuda', enabled=True, dtype=torch.float16):
            imgA = sample_curated(var, vae, B=8, label_B=label8, cfg=args.cfg,
                                 top_k=args.top_k, top_p=args.top_p, g_seed=seed,
                                 gt_idxBl=None, leak_until_si=0,
                                 prune_from_si=6, keep_ratio=0.25)
        pA = save_images(imgA, args.out_dir, 'smoke_check4_A_selfcoarse')
        real8, _ = load_real_images(args.real_img_dir, cls, 8, transform, device, seed=seed)
        with torch.inference_mode():
            gt8 = vae.img_to_idxBl(real8)
        with torch.inference_mode(), torch.autocast('cuda', enabled=True, dtype=torch.float16):
            imgB = sample_curated(var, vae, B=8, label_B=label8, cfg=args.cfg,
                                 top_k=args.top_k, top_p=args.top_p, g_seed=seed,
                                 gt_idxBl=gt8, leak_until_si=6,
                                 prune_from_si=6, keep_ratio=0.25)
        pB = save_images(imgB, args.out_dir, 'smoke_check4_B_realcoarse')
        aok = torch.isfinite(imgA).all().item() and imgA.std().item() > 1e-3
        bok = torch.isfinite(imgB).all().item() and imgB.std().item() > 1e-3
        print(f'  (A) self-coarse: {len(pA)} imgs, finite&nondegenerate={aok}')
        print(f'  (B) real-coarse: {len(pB)} imgs, finite&nondegenerate={bok}')
        valid4 = aok and bok
    except Exception as e:
        print(f'  ERROR during 2x2 generation: {e!r}')
        valid4 = False
    ok &= valid4

    print('\n' + '=' * 70)
    print(f'SMOKE TEST {"PASSED" if ok else "FAILED"}  (out_dir={args.out_dir})')
    print('=' * 70)
    print('\nTo run the FULL 2x2 experiment (per-mode generation for FID), e.g.:')
    print(f'  CUDA_VISIBLE_DEVICES=0 {os.path.basename("python")} var_leak_exp.py \\')
    print('    --mode self --keep_ratio 0.25 --prune_from_si 6 \\')
    print('    --num_samples 5000 --class_id random --out_dir runs/leak/self')
    print(f'  CUDA_VISIBLE_DEVICES=0 python var_leak_exp.py \\')
    print('    --mode real --keep_ratio 0.25 --prune_from_si 6 --leak_until_si 6 \\')
    print('    --num_samples 5000 --class_id random --out_dir runs/leak/real \\')
    print('    --real_img_dir data/imagenet/train')
    return ok


def main():
    ap = argparse.ArgumentParser(description='VAR real-coarse leak / curation experiment')
    ap.add_argument('--mode', choices=['self', 'real'], default='self')
    ap.add_argument('--keep_ratio', type=float, default=0.25)
    ap.add_argument('--prune_from_si', type=int, default=6)
    ap.add_argument('--leak_until_si', type=int, default=6)
    ap.add_argument('--num_samples', type=int, default=8)
    ap.add_argument('--class_id', default='random', help='int ImageNet class 0-999, or "random"')
    ap.add_argument('--out_dir', default='runs/leak_smoke')
    ap.add_argument('--real_img_dir', default='data/imagenet/train')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--cfg', type=float, default=4.0)
    ap.add_argument('--top_k', type=int, default=900)
    ap.add_argument('--top_p', type=float, default=0.95)
    ap.add_argument('--depth', type=int, default=16)
    ap.add_argument('--vae_ckpt', default='ckpt/vae_ch160v4096z32.pth')
    ap.add_argument('--var_ckpt', default='ckpt/var_d16.pth')
    ap.add_argument('--smoke', action='store_true', help='run correctness smoke test only')
    args = ap.parse_args()

    torch.manual_seed(args.seed); random.seed(args.seed); np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision('high')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'device={device}; building model (depth={args.depth}) ...')
    vae, var = build_model(device, args.vae_ckpt, args.var_ckpt, depth=args.depth)
    print('model ready.')

    if args.smoke:
        smoke_test(args, vae, var, device)
    else:
        run_mode(args, vae, var, device)


if __name__ == '__main__':
    main()
