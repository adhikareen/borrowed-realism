import argparse
import math
import os
import os.path as osp
import json
import random
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch

HERE = osp.dirname(osp.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import infinity.models.fastvar_utils as fvu  # noqa: E402  (we monkeypatch this)

SELECT_MODE = "smart"
SELECT_SEED = 0
_REC_ENABLE = False
_REC_CURATION = []

_ORIG_MASKED = fvu.masked_previous_scale_cache


def masked_previous_scale_cache_patched(cur_x, num_remain, cur_shape):
    B, L, c = cur_x.shape
    if SELECT_MODE == "smart":
        mean_x = cur_x.view(B, cur_shape[1], cur_shape[2], -1).permute(0, 3, 1, 2)
        mean_x = torch.nn.functional.adaptive_avg_pool2d(mean_x, (1, 1)).permute(0, 2, 3, 1).view(B, 1, c)
        score = torch.sum((cur_x - mean_x) ** 2, dim=-1, keepdim=True)
    elif SELECT_MODE == "random":
        g = torch.Generator(device=cur_x.device)
        g.manual_seed((int(SELECT_SEED) * 1_000_003 + cur_shape[1] * 7919 + cur_shape[2] * 104729 + 12345) % (2 ** 63 - 1))
        score = torch.rand(B, L, 1, generator=g, device=cur_x.device)
    else:
        raise ValueError(SELECT_MODE)

    select_indices = torch.argsort(score, dim=1, descending=True)
    filted_select_indices = select_indices[:, :num_remain, :]

    if _REC_ENABLE:
        _REC_CURATION.append(dict(
            mode=SELECT_MODE, h=int(cur_shape[1]), w=int(cur_shape[2]), L=int(L),
            num_remain=int(num_remain), kept_frac=float(num_remain) / float(L),
            idx0=filted_select_indices[0, :, 0].detach().to("cpu"),
        ))

    def merge(merged_cur_x):
        return torch.gather(merged_cur_x, dim=1, index=filted_select_indices.repeat(1, 1, c))

    def unmerge(unmerged_cur_x, unmerged_cache_x, cached_hw=None):
        unmerged_cache_x_ = unmerged_cache_x.view(B, cached_hw[0], cached_hw[1], -1).permute(0, 3, 1, 2)
        unmerged_cache_x_ = torch.nn.functional.interpolate(
            unmerged_cache_x_, size=(cur_shape[1], cur_shape[2]), mode="area"
        ).permute(0, 2, 3, 1).view(B, L, c)
        unmerged_cache_x_.scatter_(dim=1, index=filted_select_indices.repeat(1, 1, c), src=unmerged_cur_x)
        return unmerged_cache_x_

    def get_src_tgt_idx():
        return filted_select_indices

    return merge, unmerge, get_src_tgt_idx


fvu.masked_previous_scale_cache = masked_previous_scale_cache_patched

from tools.run_infinity import (  # noqa: E402
    load_tokenizer, load_visual_tokenizer, load_transformer, gen_one_img, transform,
)
from infinity.utils.dynamic_resolution import dynamic_resolution_h_w, h_div_w_templates  # noqa: E402

_REC_TOKENS_ENABLE = False
_REC_TOKENS = []


def install_token_recorder(vae):
    lfq = vae.quantizer.lfq
    if getattr(lfq, "_leak_wrapped", False):
        return
    orig = lfq.indices_to_codes

    def wrapped(indices, *a, **k):
        if _REC_TOKENS_ENABLE:
            _REC_TOKENS.append(indices.detach().to("cpu"))
        return orig(indices, *a, **k)

    lfq.indices_to_codes = wrapped
    lfq._leak_wrapped = True


def make_args(a):
    return SimpleNamespace(
        pn="1M", model_type="infinity_2b", vae_type=32, apply_spatial_patchify=0,
        use_bit_label=1, rope2d_each_sa_layer=1, rope2d_normalized_by_hw=2,
        add_lvl_embeding_only_first_block=1, use_scale_schedule_embedding=0,
        text_channels=2048, cfg=a.cfg, tau=a.tau, cfg_insertion_layer=0,
        bf16=1, checkpoint_type="torch", use_flex_attn=0,
        model_path=a.model_path, vae_path=a.vae_path,
        text_encoder_ckpt=a.text_encoder_ckpt, cache_dir="/dev/shm",
        seed=a.seed, h_div_w_template=1.0, enable_positive_prompt=0,
    )


def get_scale_schedule():
    ss = dynamic_resolution_h_w[1.0]["1M"]["scales"]
    return [(1, h, w) for (_, h, w) in ss]


def get_pixel():
    return dynamic_resolution_h_w[1.0]["1M"]["pixel"]


def build(a):
    margs = make_args(a)
    text_tokenizer, text_encoder = load_tokenizer(t5_path=margs.text_encoder_ckpt)
    vae = load_visual_tokenizer(margs)
    infinity = load_transformer(vae, margs)
    install_token_recorder(vae)
    return text_tokenizer, text_encoder, vae, infinity


@torch.no_grad()
def encode_real_prior(vae, image_path, scale_schedule, device, tgt_h, tgt_w):
    from PIL import Image
    pil = Image.open(image_path).convert("RGB")
    inp = transform(pil, tgt_h, tgt_w).unsqueeze(0).to(device)
    ss = [(int(t), int(h), int(w)) for (t, h, w) in scale_schedule]
    h, z, _, all_bit_indices, _, _ = vae.encode(inp, scale_schedule=ss)
    return all_bit_indices


@torch.no_grad()
def generate(infinity, vae, text_tokenizer, text_encoder, prompt, scale_schedule, a,
             gt_leak=0, gt_ls_Bl=None, select_mode="smart", seed=0):
    global SELECT_MODE, SELECT_SEED
    SELECT_MODE = select_mode
    SELECT_SEED = seed
    img = gen_one_img(
        infinity, vae, text_tokenizer, text_encoder, prompt,
        cfg_list=a.cfg, tau_list=a.tau, negative_prompt="",
        scale_schedule=scale_schedule, top_k=900, top_p=0.97, cfg_sc=3,
        cfg_insertion_layer=[0], vae_type=32, gumbel=0, softmax_merge_topk=-1,
        gt_leak=gt_leak, gt_ls_Bl=(gt_ls_Bl if gt_ls_Bl is not None else []),
        g_seed=seed, sampling_per_bits=1, enable_positive_prompt=0,
    )
    return img


def save_bgr(img_hwc_bgr, path):
    import cv2
    os.makedirs(osp.dirname(osp.abspath(path)), exist_ok=True)
    cv2.imwrite(path, img_hwc_bgr.cpu().numpy())


def load_mjhq(meta_path, img_root, n, seed=0, category=None):
    meta = json.load(open(meta_path))
    items = [(k, v["prompt"], v.get("category")) for k, v in meta.items()]
    if category:
        items = [it for it in items if it[2] == category]
    random.Random(seed).shuffle(items)
    out = []
    for k, prompt, cat in items:
        p = osp.join(img_root, cat, f"{k}.jpg")
        if not osp.exists(p):
            p2 = osp.join(img_root, f"{k}.jpg")
            p = p2 if osp.exists(p2) else p
        if osp.exists(p):
            out.append((k, prompt, p))
        if len(out) >= n:
            break
    return out


def smoke(a):
    global _REC_ENABLE, _REC_TOKENS_ENABLE
    print("\n" + "=" * 72 + "\nSMOKE TEST (Infinity real-coarse leak)\n" + "=" * 72)
    device = "cuda"
    ss = get_scale_schedule()
    tgt_h, tgt_w = get_pixel()
    GT_LEAK = a.gt_leak
    print(f"scale_schedule (t,h,w): {ss}")
    print(f"gt_leak={GT_LEAK} -> leaked scales si=0..{GT_LEAK-1}; pruned scales si=9(w32),10(w40)")

    tk, te, vae, inf = build(a)
    pairs = load_mjhq(a.mjhq_meta, a.mjhq_imgs, 2, seed=a.seed)
    assert len(pairs) >= 1, "no MJHQ pairs found (need images extracted)"
    (kid, prompt, imgpath) = pairs[0]
    print(f"\nsample: id={kid}\n  prompt={prompt[:90]!r}\n  real_img={imgpath}")

    gt_ls_Bl = encode_real_prior(vae, imgpath, ss, device, tgt_h, tgt_w)
    print(f"\nreal prior: len(all_bit_indices)={len(gt_ls_Bl)}; "
          f"per-scale shapes si0..8={[tuple(gt_ls_Bl[i].shape) for i in range(9)]}")

    ok = True

    print("\n[CHECK 1] leak injects real tokens at si<gt_leak and model wouldn't produce them")
    _REC_TOKENS.clear(); _REC_TOKENS_ENABLE = True
    _ = generate(inf, vae, tk, te, prompt, ss, a, gt_leak=0, gt_ls_Bl=None, select_mode="smart", seed=a.seed)
    self_tokens = list(_REC_TOKENS)
    _REC_TOKENS.clear()
    _ = generate(inf, vae, tk, te, prompt, ss, a, gt_leak=GT_LEAK, gt_ls_Bl=gt_ls_Bl, select_mode="smart", seed=a.seed)
    leak_tokens = list(_REC_TOKENS)
    _REC_TOKENS_ENABLE = False
    n_gen_scales = sum(1 for si, pn in enumerate(ss) if pn[2] not in (48, 64))
    print(f"  recorded scales: self={len(self_tokens)} leak={len(leak_tokens)} (expected {n_gen_scales})")
    valid1 = (len(leak_tokens) >= GT_LEAK) and (len(self_tokens) >= GT_LEAK)
    for si in range(GT_LEAK):
        gt = gt_ls_Bl[si].to("cpu")
        match_post = (leak_tokens[si] == gt).float().mean().item()
        agree_self = (self_tokens[si] == gt).float().mean().item()
        good = (abs(match_post - 1.0) < 1e-9) and (agree_self < 0.999)
        print(f"  si={si} (w={ss[si][2]:2d}) match_post={match_post:.4f} (=1.0) "
              f"| self-vs-real agree={agree_self:.4f} (<1.0) -> {'OK' if good else 'BAD'}")
        valid1 &= good
    print(f"  -> CHECK 1 {'OK' if valid1 else 'BAD'}")
    ok &= valid1

    print("\n[CHECK 2] curation keeps FastVAR fraction (w32:keep~0.6, w40:keep~0.5); smart!=random")
    global _REC_CURATION
    def run_capture(mode):
        global _REC_ENABLE
        _REC_CURATION.clear(); _REC_ENABLE = True
        _ = generate(inf, vae, tk, te, prompt, ss, a, gt_leak=0, gt_ls_Bl=None, select_mode=mode, seed=a.seed)
        _REC_ENABLE = False
        return list(_REC_CURATION)
    cur_smart = run_capture("smart")
    cur_rand = run_capture("random")
    exp = {}
    for pn in ss:
        w = pn[2]
        if w in (32, 40):
            L = pn[0] * pn[1] * pn[2]
            ratio = {32: 0.4, 40: 0.5}[w]
            exp[w] = (L - int(L * ratio)) / L
    print(f"  expected kept_frac by width: {exp}")
    seen_w = {}
    valid2 = len(cur_smart) > 0 and len(cur_rand) > 0
    for rec in cur_smart:
        w = rec["w"]
        good = abs(rec["kept_frac"] - exp.get(w, rec["kept_frac"])) < 1e-6
        if w not in seen_w:
            print(f"  [smart] w={w} L={rec['L']} keep={rec['num_remain']} kept_frac={rec['kept_frac']:.4f} "
                  f"(exp {exp.get(w):.4f}) -> {'OK' if good else 'BAD'}")
            seen_w[w] = True
        valid2 &= good
    def first_by_w(recs):
        d = {}
        for r in recs:
            d.setdefault(r["w"], r)
        return d
    sd, rd = first_by_w(cur_smart), first_by_w(cur_rand)
    for w in sorted(set(sd) & set(rd)):
        a_set = set(sd[w]["idx0"].tolist()); b_set = set(rd[w]["idx0"].tolist())
        keep = sd[w]["num_remain"]
        overlap = len(a_set & b_set) / max(1, keep)
        diff = overlap < 0.999
        print(f"  w={w}: smart-vs-random kept-set overlap={overlap:.4f} (<1.0 => selectors differ) "
              f"-> {'OK' if diff else 'BAD'}")
        valid2 &= diff
    print(f"  -> CHECK 2 {'OK' if valid2 else 'BAD'}")
    ok &= valid2

    print("\n[CHECK 3] generate 2x2 (smart/random x self/real); save grid; check validity")
    import cv2
    conds = [
        ("self_smart",  0,       None,     "smart"),
        ("self_random", 0,       None,     "random"),
        ("real_smart",  GT_LEAK, gt_ls_Bl, "smart"),
        ("real_random", GT_LEAK, gt_ls_Bl, "random"),
    ]
    tiles, valid3 = [], True
    for name, gl, gls, mode in conds:
        img = generate(inf, vae, tk, te, prompt, ss, a, gt_leak=gl, gt_ls_Bl=gls, select_mode=mode, seed=a.seed)
        arr = img.cpu().numpy()
        finite = np.isfinite(arr).all(); std = float(arr.std())
        v = bool(finite and std > 1.0)
        print(f"  {name:12s} shape={arr.shape} std={std:7.2f} finite={finite} -> valid={v}")
        save_bgr(img, osp.join(a.out_dir, f"smoke_{name}.png"))
        tiles.append(arr); valid3 &= v
    top = np.concatenate([tiles[0], tiles[1]], axis=1)
    bot = np.concatenate([tiles[2], tiles[3]], axis=1)
    grid = np.concatenate([top, bot], axis=0)
    cv2.imwrite(osp.join(a.out_dir, "smoke_grid_2x2.png"), grid)
    print(f"  grid saved -> {osp.join(a.out_dir, 'smoke_grid_2x2.png')} "
          f"(TL=self_smart TR=self_random BL=real_smart BR=real_random)")
    print(f"  -> CHECK 3 {'OK' if valid3 else 'BAD'}")
    ok &= valid3

    print("\n" + "=" * 72)
    print(f"SMOKE {'PASSED' if ok else 'FAILED'}  (out_dir={a.out_dir})")
    print("=" * 72)
    print_full_cmd(a)
    return ok


def print_full_cmd(a):
    print("\nFULL 2x2 run (per-condition generation, then FID vs MJHQ real):")
    for mode in ("smart", "random"):
        for tag, gl in (("self", 0), ("real", a.gt_leak)):
            print(f"  python infinity_leak_exp.py --mode gen --select {mode} --coarse {tag} "
                  f"--gt_leak {a.gt_leak} --num_samples {a.num_samples} "
                  f"--out_dir runs/leak/{tag}_{mode} "
                  f"--model_path {a.model_path} --vae_path {a.vae_path} "
                  f"--text_encoder_ckpt {a.text_encoder_ckpt} "
                  f"--mjhq_meta {a.mjhq_meta} --mjhq_imgs {a.mjhq_imgs}")
    print("  # then FID (pytorch_fid), each condition's pred/ vs the shared MJHQ real ref:")
    print("  python tools/fid_score.py runs/leak/<cond>/pred runs/leak/real_ref")


def run_gen(a):
    device = "cuda"
    ss = get_scale_schedule()
    tgt_h, tgt_w = get_pixel()
    tk, te, vae, inf = build(a)
    pairs = load_mjhq(a.mjhq_meta, a.mjhq_imgs, a.num_samples, seed=a.seed, category=a.category)
    pred_dir = osp.join(a.out_dir, "pred")
    ref_dir = osp.join(osp.dirname(a.out_dir.rstrip("/")), "real_ref")
    os.makedirs(pred_dir, exist_ok=True); os.makedirs(ref_dir, exist_ok=True)
    print(f"[gen] select={a.select} coarse={a.coarse} gt_leak={a.gt_leak if a.coarse=='real' else 0} "
          f"n={len(pairs)} -> {pred_dir}")
    import shutil
    t0 = time.time()
    for i, (kid, prompt, imgpath) in enumerate(pairs):
        gl, gls = 0, None
        if a.coarse == "real":
            gls = encode_real_prior(vae, imgpath, ss, device, tgt_h, tgt_w)
            gl = a.gt_leak
        img = generate(inf, vae, tk, te, prompt, ss, a, gt_leak=gl, gt_ls_Bl=gls,
                       select_mode=a.select, seed=a.seed)
        save_bgr(img, osp.join(pred_dir, f"{kid}.png"))
        rp = osp.join(ref_dir, f"{kid}.jpg")
        if not osp.exists(rp):
            shutil.copyfile(imgpath, rp)
        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{len(pairs)}  ({(time.time()-t0)/(i+1):.1f}s/img)")
    print(f"[gen] done {len(pairs)} imgs in {time.time()-t0:.0f}s. FID: "
          f"python tools/fid_score.py {pred_dir} {ref_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["smoke", "gen"], default="smoke")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--select", choices=["smart", "random"], default="smart")
    ap.add_argument("--coarse", choices=["self", "real"], default="self")
    ap.add_argument("--gt_leak", type=int, default=9)
    ap.add_argument("--num_samples", type=int, default=8)
    ap.add_argument("--category", default=None)
    ap.add_argument("--out_dir", default="runs/leak_smoke")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cfg", type=float, default=4.0)
    ap.add_argument("--tau", type=float, default=1.0)
    ap.add_argument("--model_path", default="weights/infinity_2b_reg.pth")
    ap.add_argument("--vae_path", default="weights/infinity_vae_d32_reg.pth")
    ap.add_argument("--text_encoder_ckpt", default="weights/flan-t5-xl")
    ap.add_argument("--mjhq_meta", default="weights/MJHQ30K/meta_data.json")
    ap.add_argument("--mjhq_imgs", default="weights/MJHQ30K/mjhq30k_imgs")
    a = ap.parse_args()

    torch.manual_seed(a.seed); random.seed(a.seed); np.random.seed(a.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    os.makedirs(a.out_dir, exist_ok=True)

    if a.smoke or a.mode == "smoke":
        smoke(a)
    else:
        run_gen(a)


if __name__ == "__main__":
    main()
