from __future__ import annotations
import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import paths

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from cosmos_tokenizer.video_lib import CausalVideoTokenizer
from cosmos_tokenizer.modules.quantizers import FSQuantizer
from tqdm import tqdm

from model import CompactVideoLM, CompactVideoLMConfig
from streams import (
    COARSE_TOKENS, FINE_TOKENS, TYPE_COARSE, TYPE_FINE, TYPE_SPECIAL,
    BOS_SOURCE_POS, EOS_SOURCE_POS, PAD_SOURCE_POS,
    COARSE_SOURCE_OFFSET, FINE_SOURCE_OFFSET, SOURCE_POS_VOCAB_SIZE,
    Layout, assert_packed_manifest_pool, special_ids,
)


def load_ckpt(path: Path, device):
    ck = torch.load(path, map_location="cpu")
    cfg = CompactVideoLMConfig(**ck["model_config"])
    model = CompactVideoLM(cfg).to(device).eval()
    model.load_state_dict(ck["model"])
    return model, cfg, ck.get("train_manifest", {}), ck.get("val_manifest", {})


class _OmniDecoderWrap:
    def __init__(self, model):
        self.model = model

    def decode(self, encodings, is_image=False):
        return self.model.decode(encodings, is_image=is_image)


def _load_omnitok_decoder(ckpt_path: str, device):
    import sys as _sys
    _sys.path.insert(0, paths.WORK)
    _sys.path.insert(0, paths.OMNITOK_DIR)
    from OmniTokenizer.omnitokenizer import VQGAN
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    hparams = ck.get("hyper_parameters", ck.get("args"))
    if hasattr(hparams, "args"):
        margs = hparams["args"]
    elif "args" in hparams:
        margs = hparams["args"]
    else:
        margs = hparams
    for fix in ("causal_in_temporal_transformer", "causal_in_peg"):
        bad = fix.replace("causal", "casual")
        if not hasattr(margs, fix) and hasattr(margs, bad):
            setattr(margs, fix, getattr(margs, bad))
    model = VQGAN(margs).eval()
    model.load_state_dict(ck["state_dict"], strict=False)
    model = model.to(device).to(torch.float32)
    return _OmniDecoderWrap(model)


class _OM2DecoderWrap:
    def __init__(self, model):
        self.model = model

    @torch.no_grad()
    def decode(self, encodings, is_image=False):
        import om2_tok
        quant = om2_tok.om2_indices_to_features(encodings.to(torch.long))
        return self.model.decode(quant.to(next(self.model.parameters()).device))


def _load_om2_decoder(ckpt_path: str, device):
    import om2_tok
    model = om2_tok.load_om2(ckpt_path=ckpt_path, device=device)
    return _OM2DecoderWrap(model)


def reconstruct_grid_from_packed(coarse_ids, fine_ids, selected_pos, device, layout: Layout = None):
    if layout is None:
        layout = Layout()
    B = coarse_ids.shape[0]
    P = layout.pool_size
    T, H, W = layout.latent_t, layout.latent_h, layout.latent_w
    coarse_grid = coarse_ids.view(B, T, layout.coarse_h, layout.coarse_w)
    coarse_up = coarse_grid.repeat_interleave(P, dim=-2).repeat_interleave(P, dim=-1)
    grid = coarse_up.clone().reshape(B, layout.fine_tokens)
    bidx = torch.arange(B, device=device).unsqueeze(1).expand_as(selected_pos)
    grid[bidx, selected_pos] = fine_ids
    return grid.view(B, T, H, W).to(torch.int32)


def reconstruct_grid_from_packed_adaptive(coarse_ids, fine_ids_padded, selected_pos, k_i,
                                          device, layout: Layout = None):
    if layout is None:
        layout = Layout()
    B = coarse_ids.shape[0]
    P = layout.pool_size
    T, H, W = layout.latent_t, layout.latent_h, layout.latent_w
    coarse_grid = coarse_ids.view(B, T, layout.coarse_h, layout.coarse_w)
    coarse_up = coarse_grid.repeat_interleave(P, dim=-2).repeat_interleave(P, dim=-1)
    grid = coarse_up.clone().reshape(B, layout.fine_tokens)
    k_max = fine_ids_padded.shape[1]
    slot = torch.arange(k_max, device=device).unsqueeze(0).expand(B, k_max)
    valid = slot < k_i.unsqueeze(1)
    sel = selected_pos.long().clamp(0, layout.fine_tokens - 1)
    write_val = torch.where(valid, fine_ids_padded.long(), torch.gather(grid, 1, sel))
    grid.scatter_(1, sel, write_val)
    return grid.view(B, T, H, W).to(torch.int32)


def reconstruct_grid_from_dense(fine_ids, device, layout: Layout = None):
    if layout is None:
        layout = Layout()
    return fine_ids.view(-1, layout.latent_t, layout.latent_h, layout.latent_w).to(torch.int32)


def build_dense_prompt(B, device, bos_id):
    ids = torch.full((B, 1), bos_id, dtype=torch.long, device=device)
    pos = torch.full((B, 1), BOS_SOURCE_POS, dtype=torch.long, device=device)
    typ = torch.full((B, 1), TYPE_SPECIAL, dtype=torch.long, device=device)
    return ids, pos, typ


def reconstruct_grid_from_dense_trunc(fine_ids_partial, n_gen, device, layout: Layout = None):
    if layout is None:
        layout = Layout()
    B = fine_ids_partial.shape[0]
    T, H, W = layout.latent_t, layout.latent_h, layout.latent_w
    F = layout.fine_tokens
    per_frame = H * W
    flat = torch.zeros(B, F, dtype=torch.long, device=device)
    flat[:, :n_gen] = fine_ids_partial.to(torch.long)
    n_full_frames = max(1, n_gen // per_frame)
    last_full = n_full_frames - 1
    grid = flat.view(B, T, H, W)
    src_frame = grid[:, last_full:last_full + 1, :, :]
    first_incomplete = n_gen // per_frame
    if first_incomplete < T:
        grid[:, first_incomplete:, :, :] = src_frame.expand(B, T - first_incomplete, H, W)
    return grid.to(torch.int32)


def build_packed_prompt_from_cache(coarse_ids, bos_id, device, layout: Layout = None):
    if layout is None:
        layout = Layout()
    coarse_n = layout.coarse_tokens
    assert coarse_ids.shape[1] == coarse_n, (
        f"coarse_ids width {coarse_ids.shape[1]} != layout.coarse_tokens {coarse_n} (pool={layout.pool_size})"
    )
    B = coarse_ids.shape[0]
    bos = torch.full((B, 1), bos_id, dtype=torch.long, device=device)
    ids = torch.cat([bos, coarse_ids.to(torch.long)], dim=1)
    pos = torch.cat([
        torch.full((B, 1), BOS_SOURCE_POS, dtype=torch.long, device=device),
        (COARSE_SOURCE_OFFSET + torch.arange(coarse_n, device=device)).unsqueeze(0).expand(B, coarse_n),
    ], dim=1)
    typ = torch.cat([
        torch.full((B, 1), TYPE_SPECIAL, dtype=torch.long, device=device),
        torch.full((B, coarse_n), TYPE_COARSE, dtype=torch.long, device=device),
    ], dim=1)
    return ids, pos, typ


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, type=Path, help="LM checkpoint .pt")
    ap.add_argument("--mode", choices=["dense", "packed", "packed_adaptive", "packed_gencoarse", "packed_gencoarse_ref", "dense_trunc"], required=True,
                    help="generation mode")
    ap.add_argument("--cosmos-ckpt", required=False, default=None, type=str,
                    help="Cosmos tokenizer dir with decoder.jit")
    ap.add_argument("--omnitok-ckpt", required=False, default=None, type=str,
                    help="OmniTok VQGAN checkpoint")
    ap.add_argument("--om2-ckpt", required=False, default=None, type=str,
                    help="OpenMAGVIT-2 checkpoint")
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--num-samples", type=int, default=1024,
                    help="1024 is usually enough for stable FVD")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--top-k", type=int, default=256)
    ap.add_argument("--top-p", type=float, default=0.0,
                    help="nucleus threshold (0 = off)")
    ap.add_argument("--repetition-penalty", type=float, default=1.0,
                    help="repetition penalty (1.0 = off)")
    ap.add_argument("--adaptive-stop", choices=["argmax", "sample", "oracle"], default="argmax",
                    help="stop rule (packed_adaptive)")
    ap.add_argument("--dense-trunc-n", type=int, default=None,
                    help="fine tokens to generate (dense_trunc)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--save-every-batches", type=int, default=1,
                    help="progress checkpoint cadence (batches)")
    ap.add_argument("--batch-timeout-sec", type=float, default=300.0,
                    help="per-batch timeout (s)")
    ap.add_argument("--packed-ref-cache", type=Path, default=None,
                    help="packed cache dir")
    ap.add_argument("--clamp-fine-to-coarse", action="store_true",
                    help="block fine-to-coarse attention")
    ap.add_argument("--coarse-mix-alpha", type=float, default=None,
                    help="probability of replacing a sampled coarse token with the real one")
    ap.add_argument("--clamp-mode", choices=["coarse", "placebo_fine"], default="coarse",
                    help="coarse | placebo_fine")
    ap.add_argument("--refiner-ckpt", type=str, default=None,
                    help="refiner checkpoint")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.out_dir / "ar_generate.log"
    progress_path = args.out_dir / "ar_generate_progress.jsonl"
    done_path = args.out_dir / "ar_generate_done.json"

    torch.manual_seed(args.seed); random.seed(args.seed); np.random.seed(args.seed)
    device = torch.device(args.device)

    def log(msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        with log_path.open("a") as f: f.write(line + "\n")

    log(f"args: {vars(args)}")
    log(f"loading ckpt {args.ckpt}")
    model, cfg, train_manifest, val_manifest = load_ckpt(args.ckpt, device)
    manifest = train_manifest or val_manifest
    base_vocab = int(manifest["base_vocab_size"])
    sp = special_ids(base_vocab)
    pool_size = int(manifest.get("pool_size", 2))
    latent_t = int(manifest.get("latent_t", manifest.get("latent_frames", 5)))
    latent_h = int(manifest.get("latent_h", 16))
    latent_w = int(manifest.get("latent_w", 16))
    layout = Layout(pool_size=pool_size, latent_t=latent_t, latent_h=latent_h, latent_w=latent_w)
    log(f"layout: pool={pool_size} coarse_tokens={layout.coarse_tokens} fine_tokens={layout.fine_tokens}")
    log(f"model {cfg.to_dict()}")
    log(f"manifest: mode={manifest.get('mode')} seq_len={manifest.get('seq_len')} vocab={manifest.get('vocab_size')}")

    if getattr(args, "om2_ckpt", None):
        log(f"loading Open-MAGVIT2 video decoder {args.om2_ckpt}")
        dec = _load_om2_decoder(args.om2_ckpt, device)
        decode_backend = "omnitok"
    elif args.omnitok_ckpt:
        log(f"loading OmniTokenizer VQGAN decoder {args.omnitok_ckpt}")
        dec = _load_omnitok_decoder(args.omnitok_ckpt, device)
        decode_backend = "omnitok"
    else:
        if not args.cosmos_ckpt:
            raise ValueError("must pass either --cosmos-ckpt or --omnitok-ckpt")
        log(f"loading cosmos decoder {args.cosmos_ckpt}")
        dec = CausalVideoTokenizer(checkpoint_dec=f"{args.cosmos_ckpt}/decoder.jit")
        decode_backend = "cosmos"

    refiner = None
    fsq_ref = None
    if args.refiner_ckpt:
        if decode_backend != "cosmos":
            raise ValueError("--refiner-ckpt only supported with the Cosmos DV decoder")
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from refiner import Refiner
        rck = torch.load(args.refiner_ckpt, map_location="cpu")
        refiner = Refiner(code_dim=6, width=int(rck.get("width", 64))).to(device).eval()
        refiner.load_state_dict(rck["refiner"])
        for p in refiner.parameters(): p.requires_grad_(False)
        fsq_ref = FSQuantizer(levels=[8, 8, 8, 5, 5, 5], dim=6).to(device).float()
        n_rp = sum(p.numel() for p in refiner.parameters())
        log(f"REFINER-FALLBACK enabled: {args.refiner_ckpt} ({n_rp/1e6:.3f}M params)")

    def apply_refiner(grid, selected_pos):
        B, T, H, W = grid.shape
        keep = torch.zeros(B, T * H * W, device=grid.device)
        bidx = torch.arange(B, device=grid.device).unsqueeze(1).expand_as(selected_pos)
        keep[bidx, selected_pos] = 1.0
        keep = keep.view(B, 1, T, H, W)
        drop = 1.0 - keep
        codes = fsq_ref.indices_to_codes(grid.reshape(B, -1).long()).float()
        codes = codes.reshape(B, T, H, W, 6).permute(0, 4, 1, 2, 3).contiguous()
        refined = refiner(codes, keep)
        out_codes = keep * codes + drop * refined
        oc = out_codes.permute(0, 2, 3, 4, 1).reshape(-1, 1, 6)
        q = fsq_ref.quantize(oc).squeeze(-2)
        ridx = fsq_ref.codes_to_indices(q.unsqueeze(1)).squeeze(1).reshape(B, T, H, W)
        keep_i = keep.long().squeeze(1)
        return (keep_i * grid + (1 - keep_i) * ridx).to(torch.int32)

    existing = sorted(args.out_dir.glob("gen_*.npy"))
    done_ids = {int(p.stem.split("_")[1]) for p in existing}
    log(f"resume: {len(done_ids)} samples already present")

    packed_ref = None
    if args.mode in ("packed", "packed_gencoarse_ref"):
        if args.packed_ref_cache is None:
            raise ValueError(f"{args.mode} mode requires --packed-ref-cache")
        refm = json.loads((args.packed_ref_cache / "manifest.json").read_text())
        if refm.get("mode") != "packed":
            raise ValueError("packed-ref-cache manifest mode must be 'packed'")
        ref_pool = int(refm.get("pool_size", 2))
        if ref_pool != pool_size:
            raise ValueError(
                f"packed-ref-cache pool_size={ref_pool} does not match LM ckpt pool_size={pool_size}. "
                f"You are mixing pool=2 and pool=4 caches — this would silently corrupt generation."
            )
        assert_packed_manifest_pool(refm, expected_pool=pool_size)
        packed_ref = {
            "manifest": refm,
            "input_ids": np.load(args.packed_ref_cache / "input_ids.npy", mmap_mode="r"),
            "source_pos": np.load(args.packed_ref_cache / "source_pos.npy", mmap_mode="r"),
            "K": int(refm["packed_fine_budget"]),
        }
        log(f"packed-ref: N={len(packed_ref['input_ids'])} K={packed_ref['K']} tpf={refm.get('tpf')} pool={ref_pool}")
        N_avail = len(packed_ref["input_ids"])
        args.num_samples = min(args.num_samples, N_avail)
    elif args.mode == "packed_adaptive":
        if args.packed_ref_cache is None:
            raise ValueError("packed_adaptive mode requires --packed-ref-cache (an adaptive cache)")
        refm = json.loads((args.packed_ref_cache / "manifest.json").read_text())
        if refm.get("mode") != "packed_adaptive":
            raise ValueError(
                f"packed_adaptive mode requires a mode='packed_adaptive' cache; got mode="
                f"{refm.get('mode')!r}. Use e.g. runs/cache_adaptive_tpf50_val.")
        ref_pool = int(refm.get("pool_size", 2))
        if ref_pool != pool_size:
            raise ValueError(
                f"adaptive-ref-cache pool_size={ref_pool} != LM ckpt pool_size={pool_size}.")
        packed_ref = {
            "manifest": refm,
            "input_ids": np.load(args.packed_ref_cache / "input_ids.npy", mmap_mode="r"),
            "source_pos": np.load(args.packed_ref_cache / "source_pos.npy", mmap_mode="r"),
            "fine_counts": np.load(args.packed_ref_cache / "fine_counts.npy"),
            "K": int(refm["packed_fine_budget"]),
        }
        log(f"adaptive-ref: N={len(packed_ref['input_ids'])} k_max={packed_ref['K']} "
            f"mean_K={refm.get('mean_fine_budget'):.2f} std_K={refm.get('std_fine_budget'):.2f} "
            f"mean_content_tok={refm.get('mean_tokens_content'):.2f} pool={ref_pool}")
        N_avail = len(packed_ref["input_ids"])
        args.num_samples = min(args.num_samples, N_avail)
        realized_kis = {}

    target_indices = [i for i in range(args.num_samples) if i not in done_ids]
    log(f"plan: {len(target_indices)} samples to generate (batch {args.batch_size})")

    if not target_indices:
        log("nothing to do, exiting"); return

    t_all0 = time.time()
    batch_idx_counter = 0
    pbar = tqdm(range(0, len(target_indices), args.batch_size), desc=f"ar-gen[{args.mode}]")
    for bstart in pbar:
        batch_start_t = time.time()
        batch_global_ids = target_indices[bstart:bstart + args.batch_size]
        B = len(batch_global_ids)

        try:
            if args.mode == "dense":
                ids, pos, typ = build_dense_prompt(B, device, sp.bos)
                max_new = layout.fine_tokens + 1
                next_type_id = TYPE_FINE
                fine_off = layout.fine_source_offset
                fine_n = layout.fine_tokens
                def next_src_fn(step):
                    if step < fine_n:
                        return fine_off + step
                    return EOS_SOURCE_POS
            elif args.mode == "dense_trunc":
                if args.dense_trunc_n is None:
                    raise ValueError("dense_trunc mode requires --dense-trunc-n")
                trunc_n = int(args.dense_trunc_n)
                ids, pos, typ = build_dense_prompt(B, device, sp.bos)
                max_new = trunc_n
                next_type_id = TYPE_FINE
                fine_off = layout.fine_source_offset
                def next_src_fn(step):
                    return fine_off + step
            elif args.mode == "packed_gencoarse":
                pass
            elif args.mode == "packed_gencoarse_ref":
                K = packed_ref["K"]
                coarse_n = layout.coarse_tokens
                fine_off = layout.fine_source_offset
                batch_spos = np.stack([packed_ref["source_pos"][i] for i in batch_global_ids])
                selected_pos = torch.from_numpy(
                    (batch_spos[:, 1 + coarse_n:1 + coarse_n + K].astype(np.int64)
                     - fine_off)
                ).to(device)
            elif args.mode == "packed_adaptive":
                k_max = packed_ref["K"]
                coarse_n = layout.coarse_tokens
                fine_off = layout.fine_source_offset
                batch_seqs = np.stack([packed_ref["input_ids"][i] for i in batch_global_ids])
                batch_spos = np.stack([packed_ref["source_pos"][i] for i in batch_global_ids]).astype(np.int64)
                coarse_ids = torch.from_numpy(batch_seqs[:, 1:1 + coarse_n].astype(np.int64)).to(device)
                sel_np = batch_spos[:, 1 + coarse_n:1 + coarse_n + k_max] - fine_off
                sel_np = np.clip(sel_np, 0, layout.fine_tokens - 1)
                selected_pos = torch.from_numpy(sel_np).to(device)
                ids, pos, typ = build_packed_prompt_from_cache(coarse_ids, sp.bos, device, layout=layout)
            else:
                K = packed_ref["K"]
                coarse_n = layout.coarse_tokens
                fine_off = layout.fine_source_offset
                batch_seqs = np.stack([packed_ref["input_ids"][i] for i in batch_global_ids])
                batch_spos = np.stack([packed_ref["source_pos"][i] for i in batch_global_ids])
                coarse_ids = torch.from_numpy(batch_seqs[:, 1:1 + coarse_n].astype(np.int64)).to(device)
                selected_pos = torch.from_numpy(
                    (batch_spos[:, 1 + coarse_n:1 + coarse_n + K].astype(np.int64)
                     - fine_off)
                ).to(device)
                ids, pos, typ = build_packed_prompt_from_cache(coarse_ids, sp.bos, device, layout=layout)
                max_new = K + 1
                next_type_id = TYPE_FINE
                sel_cpu = selected_pos.cpu().numpy()
                def next_src_fn(step, _sel=sel_cpu):
                    return int(_sel[0, step])

            t0 = time.time()
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                if args.mode == "dense":
                    gen_ids, gen_src, gen_typ = model.generate(
                        input_ids=ids, source_pos=pos, type_ids=typ,
                        max_new=max_new, next_source_pos_fn=next_src_fn,
                        next_type_id=next_type_id, temperature=args.temperature,
                        top_k=args.top_k, top_p=args.top_p,
                        repetition_penalty=args.repetition_penalty,
                        rep_pen_vocab_limit=base_vocab, eos_token=sp.eos,
                    )
                elif args.mode == "dense_trunc":
                    gen_ids, gen_src, gen_typ = model.generate(
                        input_ids=ids, source_pos=pos, type_ids=typ,
                        max_new=max_new, next_source_pos_fn=next_src_fn,
                        next_type_id=next_type_id, temperature=args.temperature,
                        top_k=args.top_k, top_p=args.top_p,
                        repetition_penalty=args.repetition_penalty,
                        rep_pen_vocab_limit=base_vocab, eos_token=None,
                    )
                elif args.mode == "packed_gencoarse":
                    gc_coarse_ids, gc_fine_ids = packed_gencoarse_generate(
                        model=model, bos_id=sp.bos, B=B, layout=layout,
                        temperature=args.temperature, top_k=args.top_k,
                        eos_id=sp.eos, device=device,
                    )
                elif args.mode == "packed_gencoarse_ref":
                    mix_real = None
                    if args.coarse_mix_alpha is not None:
                        mix_real = torch.from_numpy(np.stack(
                            [packed_ref["input_ids"][i][1:1 + layout.coarse_tokens]
                             for i in batch_global_ids]).astype(np.int64)).to(device)
                    gcr_coarse_ids, gcr_fine_ids = packed_gencoarse_ref_generate(
                        model=model, bos_id=sp.bos, B=B, layout=layout,
                        selected_pos=selected_pos, K=packed_ref["K"],
                        temperature=args.temperature, top_k=args.top_k,
                        eos_id=sp.eos, device=device,
                        fine_source_offset=layout.fine_source_offset,
                        real_coarse=mix_real, mix_alpha=args.coarse_mix_alpha,
                    )
                elif args.mode == "packed_adaptive":
                    oracle_k = None
                    if args.adaptive_stop == "oracle":
                        oracle_k = torch.from_numpy(
                            np.asarray([int(packed_ref["fine_counts"][i]) for i in batch_global_ids],
                                       dtype=np.int64)).to(device)
                    pa_fine_padded, pa_k_i = packed_adaptive_generate(
                        model=model, prompt_ids=ids, prompt_pos=pos, prompt_typ=typ,
                        selected_pos=selected_pos, k_max=packed_ref["K"],
                        temperature=args.temperature, top_k=args.top_k,
                        eos_id=sp.eos, device=device,
                        fine_source_offset=layout.fine_source_offset,
                        stop_rule=args.adaptive_stop, oracle_k=oracle_k,
                    )
                else:
                    gen_ids, gen_src, gen_typ = packed_generate(
                        model=model, prompt_ids=ids, prompt_pos=pos, prompt_typ=typ,
                        selected_pos=selected_pos, K=packed_ref["K"],
                        temperature=args.temperature, top_k=args.top_k,
                        eos_id=sp.eos, device=device,
                        fine_source_offset=layout.fine_source_offset,
                        clamp_fine_to_coarse=args.clamp_fine_to_coarse,
                        clamp_mode=args.clamp_mode,
                    )
            gen_dt = time.time() - t0

            if args.mode == "dense":
                fine_ids = gen_ids[:, 1:1 + layout.fine_tokens].clamp(0, base_vocab - 1)
                grid = reconstruct_grid_from_dense(fine_ids, device, layout=layout)
            elif args.mode == "dense_trunc":
                fine_part = gen_ids[:, 1:1 + trunc_n].clamp(0, base_vocab - 1)
                grid = reconstruct_grid_from_dense_trunc(fine_part, trunc_n, device, layout=layout)
            elif args.mode == "packed_gencoarse":
                fine_gen = gc_fine_ids.clamp(0, base_vocab - 1)
                grid = reconstruct_grid_from_dense(fine_gen, device, layout=layout)
            elif args.mode == "packed_gencoarse_ref":
                gcr_coarse = gcr_coarse_ids.clamp(0, base_vocab - 1)
                fine_gen = gcr_fine_ids.clamp(0, base_vocab - 1)
                grid = reconstruct_grid_from_packed(gcr_coarse, fine_gen, selected_pos, device, layout=layout)
            elif args.mode == "packed_adaptive":
                grid = reconstruct_grid_from_packed_adaptive(
                    coarse_ids, pa_fine_padded, selected_pos, pa_k_i, device, layout=layout)
                kis_b = pa_k_i.detach().cpu().numpy()
                for j, gid in enumerate(batch_global_ids):
                    realized_kis[int(gid)] = int(kis_b[j])
            else:
                K = packed_ref["K"]
                coarse_n = layout.coarse_tokens
                coarse_ids = gen_ids[:, 1:1 + coarse_n]
                fine_gen = gen_ids[:, 1 + coarse_n:1 + coarse_n + K].clamp(0, base_vocab - 1)
                grid = reconstruct_grid_from_packed(coarse_ids, fine_gen, selected_pos, device, layout=layout)
                if refiner is not None:
                    grid = apply_refiner(grid, selected_pos)

            with torch.no_grad():
                if decode_backend == "omnitok":
                    rec = dec.decode(grid.to(torch.long), is_image=False).float().clamp(-1, 1)
                else:
                    rec = dec.decode(grid).float().clamp(-1, 1)
            rec01 = ((rec + 1) / 2).to(torch.float16).cpu().numpy()

            for j, gid in enumerate(batch_global_ids):
                cls = 0
                np.save(args.out_dir / f"gen_{gid:07d}_cls{cls:04d}.npy", rec01[j])

            if args.mode == "packed_adaptive":
                with (args.out_dir / "realized_k.jsonl").open("a") as f:
                    for j, gid in enumerate(batch_global_ids):
                        f.write(json.dumps({"idx": int(gid), "k": int(kis_b[j])}) + "\n")

            batch_dt = time.time() - batch_start_t
            gpu_mem_gb = torch.cuda.max_memory_allocated(device) / 1e9
            prog = {"batch": batch_idx_counter, "from_idx": batch_global_ids[0],
                    "to_idx": batch_global_ids[-1], "bsz": B, "batch_sec": batch_dt,
                    "gen_sec": gen_dt, "gpu_mem_gb": gpu_mem_gb,
                    "elapsed_sec": time.time() - t_all0,
                    "done": len(done_ids) + bstart + B, "total": args.num_samples}
            with progress_path.open("a") as f: f.write(json.dumps(prog) + "\n")
            if args.batch_timeout_sec > 0 and batch_dt > args.batch_timeout_sec:
                log(f"[slow] batch {batch_idx_counter} took {batch_dt:.1f}s (>{args.batch_timeout_sec}s)")
            pbar.set_postfix_str(f"{batch_dt:.1f}s/b mem={gpu_mem_gb:.1f}GB")

            if (batch_idx_counter % 8) == 0:
                torch.cuda.empty_cache()

        except torch.cuda.OutOfMemoryError as e:
            log(f"[oom] batch {batch_idx_counter} starting at {batch_global_ids[0]}: {e}. "
                f"Skipping; reduce --batch-size to recover.")
            torch.cuda.empty_cache()
        except KeyboardInterrupt:
            log("[interrupt] saving partial progress and exiting"); break
        except Exception as e:
            log(f"[error] batch {batch_idx_counter}: {type(e).__name__}: {e}. skipping.")
            torch.cuda.empty_cache()

        batch_idx_counter += 1

    total = time.time() - t_all0
    n_saved = len(list(args.out_dir.glob("gen_*.npy")))
    summary = {"finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
               "mode": args.mode, "num_saved": n_saved,
               "total_sec": total, "videos_per_sec": n_saved / max(total, 1e-9)}
    if args.mode == "packed_adaptive":
        rk_path = args.out_dir / "realized_k.jsonl"
        ks = []
        if rk_path.exists():
            seen = {}
            for line in rk_path.read_text().splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                    seen[int(d["idx"])] = int(d["k"])
                except Exception:
                    pass
            ks = list(seen.values())
        if ks:
            ks_np = np.asarray(ks, dtype=np.float64)
            coarse_n = int(packed_ref["manifest"]["coarse_tokens"]) if packed_ref else 320
            summary.update({
                "realized_mean_K": float(ks_np.mean()),
                "realized_std_K": float(ks_np.std()),
                "realized_min_K": int(ks_np.min()),
                "realized_max_K": int(ks_np.max()),
                "realized_median_K": float(np.median(ks_np)),
                "realized_n_clips_with_K": int(ks_np.shape[0]),
                "realized_mean_content_tokens": float(coarse_n + ks_np.mean()),
                "realized_mean_tokens_incl_special": float(coarse_n + ks_np.mean() + 2),
                "cache_mean_K": float(packed_ref["manifest"].get("mean_fine_budget")) if packed_ref else None,
                "cache_mean_content_tokens": float(packed_ref["manifest"].get("mean_tokens_content")) if packed_ref else None,
            })
            log(f"ADAPTIVE realized K: mean={ks_np.mean():.2f} std={ks_np.std():.2f} "
                f"min={int(ks_np.min())} max={int(ks_np.max())} | "
                f"mean content tokens (320+K) = {coarse_n + ks_np.mean():.2f} "
                f"(cache target {packed_ref['manifest'].get('mean_tokens_content'):.2f})")
    done_path.write_text(json.dumps(summary, indent=2))
    log(f"done: {n_saved} videos in {total:.1f}s ({n_saved/max(total, 1e-9):.2f} vid/s)")


@torch.no_grad()
def packed_generate(model, prompt_ids, prompt_pos, prompt_typ,
                    selected_pos, K, temperature, top_k, eos_id, device,
                    fine_source_offset: int = FINE_SOURCE_OFFSET,
                    clamp_fine_to_coarse: bool = False,
                    clamp_mode: str = "coarse"):
    model.eval()
    B = prompt_ids.shape[0]
    kv_caches = [{} for _ in range(model.cfg.n_layers)]

    logits, _ = model(input_ids=prompt_ids, source_pos=prompt_pos, type_ids=prompt_typ,
                      kv_caches=kv_caches, start_pos=0,
                      clamp_fine_to_coarse=clamp_fine_to_coarse, clamp_mode=clamp_mode)
    cur_pos = prompt_ids.shape[1]
    all_ids = [prompt_ids]; all_src = [prompt_pos]; all_typ = [prompt_typ]
    next_logits = logits[:, -1, :]

    for step in range(K + 1):
        nl = next_logits / max(temperature, 1e-5)
        if top_k and top_k > 0:
            v, _ = torch.topk(nl, min(top_k, nl.size(-1)))
            nl = torch.where(nl < v[:, -1:], torch.full_like(nl, -float("inf")), nl)
        probs = F.softmax(nl, dim=-1)
        next_tok = torch.multinomial(probs, num_samples=1)

        if step < K:
            next_src = (fine_source_offset + selected_pos[:, step]).unsqueeze(1)
            next_typ = torch.full((B, 1), TYPE_FINE, dtype=torch.long, device=device)
        else:
            next_src = torch.full((B, 1), EOS_SOURCE_POS, dtype=torch.long, device=device)
            next_typ = torch.full((B, 1), TYPE_SPECIAL, dtype=torch.long, device=device)

        all_ids.append(next_tok); all_src.append(next_src); all_typ.append(next_typ)
        if step == K: break
        logits, _ = model(input_ids=next_tok, source_pos=next_src, type_ids=next_typ,
                          kv_caches=kv_caches, start_pos=cur_pos,
                          clamp_fine_to_coarse=clamp_fine_to_coarse, clamp_mode=clamp_mode)
        cur_pos += 1
        next_logits = logits[:, -1, :]

    return (torch.cat(all_ids, dim=1), torch.cat(all_src, dim=1), torch.cat(all_typ, dim=1))


@torch.no_grad()
def packed_adaptive_generate(model, prompt_ids, prompt_pos, prompt_typ,
                             selected_pos, k_max, temperature, top_k, eos_id, device,
                             fine_source_offset: int = FINE_SOURCE_OFFSET,
                             stop_rule: str = "argmax", oracle_k=None):
    assert stop_rule in ("sample", "argmax", "oracle"), f"unknown stop_rule {stop_rule}"
    if stop_rule == "oracle":
        assert oracle_k is not None, "oracle stop_rule requires oracle_k (per-clip budget)"
        oracle_k = oracle_k.to(device=device, dtype=torch.long).clamp(0, k_max)
    model.eval()
    B = prompt_ids.shape[0]
    kv_caches = [{} for _ in range(model.cfg.n_layers)]

    logits, _ = model(input_ids=prompt_ids, source_pos=prompt_pos, type_ids=prompt_typ,
                      kv_caches=kv_caches, start_pos=0)
    cur_pos = prompt_ids.shape[1]
    next_logits = logits[:, -1, :]

    pad_id = model.cfg.pad_token_id
    fine_ids_padded = torch.zeros(B, k_max, dtype=torch.long, device=device)
    k_i = torch.zeros(B, dtype=torch.long, device=device)
    finished = torch.zeros(B, dtype=torch.bool, device=device)

    for step in range(k_max):
        if stop_rule == "sample":
            nl = next_logits / max(temperature, 1e-5)
            if top_k and top_k > 0:
                v, _ = torch.topk(nl, min(top_k, nl.size(-1)))
                nl = torch.where(nl < v[:, -1:], torch.full_like(nl, -float("inf")), nl)
            probs = F.softmax(nl, dim=-1)
            tok = torch.multinomial(probs, num_samples=1).squeeze(1)
            is_eos = (tok == eos_id)
        elif stop_rule == "oracle":
            is_eos = (k_i >= oracle_k)
            nl = next_logits.clone()
            nl[:, eos_id] = -float("inf")
            nl = nl / max(temperature, 1e-5)
            if top_k and top_k > 0:
                v, _ = torch.topk(nl, min(top_k, nl.size(-1)))
                nl = torch.where(nl < v[:, -1:], torch.full_like(nl, -float("inf")), nl)
            probs = F.softmax(nl, dim=-1)
            tok = torch.multinomial(probs, num_samples=1).squeeze(1)
        else:
            is_eos = (next_logits.argmax(dim=-1) == eos_id)
            nl = next_logits.clone()
            nl[:, eos_id] = -float("inf")
            nl = nl / max(temperature, 1e-5)
            if top_k and top_k > 0:
                v, _ = torch.topk(nl, min(top_k, nl.size(-1)))
                nl = torch.where(nl < v[:, -1:], torch.full_like(nl, -float("inf")), nl)
            probs = F.softmax(nl, dim=-1)
            tok = torch.multinomial(probs, num_samples=1).squeeze(1)

        newly_finished = is_eos & (~finished)
        active = (~finished)
        record = active & (~is_eos)
        if record.any():
            rows = record.nonzero(as_tuple=False).squeeze(1)
            cursors = k_i[rows]
            fine_ids_padded[rows, cursors] = tok[rows].clamp(0, pad_id - 1)
            k_i[rows] = cursors + 1
        finished = finished | newly_finished

        if finished.all():
            break
        if step == k_max - 1:
            break

        nxt_tok = torch.where(record, tok, torch.full_like(tok, pad_id)).unsqueeze(1)
        pos_idx = (k_i - 1).clamp(min=0)
        sel_here = selected_pos.gather(1, pos_idx.unsqueeze(1)).squeeze(1)
        nxt_src = torch.where(record, fine_source_offset + sel_here,
                              torch.full_like(sel_here, PAD_SOURCE_POS)).unsqueeze(1)
        nxt_typ = torch.where(record, torch.full_like(tok, TYPE_FINE),
                              torch.full_like(tok, TYPE_SPECIAL)).unsqueeze(1)

        logits, _ = model(input_ids=nxt_tok, source_pos=nxt_src.to(torch.long),
                          type_ids=nxt_typ, kv_caches=kv_caches, start_pos=cur_pos)
        cur_pos += 1
        next_logits = logits[:, -1, :]

    return fine_ids_padded, k_i


def _sample_next(next_logits, temperature, top_k):
    nl = next_logits / max(temperature, 1e-5)
    if top_k and top_k > 0:
        v, _ = torch.topk(nl, min(top_k, nl.size(-1)))
        nl = torch.where(nl < v[:, -1:], torch.full_like(nl, -float("inf")), nl)
    probs = F.softmax(nl, dim=-1)
    return torch.multinomial(probs, num_samples=1)


@torch.no_grad()
def packed_gencoarse_generate(model, bos_id, B, layout,
                              temperature, top_k, eos_id, device):
    model.eval()
    C = layout.coarse_tokens
    Fn = layout.fine_tokens
    coarse_off = layout.coarse_source_offset
    fine_off = layout.fine_source_offset
    kv_caches = [{} for _ in range(model.cfg.n_layers)]

    ids = torch.full((B, 1), bos_id, dtype=torch.long, device=device)
    pos = torch.full((B, 1), BOS_SOURCE_POS, dtype=torch.long, device=device)
    typ = torch.full((B, 1), TYPE_SPECIAL, dtype=torch.long, device=device)
    logits, _ = model(input_ids=ids, source_pos=pos, type_ids=typ,
                      kv_caches=kv_caches, start_pos=0)
    cur_pos = 1
    next_logits = logits[:, -1, :]

    coarse_out = []
    fine_out = []
    total = C + Fn
    for step in range(total + 1):
        next_tok = _sample_next(next_logits, temperature, top_k)
        if step < C:
            coarse_out.append(next_tok)
            nxt_src = torch.full((B, 1), coarse_off + step, dtype=torch.long, device=device)
            nxt_typ = torch.full((B, 1), TYPE_COARSE, dtype=torch.long, device=device)
        elif step < total:
            fidx = step - C
            fine_out.append(next_tok)
            nxt_src = torch.full((B, 1), fine_off + fidx, dtype=torch.long, device=device)
            nxt_typ = torch.full((B, 1), TYPE_FINE, dtype=torch.long, device=device)
        else:
            break
        logits, _ = model(input_ids=next_tok, source_pos=nxt_src, type_ids=nxt_typ,
                          kv_caches=kv_caches, start_pos=cur_pos)
        cur_pos += 1
        next_logits = logits[:, -1, :]

    coarse_ids = torch.cat(coarse_out, dim=1)
    fine_ids = torch.cat(fine_out, dim=1)
    return coarse_ids, fine_ids


@torch.no_grad()
def packed_gencoarse_ref_generate(model, bos_id, B, layout, selected_pos, K,
                                  temperature, top_k, eos_id, device,
                                  fine_source_offset: int = FINE_SOURCE_OFFSET,
                                  real_coarse=None, mix_alpha=None):
    model.eval()
    C = layout.coarse_tokens
    coarse_off = layout.coarse_source_offset
    kv_caches = [{} for _ in range(model.cfg.n_layers)]

    ids = torch.full((B, 1), bos_id, dtype=torch.long, device=device)
    pos = torch.full((B, 1), BOS_SOURCE_POS, dtype=torch.long, device=device)
    typ = torch.full((B, 1), TYPE_SPECIAL, dtype=torch.long, device=device)
    logits, _ = model(input_ids=ids, source_pos=pos, type_ids=typ,
                      kv_caches=kv_caches, start_pos=0)
    cur_pos = 1
    next_logits = logits[:, -1, :]

    coarse_out = []
    fine_out = []
    total = C + K
    for step in range(total + 1):
        next_tok = _sample_next(next_logits, temperature, top_k)
        if step < C:
            if mix_alpha is not None and real_coarse is not None and mix_alpha > 0:
                keep_real = (torch.rand(B, 1, device=device) < mix_alpha)
                next_tok = torch.where(keep_real, real_coarse[:, step:step + 1], next_tok)
            coarse_out.append(next_tok)
            nxt_src = torch.full((B, 1), coarse_off + step, dtype=torch.long, device=device)
            nxt_typ = torch.full((B, 1), TYPE_COARSE, dtype=torch.long, device=device)
        elif step < total:
            fidx = step - C
            fine_out.append(next_tok)
            nxt_src = (fine_source_offset + selected_pos[:, fidx]).unsqueeze(1).to(torch.long)
            nxt_typ = torch.full((B, 1), TYPE_FINE, dtype=torch.long, device=device)
        else:
            break
        logits, _ = model(input_ids=next_tok, source_pos=nxt_src, type_ids=nxt_typ,
                          kv_caches=kv_caches, start_pos=cur_pos)
        cur_pos += 1
        next_logits = logits[:, -1, :]

    coarse_ids = torch.cat(coarse_out, dim=1)
    fine_ids = torch.cat(fine_out, dim=1)
    return coarse_ids, fine_ids


if __name__ == "__main__":
    main()
