from __future__ import annotations
import argparse, json, time
from pathlib import Path
import numpy as np
from tqdm import tqdm
from scipy.stats import mode as _scipy_mode

from streams import (Layout, build_packed_sequence_pool, build_dense_sequence_pool,
                     special_ids)

OMNI_VOCAB = 8192


def coarse_mode_pool(indices, latent_t, latent_h, latent_w, pool):
    B = indices.shape[0]
    ch, cw = latent_h // pool, latent_w // pool
    win = (indices.reshape(B, latent_t, ch, pool, cw, pool)
                  .transpose(0, 1, 2, 4, 3, 5).reshape(B, latent_t, ch, cw, pool * pool))
    cmode = _scipy_mode(win, axis=-1, keepdims=False).mode.astype(np.int32)
    return cmode.reshape(B, -1)


def iter_long_shards(raw_dir, num_videos):
    shards = sorted((raw_dir / "shards").glob("shard_*.npz"))
    if not shards:
        raise FileNotFoundError(f"no shards under {raw_dir}/shards")
    remaining = num_videos
    for sh in shards:
        with np.load(sh) as d:
            cnt = int(d["indices"].shape[0])
        take = cnt if remaining is None else min(remaining, cnt)
        if take <= 0:
            break
        yield sh, take
        if remaining is not None:
            remaining -= take
            if remaining <= 0:
                break


GENSCORE_ALPHA = {"genscore": 1.0, "genscore05": 0.5}
SAL_SELECTORS = ("gensal", "attninf")


def contigmax_fixed_mask(latent_t: int, latent_h: int, latent_w: int, K: int) -> np.ndarray:
    h = np.arange(latent_h)[:, None]
    w = np.arange(latent_w)[None, :]
    ch, cw = (latent_h - 1) / 2.0, (latent_w - 1) / 2.0
    dist2d = (h - ch) ** 2 + (w - cw) ** 2
    dist = np.broadcast_to(dist2d[None], (latent_t, latent_h, latent_w))
    order = np.argsort(dist.reshape(-1), kind="stable")
    return np.sort(order[:K]).astype(np.int64)


def smooth_scores_3x3(score_flat: np.ndarray, latent_t: int, latent_h: int, latent_w: int) -> np.ndarray:
    from scipy.ndimage import uniform_filter
    s = score_flat.reshape(latent_t, latent_h, latent_w).astype(np.float32)
    return uniform_filter(s, size=(1, 3, 3), mode="nearest").reshape(-1)


def _zscore(x: np.ndarray) -> np.ndarray:
    m = x.mean()
    s = x.std()
    return (x - m) / max(s, 1e-8)


def build(raw_dir: Path, out_dir: Path, sel: str, keep_frac: float,
          num_videos: int, seed: int, pool: int = 2, nll_dir: Path | None = None,
          sal_dir: Path | None = None):
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    shards = sorted((raw_dir / "shards").glob("shard_*.npz"))
    with np.load(shards[0]) as d0:
        idx_shape = d0["indices"].shape
        has_gap = "l2gap" in d0.files
        has_dec = "decode_imp" in d0.files
        has_pred = "pred" in d0.files
        has_sgimp = "sg_imp" in d0.files
    latent_t, latent_h, latent_w = int(idx_shape[1]), int(idx_shape[2]), int(idx_shape[3])
    layout = Layout(pool_size=pool, latent_t=latent_t, latent_h=latent_h, latent_w=latent_w)
    fine_n = layout.fine_tokens
    coarse_n = layout.coarse_tokens

    is_dense = (sel == "dense")
    if sel in ("full", "dense"):
        K = fine_n
    else:
        K = int(round(keep_frac * fine_n))
        K = max(1, min(K, fine_n))
    seq_len = layout.dense_seq_len() if is_dense else layout.packed_seq_len(K)
    if sel in ("l2gap",) and not has_gap:
        raise ValueError("l2gap selection requires l2gap in shards")
    if sel == "decode" and not has_dec:
        raise ValueError("decode selection requires decode_imp in shards "
                         "(re-extract with src/extract_omnitok_decode.py)")
    if sel == "learned" and not has_pred:
        raise ValueError("learned selection requires `pred` in shards "
                         "(produce with src/train_learned_selector.py)")
    if sel == "sgoracle" and not has_sgimp:
        raise ValueError("sgoracle selection requires `sg_imp` in shards "
                         "(produce with src/extract_sg_importance.py)")
    if sel in GENSCORE_ALPHA:
        if not has_dec:
            raise ValueError(f"{sel} requires decode_imp in shards")
        if nll_dir is None or not (Path(nll_dir) / "shards").is_dir():
            raise ValueError(f"{sel} requires --nll-dir pointing at per-cell NLL shards "
                             "(produce with src/extract_cell_nll.py)")
    if sel in SAL_SELECTORS:
        if sal_dir is None or not (Path(sal_dir) / "shards").is_dir():
            raise ValueError(f"{sel} requires --sal-dir pointing at per-cell score shards "
                             "(produce with src/extract_gensal.py or src/extract_attninf.py)")
    if sel == "l2gap_smooth" and not has_gap:
        raise ValueError("l2gap_smooth selection requires l2gap in shards")
    contig_selpos = contigmax_fixed_mask(latent_t, latent_h, latent_w, K) if sel == "contigmax" else None

    print(f"[long-cache] sel={sel} latent=({latent_t},{latent_h},{latent_w}) coarse_n={coarse_n} "
          f"fine_n={fine_n} K={K} keep={K/fine_n:.3f} seq_len={seq_len}", flush=True)

    take_list = list(iter_long_shards(raw_dir, num_videos))
    total = sum(t for _, t in take_list)
    print(f"[long-cache] packing {total} examples", flush=True)

    input_ids = np.lib.format.open_memmap(out_dir / "input_ids.npy", mode="w+",
                                          dtype=np.int32, shape=(total, seq_len))
    type_ids = np.lib.format.open_memmap(out_dir / "type_ids.npy", mode="w+",
                                         dtype=np.uint8, shape=(total, seq_len))
    source_pos = np.lib.format.open_memmap(out_dir / "source_pos.npy", mode="w+",
                                           dtype=np.int32, shape=(total, seq_len))

    cursor = 0
    for sh, take in tqdm(take_list, desc=f"build-long[{sel}]"):
        with np.load(sh) as d:
            indices = d["indices"][:take].astype(np.int32)
            gap = d["l2gap"][:take].astype(np.float32) if (has_gap and sel in ("l2gap", "l2gap_smooth")) else None
            dec = d["decode_imp"][:take].astype(np.float32) \
                if (has_dec and sel in ("decode", "decode_deranged") + tuple(GENSCORE_ALPHA)) else None
            prd = d["pred"][:take].astype(np.float32) if (has_pred and sel == "learned") else None
            sgi = d["sg_imp"][:take].astype(np.float32) if (has_sgimp and sel == "sgoracle") else None
        nll = None
        if sel in GENSCORE_ALPHA:
            nll_path = Path(nll_dir) / "shards" / sh.name
            with np.load(nll_path) as dn:
                nll = dn["nll"][:take].astype(np.float32)
            if nll.shape != indices.shape:
                raise ValueError(f"nll shard {nll_path} shape {nll.shape} != indices {indices.shape}")
        sal = None
        if sel in SAL_SELECTORS:
            sal_path = Path(sal_dir) / "shards" / sh.name
            with np.load(sal_path) as ds:
                sal = ds["score"][:take].astype(np.float32)
            if sal.shape != indices.shape:
                raise ValueError(f"sal shard {sal_path} shape {sal.shape} != indices {indices.shape}")
        B = take
        coarse_flat = coarse_mode_pool(indices, latent_t, latent_h, latent_w, pool)
        fine_flat = indices.reshape(B, -1)
        gap_flat = gap.reshape(B, -1) if gap is not None else None
        dec_flat = dec.reshape(B, -1) if dec is not None else None
        derange_idx = (np.arange(B) + 1) % B
        prd_flat = prd.reshape(B, -1) if prd is not None else None
        sgi_flat = sgi.reshape(B, -1) if sgi is not None else None
        nll_flat = nll.reshape(B, -1) if nll is not None else None
        sal_flat = sal.reshape(B, -1) if sal is not None else None

        if is_dense:
            for j in range(B):
                seq, typ, pos = build_dense_sequence_pool(
                    fine_flat=fine_flat[j], base_vocab_size=OMNI_VOCAB, layout=layout)
                input_ids[cursor] = seq
                type_ids[cursor] = typ
                source_pos[cursor] = pos
                cursor += 1
            continue

        for j in range(B):
            if sel == "full":
                selpos = np.arange(fine_n, dtype=np.int64)
            elif sel == "l2gap":
                selpos = np.argpartition(gap_flat[j], -K)[-K:]
                selpos.sort()
            elif sel == "decode":
                selpos = np.argpartition(dec_flat[j], -K)[-K:]
                selpos.sort()
            elif sel == "decode_deranged":
                src = int(derange_idx[j])
                selpos = np.argpartition(dec_flat[src], -K)[-K:]
                selpos.sort()
            elif sel == "learned":
                selpos = np.argpartition(prd_flat[j], -K)[-K:]
                selpos.sort()
            elif sel == "sgoracle":
                selpos = np.argpartition(sgi_flat[j], -K)[-K:]
                selpos.sort()
            elif sel in GENSCORE_ALPHA:
                alpha = GENSCORE_ALPHA[sel]
                score = _zscore(dec_flat[j]) - alpha * _zscore(nll_flat[j])
                selpos = np.argpartition(score, -K)[-K:]
                selpos.sort()
            elif sel in SAL_SELECTORS:
                selpos = np.argpartition(sal_flat[j], -K)[-K:]
                selpos.sort()
            elif sel == "contigmax":
                selpos = contig_selpos
            elif sel == "l2gap_smooth":
                sm = smooth_scores_3x3(gap_flat[j], latent_t, latent_h, latent_w)
                selpos = np.argpartition(sm, -K)[-K:]
                selpos.sort()
            elif sel == "taildrop":
                selpos = np.arange(K, dtype=np.int64)
            elif sel == "random":
                selpos = rng.choice(fine_n, size=K, replace=False); selpos.sort()
            else:
                raise ValueError(sel)
            seq, typ, pos = build_packed_sequence_pool(
                coarse_flat=coarse_flat[j], fine_flat=fine_flat[j],
                selected_pos=selpos, base_vocab_size=OMNI_VOCAB, layout=layout)
            input_ids[cursor] = seq
            type_ids[cursor] = typ
            source_pos[cursor] = pos
            cursor += 1

    sids = special_ids(OMNI_VOCAB)
    clip_frames = (latent_t - 1) * 4 + 1
    manifest = {
        "kind": "compact_token_lm_stream_cache",
        "mode": "dense" if is_dense else "packed",
        "cache_dir": str(raw_dir),
        "tokenizer": "OmniTokenizer_VQGAN_K600_8K_LONG",
        "num_examples": int(total),
        "latent_frames": int(latent_t),
        "clip_frames": int(clip_frames),
        "base_vocab_size": OMNI_VOCAB,
        "vocab_size": OMNI_VOCAB + 3,
        "pad_token_id": sids.pad, "bos_token_id": sids.bos, "eos_token_id": sids.eos,
        "seq_len": int(seq_len),
        "tpf": float(fine_n / clip_frames) if is_dense else float((coarse_n + K) / clip_frames),
        "packed_fine_budget": int(fine_n if is_dense else K),
        "keep_frac": float(K / fine_n),
        "selector_name": sel,
        "random_seed": int(seed),
        **layout.to_manifest_fields(),
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps({k: manifest[k] for k in
                      ["selector_name", "latent_frames", "clip_frames", "packed_fine_budget",
                       "keep_frac", "seq_len", "tpf", "num_examples"]}, indent=2), flush=True)
    return manifest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--sel", choices=["full", "l2gap", "random", "decode", "decode_deranged", "dense", "learned",
                                      "genscore", "genscore05", "gensal", "attninf",
                                      "contigmax", "l2gap_smooth", "sgoracle", "taildrop"], required=True)
    ap.add_argument("--keep-frac", type=float, default=0.25)
    ap.add_argument("--num-videos", type=int, default=100000)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--pool-size", type=int, default=2)
    ap.add_argument("--nll-dir", type=Path, default=None,
                    help="per-cell NLL dir")
    ap.add_argument("--sal-dir", type=Path, default=None,
                    help="per-cell score dir")
    args = ap.parse_args()
    build(args.raw_dir, args.out_dir, args.sel, args.keep_frac,
          args.num_videos, args.seed, args.pool_size, nll_dir=args.nll_dir,
          sal_dir=args.sal_dir)


if __name__ == "__main__":
    main()
