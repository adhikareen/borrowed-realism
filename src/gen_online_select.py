from __future__ import annotations
import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import paths
import argparse, json, time, sys
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
from ar_generate import (load_ckpt, _load_omnitok_decoder,  # noqa: E402
                         reconstruct_grid_from_packed, _sample_next)
from streams import (Layout, TYPE_SPECIAL, TYPE_COARSE, TYPE_FINE,  # noqa: E402
                     BOS_SOURCE_POS, special_ids)
from train_coarse_only_selector import coarse_only_features, FEAT_DIM  # noqa: E402
from train_learned_selector import SelectorMLP  # noqa: E402


def select_from_generated_coarse(coarse_ids, mode, head, emb_np, mu, sd, layout, K, device, gen):
    B = coarse_ids.shape[0]
    if mode == "random":
        scores = torch.rand(B, layout.fine_tokens, device=device, generator=gen)
    else:
        cg = coarse_ids.view(B, layout.latent_t, layout.coarse_h, layout.coarse_w).cpu().numpy()
        X = coarse_only_features(cg, emb_np).reshape(-1, FEAT_DIM)
        Xn = torch.from_numpy((X - mu) / sd).to(device)
        with torch.no_grad():
            scores = head(Xn).view(B, -1).float()
    idx = torch.topk(scores, K, dim=1).indices
    return torch.sort(idx, dim=1).values


@torch.no_grad()
def generate_batch(model, layout, K, B, bos_id, temperature, top_k, device,
                   mode, head, emb_np, mu, sd, gen, n_codes):
    C = layout.coarse_tokens
    coarse_off, fine_off = layout.coarse_source_offset, layout.fine_source_offset
    kv = [{} for _ in range(model.cfg.n_layers)]
    ids = torch.full((B, 1), bos_id, dtype=torch.long, device=device)
    pos = torch.full((B, 1), BOS_SOURCE_POS, dtype=torch.long, device=device)
    typ = torch.full((B, 1), TYPE_SPECIAL, dtype=torch.long, device=device)
    logits, _ = model(input_ids=ids, source_pos=pos, type_ids=typ, kv_caches=kv, start_pos=0)
    cur, nl = 1, logits[:, -1, :]

    coarse_out = []
    for step in range(C):
        t = _sample_next(nl, temperature, top_k)
        coarse_out.append(t)
        s = torch.full((B, 1), coarse_off + step, dtype=torch.long, device=device)
        ty = torch.full((B, 1), TYPE_COARSE, dtype=torch.long, device=device)
        logits, _ = model(input_ids=t, source_pos=s, type_ids=ty, kv_caches=kv, start_pos=cur)
        cur += 1; nl = logits[:, -1, :]
    coarse_ids = torch.cat(coarse_out, dim=1).clamp(0, n_codes - 1)

    selected_pos = select_from_generated_coarse(coarse_ids, mode, head, emb_np, mu, sd,
                                                layout, K, device, gen)

    fine_out = []
    for j in range(K):
        t = _sample_next(nl, temperature, top_k)
        fine_out.append(t)
        s = (fine_off + selected_pos[:, j]).unsqueeze(1).long()
        ty = torch.full((B, 1), TYPE_FINE, dtype=torch.long, device=device)
        logits, _ = model(input_ids=t, source_pos=s, type_ids=ty, kv_caches=kv, start_pos=cur)
        cur += 1; nl = logits[:, -1, :]
    return coarse_ids, torch.cat(fine_out, dim=1).clamp(0, n_codes - 1), selected_pos


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, type=Path)
    ap.add_argument("--mode", choices=["coarse_only", "random"], required=True)
    ap.add_argument("--head", type=Path, default=Path("runs/coarse_only_selector.pt"))
    ap.add_argument("--omnitok-ckpt", default=paths.OMNITOK_CKPT)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--num-samples", type=int, default=2000)
    ap.add_argument("--batch-size", type=int, default=25)
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--top-k", type=int, default=256)
    ap.add_argument("--keep-frac", type=float, default=0.6)
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    t0 = time.time()
    device = "cuda"
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    layout = Layout()
    K = int(round(args.keep_frac * layout.fine_tokens))
    args.out_dir.mkdir(parents=True, exist_ok=True)

    model, cfg, train_manifest, val_manifest = load_ckpt(args.ckpt, device)
    manifest = train_manifest or val_manifest
    base_vocab = int(manifest["base_vocab_size"])
    sp = special_ids(base_vocab)
    bos_id = sp.bos
    dec = _load_omnitok_decoder(args.omnitok_ckpt, device)
    emb_np = dec.model.codebook.embeddings.detach().cpu().numpy().astype(np.float32)

    head = mu = sd = None
    if args.mode == "coarse_only":
        hd = torch.load(args.head, map_location="cpu", weights_only=False)
        head = SelectorMLP(in_dim=hd.get("feat_dim", FEAT_DIM),
                           hidden=hd["args"]["hidden"], layers=hd["args"]["layers"]).to(device)
        head.load_state_dict(hd["state_dict"]); head.eval()
        mu, sd = hd["mu"], hd["sd"]
    gen = torch.Generator(device=device); gen.manual_seed(args.seed)

    print(f"[online] mode={args.mode} K={K} n={args.num_samples} ckpt={args.ckpt}", flush=True)
    n = 0
    while n < args.num_samples:
        B = min(args.batch_size, args.num_samples - n)
        c, f, sp = generate_batch(model, layout, K, B, bos_id, args.temperature, args.top_k,
                                  device, args.mode, head, emb_np, mu, sd, gen, base_vocab)
        c = c.clamp(0, base_vocab - 1)
        f = f.clamp(0, base_vocab - 1)
        grid = reconstruct_grid_from_packed(c, f, sp, device, layout)
        with torch.no_grad():
            vid = dec.decode(grid, is_image=False).float()
        vid = ((vid + 1.0) / 2.0).clamp(0, 1).detach().cpu().numpy().astype(np.float16)
        for i in range(B):
            np.save(args.out_dir / f"gen_{n + i:05d}.npy", vid[i])
        n += B
        print(f"[online] {n}/{args.num_samples}  ({time.time()-t0:.0f}s)", flush=True)
    meta = {"mode": args.mode, "ckpt": str(args.ckpt), "K": K, "n": n,
            "temperature": args.temperature, "top_k": args.top_k, "seed": args.seed,
            "elapsed_sec": time.time() - t0,
            "note": "positions chosen from the GENERATED coarse scaffold only"}
    (args.out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2), flush=True)


if __name__ == "__main__":
    main()
