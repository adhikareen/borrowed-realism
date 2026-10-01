from __future__ import annotations
import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import paths
import sys
import torch

SEED_ROOT = paths.SEED_VOKEN
OM2_CONFIG = f"{SEED_ROOT}/configs/Open-MAGVIT2/gpu/ucf101_lfqfan_128_L.yaml"
OM2_CKPT = paths.OM2_CKPT
OM2_ZCH = 18
OM2_VOCAB = 2 ** OM2_ZCH


def _import_seed_vqmodel():
    import importlib
    src_dir = str(__import__("pathlib").Path(__file__).resolve().parent)
    repo_root = str(__import__("pathlib").Path(__file__).resolve().parent.parent)
    saved_path = list(sys.path)
    saved_mods = {k: sys.modules[k] for k in list(sys.modules)
                  if k == "src" or k.startswith("src.")}
    try:
        sys.path = [p for p in sys.path
                    if p not in ("", ".", src_dir, repo_root)]
        for k in list(saved_mods):
            del sys.modules[k]
        sys.path.insert(0, SEED_ROOT)
        mod = importlib.import_module("src.Open_MAGVIT2.models.video_lfqgan")
        return mod.VQModel
    finally:
        sys.path = saved_path
        for k, v in saved_mods.items():
            sys.modules[k] = v


def load_om2(ckpt_path: str = OM2_CKPT, config_path: str = OM2_CONFIG,
             device: torch.device | str = "cuda") -> torch.nn.Module:
    from omegaconf import OmegaConf
    VQModel = _import_seed_vqmodel()

    config = OmegaConf.load(config_path)
    init_args = OmegaConf.to_container(config.model.init_args, resolve=True)
    init_args["image_pretrain_path"] = None
    model = VQModel(**init_args)
    state = torch.load(ckpt_path, map_location="cpu")
    sd = state.get("state_dict", state)
    model_keys = set(model.state_dict().keys())
    sd = {k: v for k, v in sd.items() if k in model_keys}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"[om2] loaded ckpt: missing={len(missing)} unexpected={len(unexpected)}", flush=True)
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


@torch.no_grad()
def om2_encode_indices(model, videos_pm1: torch.Tensor) -> torch.Tensor:
    quant, _, _, _ = model.encode(videos_pm1.float())
    bits = (quant > 0).to(torch.int64)
    weights = (1 << torch.arange(OM2_ZCH, device=quant.device, dtype=torch.int64))
    idx = (bits * weights.view(1, -1, 1, 1, 1)).sum(dim=1)
    return idx.to(torch.int32)


@torch.no_grad()
def om2_indices_to_features(indices: torch.Tensor) -> torch.Tensor:
    idx = indices.to(torch.long)
    B, T, H, W = idx.shape
    shifts = torch.arange(OM2_ZCH, device=idx.device, dtype=torch.long)
    bits = ((idx.unsqueeze(1) >> shifts.view(1, -1, 1, 1, 1)) & 1).float()
    return 2.0 * bits - 1.0


@torch.no_grad()
def om2_decode_indices(model, indices: torch.Tensor) -> torch.Tensor:
    quant = om2_indices_to_features(indices).to(next(model.parameters()).device)
    rec = model.decode(quant)
    return rec.clamp(-1, 1)


def om2_sign_codebook(device="cpu") -> torch.Tensor:
    idx = torch.arange(OM2_VOCAB, dtype=torch.long, device=device)
    shifts = torch.arange(OM2_ZCH, dtype=torch.long, device=device)
    bits = ((idx.unsqueeze(1) >> shifts.unsqueeze(0)) & 1).float()
    return 2.0 * bits - 1.0
