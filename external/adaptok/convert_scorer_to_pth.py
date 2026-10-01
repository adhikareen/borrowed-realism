import argparse
import json
import os

import torch
from safetensors.torch import load_file


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="ckpts/scorer", help="dir with model.safetensors + config.json")
    ap.add_argument("--out", default="ckpts/scorer/scorer_labeler.pth")
    args = ap.parse_args()

    sd = load_file(os.path.join(args.src, "model.safetensors"))
    with open(os.path.join(args.src, "config.json")) as f:
        cfg = json.load(f)
    cfg.pop("transformers_version", None)

    ckpt = {
        "model": {"name": "adaptok", "args": cfg, "sd": sd},
        "epoch": 0,
        "iter": 0,
    }
    torch.save(ckpt, args.out)
    print(f"wrote {args.out} | {len(sd)} tensors | config keys: {len(cfg)}")


if __name__ == "__main__":
    main()
