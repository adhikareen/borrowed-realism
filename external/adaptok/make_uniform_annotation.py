#!/usr/bin/env python3
import argparse

import torch


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="adaptive ILP-212 annotation (.pt)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--per_block", type=int, default=212)
    ap.add_argument("--check", default=None, help="uniform annotation to compare")
    a = ap.parse_args()
    ann = torch.load(a.src, map_location="cpu", weights_only=False)
    uni = {"id": ann["id"], "latent_nums": torch.full_like(ann["latent_nums"], a.per_block),
           "bottleneck_rep": ann["bottleneck_rep"]}
    torch.save(uni, a.out)
    print(f"wrote {a.out}: {len(uni['id'])} clips, latent_nums == {a.per_block} per block")
    if a.check:
        ref = torch.load(a.check, map_location="cpu", weights_only=False)
        same = (ref["id"] == uni["id"] and torch.equal(ref["latent_nums"], uni["latent_nums"])
                and torch.equal(ref["bottleneck_rep"], uni["bottleneck_rep"]))
        print("identical to --check file:", same)


if __name__ == "__main__":
    main()
