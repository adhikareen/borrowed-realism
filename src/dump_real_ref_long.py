import argparse, sys
from pathlib import Path
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", required=True, type=Path, help="long extract dir with shards/ (must have frames)")
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--limit", type=int, default=2000)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    shards = sorted((args.raw_dir / "shards").glob("shard_*.npz"))
    if not shards:
        raise FileNotFoundError(f"no shards under {args.raw_dir}/shards")
    written = 0
    for sh in shards:
        with np.load(sh) as d:
            if "frames" not in d.files:
                raise ValueError(f"{sh} has no 'frames' — re-extract with --store-frames")
            frames = d["frames"]
            for i in range(frames.shape[0]):
                np.save(args.out_dir / f"real_{written:07d}.npy", frames[i].astype(np.float16, copy=False))
                written += 1
                if written >= args.limit:
                    print(f"[real-ref] wrote {written} to {args.out_dir}", flush=True)
                    return
    print(f"[real-ref] wrote {written} to {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
