from __future__ import annotations
import argparse, json
from pathlib import Path

import build_stream_cache_omni_long as B

OMAG_VOCAB = 262144


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--sel", choices=["full", "l2gap", "random", "decode", "dense", "learned", "contigmax"],
                    required=True)
    ap.add_argument("--keep-frac", type=float, default=0.60)
    ap.add_argument("--num-videos", type=int, default=100000)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--pool-size", type=int, default=2)
    args = ap.parse_args()

    B.OMNI_VOCAB = OMAG_VOCAB
    manifest = B.build(args.raw_dir, args.out_dir, args.sel, args.keep_frac,
                       args.num_videos, args.seed, args.pool_size)
    mpath = args.out_dir / "manifest.json"
    m = json.loads(mpath.read_text())
    m["tokenizer"] = "OpenMAGVIT2_video_LFQ_262144"
    assert m["base_vocab_size"] == OMAG_VOCAB, m["base_vocab_size"]
    assert m["vocab_size"] == OMAG_VOCAB + 3
    mpath.write_text(json.dumps(m, indent=2))
    print(f"[omag-cache] manifest tokenizer tag fixed -> {m['tokenizer']}", flush=True)


if __name__ == "__main__":
    main()
