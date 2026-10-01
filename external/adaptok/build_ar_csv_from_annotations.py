import argparse
import csv
import os

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--annot", required=True)
    ap.add_argument("--src_csv", required=True)
    ap.add_argument("--out_csv", required=True)
    args = ap.parse_args()

    d = torch.load(args.annot, weights_only=False)
    annotated = set()
    for _id in d["id"]:
        annotated.add(_id.split("frames__")[-1])
    print(f"annotation has {len(annotated)} unique video basenames")

    kept = []
    with open(args.src_csv) as f:
        r = csv.reader(f)
        header = next(r)
        for row in r:
            base = os.path.basename(row[1])
            if base in annotated:
                kept.append(row)
    kept.sort(key=lambda x: (int(x[3]), x[1]))
    with open(args.out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for i, row in enumerate(kept):
            w.writerow([i, row[1], row[2], row[3]])
    print(f"wrote {args.out_csv} with {len(kept)} annotated videos (from {args.src_csv})")


if __name__ == "__main__":
    main()
