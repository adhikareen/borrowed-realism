#!/usr/bin/env python3
import argparse, json, os, sys

BOOL = {"amp_bf16", "untied_embeddings", "clamp_fine_to_coarse"}
PATHS = {"train_dir", "val_dir", "out_dir", "resume_from"}


def resolve(p, runs, repo):
    p = p.replace("${RUNS}", runs).replace("${REPO}", repo)
    if p.startswith("runs/"):
        p = os.path.join(runs, p[len("runs/"):])
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--runs", required=True)
    ap.add_argument("--repo", default="")
    ap.add_argument("--set", nargs="*", default=[])
    a = ap.parse_args()
    cfg = json.load(open(a.config))[a.run]
    for kv in a.set:
        k, v = kv.split("=", 1)
        cfg[k] = v
    out = []
    for k, v in cfg.items():
        flag = "--" + k.replace("_", "-")
        if k in BOOL:
            if v:
                out.append(flag)
            continue
        if v is None:
            continue
        if isinstance(v, float) and v.is_integer() and k == "target_train_tokens":
            v = int(v)
        if k in PATHS:
            v = resolve(str(v), a.runs.rstrip("/"), a.repo.rstrip("/"))
        out += [flag, str(v)]
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
