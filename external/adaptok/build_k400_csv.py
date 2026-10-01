import csv
import os
import sys
from pathlib import Path

ADAPTOK_ROOT = Path(__file__).resolve().parent
K400_ANN = Path("data/k400/annotations")
FILELISTS = {
    "k400_train": Path("data/k400/filelists/k400_train.txt"),
    "k400_val": Path("data/k400/filelists/k400_val.txt"),
}


def load_ytid2action():
    yt2act = {}
    for split in ["train", "val", "test"]:
        csv_path = K400_ANN / f"{split}.csv"
        if not csv_path.exists():
            continue
        with csv_path.open() as f:
            r = csv.DictReader(f)
            for row in r:
                yt2act[row["youtube_id"]] = row["label"].strip()
    return yt2act


def build_action2label(yt2act):
    actions = sorted(set(yt2act.values()))
    return {a: i for i, a in enumerate(actions)}, actions


def ytid_from_path(p):
    stem = Path(p).stem
    parts = stem.rsplit("_", 2)
    if len(parts) == 3:
        return parts[0]
    return stem


def main():
    yt2act = load_ytid2action()
    action2label, actions = build_action2label(yt2act)
    print(f"K400: {len(actions)} unique action classes; {len(yt2act)} ytid->action entries")

    out_dir = ADAPTOK_ROOT / "data" / "metadata"
    out_dir.mkdir(parents=True, exist_ok=True)

    for name, fl in FILELISTS.items():
        rows = []
        n_missing = 0
        with fl.open() as f:
            paths = [ln.strip() for ln in f if ln.strip()]
        for path in paths:
            ytid = ytid_from_path(path)
            act = yt2act.get(ytid)
            if act is None:
                n_missing += 1
                continue
            label = action2label[act]
            rows.append((path, act, label))
        rows.sort(key=lambda r: (r[2], r[0]))
        out_csv = out_dir / f"{name}.csv"
        with out_csv.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["id", "path", "action", "label"])
            for i, (path, act, label) in enumerate(rows):
                w.writerow([i, path, act, label])
        print(f"{name}: wrote {len(rows)} rows to {out_csv} (missing action for {n_missing})")


if __name__ == "__main__":
    main()
