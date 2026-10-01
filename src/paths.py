import os

WORK = os.environ.get("BR_WORK", "third_party")
OMNITOK_DIR = os.environ.get("OMNITOK_DIR", os.path.join(WORK, "OmniTokenizer"))
OMNITOK_CKPT = os.environ.get("OMNITOK_CKPT", os.path.join(OMNITOK_DIR, "weights", "imagenet_k600.ckpt"))
SEED_VOKEN = os.environ.get("SEED_VOKEN_DIR", os.path.join(WORK, "SEED-Voken"))
OM2_CKPT = os.environ.get("OM2_CKPT", os.path.join(WORK, "checkpoints", "OpenMAGVIT2", "video_128_262144.ckpt"))
