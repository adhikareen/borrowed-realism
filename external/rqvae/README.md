# RQ-VAE (Table S6)

Upstream: https://github.com/kakaobrain/rq-vae-transformer @ `341395e`, ImageNet RQ-Transformer 480M.

```bash
git clone https://github.com/kakaobrain/rq-vae-transformer "$EXT_ROOT/rqvae" && git -C "$EXT_ROOT/rqvae" checkout 341395e
cp external/rqvae/*.py "$EXT_ROOT/rqvae/"
EXT_ROOT=... IMAGENET_TRAIN=... IMAGENET_VAL=... bash scripts/external/rqvae_run.sh
```
