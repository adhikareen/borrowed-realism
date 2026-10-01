# VAR (Table 3, Fig. 6)

Upstream: https://github.com/FoundationVision/VAR @ `78b9539`, VAR-d16 + `vae_ch160v4096z32.pth`.

```bash
git clone https://github.com/FoundationVision/VAR "$EXT_ROOT/VAR" && git -C "$EXT_ROOT/VAR" checkout 78b9539
cp external/var/*.py "$EXT_ROOT/VAR/"
EXT_ROOT=... IMAGENET_TRAIN=... IMAGENET_VAL=... bash scripts/external/var_run.sh   # ref table3 sweep collect
```
