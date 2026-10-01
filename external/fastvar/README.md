# FastVAR / Infinity-2B (Table S8)

Upstream: https://github.com/csguoh/FastVAR @ `de2b489`, Infinity-2B weights, MJHQ-30K. Unmodified; real coarse scales via `gt_leak`.

```bash
git clone https://github.com/csguoh/FastVAR "$EXT_ROOT/FastVAR" && git -C "$EXT_ROOT/FastVAR" checkout de2b489
cp external/fastvar/infinity_leak_exp.py "$EXT_ROOT/FastVAR/Infinity/"
EXT_ROOT=... bash scripts/external/fastvar_run.sh   # run1 runs23 dial collect
```
