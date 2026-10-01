# AdapTok (Sec. 4.7)

Upstream: https://github.com/VisionXLab/AdapTok @ `a72076c`, released tokenizer and scorer.

```bash
git clone https://github.com/VisionXLab/AdapTok "$EXT_ROOT/AdapTok" && cd "$EXT_ROOT/AdapTok" && git checkout a72076c
git apply "$REPO/external/adaptok/adaptok_ours.patch"
cp -r "$REPO"/external/adaptok/{*.py,cfgs,data} .
EXT_ROOT=... bash "$REPO/scripts/external/adaptok_run.sh"   # annot train eval harvest
```
