# Phase 04-02 Summary: generate_fake_dataset.py

**Status:** Complete  
**Commit:** 5f4a450  
**Date:** 2025-05-28  
**Requirements closed:** VAL-01

## What Was Built

Created `scripts/generate_fake_dataset.py` — a standalone script that generates a minimal synthetic COCO dataset for CI smoke-testing the end-to-end training pipeline without real data.

**Output:**
- `<out>/images/fake_0001.png` through `fake_0005.png` — 5 synthetic 64×64 RGB PNGs
- `<out>/train.json` and `<out>/val.json` — valid COCO JSON (both contain all 5 images)

**COCO format:** 1-based IDs, `[x, y, w, h]` bbox, 8-element rectangle polygon segmentation, `iscrowd=0`, single `{"id":1,"name":"defect","supercategory":"defect"}` category.

## Files Created

| File | Description |
|------|-------------|
| `scripts/generate_fake_dataset.py` | 119 lines; CLI: `--out` (required), `--n-images` (default 5), `--img-size` (default 64) |

## Acceptance Criteria — All Passed

- ✅ Syntax: OK
- ✅ Exit 0 with `--out /tmp/fake_defect_ci`
- ✅ Output: `train.json`, `val.json`, `images/fake_0001.png`–`fake_0005.png`
- ✅ JSON: 5 images, 5 annotations, 1 category — all 1-based IDs
- ✅ Segmentation: 8-element rectangle polygon per annotation
- ✅ `generate_fake_dataset` appears 4× (def + docstring + 2 calls)

## Self-Check: PASSED
