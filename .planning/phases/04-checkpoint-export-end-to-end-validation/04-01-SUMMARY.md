# Phase 04-01 Summary: Patch trainer.py → best_checkpoint.pth

**Status:** Complete  
**Commit:** bc948ae  
**Date:** 2025-05-28  
**Requirements closed:** CKPT-01, CKPT-02

## What Was Built

Patched `sam3/train/trainer.py::save_checkpoint()` to export a model-weights-only `best_checkpoint.pth` file whenever a best-meter checkpoint fires during training.

**Patch location:** Lines 377–402 of `sam3/train/trainer.py`, inserted after the existing `for checkpoint_path in checkpoint_paths:` loop.

**Export format:** `{"model": {"detector.<key>": <tensor>}}` — satisfies all three constraints of `_load_checkpoint`:
1. `weights_only=True` — only tensors in the dict, no Python objects (no optimizer, epoch, loss, scaler)
2. `"model"` top-level key — `_load_checkpoint` extracts `ckpt["model"]`
3. `"detector."` prefix — `_load_checkpoint` filters for keys containing "detector"

**Discrimination guard:** Set intersection of `checkpoint_names` with normalized `save_best_meters` keys (`val_custom/detection` → `val_custom_detection`). Regular epoch saves use `["checkpoint"]` — never match. Best-meter saves use `["val_custom_detection"]` — always match.

## Files Modified

| File | Change |
|------|--------|
| `sam3/train/trainer.py` | +24 lines in `save_checkpoint()` after the existing checkpoint loop |

## Acceptance Criteria — All Passed

- ✅ `best_checkpoint.pth` string at line 397 (> 375 — after existing loop)
- ✅ Python AST syntax check: OK
- ✅ `_best_keys_norm` present (1 assignment + 1 usage = 2 occurrences)
- ✅ `distributed_rank != 0` count: exactly 1 (no duplicate rank guard added)
- ✅ Patch is after `for checkpoint_path in checkpoint_paths:` loop

## Self-Check: PASSED
