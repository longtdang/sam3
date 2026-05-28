# Phase 04-03 Summary: test_checkpoint_compatibility.py

**Status:** Complete  
**Commit:** 5f5c8aa  
**Date:** 2025-05-28  
**Requirements closed:** CKPT-02, VAL-02

## What Was Built

Created `scripts/test_checkpoint_compatibility.py` — a standalone smoke test that loads `best_checkpoint.pth` via `build_sam3_image_model()` and asserts the model has non-zero parameters and is in eval mode.

**Invocation:** `python scripts/test_checkpoint_compatibility.py --checkpoint /path/to/best_checkpoint.pth`

**Exit 0 (pass):**
```
[OK] Loaded checkpoint: /path/to/best_checkpoint.pth
[OK] Model params: <N>
```

**Exit 1 (fail):** `[FAIL] <error>` to stderr

**Load call:** `build_sam3_image_model(checkpoint_path=..., enable_segmentation=True, load_from_HF=False, device="cpu", eval_mode=True)` — exercises the full `_load_checkpoint` code path including `weights_only=True` and `"detector."` prefix filtering.

**No forward pass:** `Sam3Image.forward()` requires `BatchedDatapoint`; user approved `len(model.state_dict()) > 0` substitution (D-04-03 revised 2025-05-28).

## Files Created

| File | Description |
|------|-------------|
| `scripts/test_checkpoint_compatibility.py` | 75 lines; `--checkpoint` required arg |

## Acceptance Criteria — All Passed

- ✅ Syntax: OK
- ✅ `build_sam3_image_model` appears 4× (import comment, docstring, call)
- ✅ `load_from_HF=False` present (1 occurrence)
- ✅ `device="cpu"` present
- ✅ `[OK] Loaded checkpoint` and `[OK] Model params` output lines present
- ✅ `[FAIL]` stderr output present
- ✅ `sys.exit(1)` on failure
- ✅ `_SAM3_ROOT` bootstrap present
- ✅ Missing checkpoint → exit 1 + `[FAIL]` to stderr

## Self-Check: PASSED
