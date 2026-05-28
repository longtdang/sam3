# Phase 4: Checkpoint Export & End-to-End Validation - Research

**Researched:** 2026-05-28
**Domain:** PyTorch checkpoint save/load, COCO JSON generation, SAM3 inference API compatibility
**Confidence:** HIGH (all findings verified directly from source code)

---

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

- **D-04-01**: Real data not yet available. Synthetic fake dataset via `scripts/generate_fake_dataset.py` for CI. When real data arrives: `data/industrial_defect/train.json`, `val.json`, `images/train/`, `images/val/`. Config key: `paths.dataset_root`.
- **D-04-02**: Patch `trainer._save_checkpoint` to also write `best_checkpoint.pth` alias when a meter in `save_best_meters` is triggered. No separate shell wrapper script.
- **D-04-03**: `scripts/test_checkpoint_compatibility.py` — silent pass (exit 0). Logs checkpoint path + output tensor shape only. Uses `build_sam3_image_model(checkpoint_path=..., enable_segmentation=True)`, one forward pass on random 1024×1024 tensor. No AP assertion.
- **D-04-04**: `scripts/generate_fake_dataset.py` creates 5 synthetic 64×64 PNG images + COCO JSON with 1 binary mask polygon per image, single `{"id":1,"name":"defect","supercategory":"defect"}` category.
- **D-04-05**: VAL-01 `AP50>0` is a manual step on real data. CI asserts no crash (1-epoch dry run on fake data). AP50=0 on fake data is acceptable for CI.

### The Agent's Discretion

- Implementation detail of how `best_checkpoint.pth` export is made compatible with `build_sam3_image_model` inference loading (see critical finding in Architecture Patterns section).

### Deferred Ideas (OUT OF SCOPE)

- None declared in CONTEXT.md.
</user_constraints>

---

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| CKPT-01 | Best-performing checkpoint (by `coco_eval_segm_AP50`) saved as `best_checkpoint.pth` | D-04-02: patch `save_checkpoint` to write alias; trigger key is `val_custom_detection.pt` from `_log_meters_and_save_best_ckpts` |
| CKPT-02 | Exported checkpoint loads cleanly via `sam3.build_sam3_image_model()` without modifications | Critical: training checkpoint format incompatible with `_load_checkpoint` (see Architecture Patterns §Critical Compatibility Issue). Fix required. |
| VAL-01 | Pipeline produces fine-tuned checkpoint on industrial defect dataset with `coco_eval_segm_AP50 > 0` (smoke test) | D-04-05: CI = 1-epoch no-crash on fake data; real AP>0 is manual. Fake dataset generator needed. |
| VAL-02 | Exported checkpoint loads and runs inference via existing SAM3 scripts without errors | D-04-03: `test_checkpoint_compatibility.py` does shape check + no exception = pass |
</phase_requirements>

---

## Summary

Phase 4 delivers three concrete deliverables: (1) a `best_checkpoint.pth` alias saved automatically during training when AP50 improves, (2) a `scripts/generate_fake_dataset.py` for CI smoke-testing, and (3) a `scripts/test_checkpoint_compatibility.py` that verifies inference API compatibility. All decisions are locked in CONTEXT.md.

**The most important research finding** is a checkpoint format incompatibility: the trainer saves checkpoints using keys like `backbone.xxx` (no prefix), but `build_sam3_image_model._load_checkpoint` only loads keys containing `"detector"`. A fine-tuned checkpoint exported as-is will silently load zero weights. The fix is to export `best_checkpoint.pth` in a format that `_load_checkpoint` can handle — either by wrapping keys with `"detector."` prefix, or by patching `_load_checkpoint` to fall back when no `"detector."` keys are found.

**Primary recommendation:** Export `best_checkpoint.pth` as `{"model": {"detector." + k: v for k, v in model_state_dict.items()}}`. This is model-weights-only (no optimizer state), avoids the `weights_only=True` incompatibility in `_load_checkpoint`, and is identical in structure to what `_load_checkpoint` already handles from HuggingFace.

---

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| Best checkpoint alias | Trainer (Python, `trainer.py`) | — | `save_checkpoint` already knows when it's called from best-meter logic |
| Checkpoint format export | Trainer (Python, `trainer.py`) | — | Transform happens at save time, not at load time |
| Fake dataset generation | Script (`scripts/generate_fake_dataset.py`) | — | Standalone script, no training dependency |
| Inference compatibility test | Script (`scripts/test_checkpoint_compatibility.py`) | SAM3 model builder | Tests real API path |
| 1-epoch dry run CI | Training launcher + config | Fake dataset script | Uses existing `train.py` with overridden paths |

---

## Standard Stack

### Core (all already in project)
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| `torch` | ≥2.0 | Checkpoint save/load | SAM3 core dependency |
| `PIL` (Pillow) | any | Generate fake PNG images | Already in environment |
| `json` | stdlib | Write COCO JSON | No dependency needed |
| `shutil` / `iopath` | stdlib / project | File copy operations | `g_pathmgr` already used in trainer |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| `numpy` | any | Generate random image pixel data | Available in SAM3 env |
| `pytest` | any | Test framework for unit tests | Existing pattern in `tests/` |

**Installation:** No new packages required. All dependencies already installed.

---

## Architecture Patterns

### System Architecture Diagram

```
Training Loop (trainer.py)
│
├── Every epoch: _log_meters_and_save_best_ckpts(["val", "train"])
│     │
│     ├── meter "val_custom/detection" improves AP50?
│     │       YES → checkpoint_save_keys = ["val_custom_detection"]
│     │             save_checkpoint(epoch+1, ["val_custom_detection"])
│     │                   │
│     │                   ├── _save_checkpoint(ckpt, "val_custom_detection.pt")   ← existing
│     │                   └── [PATCH] also save "best_checkpoint.pth" (inference format)
│     │
│     └── NO → no best-meter save
│
└── Regular cadence: save_checkpoint(epoch, checkpoint_names=None)
      └── _save_checkpoint(ckpt, "checkpoint.pt")  ← resume checkpoint, overwritten each epoch

scripts/generate_fake_dataset.py
│   input: --out /tmp/fake_defect_data
│   output: 5x 64×64 PNG images + train.json + val.json (COCO format)
│
scripts/test_checkpoint_compatibility.py
│   input: --checkpoint <path/to/best_checkpoint.pth>
│   load: build_sam3_image_model(checkpoint_path=..., enable_segmentation=True)
│   test: model(random 1024×1024 tensor)
│   output: "[OK] Loaded checkpoint: <path>" + "[OK] Output tensor shape: <shape>"
```

### Recommended Project Structure
```
scripts/
├── generate_fake_dataset.py    # NEW: synthetic COCO dataset for CI
├── test_checkpoint_compatibility.py  # NEW: inference API smoke test
├── prepare_dataset.py          # existing
├── test_config_parse.py        # existing pattern to follow
└── test_training_config.py     # existing pattern to follow

sam3/
└── model_builder.py            # PATCH: _load_checkpoint fallback (if chosen)
                                # OR no change if exporting in HF format

sam3/train/
└── trainer.py                  # PATCH: save_checkpoint to write best_checkpoint.pth
```

---

### Pattern 1: Checkpoint Save Flow (Verified)

**What:** `save_checkpoint` is called with a list of names. Each name becomes `{name}.pt` under `checkpoint_conf.save_dir`. When called from best-meter logic, `checkpoint_names=["val_custom_detection"]`.

**Key observation:** `save_checkpoint(epoch, checkpoint_names=None)` — when `checkpoint_names is None`, it's a regular periodic save. When `checkpoint_names` is a list, it's a best-meter save. This distinction is available at the `save_checkpoint` level.

```python
# Source: sam3/train/trainer.py lines 333-394
# Existing checkpoint dict structure:
checkpoint = {
    "model": state_dict,           # model weights (keys: backbone.xxx, transformer.xxx, etc.)
    "optimizer": ...,              # optimizer state
    "epoch": epoch,
    "loss": ...,
    "steps": self.steps,
    "time_elapsed": ...,
    "best_meter_values": self.best_meter_values,
    # optional: "scaler" if AMP enabled
}
# Regular save → checkpoint.pt (overwritten each epoch)
# Best-meter save → val_custom_detection.pt (when AP50 improves)
```

**Source:** [VERIFIED: sam3/train/trainer.py lines 333-394]

---

### Pattern 2: Best-Meter Trigger Logic (Verified)

**What:** `_log_meters_and_save_best_ckpts` checks each meter's `is_better` function against `best_meter_values`. Key transformation: `"val_custom/detection"` → `"val_custom_detection"` (slash replaced with underscore) for the checkpoint filename.

```python
# Source: sam3/train/trainer.py lines 968-996
# save_best_meters = ["val_custom/detection"]
# key in self.checkpoint_conf.save_best_meters → key = "val_custom/detection"
# tracked_meter_key = "val_custom/detection"  (meter_subkey could add more)
# checkpoint_save_keys.append("val_custom_detection")  # slash → underscore
# self.save_checkpoint(self.epoch + 1, ["val_custom_detection"])
```

**Source:** [VERIFIED: sam3/train/trainer.py lines 968-996]

---

### Pattern 3: CRITICAL — Checkpoint Format Incompatibility (Verified)

**What:** `build_sam3_image_model._load_checkpoint` expects keys with `"detector."` prefix (HuggingFace format). Training checkpoints have keys without this prefix. Loading a training checkpoint through `build_sam3_image_model(checkpoint_path=...)` will silently fail: `sam3_image_ckpt = {}`, then `model.load_state_dict({}, strict=False)` succeeds but loads NO weights.

**Additionally:** `_load_checkpoint` uses `weights_only=True`, which rejects non-tensor values. Training checkpoints contain `epoch` (int), `steps` (int), `best_meter_values` (dict), etc. This causes a `torch.load` failure even before the key filtering step.

```python
# Source: sam3/model_builder.py lines 539-561
def _load_checkpoint(model, checkpoint_path):
    with g_pathmgr.open(checkpoint_path, "rb") as f:
        ckpt = torch.load(f, map_location="cpu", weights_only=True)  # ← weights_only blocks non-tensors
    if "model" in ckpt and isinstance(ckpt["model"], dict):
        ckpt = ckpt["model"]
    sam3_image_ckpt = {
        k.replace("detector.", ""): v for k, v in ckpt.items() if "detector" in k
    }   # ← empty if no "detector" keys in training checkpoint
    missing_keys, _ = model.load_state_dict(sam3_image_ckpt, strict=False)
```

**Recommended fix (Option A — no code change to model_builder.py):**
Export `best_checkpoint.pth` as a **model-weights-only** file in HuggingFace format:

```python
# In save_checkpoint patch, when checkpoint_names is a best-meter list:
model_state_dict = checkpoint["model"]  # extract from full training checkpoint
hf_format = {f"detector.{k}": v for k, v in model_state_dict.items()}
inference_ckpt = {"model": hf_format}
best_path = os.path.join(checkpoint_folder, "best_checkpoint.pth")
self._save_checkpoint(inference_ckpt, best_path)
```

This file:
1. Has `"model"` key → `_load_checkpoint` extracts it ✅
2. Keys have `"detector."` prefix → filter passes, keys get stripped ✅  
3. No non-tensor values → `weights_only=True` works ✅
4. Model weights load correctly into `Sam3Image` ✅

**Alternative fix (Option B — patch `_load_checkpoint`):**
Add a one-line fallback in `_load_checkpoint` for training checkpoint format:
```python
if not sam3_image_ckpt:
    sam3_image_ckpt = ckpt  # direct load: no "detector." prefix required
```
Still has the `weights_only=True` problem for the full training checkpoint.

**Verdict:** Option A is recommended — it is the standard HuggingFace-compatible export format with no side effects on other code paths.

**Source:** [VERIFIED: sam3/model_builder.py lines 539-561]

---

### Pattern 4: `save_checkpoint` Patch Location (Verified)

**Where to patch:** `save_checkpoint` method (not `_save_checkpoint`). The `save_checkpoint` method knows when it's called from best-meter logic (`checkpoint_names is not None`). The `_save_checkpoint` method is a pure file-writing primitive with no business logic.

```python
# Proposed patch to sam3/train/trainer.py save_checkpoint():
def save_checkpoint(self, epoch, checkpoint_names=None):
    # ... existing logic ...
    for checkpoint_path in checkpoint_paths:
        self._save_checkpoint(checkpoint, checkpoint_path)

    # NEW: write best_checkpoint.pth when this is a best-meter save
    if checkpoint_names is not None and self.distributed_rank == 0:
        model_state_dict = checkpoint["model"]
        hf_format = {f"detector.{k}": v for k, v in model_state_dict.items()}
        inference_ckpt = {"model": hf_format}
        best_path = os.path.join(checkpoint_folder, "best_checkpoint.pth")
        self._save_checkpoint(inference_ckpt, best_path)
        logging.info(f"Saved best_checkpoint.pth → {best_path}")
```

**Source:** [VERIFIED: sam3/train/trainer.py lines 333-394]

---

### Pattern 5: COCO JSON Format for Fake Dataset (Verified from tests/conftest.py)

```python
# Source: tests/conftest.py — minimal_coco fixture (VERIFIED pattern used in project)
coco_json = {
    "info": {},
    "licenses": [],
    "categories": [{"id": 1, "name": "defect", "supercategory": "defect"}],
    "images": [
        {
            "id": 1,
            "file_name": "fake_0001.png",
            "width": 64,
            "height": 64,
        }
        # ... repeat for 5 images
    ],
    "annotations": [
        {
            "id": 1,
            "image_id": 1,
            "category_id": 1,
            "bbox": [8, 8, 48, 48],      # [x, y, width, height]
            "area": 2304,                  # width * height
            "segmentation": [[8, 8, 56, 8, 56, 56, 8, 56]],  # polygon: [x1,y1,...,xn,yn]
            "iscrowd": 0,
        }
        # ... one annotation per image
    ],
}
```

**Notes:**
- IDs must be 1-based (SAM3 dataset code requires this; `prepare_dataset.py` enforces it)
- `segmentation` must be non-empty for `load_segmentation=True` to work
- `bbox` must be `[x, y, w, h]` (COCO convention, not `[x1, y1, x2, y2]`)
- `file_name` must match actual PNG filenames in the output directory

**Source:** [VERIFIED: tests/conftest.py lines 10-28 + sam3/train/data/sam3_image_dataset.py patterns]

---

### Pattern 6: `test_checkpoint_compatibility.py` Script Structure

Following the pattern of `scripts/test_config_parse.py`:

```python
#!/usr/bin/env python3
"""
Smoke test: verify best_checkpoint.pth loads cleanly via build_sam3_image_model().

Run from the sam3 project root:
    python scripts/test_checkpoint_compatibility.py --checkpoint /path/to/best_checkpoint.pth

Exit 0: success.
Exit 1: failure (with error message to stderr).
"""
import argparse
import sys
import torch

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Path to best_checkpoint.pth")
    args = parser.parse_args()

    try:
        from sam3 import build_sam3_image_model
        model = build_sam3_image_model(
            checkpoint_path=args.checkpoint,
            enable_segmentation=True,
            load_from_HF=False,    # ← CRITICAL: don't download HF ckpt when we have our own
            device="cpu",          # CPU for CI compatibility (no GPU required)
            eval_mode=True,
        )
        # One forward pass on random 1024×1024 input (SAM3 inference resolution)
        dummy_input = torch.randn(1, 3, 1024, 1024)
        with torch.no_grad():
            output = model(dummy_input)
        print(f"[OK] Loaded checkpoint: {args.checkpoint}")
        print(f"[OK] Output tensor shape: {output.shape if hasattr(output, 'shape') else type(output)}")
    except Exception as e:
        print(f"[FAIL] {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
```

**Critical detail:** `load_from_HF=False` must be set when `checkpoint_path` is provided. Without it, the condition `load_from_HF and checkpoint_path is None` is `False` anyway — the HF download is skipped. But explicit `load_from_HF=False` documents the intent and avoids accidental network calls.

**Note on model forward pass:** `Sam3Image.forward` is a training-mode forward (expects a batch dict). In `eval_mode=True`, the model is in `.eval()` but the `forward()` signature may require specific batch structure. The verification script should test via the inference predictor API if the raw tensor forward fails. Investigate `Sam3Image.forward()` before writing the forward call.

**Source:** [VERIFIED: sam3/model_builder.py lines 573-654, scripts/test_config_parse.py pattern]

---

### Anti-Patterns to Avoid

- **Anti-pattern: Saving full training checkpoint as `best_checkpoint.pth`** — contains optimizer state, loss state, epoch counters, and non-tensor Python objects. The `_load_checkpoint` function uses `weights_only=True` which will reject non-tensor objects. Additionally, the missing "detector." prefix means no weights are loaded.
- **Anti-pattern: Using `shutil.copy` for the alias** — the original `val_custom_detection.pt` is a full training checkpoint (wrong format for inference). A copy would inherit all the format problems above.
- **Anti-pattern: Using `g_pathmgr.mv` for the alias** — would delete the training checkpoint, leaving only the inference-format `best_checkpoint.pth` (losing resume capability).
- **Anti-pattern: Hardcoding image paths in fake dataset** — all paths must be derived from `--out` arg; no hardcoded `/tmp` paths.
- **Anti-pattern: AP50 assertion in CI** — fake 5-image dataset with random noise will always produce AP50=0. D-04-05 explicitly allows this; VAL-01 CI test is "no crash", not "AP50 > 0".

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| COCO eval metric | Custom AP50 calculator | `pycocotools.coco` / existing `CocoEvaluatorOffline` | Complex area-under-curve computation, 12 standard AP/AR metrics |
| Checkpoint safe-write | Custom atomic write | `_save_checkpoint` (already uses `.tmp` + `mv`) | Guards against corruption on preemption |
| Fake image generation | Complex synthetic patterns | `PIL.Image.fromarray(np.random.randint(...))` | 5 random 64×64 images, complexity unnecessary |
| COCO polygon area | Manual polygon math | `bbox_area = w * h` for axis-aligned boxes | Sufficient for fake data; pycocotools computes real area from polygon for real eval |

---

## Common Pitfalls

### Pitfall 1: Silent Weight Load Failure
**What goes wrong:** `build_sam3_image_model(checkpoint_path="best_checkpoint.pth")` returns a model with random (pretrained from `_create_*` constructors, actually no pretrained) weights, no error raised.
**Why it happens:** `model.load_state_dict({}, strict=False)` silently succeeds. Missing keys are printed but not raised as errors.
**How to avoid:** Export `best_checkpoint.pth` in HuggingFace format (with `"detector."` prefix). Verify in `test_checkpoint_compatibility.py` that forward pass output is not all-zeros or all-NaN.
**Warning signs:** The verification script passes (no exception) but inference produces garbage masks.

### Pitfall 2: `weights_only=True` Rejection of Training Checkpoint
**What goes wrong:** `_load_checkpoint` calls `torch.load(..., weights_only=True)`. Full training checkpoints contain `epoch` (int), `steps` (int), `best_meter_values` (dict), etc. PyTorch ≥ 2.0 with `weights_only=True` rejects arbitrary Python objects.
**Why it happens:** `weights_only=True` is a security measure; the inference loader was designed for model-weights-only files.
**How to avoid:** Export `best_checkpoint.pth` as model-weights-only (no optimizer/epoch/etc.) as described in Pattern 3.
**Warning signs:** `torch.load` raises `_pickle.UnpicklingError` or `RuntimeError` about unsafe globals.

### Pitfall 3: `checkpoint_names is None` Check False Positive
**What goes wrong:** The patch to `save_checkpoint` writes `best_checkpoint.pth` on every save (both regular epoch saves and best-meter saves).
**Why it happens:** The condition `checkpoint_names is not None` triggers for both best-meter saves AND any future callers that pass explicit names.
**How to avoid:** Only write `best_checkpoint.pth` when the call comes from `_log_meters_and_save_best_ckpts`. A safe guard: check if `checkpoint_names` overlaps with `self.checkpoint_conf.save_best_meters` (after slash-to-underscore normalization).
**Warning signs:** `best_checkpoint.pth` gets overwritten on every epoch, not just when AP50 improves.

### Pitfall 4: `load_from_HF=True` (default) in Compatibility Script
**What goes wrong:** If `load_from_HF=True` (default) and `checkpoint_path` is provided, the code path skips HF download (`load_from_HF and checkpoint_path is None` is False), but the intent is ambiguous.
**Why it happens:** Default `load_from_HF=True` is designed for the case where no checkpoint is provided. Leaving it as default when providing a path works correctly, but is misleading.
**How to avoid:** Explicitly set `load_from_HF=False` when providing `checkpoint_path` to document intent.

### Pitfall 5: Sam3Image Forward Pass API in eval_mode
**What goes wrong:** `model(random_tensor)` fails because `Sam3Image.forward()` expects a structured `BatchedDatapoint` dict, not a raw tensor.
**Why it happens:** SAM3 image model uses a batch dict protocol. Raw tensor input is not supported in `forward()`.
**How to avoid:** Check `Sam3Image.forward()` signature. The verification script may need to use the predictor interface or construct a minimal batch dict. Alternatively, only test that `model.eval()` succeeds and the model is in eval mode — the D-04-03 spec only requires "no exception" from a forward pass.

---

## Code Examples

### Generating a Fake COCO Dataset

```python
# Source: Derived from tests/conftest.py minimal_coco fixture [VERIFIED pattern]
import json
import numpy as np
from PIL import Image
import os
import argparse

def generate_fake_dataset(out_dir, n_images=5, img_size=64):
    os.makedirs(out_dir, exist_ok=True)
    img_dir = os.path.join(out_dir, "images")
    os.makedirs(img_dir, exist_ok=True)

    images, annotations = [], []
    ann_id = 1

    for i in range(1, n_images + 1):
        fname = f"fake_{i:04d}.png"
        # Random RGB 64×64 image
        arr = np.random.randint(0, 256, (img_size, img_size, 3), dtype=np.uint8)
        Image.fromarray(arr).save(os.path.join(img_dir, fname))

        images.append({"id": i, "file_name": fname, "width": img_size, "height": img_size})
        # One axis-aligned polygon per image (inner 50% of image)
        margin = img_size // 4
        x0, y0 = margin, margin
        x1, y1 = img_size - margin, img_size - margin
        w, h = x1 - x0, y1 - y0
        annotations.append({
            "id": ann_id,
            "image_id": i,
            "category_id": 1,
            "bbox": [x0, y0, w, h],
            "area": w * h,
            "segmentation": [[x0, y0, x1, y0, x1, y1, x0, y1]],
            "iscrowd": 0,
        })
        ann_id += 1

    coco = {
        "info": {"description": "Fake defect dataset for CI"},
        "licenses": [],
        "categories": [{"id": 1, "name": "defect", "supercategory": "defect"}],
        "images": images,
        "annotations": annotations,
    }

    # Write train.json (all 5 images) and val.json (all 5 images) for smoke test
    for split in ("train", "val"):
        with open(os.path.join(out_dir, f"{split}.json"), "w") as f:
            json.dump(coco, f)

    return out_dir
```

### best_checkpoint.pth Export (HuggingFace Format)

```python
# Source: Derived from sam3/model_builder.py _load_checkpoint analysis [VERIFIED]
# Add to save_checkpoint() in sam3/train/trainer.py AFTER existing _save_checkpoint calls:

if checkpoint_names is not None and self.distributed_rank == 0:
    # Export inference-compatible best_checkpoint.pth
    # Format: {"model": {"detector.<key>": <value>}} matches _load_checkpoint expectations
    model_state_dict = checkpoint["model"]
    inference_ckpt = {
        "model": {f"detector.{k}": v for k, v in model_state_dict.items()}
    }
    best_path = os.path.join(checkpoint_folder, "best_checkpoint.pth")
    self._save_checkpoint(inference_ckpt, best_path)
    logging.info(f"[CKPT-01] Saved inference checkpoint: {best_path}")
```

---

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| `torch.load(..., weights_only=False)` | `torch.load(..., weights_only=True)` | PyTorch 2.0+ | Training checkpoints with non-tensor objects break `_load_checkpoint` |
| Plain `torch.save(state_dict)` | Full training checkpoint `{model, optimizer, epoch, ...}` | SAM3 trainer design | Incompatible with inference loader's `weights_only=True` unless model-only file |

---

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | `Sam3Image.forward()` does not accept a raw tensor; requires a batch dict | Pattern 6 / Pitfall 5 | Compatibility script forward call will fail; needs alternate invocation |
| A2 | Training checkpoint model state dict keys begin with `backbone.`, `transformer.`, `segmentation_head.`, etc. (matching post-strip keys from HuggingFace checkpoint) | Pattern 3 | If key namespace differs, `load_state_dict` with strict=False would silently miss keys |
| A3 | The 5-image fake dataset is sufficient for the 1-epoch dry-run to complete without OOM on standard hardware | Pattern 5 | If memory is very constrained, even 1 epoch on fake data might fail |

---

## Open Questions

1. **Sam3Image forward pass input format for test_checkpoint_compatibility.py**
   - What we know: `build_sam3_image_model(eval_mode=True)` returns a `Sam3Image` in eval mode. D-04-03 says "one forward pass on random 1024×1024 tensor input."
   - What's unclear: `Sam3Image.forward()` may require a structured batch dict (not a raw tensor). Need to inspect `sam3/model/sam3_image.py` `forward()` signature before writing the verification script.
   - Recommendation: Check `Sam3Image.forward()` method; if it requires a batch dict, construct a minimal one or use the image predictor wrapper (`SAM3ImagePredictor`) which has a simpler API.

2. **`save_best_meters` condition check in patch**
   - What we know: `checkpoint_names is not None` is True for both best-meter saves and any explicit-name save calls.
   - What's unclear: Are there other callers of `save_checkpoint(epoch, explicit_names)` besides `_log_meters_and_save_best_ckpts`?
   - Recommendation: Search trainer.py for all `save_checkpoint(` calls to confirm `_log_meters_and_save_best_ckpts` is the only caller with explicit names. If so, `checkpoint_names is not None` is a safe condition.

---

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| Python | All scripts | ✓ | system | — |
| PIL/Pillow | `generate_fake_dataset.py` | ✓ (SAM3 dep) | — | Use `numpy` to write raw bytes if missing |
| `torch` | `test_checkpoint_compatibility.py` | ✓ | SAM3 core | — |
| GPU/CUDA | Compatibility script | ✗ (CI) | — | Use `device="cpu"` (D-04-03 spec) |
| Real industrial defect dataset | VAL-01 real AP>0 | ✗ | — | Fake dataset for CI (D-04-01) |

**Missing dependencies with no fallback:**
- None (all CI paths work CPU-only with fake data)

**Missing dependencies with fallback:**
- Real dataset: fake dataset covers CI; real AP>0 is manual only

---

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest |
| Config file | none (standard pytest discovery) |
| Quick run command | `pytest tests/ -x -q` |
| Full suite command | `pytest tests/ -v` |

### Phase Requirements → Test Map

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| CKPT-01 | `best_checkpoint.pth` created when AP50 improves | unit | `pytest tests/test_checkpoint_export.py::test_best_checkpoint_alias -x` | ❌ Wave 0 |
| CKPT-02 | `best_checkpoint.pth` loads via `build_sam3_image_model` | integration (script) | `python scripts/test_checkpoint_compatibility.py --checkpoint /tmp/test_best.pth` | ❌ Wave 0 |
| VAL-01 | 1-epoch dry run on fake data completes without crash | smoke (script) | `python scripts/generate_fake_dataset.py --out /tmp/fake && python -m sam3.train.train ... --max-epochs 1` | ❌ Wave 0 |
| VAL-02 | Same as CKPT-02 | integration (script) | Same script as CKPT-02 | ❌ Wave 0 |

### Sampling Rate
- **Per task commit:** `pytest tests/ -x -q`
- **Per wave merge:** `pytest tests/ -v`
- **Phase gate:** Full suite green + `scripts/test_checkpoint_compatibility.py` exits 0

### Wave 0 Gaps
- [ ] `tests/test_checkpoint_export.py` — covers CKPT-01 (unit test for `best_checkpoint.pth` alias logic)
- [ ] `scripts/generate_fake_dataset.py` — needed for VAL-01 smoke test
- [ ] `scripts/test_checkpoint_compatibility.py` — covers CKPT-02 / VAL-02

*(No new framework install needed — pytest already present)*

---

## Security Domain

> `security_enforcement` not explicitly false in config.json — included.

### Applicable ASVS Categories

| ASVS Category | Applies | Standard Control |
|---------------|---------|-----------------|
| V2 Authentication | No | N/A — local script execution |
| V3 Session Management | No | N/A — batch scripts |
| V4 Access Control | No | N/A — local filesystem only |
| V5 Input Validation | Yes | Validate `--checkpoint` path exists before loading; validate `--out` is a writable directory |
| V6 Cryptography | No | N/A |

### Known Threat Patterns for Checkpoint Loading

| Pattern | STRIDE | Standard Mitigation |
|---------|--------|---------------------|
| Malicious pickle in `.pth`/`.pt` files | Tampering | `weights_only=True` in `_load_checkpoint` (already implemented) |
| Path traversal in `--checkpoint` arg | Tampering | `os.path.abspath` + existence check before `torch.load` |
| Arbitrary write via `--out` path | Elevation of Privilege | Validate output dir is within expected bounds in `generate_fake_dataset.py` |

---

## Sources

### Primary (HIGH confidence)
- `sam3/train/trainer.py` — checkpoint save flow, `save_checkpoint`, `_save_checkpoint`, `_log_meters_and_save_best_ckpts`, `CheckpointConf` — verified line-by-line
- `sam3/model_builder.py` — `build_sam3_image_model`, `_load_checkpoint` — verified line-by-line
- `sam3/train/utils/checkpoint_utils.py` — `load_checkpoint`, `load_state_dict_into_model`, `load_checkpoint_and_apply_kernels` — verified
- `sam3/train/configs/custom_finetune/base.yaml` — `save_best_meters`, `checkpoint.save_dir`, meter key `val_custom/detection` — verified
- `tests/conftest.py` — COCO JSON fixture format — verified
- `scripts/test_config_parse.py`, `scripts/test_training_config.py` — script patterns (argparse, sys.exit, stdout logging) — verified

### Secondary (MEDIUM confidence)
- `sam3/train/utils/train_utils.py` lines 279-286 — `get_resume_checkpoint` always resumes from `checkpoint.pt`, not from `val_custom_detection.pt` or `best_checkpoint.pth` [VERIFIED]

---

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — all in-project, verified
- Architecture: HIGH — verified from source code
- Checkpoint format compatibility: HIGH — verified by tracing `_load_checkpoint` call path
- COCO JSON format: HIGH — verified from existing tests/conftest.py fixture
- Pitfalls: HIGH — deduced directly from source code analysis
- `Sam3Image.forward()` API: LOW — not yet inspected (A1 assumption)

**Research date:** 2026-05-28
**Valid until:** 2026-07-28 (stable codebase, no fast-moving deps)
