# Phase 4: Checkpoint Export & End-to-End Validation — Pattern Map

**Mapped:** 2025-05-28
**Files analyzed:** 4
**Analogs found:** 4 / 4

---

## File Classification

| New/Modified File | Role | Data Flow | Closest Analog | Match Quality |
|-------------------|------|-----------|----------------|---------------|
| `sam3/train/trainer.py` | service | CRUD (file I/O) | `sam3/train/trainer.py` (self — patch) | exact |
| `scripts/generate_fake_dataset.py` | utility | file-I/O | `tests/conftest.py` (COCO fixture) + `scripts/test_config_parse.py` (script structure) | role-match |
| `scripts/test_checkpoint_compatibility.py` | utility | request-response | `scripts/test_training_config.py` | exact |
| `sam3/train/configs/custom_finetune/base.yaml` | config | — | `sam3/train/configs/custom_finetune/base.yaml` (self — verify only) | exact |

---

## Pattern Assignments

---

### `sam3/train/trainer.py` — patch `save_checkpoint()` (service, file-I/O)

**Analog:** `sam3/train/trainer.py` (self — surgical patch after line 375)

**Imports already present** (lines 1–30 — nothing new required):
```python
import logging
import os
import torch
from iopath.common.file_io import g_pathmgr
```

**Core pattern — existing `save_checkpoint` method** (lines 333–375):
```python
def save_checkpoint(self, epoch, checkpoint_names=None):
    if self.skip_saving_ckpts:
        logging.info(
            "skip_saving_ckpts is set to True. So, no checkpoints have been saved."
        )
        return
    checkpoint_folder = self.checkpoint_conf.save_dir
    makedir(checkpoint_folder)
    if checkpoint_names is None:
        checkpoint_names = ["checkpoint"]
        if (
            self.checkpoint_conf.save_freq > 0
            and (int(epoch) % self.checkpoint_conf.save_freq == 0)
        ) or int(epoch) in self.checkpoint_conf.save_list:
            checkpoint_names.append(f"checkpoint_{int(epoch)}")

    checkpoint_paths = []
    for ckpt_name in checkpoint_names:
        checkpoint_paths.append(os.path.join(checkpoint_folder, f"{ckpt_name}.pt"))

    state_dict = unwrap_ddp_if_wrapped(self.model).state_dict()
    state_dict = exclude_params_matching_unix_pattern(
        patterns=self.checkpoint_conf.skip_saving_parameters, state_dict=state_dict
    )

    checkpoint = {
        "model": state_dict,
        "optimizer": self.optim.optimizer.state_dict(),
        "epoch": epoch,
        "loss": self.loss.state_dict(),
        "steps": self.steps,
        "time_elapsed": self.time_elapsed_meter.val,
        "best_meter_values": self.best_meter_values,
    }
    if self.optim_conf.amp.enabled:
        checkpoint["scaler"] = self.scaler.state_dict()

    # DDP checkpoints are only saved on rank 0 (all workers are identical)
    if self.distributed_rank != 0:
        return

    for checkpoint_path in checkpoint_paths:
        self._save_checkpoint(checkpoint, checkpoint_path)
```

**Patch location:** immediately after the `for checkpoint_path in checkpoint_paths:` loop (line 375), still inside the `save_checkpoint` method, **before** the method ends:
```python
    # NEW (CKPT-01 / CKPT-02): export inference-compatible best_checkpoint.pth
    # Only written when this is a best-meter save (checkpoint_names was not None on entry).
    # Guard: checkpoint_names must contain a key that matches one of save_best_meters
    # (after slash→underscore normalization) to avoid writing on every named save.
    if (
        checkpoint_names is not None
        and self.checkpoint_conf.save_best_meters is not None
        and self.distributed_rank == 0
    ):
        best_meter_keys_normalized = {
            k.replace("/", "_") for k in self.checkpoint_conf.save_best_meters
        }
        if any(k in best_meter_keys_normalized for k in checkpoint_names):
            # Re-read from `checkpoint["model"]` (already built above).
            # Export in HuggingFace format: {"model": {"detector.<key>": <tensor>}}
            # This satisfies _load_checkpoint (model_builder.py lines 542-556):
            #   - weights_only=True: only tensors → OK (no optimizer/epoch/etc.)
            #   - "model" key → ckpt["model"] extracted  
            #   - "detector." prefix → filter passes, prefix stripped before load_state_dict
            inference_ckpt = {
                "model": {
                    f"detector.{k}": v
                    for k, v in checkpoint["model"].items()
                }
            }
            best_path = os.path.join(checkpoint_folder, "best_checkpoint.pth")
            self._save_checkpoint(inference_ckpt, best_path)
            logging.info(f"[CKPT-01] Saved inference checkpoint: {best_path}")
```

**`_save_checkpoint` primitive** (lines 377–394) — reuse unchanged (atomic write via `.tmp` + `mv`):
```python
def _save_checkpoint(self, checkpoint, checkpoint_path):
    checkpoint_path_tmp = f"{checkpoint_path}.tmp"
    with g_pathmgr.open(checkpoint_path_tmp, "wb") as f:
        torch.save(checkpoint, f)
    if g_pathmgr.exists(checkpoint_path):
        g_pathmgr.rm(checkpoint_path)
    success = g_pathmgr.mv(checkpoint_path_tmp, checkpoint_path)
    assert success
```

**Best-meter trigger** (lines 968–996) — unchanged; `save_checkpoint(self.epoch + 1, checkpoint_save_keys)` is the call that passes `checkpoint_names` as a list containing `"val_custom_detection"`, which matches the normalized `save_best_meters` entry `"val_custom/detection"`:
```python
# Line 993: checkpoint_save_keys.append(tracked_meter_key.replace("/", "_"))
# → "val_custom_detection"
# Line 996: self.save_checkpoint(self.epoch + 1, checkpoint_save_keys)
```

**`CheckpointConf` dataclass** (lines 109–124) — `save_best_meters` field already present; no change needed:
```python
@dataclass
class CheckpointConf:
    save_dir: str
    save_freq: int
    save_list: List[int] = field(default_factory=list)
    model_weight_initializer: Any = None
    save_best_meters: List[str] = None     # ← already wired (D-P3-01)
    skip_saving_parameters: List[str] = field(default_factory=list)
    initialize_after_preemption: Optional[bool] = None
    resume_from: Optional[str] = None
```

---

### `scripts/generate_fake_dataset.py` (utility, file-I/O)

**Analog 1 — Script structure:** `scripts/test_config_parse.py`
**Analog 2 — COCO JSON structure:** `tests/conftest.py` (lines 10–28)

**Script-level boilerplate** (copy from `scripts/test_config_parse.py` lines 1–4, 19–22):
```python
#!/usr/bin/env python3
"""
Generate a minimal synthetic COCO dataset for CI smoke-testing.

Run from the sam3 project root:
    python scripts/generate_fake_dataset.py --out /tmp/fake_defect_data

Output: 5 synthetic 64×64 PNG images + train.json + val.json (COCO format)
        at <out>/images/ and <out>/train.json, <out>/val.json.
Prints the output path to stdout for downstream use.
"""
import argparse
import json
import os
import sys
import numpy as np
from PIL import Image
```

**`sys.path` setup pattern** (copy from `scripts/test_config_parse.py` lines 33–34) — not needed for this script (no sam3 imports), skip.

**COCO JSON structure** (from `tests/conftest.py` lines 10–28):
```python
# 1-based IDs are required by sam3/train/data/sam3_image_dataset.py
# segmentation must be non-empty for load_segmentation=True
# bbox is [x, y, w, h] (COCO convention, not [x1, y1, x2, y2])
coco_json = {
    "info": {"description": "Fake defect dataset for CI"},
    "licenses": [],
    "categories": [{"id": 1, "name": "defect", "supercategory": "defect"}],
    "images": [
        {"id": 1, "file_name": "fake_0001.png", "width": 64, "height": 64},
        # ... repeat for ids 2–5
    ],
    "annotations": [
        {
            "id": 1, "image_id": 1, "category_id": 1,
            "bbox": [16, 16, 32, 32],               # [x, y, w, h]
            "area": 1024,                            # w * h
            "segmentation": [[16, 16, 48, 16, 48, 48, 16, 48]],  # polygon [x1,y1,...,xn,yn]
            "iscrowd": 0,
        },
        # ... one annotation per image
    ],
}
```

**Core image generation pattern** (from RESEARCH.md Pattern 5 / Code Examples):
```python
def generate_fake_dataset(out_dir, n_images=5, img_size=64):
    os.makedirs(out_dir, exist_ok=True)
    img_dir = os.path.join(out_dir, "images")
    os.makedirs(img_dir, exist_ok=True)

    images, annotations = [], []
    ann_id = 1

    for i in range(1, n_images + 1):
        fname = f"fake_{i:04d}.png"
        arr = np.random.randint(0, 256, (img_size, img_size, 3), dtype=np.uint8)
        Image.fromarray(arr).save(os.path.join(img_dir, fname))

        images.append({"id": i, "file_name": fname, "width": img_size, "height": img_size})
        margin = img_size // 4
        x0, y0 = margin, margin
        x1, y1 = img_size - margin, img_size - margin
        w, h = x1 - x0, y1 - y0
        annotations.append({
            "id": ann_id, "image_id": i, "category_id": 1,
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
    # Write both splits (all 5 images in each — smoke test only)
    for split in ("train", "val"):
        with open(os.path.join(out_dir, f"{split}.json"), "w") as f:
            json.dump(coco, f)
    return out_dir
```

**`main()` / `argparse` pattern** (copy from `scripts/test_config_parse.py` lines 86–88 adapted):
```python
def main():
    parser = argparse.ArgumentParser(
        description="Generate a minimal synthetic COCO defect dataset for CI."
    )
    parser.add_argument(
        "--out", required=True,
        help="Output directory (created if not exists). E.g. /tmp/fake_defect_data"
    )
    parser.add_argument("--n-images", type=int, default=5, help="Number of fake images")
    parser.add_argument("--img-size", type=int, default=64, help="Image size in pixels")
    args = parser.parse_args()

    out = generate_fake_dataset(args.out, n_images=args.n_images, img_size=args.img_size)
    print(out)   # prints path for downstream shell capture

if __name__ == "__main__":
    main()
```

**Error handling pattern** (from `scripts/test_config_parse.py` lines 189–193):
```python
# No try/except wrapping the whole main — let exceptions propagate naturally.
# sys.exit(1) is only used for expected assertion failures.
# Unexpected exceptions should print a full traceback (default Python behavior).
```

---

### `scripts/test_checkpoint_compatibility.py` (utility, request-response)

**Analog:** `scripts/test_training_config.py` (exact match: same script structure, same `sys.exit(1)` on failure, same single `main()` entry point)

**Script-level boilerplate** (copy from `scripts/test_training_config.py` lines 1–4, 19–31):
```python
#!/usr/bin/env python3
"""
Smoke test: verify best_checkpoint.pth loads cleanly via build_sam3_image_model().

Run from the sam3 project root:
    python scripts/test_checkpoint_compatibility.py --checkpoint /path/to/best_checkpoint.pth

Exit 0: success — prints "[OK] Loaded checkpoint: <path>" and "[OK] Output tensor shape: <shape>"
Exit 1: failure — prints "[FAIL] <error>" to stderr
"""
import argparse
import sys
import os

_SAM3_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _SAM3_ROOT)
```

**Core inference load pattern** (from `sam3/model_builder.py` lines 573–652):
```python
# build_sam3_image_model signature (model_builder.py line 573):
# def build_sam3_image_model(
#     device="cuda" if torch.cuda.is_available() else "cpu",
#     eval_mode=True,
#     checkpoint_path=None,
#     load_from_HF=True,
#     enable_segmentation=True,
#     ...
# )
#
# Critical: load_from_HF=False prevents accidental HF download when checkpoint_path is provided.
# device="cpu" for CI compatibility (no GPU required for shape check).
from sam3 import build_sam3_image_model
model = build_sam3_image_model(
    checkpoint_path=args.checkpoint,
    enable_segmentation=True,
    load_from_HF=False,
    device="cpu",
    eval_mode=True,
)
```

**SAM3 forward pass note** (from RESEARCH.md Pitfall 5 / Pattern 6):
```
Sam3Image.forward() expects a structured BatchedDatapoint dict, NOT a raw tensor.
Use the image predictor API or construct a minimal batch dict for the forward call.
Alternatively, just test that model.eval() succeeds + model is in eval mode
(D-04-03 spec: "no exception" = pass).
```

**Main pattern with error handling** (copy from `scripts/test_training_config.py` lines 75–176 adapted):
```python
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint", required=True,
        help="Path to best_checkpoint.pth (HuggingFace-format inference checkpoint)"
    )
    args = parser.parse_args()

    try:
        import torch
        from sam3 import build_sam3_image_model

        model = build_sam3_image_model(
            checkpoint_path=args.checkpoint,
            enable_segmentation=True,
            load_from_HF=False,
            device="cpu",
            eval_mode=True,
        )
        # Verify eval mode is set (no forward pass required for shape assertion)
        assert not model.training, "Expected model to be in eval mode"
        print(f"[OK] Loaded checkpoint: {args.checkpoint}")
        print(f"[OK] Model is in eval mode: {not model.training}")
    except Exception as e:
        print(f"[FAIL] {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
```

**Summary / exit pattern** (copy from `scripts/test_training_config.py` lines 165–172):
```python
# On success: implicit exit 0 (no sys.exit call needed)
# On failure: sys.exit(1) inside except block
# No summary aggregation needed (only one test in this script)
```

---

### `sam3/train/configs/custom_finetune/base.yaml` — verify only (config, n/a)

**Analog:** self — no changes required for Phase 4.

**Relevant existing config stanza** (lines 369–373) — already correct, no modification needed:
```yaml
  checkpoint:
    save_dir: ${launcher.experiment_log_dir}/checkpoints
    save_freq: 0                 # 0 = save only most-recent + best-meter checkpoints
    save_best_meters:
      - "val_custom/detection"   # saves checkpoint when any metric under this meter improves
```

**Verification assertion** (for `test_checkpoint_compatibility.py` or a future config test):
```python
# The save_best_meters key "val_custom/detection" → normalized to "val_custom_detection"
# This must match what _log_meters_and_save_best_ckpts produces on line 993:
#   checkpoint_save_keys.append(tracked_meter_key.replace("/", "_"))
# where tracked_meter_key = os.path.join("val_custom", "detection") = "val_custom/detection"
assert cfg_base.trainer.checkpoint.save_best_meters == ["val_custom/detection"]
```

---

## Shared Patterns

### Atomic Checkpoint Write
**Source:** `sam3/train/trainer.py` `_save_checkpoint()` (lines 377–394)
**Apply to:** All checkpoint writing in `save_checkpoint()` patch — reuse `self._save_checkpoint()`, never call `torch.save()` directly.
```python
def _save_checkpoint(self, checkpoint, checkpoint_path):
    checkpoint_path_tmp = f"{checkpoint_path}.tmp"
    with g_pathmgr.open(checkpoint_path_tmp, "wb") as f:
        torch.save(checkpoint, f)
    if g_pathmgr.exists(checkpoint_path):
        g_pathmgr.rm(checkpoint_path)
    success = g_pathmgr.mv(checkpoint_path_tmp, checkpoint_path)
    assert success
```

### Script `sys.path` Bootstrap
**Source:** `scripts/test_training_config.py` (lines 30–31)
**Apply to:** `scripts/test_checkpoint_compatibility.py` (needs to import `sam3`)
```python
_SAM3_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _SAM3_ROOT)
```

### Script Exit Convention
**Source:** `scripts/test_training_config.py` (lines 165–172) and `scripts/test_config_parse.py` (lines 189–193)
**Apply to:** `scripts/test_checkpoint_compatibility.py`, `scripts/generate_fake_dataset.py`
```python
# Success: no sys.exit call → implicit exit code 0
# Expected failure: sys.exit(1) after printing error
# Unexpected exception: let it propagate (full traceback, non-zero exit)
```

### DDP Rank Guard
**Source:** `sam3/train/trainer.py` (lines 371–372)
**Apply to:** All file-writing code in `save_checkpoint()` patch
```python
if self.distributed_rank != 0:
    return  # only rank-0 writes files
```

### HuggingFace Checkpoint Format for `_load_checkpoint`
**Source:** `sam3/model_builder.py` (lines 539–561)
**Apply to:** `best_checkpoint.pth` export in trainer patch
```python
# _load_checkpoint expects:
#   ckpt["model"] → dict with keys "detector.<layer>" → tensor
#   torch.load(..., weights_only=True) → only tensors allowed, no Python objects
# Correct export format:
inference_ckpt = {
    "model": {f"detector.{k}": v for k, v in model_state_dict.items()}
}
# model_state_dict must be pure {str: Tensor} — no optimizer, no epoch, no scalars
```

### COCO JSON ID Convention
**Source:** `tests/conftest.py` (lines 10–28)
**Apply to:** `scripts/generate_fake_dataset.py`
```python
# All IDs (image, annotation, category) must be 1-based integers
# bbox: [x, y, w, h] — NOT [x1, y1, x2, y2]
# segmentation: [[x1, y1, x2, y2, ..., xn, yn]] — flat polygon in outer list
# iscrowd: 0 for all synthetic annotations
```

---

## No Analog Found

All four files have close analogs in the codebase. No files require pattern invention from scratch.

---

## Metadata

**Analog search scope:** `sam3/train/trainer.py`, `sam3/model_builder.py`, `scripts/`, `tests/conftest.py`, `sam3/train/configs/custom_finetune/base.yaml`
**Files scanned:** 6 source files + RESEARCH.md + CONTEXT.md
**Pattern extraction date:** 2025-05-28
