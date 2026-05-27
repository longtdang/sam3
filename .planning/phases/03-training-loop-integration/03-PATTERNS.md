# Phase 3: Training Loop Integration - Pattern Map

**Mapped:** 2026-05-27
**Files analyzed:** 3 new/modified files
**Analogs found:** 3 / 3

---

## File Classification

| New/Modified File | Role | Data Flow | Closest Analog | Match Quality |
|---|---|---|---|---|
| `sam3/train/transforms/basic.py` | transform (utility class) | transform (PIL→PIL, tensor→tensor) | `sam3/train/transforms/basic_for_api.py` (ToTensorAPI, NormalizeAPI, ColorJitter) + `basic.py::RandomErasing` | exact — same file, same wrapping architecture |
| `sam3/train/configs/custom_finetune/base.yaml` | config | YAML instantiation via Hydra | `sam3/train/configs/odinw13/odinw_text_only_train.yaml` + current `base.yaml` | exact — same file being modified |
| `scripts/test_training_config.py` | test / dry-run script | request-response (Hydra compose + assert) | `scripts/test_config_parse.py` | exact — same pattern, same codebase |

---

## Pattern Assignments

---

### `sam3/train/transforms/basic.py` — Add `ColorJitter`, `GaussianBlur`, `RandomErasingAPI` classes

**Analog 1 (wrapping architecture):** `sam3/train/transforms/basic.py` lines 381–388 — `RandomErasing`

**Analog 2 (API-compatible `datapoint` signature):** `sam3/train/transforms/basic_for_api.py` lines 867–919 — `ToTensorAPI` and `NormalizeAPI`

**Critical interface constraint:** The `(img, target)` signature of the *existing* `RandomErasing` class (basic.py:381) is **NOT compatible with `ComposeAPI`**. All transforms used inside `ComposeAPI` in `base.yaml` receive a `Datapoint` object via `(datapoint, **kwargs)`. The new classes must use the API-compatible signature (see Analog 2).

---

#### Imports pattern (basic.py lines 1–18):
```python
# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved
# pyre-unsafe

import math
import random
from typing import Iterable

import PIL
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F
from sam3.model.box_ops import box_xyxy_to_cxcywh
from sam3.model.data_misc import interpolate
```

**Note:** No new imports are required. `T.ColorJitter`, `T.GaussianBlur`, and `T.RandomErasing` are all available via the existing `import torchvision.transforms as T`. `Datapoint` is NOT imported in `basic.py` — the new API classes should type-hint loosely (no import needed for runtime; the `datapoint` arg is duck-typed).

---

#### Existing `RandomErasing` pattern (basic.py lines 381–388) — wrapping architecture to copy:
```python
class RandomErasing:
    def __init__(self, *args, **kwargs):
        self.eraser = T.RandomErasing(*args, **kwargs)

    def __call__(self, img, target):
        return self.eraser(img), target
```
**Copy:** The `__init__(*args, **kwargs)` → delegate pattern. Do NOT copy the `(img, target)` call signature for new classes.

---

#### API-compatible `__call__` signature — copy from basic_for_api.py lines 867–879 (`ToTensorAPI`):
```python
class ToTensorAPI:
    def __init__(self, v2=False):
        self.v2 = v2

    def __call__(self, datapoint: Datapoint, **kwargs):
        for img in datapoint.images:
            if self.v2:
                img.data = Fv2.to_image_tensor(img.data)
            else:
                img.data = F.to_tensor(img.data)
        return datapoint
```
**Key pattern to copy:**
- `__call__(self, datapoint, **kwargs)` signature
- `for img in datapoint.images:` loop
- `img.data = transform(img.data)` — mutate `img.data` in-place within the loop
- `return datapoint`

---

#### Core pattern for new `ColorJitter` class (combine both analogs):
```python
# Copy __init__ wrapping from basic.py::RandomErasing (lines 382-383)
# Copy __call__ datapoint loop from basic_for_api.py::ToTensorAPI (lines 871-878)
class ColorJitter:
    def __init__(self, *args, **kwargs):
        self.jitter = T.ColorJitter(*args, **kwargs)   # delegate to torchvision

    def __call__(self, datapoint, **kwargs):
        for img in datapoint.images:
            img.data = self.jitter(img.data)           # PIL or tensor → PIL or tensor
        return datapoint
```
**YAML parameters (D-03-05):** `brightness: 0.2, contrast: 0.2, saturation: 0.2, hue: 0.0`

---

#### Core pattern for new `GaussianBlur` class:
```python
class GaussianBlur:
    def __init__(self, *args, **kwargs):
        self.blur = T.GaussianBlur(*args, **kwargs)    # delegate to torchvision

    def __call__(self, datapoint, **kwargs):
        for img in datapoint.images:
            img.data = self.blur(img.data)             # PIL or tensor → PIL or tensor
        return datapoint
```
**YAML parameters (D-03-05):** `kernel_size: 3, sigma: [0.1, 2.0]`

---

#### Core pattern for new `RandomErasingAPI` class (API-compatible version alongside existing `RandomErasing`):
```python
# Distinct name avoids clash with existing basic.py::RandomErasing (img, target) class
class RandomErasingAPI:
    def __init__(self, *args, **kwargs):
        self.eraser = T.RandomErasing(*args, **kwargs)  # delegate to torchvision

    def __call__(self, datapoint, **kwargs):
        for img in datapoint.images:
            img.data = self.eraser(img.data)            # tensor required (place AFTER ToTensorAPI)
        return datapoint
```
**YAML parameters (D-03-05):** `p: 0.2, scale: [0.02, 0.1]`
**Placement constraint:** Must go **after** `ToTensorAPI` and **before** `NormalizeAPI` — `T.RandomErasing` requires tensor input.

---

#### Placement in file:
Append all three new classes at the **end of `basic.py`**, after the existing `RandomErasing` class (line 386). This keeps the `(img, target)` legacy classes together and the API-compatible classes clearly separated at the bottom.

---

### `sam3/train/configs/custom_finetune/base.yaml` — Two targeted changes

**Analog:** The same file being modified. The existing content is the pattern.

---

#### Change 1 — `val_epoch_freq` (base.yaml line 215):

**Current (lines 207–216):**
```yaml
trainer:
  _target_: sam3.train.trainer.Trainer
  skip_saving_ckpts: false
  empty_gpu_mem_cache_after_eval: True
  skip_first_val: True
  max_epochs: ${scratch.max_data_epochs}
  accelerator: cuda
  seed_value: 42
  val_epoch_freq: 10                # ← CHANGE THIS to 1
  mode: train
  gradient_accumulation_steps: ${scratch.gradient_accumulation_steps}
```

**After change:**
```yaml
  val_epoch_freq: 1                 # D-03-06: evaluate every epoch (changed from 10)
```

---

#### Change 2 — Augmentation entries in `scratch.train_transforms` (base.yaml lines 111–148):

**Existing pipeline structure (inside `ComposeAPI.transforms` list):**
```yaml
scratch:
  train_transforms:
    - _target_: sam3.train.transforms.basic_for_api.ComposeAPI
      transforms:
        - _target_: ...FilterCrowds
        - _target_: ...RandomizeInputBbox
        - _target_: ...DecodeRle
        - _target_: ...RandomResizeAPI
        - _target_: ...PadToSizeAPI          # ← INSERT ColorJitter + GaussianBlur AFTER here
        - _target_: ...ToTensorAPI           # ← INSERT RandomErasingAPI AFTER here
        - _target_: ...FlexibleFilterFindGetQueries  # FilterEmptyTargets
        - _target_: ...NormalizeAPI          # ← RandomErasingAPI goes BEFORE here
        - _target_: ...FlexibleFilterFindGetQueries  # FilterEmptyTargets
```

**Entries to add — after `PadToSizeAPI`, before `ToTensorAPI` (PIL stage):**
```yaml
        - _target_: sam3.train.transforms.basic.ColorJitter
          brightness: 0.2
          contrast: 0.2
          saturation: 0.2
          hue: 0.0
        - _target_: sam3.train.transforms.basic.GaussianBlur
          kernel_size: 3
          sigma: [0.1, 2.0]
```

**Entry to add — after `ToTensorAPI`, before `NormalizeAPI` (tensor stage):**
```yaml
        - _target_: sam3.train.transforms.basic.RandomErasingAPI
          p: 0.2
          scale: [0.02, 0.1]
```

**Val transforms are NOT modified** — augmentation on train pipeline only (D-03-05, standard practice).

---

#### YAML `_target_:` instantiation pattern (copy from existing entries):
All transforms follow this single-entry format:
```yaml
- _target_: sam3.train.transforms.basic_for_api.PadToSizeAPI
  size: ${scratch.resolution}
  consistent_transform: ${scratch.consistent_transform}
```
New entries follow the same pattern:
- `_target_:` points to fully-qualified Python class path
- Parameters are keyword arguments matching `__init__` signature
- Indentation: 8 spaces (inside `transforms:` list which is inside `ComposeAPI` entry)

---

### `scripts/test_training_config.py` — New dry-run validation script

**Analog:** `scripts/test_config_parse.py` (full file, lines 1–199) — exact same structure.

---

#### Imports pattern (test_config_parse.py lines 19–28):
```python
import math
import os
import sys
import types

import numpy as np
import hydra.utils
from hydra import compose, initialize_config_module
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf
```

---

#### Module stub pattern — avoid PyTorch model deps (test_config_parse.py lines 33–48):
```python
_SAM3_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _SAM3_ROOT)

def _make_package_stub(name: str, path: str) -> types.ModuleType:
    import importlib.util
    init_file = os.path.join(path, "__init__.py")
    spec = importlib.util.spec_from_file_location(name, init_file, submodule_search_locations=[path])
    m = types.ModuleType(name)
    m.__path__ = [path]
    m.__package__ = name
    m.__file__ = init_file
    m.__spec__ = spec
    return m

sys.modules.setdefault("sam3", _make_package_stub("sam3", os.path.join(_SAM3_ROOT, "sam3")))
sys.modules.setdefault("sam3.train", _make_package_stub("sam3.train", os.path.join(_SAM3_ROOT, "sam3", "train")))
```
**Copy verbatim.** This stubs `sam3/__init__.py` to prevent `model_builder.py → decoder.py → torch.nn.attention` from loading (requires PyTorch ≥ 2.3).

---

#### OmegaConf resolver registration pattern (test_config_parse.py lines 51–69):
```python
def register_omegaconf_resolvers():
    OmegaConf.register_new_resolver("get_method", hydra.utils.get_method, replace=True)
    OmegaConf.register_new_resolver("get_class", hydra.utils.get_class, replace=True)
    OmegaConf.register_new_resolver("add", lambda x, y: x + y, replace=True)
    OmegaConf.register_new_resolver("times", lambda *a: np.prod(np.array(a)).item(), replace=True)
    OmegaConf.register_new_resolver("divide", lambda x, y: x / y, replace=True)
    OmegaConf.register_new_resolver("pow", lambda x, y: x**y, replace=True)
    OmegaConf.register_new_resolver("subtract", lambda x, y: x - y, replace=True)
    OmegaConf.register_new_resolver("range", lambda x: list(range(x)), replace=True)
    OmegaConf.register_new_resolver("int", lambda x: int(x), replace=True)
    OmegaConf.register_new_resolver("ceil_int", lambda x: int(math.ceil(x)), replace=True)
    OmegaConf.register_new_resolver("merge", lambda *x: OmegaConf.merge(*x), replace=True)
    OmegaConf.register_new_resolver("string", lambda x: str(x), replace=True)
```
**Copy verbatim.** This matches the resolver set registered by `train.py` at runtime.

---

#### Hydra initialization and reset pattern (test_config_parse.py lines 72–83):
```python
def reset_hydra():
    GlobalHydra.instance().clear()

def compose_config(config_name: str) -> object:
    cfg = compose(config_name=config_name)
    OmegaConf.to_container(cfg, resolve=True, throw_on_missing=False)
    return cfg
```
**Copy verbatim.** `throw_on_missing=False` allows `paths.*: null` (REQUIRED sentinel) without raising at parse time.

---

#### Core main() pattern (test_config_parse.py lines 86–195):
```python
def main():
    register_omegaconf_resolvers()
    reset_hydra()
    initialize_config_module("sam3.train", version_base="1.2")

    errors = []

    try:
        cfg_base = compose_config("configs/custom_finetune/base")
        # assert specific values...
        print("✓ custom_finetune/base")
    except Exception as e:
        errors.append(f"✗ custom_finetune/base: {e}")
        print(f"✗ custom_finetune/base: {e}")

    # ... additional test blocks ...

    if errors:
        print(f"\n{len(errors)} config(s) FAILED:")
        for err in errors:
            print(f"  {err}")
        sys.exit(1)
    else:
        print("\nAll configs parsed successfully.")

if __name__ == "__main__":
    main()
```

---

#### New Phase 3 assertions to add inside the test block:
```python
# Verify val_epoch_freq changed from 10 → 1 (D-03-06)
assert cfg_base.trainer.val_epoch_freq == 1, (
    f"Expected val_epoch_freq=1, got {cfg_base.trainer.val_epoch_freq}"
)

# Verify TensorBoard block present (D-03-07 — already in base.yaml from Phase 2)
assert cfg_base.trainer.logging.tensorboard_writer is not None, (
    "TensorBoard writer not configured"
)
assert cfg_base.trainer.logging.tensorboard_writer._target_ == (
    "sam3.train.utils.logger.make_tensorboard_logger"
), f"Unexpected TensorBoard target: {cfg_base.trainer.logging.tensorboard_writer._target_}"

# Verify augmentation entries present in train_transforms ComposeAPI (D-03-04)
inner_targets = [
    t.get("_target_", "")
    for t in cfg_base.scratch.train_transforms[0].transforms
]
assert any("ColorJitter" in t for t in inner_targets), (
    f"ColorJitter not in train_transforms. Found: {inner_targets}"
)
assert any("GaussianBlur" in t for t in inner_targets), (
    f"GaussianBlur not in train_transforms. Found: {inner_targets}"
)
assert any("RandomErasing" in t for t in inner_targets), (
    f"RandomErasing not in train_transforms. Found: {inner_targets}"
)

# Verify segmentation enabled (inherited from Phase 2)
assert cfg_base.scratch.enable_segmentation is True, (
    f"Expected enable_segmentation=True, got {cfg_base.scratch.enable_segmentation}"
)

# Verify iou_type = segm (EVAL-02)
assert cfg_base.trainer.meters.val.custom.detection.iou_type == "segm", (
    f"Expected iou_type=segm, got {cfg_base.trainer.meters.val.custom.detection.iou_type}"
)
```

---

#### Script docstring pattern (test_config_parse.py lines 2–17):
```python
"""
Smoke test: validate that all custom_finetune Hydra configs compose without errors.

Run from the sam3 project root:
    python scripts/test_training_config.py

Tests:
  1. base.yaml: val_epoch_freq == 1, TensorBoard configured, ColorJitter/GaussianBlur/RandomErasingAPI
     in train_transforms, segmentation enabled, iou_type == "segm"
  2. decoder_only.yaml: inherits base augmentations, val_epoch_freq still 1
  3. full_finetune.yaml: inherits base augmentations, val_epoch_freq still 1

Uses the Hydra compose API with initialize_config_module("sam3.train"), matching
train.py exactly. Stubs out sam3 top-level __init__ to avoid requiring PyTorch >= 2.3.
Requires: hydra-core, omegaconf, numpy.
"""
```

---

## Shared Patterns

### Module-stub anti-import pattern
**Source:** `scripts/test_config_parse.py` lines 33–48
**Apply to:** `scripts/test_training_config.py`
```python
_SAM3_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _SAM3_ROOT)
sys.modules.setdefault("sam3", _make_package_stub("sam3", os.path.join(_SAM3_ROOT, "sam3")))
sys.modules.setdefault("sam3.train", _make_package_stub("sam3.train", os.path.join(_SAM3_ROOT, "sam3", "train")))
```
This pattern must precede any `from hydra import ...` or `from omegaconf import ...` calls.

---

### Hydra compose with `throw_on_missing=False`
**Source:** `scripts/test_config_parse.py` lines 77–83
**Apply to:** `scripts/test_training_config.py`
```python
cfg = compose(config_name=config_name)
OmegaConf.to_container(cfg, resolve=True, throw_on_missing=False)
```
`throw_on_missing=False` is required because `paths.*: null` are REQUIRED sentinels — they intentionally have no value at config-parse time.

---

### Hydra `_target_:` instantiation pattern in YAML
**Source:** `sam3/train/configs/custom_finetune/base.yaml` (entire file)
**Apply to:** new augmentation entries in `base.yaml`

Every class instantiated via Hydra uses:
```yaml
_target_: fully.qualified.module.ClassName
param1: value1
param2: value2
```
List entries inside `transforms:` start with `- _target_: ...` at 8-space indentation (inside `ComposeAPI.transforms`).

---

### torchvision delegate wrapping pattern
**Source:** `sam3/train/transforms/basic.py` lines 381–386 (`RandomErasing`)
**Apply to:** new `ColorJitter`, `GaussianBlur`, `RandomErasingAPI` classes in `basic.py`
```python
class SomeTransform:
    def __init__(self, *args, **kwargs):
        self.delegate = T.SomeTransform(*args, **kwargs)  # pass-through all params

    def __call__(self, datapoint, **kwargs):
        for img in datapoint.images:
            img.data = self.delegate(img.data)
        return datapoint
```
`*args, **kwargs` pass-through means YAML can specify any keyword that `torchvision.transforms` accepts without needing to list them explicitly in the wrapper.

---

### `datapoint.images` loop pattern
**Source:** `sam3/train/transforms/basic_for_api.py` lines 871–879 (`ToTensorAPI.__call__`)
**Apply to:** new `ColorJitter`, `GaussianBlur`, `RandomErasingAPI` in `basic.py`
```python
def __call__(self, datapoint, **kwargs):
    for img in datapoint.images:
        img.data = transform(img.data)
    return datapoint
```
`datapoint.images` is a **list of `Image` objects**, each having a `.data` attribute (PIL Image or tensor depending on pipeline stage). The loop iterates all images in the datapoint (typically 1 for image-only fine-tuning, >1 for video).

---

### Error collection + sys.exit(1) pattern
**Source:** `scripts/test_config_parse.py` lines 93–195
**Apply to:** `scripts/test_training_config.py`
```python
errors = []
try:
    # ... assertions ...
    print("✓ config_name")
except Exception as e:
    errors.append(f"✗ config_name: {e}")
    print(f"✗ config_name: {e}")

if errors:
    print(f"\n{len(errors)} config(s) FAILED:")
    for err in errors: print(f"  {err}")
    sys.exit(1)
else:
    print("\nAll configs parsed successfully.")
```

---

## No Analog Found

All three files have strong analogs. No files are without a match.

| File | Analog Used | Match Quality |
|---|---|---|
| `sam3/train/transforms/basic.py` (new classes) | `basic.py::RandomErasing` + `basic_for_api.py::ToTensorAPI` | exact (same file + same repo) |
| `sam3/train/configs/custom_finetune/base.yaml` (changes) | same file + `odinw_text_only_train.yaml` | exact |
| `scripts/test_training_config.py` | `scripts/test_config_parse.py` | exact |

---

## Key Warnings for Planner

1. **Interface mismatch is the critical risk:** The existing `basic.py::RandomErasing` uses `(img, target)` — **do NOT copy that call signature**. New classes must use `(datapoint, **kwargs)` + `for img in datapoint.images: img.data = ...` pattern from `basic_for_api.py`.

2. **PIL vs tensor stage split:** `ColorJitter` and `GaussianBlur` must be placed **before `ToTensorAPI`** (PIL stage); `RandomErasingAPI` must be placed **after `ToTensorAPI`** and **before `NormalizeAPI`** (tensor stage). Wrong placement raises `TypeError`.

3. **TensorBoard block already present:** `base.yaml` already has the full `trainer.logging.tensorboard_writer` block from Phase 2. D-03-07 requires NO change.

4. **Only two changes needed in base.yaml:** (1) `val_epoch_freq: 10 → 1`, (2) three augmentation entries inserted at correct pipeline positions.

5. **`--config` not `--config-name`:** `train.py` uses argparse with `--config` (or `-c`). `--config-name` is a Hydra `@hydra.main` convention not used here.

6. **`RandomErasingAPI` naming:** Use `RandomErasingAPI` (not `RandomErasing`) to avoid shadowing the existing `(img, target)` class. The YAML `_target_` must reference `sam3.train.transforms.basic.RandomErasingAPI`.

---

## Metadata

**Analog search scope:** `sam3/train/transforms/`, `sam3/train/configs/custom_finetune/`, `sam3/train/configs/odinw13/`, `scripts/`
**Files scanned:** 5 (basic.py, basic_for_api.py, base.yaml, test_config_parse.py, odinw_text_only_train.yaml)
**Pattern extraction date:** 2026-05-27
