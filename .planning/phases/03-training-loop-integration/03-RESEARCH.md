# Phase 3: Training Loop Integration - Research

**Researched:** 2026-05-27
**Domain:** SAM3 Hydra/Submitit training stack — config extension, transform augmentation, eval frequency
**Confidence:** HIGH

---

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- **D-03-01:** Use the existing `python sam3/train/train.py` launcher (NOT `torchrun`). The training stack uses `torch.multiprocessing.start_processes` internally via `submitit` LocalExecutor — `torchrun` is not the entry point.
- **D-03-02:** GPU count is controlled via `--num_gpus N` CLI flag (not config YAML). The canonical 2-GPU command is:
  ```bash
  python sam3/train/train.py --config configs/custom_finetune/base --num-gpus 2
  ```
  Single-GPU: `--num-gpus 1`. Do NOT document `torchrun`.
- **D-03-03:** Add `ColorJitter` and `GaussianBlur` wrapper classes to `sam3/train/transforms/basic.py`, following the exact `RandomErasing` pattern (wraps `torchvision.transforms.v2.ColorJitter` / `.GaussianBlur`).
- **D-03-04:** Add all three augmentation transforms (`ColorJitter`, `GaussianBlur`, `RandomErasing`) to `base.yaml`'s `train_transforms` pipeline. They should appear AFTER the resize/pad/normalize steps.
- **D-03-05:** Augmentation defaults for industrial defect data:
  - `ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.0)`
  - `GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))`
  - `RandomErasing(p=0.2, scale=(0.02, 0.1))`
- **D-03-06:** Set `val_epoch_freq: 1` in the trainer config block — run validation after every epoch.
- **D-03-07:** Add TensorBoard logging block to `base.yaml` using the existing pattern from `sam3/train/configs/odinw13/`.
- **D-03-08:** Create `scripts/test_training_config.py` — validates that the training config assembles correctly (datasets, trainer, loss instantiate via Hydra) without launching a real training run.

### the agent's Discretion
- Exact placement of augmentation transforms in the `train_transforms` list (before or after existing filter steps is the planner's call based on what makes semantic sense)
- Whether to add a `use_augmentation: true` flag in `scratch` to allow disabling augmentation without editing the transform list
- Val transform pipeline is NOT augmented (standard practice — only train pipeline gets augmentation)

### Deferred Ideas (OUT OF SCOPE)
- `torchrun` support
- `use_augmentation: true/false` flag (deferred to runbook phase)
</user_constraints>

---

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| TRAIN-01 | Fine-tuning runs end-to-end with `python -m sam3.train.train --config-name custom_finetune/base` on a multi-GPU workstation | Launcher is `sam3/train/train.py` with `--config` flag; DDP via `torch.multiprocessing.start_processes` already implemented |
| TRAIN-02 | Multi-GPU DDP training works automatically using existing `torch.distributed` infrastructure | `single_node_runner()` in `train.py` calls `torch.multiprocessing.start_processes`; no code changes needed |
| TRAIN-03 | Default hyperparameters set for small datasets: epochs=40, target_epoch_size=500, batch=1, grad_accum×4, LR=8e-5/2.5e-6 | Already in `base.yaml` from Phase 2; no changes needed for TRAIN-03 |
| TRAIN-04 | Data augmentation config includes ColorJitter, GaussianBlur, RandomErasing on top of existing pipeline | Requires: new classes in `basic.py` + 3 YAML entries in `train_transforms` |
| TRAIN-05 | Training checkpoints save to user-configurable output directory | Already in `base.yaml`: `checkpoint.save_dir: ${launcher.experiment_log_dir}/checkpoints`; no changes needed |
| TRAIN-06 | TensorBoard logging enabled for loss curves and evaluation metrics | Already in `base.yaml`: full `trainer.logging.tensorboard_writer` block present; no changes needed |
| EVAL-01 | Val loop evaluates and reports `coco_eval_segm_AP50`, `coco_eval_segm_APs`, `coco_eval_segm_AP` | Already in `base.yaml` via `PredictionDumper` + `CocoEvaluatorOfflineWithPredFileEvaluators`; requires `val_epoch_freq: 1` change |
| EVAL-02 | `iou_type: "segm"` set in evaluation config | Already in `base.yaml`: multiple `iou_type: "segm"` occurrences; no changes needed |
</phase_requirements>

---

## Summary

Phase 3 has a **very small surface area** compared to the ROADMAP description. The bulk of the training loop wiring (TensorBoard, eval metrics, DDP, checkpoints, hyperparameters) was completed in Phase 2 as part of `base.yaml`. The actual Phase 3 work is limited to four targeted changes: (1) change `val_epoch_freq` from 10 to 1, (2) add two new transform wrapper classes to `basic.py`, (3) insert three augmentation entries into `base.yaml`'s `train_transforms`, and (4) create the dry-run validation script.

**Critical discovery:** The CONTEXT.md describes `D-03-07` (TensorBoard block) and `D-03-06` (val_epoch_freq) as Phase 3 work, but inspection of `base.yaml` shows the TensorBoard block is **already fully present** from Phase 2 (complete with `flush_secs: 120`, `should_log: True`). The only eval-frequency change needed is `val_epoch_freq: 10` → `val_epoch_freq: 1` at line containing `val_epoch_freq` in `base.yaml`'s `trainer:` block.

**Transform pipeline interface constraint:** There is a critical interface mismatch that planners must resolve: `basic.py` transforms use `(img, target)` signatures, but the `base.yaml` pipeline is entirely API-style (`(datapoint, **kwargs)` via `ComposeAPI`). The new `ColorJitter` and `GaussianBlur` classes must have API-compatible signatures to work in the YAML pipeline. See Architecture Patterns → Transform Interface Decision for the recommended resolution.

**Primary recommendation:** Four atomic tasks, each independent; plan as Wave 1 (config/code) + Wave 2 (dry-run script + tests).

---

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| Multi-GPU launch | Training runner (`train.py`) | OS/process layer | `torch.multiprocessing.start_processes` already handles DDP process spawning |
| Augmentation logic | Transform classes (`basic.py`) | YAML config | Wrapper classes hold augmentation params; YAML wires them into pipeline |
| Eval frequency control | Trainer (`trainer.py`) | YAML config | `val_epoch_freq` param in `Trainer.__init__`; YAML sets value |
| TensorBoard output | Logger (`utils/logger.py`) | YAML config | `make_tensorboard_logger` already implemented; YAML selects it |
| Config validation | Dry-run script | Hydra compose API | Script uses `hydra.compose` + OmegaConf resolution to validate without training |

---

## Standard Stack

### Core (no new dependencies required)
| Library | Version | Purpose | Status |
|---------|---------|---------|--------|
| torchvision.transforms | bundled with torch | `T.ColorJitter`, `T.GaussianBlur`, `T.RandomErasing` wrappers | Already imported in `basic.py` as `import torchvision.transforms as T` |
| hydra-core | bundled | Config composition for dry-run script | Already used in `test_config_parse.py` |
| torch.utils.tensorboard | bundled | `SummaryWriter` via `make_tensorboard_logger` | Already imported in `utils/logger.py` |
| submitit | bundled | LocalExecutor for single-node training | Already used in `train.py` |

**No `pip install` required for Phase 3.** All required libraries are already present.

---

## Architecture Patterns

### System Architecture Diagram

```
CLI: python sam3/train/train.py --config configs/custom_finetune/base --num-gpus N
         │
         ▼
    main() [train.py]
         │
         ├── Hydra compose(config_name="configs/custom_finetune/base")
         │       └── base.yaml resolved: trainer, data, launcher, submitit
         │
         ├── submitit LocalExecutor (use_cluster=False)
         │       └── single_node_runner(cfg, main_port)
         │               │
         │               └── torch.multiprocessing.start_processes(single_proc_run, nprocs=N)
         │                       │
         │                       └── instantiate(cfg.trainer) → Trainer.run()
         │                               │
         │                               ├── Train loop
         │                               │     ├── Sam3ImageDataset.__getitem__
         │                               │     │     └── train_transforms pipeline:
         │                               │     │           ComposeAPI [
         │                               │     │             FilterCrowds → RandomizeInputBbox →
         │                               │     │             DecodeRle → RandomResizeAPI →
         │                               │     │             PadToSizeAPI →
         │                               │     │    [NEW] ColorJitter → GaussianBlur →
         │                               │     │             ToTensorAPI →
         │                               │     │    [NEW] RandomErasing →
         │                               │     │             FilterEmptyTargets →
         │                               │     │             NormalizeAPI →
         │                               │     │             FilterEmptyTargets
         │                               │     │           ]
         │                               │     └── Loss: Sam3LossWrapper (Boxes+BCE+Masks)
         │                               │
         │                               ├── Every epoch (val_epoch_freq=1) [CHANGED from 10]
         │                               │     └── PredictionDumper → CocoEvaluatorOffline
         │                               │           └── coco_eval_segm_AP, AP50, APs
         │                               │
         │                               └── Logging
         │                                     └── TensorBoard → experiment_log_dir/tensorboard/
         │                                           [ALREADY PRESENT in base.yaml]
         │
         └── Checkpoints → experiment_log_dir/checkpoints/
               save_best_meters: ["val_custom/detection"]
```

### Recommended Project Structure (Phase 3 changes only)
```
sam3/train/transforms/
├── basic.py              # ADD: ColorJitter class, GaussianBlur class (new)
│                         # EXISTING: RandomErasing class (reference pattern)
│
sam3/train/configs/custom_finetune/
├── base.yaml             # CHANGE: val_epoch_freq 10→1
│                         # ADD: 3 augmentation entries in scratch.train_transforms
│
scripts/
├── test_config_parse.py  # EXISTING (Phase 2 smoke test — reference pattern)
└── test_training_config.py  # NEW: dry-run config validation (Phase 3)
```

---

## Transform Interface Decision (Critical)

### The Interface Mismatch Problem [VERIFIED: codebase inspection]

The `basic.py` `RandomErasing` class uses a **non-API signature**:
```python
# basic.py — (img, target) style — NOT compatible with ComposeAPI
class RandomErasing:
    def __init__(self, *args, **kwargs):
        self.eraser = T.RandomErasing(*args, **kwargs)
    def __call__(self, img, target):       # ← takes (img, target)
        return self.eraser(img), target    # ← returns (img, target)
```

All transforms inside `ComposeAPI` in `base.yaml` use **API-style signatures**:
```python
# ComposeAPI.__call__ passes datapoint to each sub-transform
def __call__(self, datapoint, **kwargs):
    for t in self.transforms:
        datapoint = t(datapoint, **kwargs)   # ← expects (datapoint, **kwargs)
    return datapoint
```

`Sam3ImageDataset.__getitem__` also calls outer transforms as `transform(datapoint, epoch=...)`. **There is no section of the pipeline that uses `(img, target)` style.**

### Resolution (Recommended) [VERIFIED: basic_for_api.py inspection]

The new `ColorJitter` and `GaussianBlur` classes should be added to `basic.py` with **API-compatible `(datapoint, **kwargs)` signatures** that operate on `datapoint.images[i].data`, following the pattern of `ToTensorAPI` and `NormalizeAPI` in `basic_for_api.py`. The "RandomErasing pattern" refers to the **wrapping architecture** (delegate to torchvision with `*args, **kwargs`), not the exact call signature.

**Note:** `basic_for_api.py` already has a `ColorJitter` class at line 958, but it requires a `consistent_transform` parameter and has complex logic for multi-image video consistency. The simpler version in `basic.py` (without `consistent_transform`) is appropriate for single-image fine-tuning.

### Augmentation Placement in Pipeline

Apply at the correct stage based on torchvision requirements:

| Transform | Input Type Required | Correct Placement |
|-----------|--------------------|--------------------|
| `ColorJitter` | PIL image OR tensor | **AFTER `PadToSizeAPI`, BEFORE `ToTensorAPI`** (PIL preferred — natural pixel space) |
| `GaussianBlur` | PIL image OR tensor | **AFTER `PadToSizeAPI`, BEFORE `ToTensorAPI`** (PIL preferred) |
| `RandomErasing` | Tensor only | **AFTER `ToTensorAPI`, BEFORE `NormalizeAPI`** (tensor required; before normalization = natural pixel range) |

**Recommended final order inside `ComposeAPI.transforms`:**
```
FilterCrowds → RandomizeInputBbox → DecodeRle →
RandomResizeAPI → PadToSizeAPI →
[NEW] ColorJitter → [NEW] GaussianBlur →        ← PIL stage
ToTensorAPI → FilterEmptyTargets →
[NEW] RandomErasing →                           ← tensor stage (pre-norm)
NormalizeAPI → FilterEmptyTargets
```

### New Class Patterns for `basic.py`

```python
# Source: based on existing basic.py RandomErasing + basic_for_api.py API conventions
# [VERIFIED: basic.py:381-388, basic_for_api.py:867-899]

class ColorJitter:
    def __init__(self, *args, **kwargs):
        self.jitter = T.ColorJitter(*args, **kwargs)

    def __call__(self, datapoint, **kwargs):
        for img in datapoint.images:
            img.data = self.jitter(img.data)  # PIL or tensor → PIL or tensor
        return datapoint


class GaussianBlur:
    def __init__(self, *args, **kwargs):
        self.blur = T.GaussianBlur(*args, **kwargs)

    def __call__(self, datapoint, **kwargs):
        for img in datapoint.images:
            img.data = self.blur(img.data)  # PIL or tensor → PIL or tensor
        return datapoint
```

**Note on RandomErasing:** The existing `RandomErasing` in `basic.py` uses `(img, target)` which is NOT compatible with the API pipeline. For YAML usage, the planner should either: (a) add an API-compatible `RandomErasing` wrapper alongside the existing one in `basic.py`, or (b) use the existing `(img, target)` class only in non-API contexts. Given D-03-04 locks the placement, **the planner should add an API wrapper** that calls `T.RandomErasing` on `img.data`:

```python
# New API-compatible version (alongside existing (img,target) version)
class RandomErasingAPI:
    def __init__(self, *args, **kwargs):
        self.eraser = T.RandomErasing(*args, **kwargs)

    def __call__(self, datapoint, **kwargs):
        for img in datapoint.images:
            img.data = self.eraser(img.data)  # tensor required
        return datapoint
```

OR the planner can simply rename the new class `RandomErasing` and reference it as `sam3.train.transforms.basic.RandomErasing` — but the existing `RandomErasing` uses `(img, target)`. The cleanest solution is to add the API-compatible versions with distinct names or simply make the new basic.py `ColorJitter` and `GaussianBlur` API-compatible and add an `RandomErasing` API-compatible version alongside the existing one.

---

## YAML Changes Required

### 1. `val_epoch_freq` Change [VERIFIED: base.yaml inspection]

**File:** `sam3/train/configs/custom_finetune/base.yaml`
**Location:** `trainer:` block
**Change:** `val_epoch_freq: 10` → `val_epoch_freq: 1`

```yaml
trainer:
  _target_: sam3.train.trainer.Trainer
  # ... other params ...
  val_epoch_freq: 1    # CHANGED from 10 → evaluate every epoch (D-03-06)
```

### 2. Augmentation in `train_transforms` [VERIFIED: base.yaml inspection]

**File:** `sam3/train/configs/custom_finetune/base.yaml`
**Location:** `scratch.train_transforms[0].transforms` list (inside `ComposeAPI`)

Add three entries in the recommended positions:

```yaml
# After PadToSizeAPI, before ToTensorAPI — PIL image stage:
- _target_: sam3.train.transforms.basic.ColorJitter
  brightness: 0.2
  contrast: 0.2
  saturation: 0.2
  hue: 0.0

- _target_: sam3.train.transforms.basic.GaussianBlur
  kernel_size: 3
  sigma: [0.1, 2.0]

# After ToTensorAPI, before NormalizeAPI — tensor stage:
- _target_: sam3.train.transforms.basic.RandomErasingAPI  # or renamed class
  p: 0.2
  scale: [0.02, 0.1]
```

**Val transforms are NOT modified** — augmentation on train pipeline only (D-03-05).

### 3. TensorBoard Block — Already Present [VERIFIED: base.yaml inspection]

**Finding:** The TensorBoard block is ALREADY in `base.yaml` from Phase 2. No change needed for D-03-07.

```yaml
# Already present in base.yaml trainer.logging:
logging:
  tensorboard_writer:
    _target_: sam3.train.utils.logger.make_tensorboard_logger
    log_dir: ${launcher.experiment_log_dir}/tensorboard
    flush_secs: 120
    should_log: True
  wandb_writer: null
  log_dir: ${launcher.experiment_log_dir}/logs
  log_freq: 10
```

---

## What's Already Done in Phase 2

The following requirements are **fully satisfied by Phase 2's `base.yaml`** — NO changes needed:

| Requirement | What's Already in base.yaml |
|-------------|------------------------------|
| TRAIN-01 (end-to-end launch) | `launcher:`, `submitit:`, `trainer:` all wired; `use_cluster: False` for local |
| TRAIN-02 (multi-GPU DDP) | `distributed.backend: nccl`, `find_unused_parameters: True`, `gradient_as_bucket_view: True` |
| TRAIN-03 (small-dataset defaults) | `max_epochs: 40`, `train_batch_size: 1`, `gradient_accumulation_steps: 4`, `lr_transformer: 8e-5`, `lr_vision_backbone: 2.5e-6` |
| TRAIN-05 (checkpoints) | `checkpoint.save_dir: ${launcher.experiment_log_dir}/checkpoints`, `save_best_meters: ["val_custom/detection"]` |
| TRAIN-06 (TensorBoard) | Full `tensorboard_writer` block with `make_tensorboard_logger` and correct `log_dir` |
| EVAL-01 (metrics) | `PredictionDumper` + `CocoEvaluatorOfflineWithPredFileEvaluators` with `iou_type: "segm"` |
| EVAL-02 (`iou_type: segm`) | `iou_type: "segm"` set in both `segm_postprocessor` and evaluator |

---

## CLI Command Correction

**CONTEXT.md says:** `--config-name` and `--num_gpus`
**Actual `train.py` argparse:** `--config` (alias `-c`) and `--num-gpus` (hyphen, not underscore)

[VERIFIED: `sam3/train/train.py` argparse inspection]

```bash
# Correct command (from actual argparse definition):
python sam3/train/train.py --config configs/custom_finetune/base --num-gpus 2

# Python internally converts --num-gpus → args.num_gpus (argparse hyphen→underscore)
# But the CLI flag must use a hyphen: --num-gpus N
```

The CONTEXT.md D-03-02 documents `--num_gpus` (underscore) but the CLI flag is `--num-gpus` (hyphen). Both work from the shell because argparse maps them, but documentation should use `--num-gpus`.

---

## Dry-Run Script Pattern

### Reference: `scripts/test_config_parse.py` [VERIFIED: file inspection]

Pattern:
1. Stub `sam3` package to avoid PyTorch ≥ 2.3 model deps
2. Register OmegaConf resolvers inline (same set as `train.py`)
3. `initialize_config_module("sam3.train", version_base="1.2")`
4. `compose(config_name="configs/custom_finetune/base")`
5. `OmegaConf.to_container(cfg, resolve=True, throw_on_missing=False)`
6. Assert specific values; collect errors; `sys.exit(1)` if any

### New `scripts/test_training_config.py` Assertions

The new dry-run script should verify the Phase 3 additions:

```python
# Verify val_epoch_freq = 1
assert cfg.trainer.val_epoch_freq == 1, f"Expected 1, got {cfg.trainer.val_epoch_freq}"

# Verify TensorBoard block present
assert cfg.trainer.logging.tensorboard_writer is not None, "TensorBoard writer not configured"
assert cfg.trainer.logging.tensorboard_writer._target_ == "sam3.train.utils.logger.make_tensorboard_logger"

# Verify augmentation entries in train_transforms
# (check at least one of the ComposeAPI inner transforms references basic.ColorJitter)
inner_targets = [
    t.get("_target_", "") 
    for t in cfg.scratch.train_transforms[0].transforms
]
assert any("ColorJitter" in t for t in inner_targets), "ColorJitter not in train_transforms"
assert any("GaussianBlur" in t for t in inner_targets), "GaussianBlur not in train_transforms"
assert any("RandomErasing" in t for t in inner_targets), "RandomErasing not in train_transforms"

# Verify iou_type = segm
assert cfg.trainer.meters.val.custom.detection.iou_type == "segm"

# Verify segmentation enabled
assert cfg.scratch.enable_segmentation is True
```

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Multi-GPU training | Custom DDP loop | `torch.multiprocessing.start_processes` (already in `train.py`) | Handles process spawning, env var setup, port assignment |
| TensorBoard logging | Custom file writer | `torch.utils.tensorboard.SummaryWriter` via `make_tensorboard_logger` | Already implemented and wired |
| Color augmentation | Custom pixel jitter | `torchvision.transforms.ColorJitter` | Handles brightness/contrast/saturation/hue correctly |
| Blur augmentation | Custom convolution | `torchvision.transforms.GaussianBlur` | Handles kernel size / sigma correctly |
| Random erasing | Custom masking | `torchvision.transforms.RandomErasing` | Handles scale/ratio/value params correctly |
| COCO AP evaluation | Custom metric | `CocoEvaluatorOfflineWithPredFileEvaluators` (already in base.yaml) | AP50, APs, AP correctly computed |

**Key insight:** All the hard infrastructure was built in Phase 2. Phase 3 is config and thin wrapper code only.

---

## Common Pitfalls

### Pitfall 1: Wrong Transform Placement Breaks PIL/Tensor Contract
**What goes wrong:** `T.RandomErasing` requires a tensor input. Placing it before `ToTensorAPI` raises `TypeError: expected Tensor as image, but got PIL.Image.Image`.
**Why it happens:** PIL stage transforms (ColorJitter, GaussianBlur) and tensor stage transforms (RandomErasing) are mixed in the pipeline.
**How to avoid:** ColorJitter + GaussianBlur go BEFORE `ToTensorAPI`; RandomErasing goes AFTER `ToTensorAPI`.
**Warning signs:** `TypeError: expected Tensor` during dataset iteration.

### Pitfall 2: `(img, target)` Classes Don't Work in ComposeAPI
**What goes wrong:** If the new `ColorJitter`/`GaussianBlur` use `(img, target)` signature like the existing `RandomErasing` in `basic.py`, they fail when called as `transform(datapoint, epoch=...)` by `ComposeAPI`.
**Why it happens:** `ComposeAPI.__call__` passes a `Datapoint` object, not `(img, target)`.
**How to avoid:** New classes in `basic.py` must have `(datapoint, **kwargs)` signatures that access `datapoint.images[i].data`.
**Warning signs:** `AttributeError: 'Datapoint' object has no attribute 'size'` or similar during dataset init.

### Pitfall 3: `--config-name` vs `--config` Mismatch
**What goes wrong:** `python sam3/train/train.py --config-name configs/custom_finetune/base` fails with `unrecognized arguments: --config-name`.
**Why it happens:** The train.py argparse uses `--config` (not `--config-name` which is a Hydra CLI convention for `@hydra.main`-decorated functions). `train.py` uses the Hydra compose API, not `@hydra.main`.
**How to avoid:** Always use `--config` (or `-c`).
**Warning signs:** `argparse: unrecognized arguments`.

### Pitfall 4: `val_epoch_freq=10` Means Eval Only at Epoch 10, 20, 30, 40
**What goes wrong:** With 40 epochs and `val_epoch_freq=10`, validation only runs 4 times total, and no checkpoint is saved until epoch 10.
**Why it happens:** Phase 2 left the default at `val_epoch_freq: 10`.
**How to avoid:** Set to `1` per D-03-06.
**Warning signs:** Training runs for 10 epochs with no validation output in logs.

### Pitfall 5: `Datapoint.images[i].data` Access Pattern
**What goes wrong:** New wrapper classes access `datapoint.images.data` (flat) instead of `datapoint.images[i].data` (indexed).
**Why it happens:** Datapoint stores a list of Image objects at `datapoint.images`.
**How to avoid:** Follow `ToTensorAPI`/`NormalizeAPI` pattern: `for img in datapoint.images: img.data = transform(img.data)`.
**Warning signs:** `AttributeError: 'list' object has no attribute 'data'`.

---

## Code Examples

### Existing `RandomErasing` Pattern (Reference, `basic.py`) [VERIFIED: basic.py:381-388]
```python
# basic.py — (img, target) style — NOT for API pipeline use
class RandomErasing:
    def __init__(self, *args, **kwargs):
        self.eraser = T.RandomErasing(*args, **kwargs)

    def __call__(self, img, target):
        return self.eraser(img), target
```

### API-Compatible Pattern (Follow for new classes) [VERIFIED: basic_for_api.py:867-899]
```python
# Pattern from ToTensorAPI and NormalizeAPI in basic_for_api.py
class ToTensorAPI:
    def __call__(self, datapoint: Datapoint, **kwargs):
        for img in datapoint.images:
            img.data = F.to_tensor(img.data)   # ← access via img.data
        return datapoint                        # ← return datapoint
```

### `val_epoch_freq` Usage in Trainer [VERIFIED: trainer.py:159,464]
```python
# trainer.py line 159 — constructor param with default=1
def __init__(self, ..., val_epoch_freq: int = 1, ...):
    self.val_epoch_freq = val_epoch_freq

# trainer.py line 464 — used in eval scheduling
epoch % self.val_epoch_freq == 0
```

### `make_tensorboard_logger` Signature [VERIFIED: utils/logger.py:22]
```python
# utils/logger.py — already implemented
def make_tensorboard_logger(log_dir: str, **writer_kwargs: Any):
    # Returns a TBLogger wrapping SummaryWriter
    # writer_kwargs: flush_secs, max_queue, purge_step, etc.
```

### DDP Launch Path [VERIFIED: train.py:single_node_runner]
```python
# train.py — local multi-GPU (use_cluster=False)
def single_node_runner(cfg, main_port: int):
    assert cfg.launcher.num_nodes == 1
    num_proc = cfg.launcher.gpus_per_node
    torch.multiprocessing.set_start_method("spawn")
    if num_proc == 1:
        single_proc_run(local_rank=0, ...)
    else:
        torch.multiprocessing.start_processes(
            single_proc_run, args=(main_port, cfg, num_proc),
            nprocs=num_proc, start_method="spawn"
        )
```

---

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | `T.ColorJitter` and `T.GaussianBlur` work on PIL images in the transform position after `PadToSizeAPI` | Transform Placement | Low — both classes support PIL; confirmed by torchvision docs [ASSUMED: training data, not verified via live import] |
| A2 | `basic_for_api.py::ColorJitter` requires `consistent_transform` param making it unsuitable for simple wrapping | Transform Classes | Low — if incorrect, could use existing class instead of adding new one |

---

## Open Questions (RESOLVED)

1. **`RandomErasing` naming conflict in `basic.py`**
   - What we know: `basic.py` already has `RandomErasing` with `(img, target)` signature; planner needs API-compatible version
   - What's unclear: Should planner add `RandomErasingAPI` (new name in `basic.py`) or replace the existing class?
   - RESOLVED: Add `RandomErasingAPI` alongside existing — preserves backward compatibility, avoids breaking any code that imports the `(img, target)` version

2. **Augmentation on `decoder_only.yaml` and `full_finetune.yaml`**
   - What we know: These configs inherit from `base.yaml`; `train_transforms` is under `scratch` namespace
   - What's unclear: Do decoder_only/full_finetune configs override `train_transforms`? If so, augmentation must be added there too
   - RESOLVED: Check both files. Since they currently only override LR-related `scratch` fields, augmentation entries added to `base.yaml` will be inherited. No changes needed to override configs.

3. **`test_training_config.py` scope: validate all 3 configs or just base?**
   - What we know: `test_config_parse.py` validates all 3 (base, decoder_only, full_finetune)
   - What's unclear: CONTEXT.md says "all three configs (base, decoder_only, full_finetune) can instantiate their datasets and trainers" 
   - RESOLVED: New script should test all 3 configs, asserting val_epoch_freq=1, TensorBoard block presence, and augmentation entries for each

---

## Environment Availability

Step 2.6: All dependencies are already available — this phase installs nothing new.

| Dependency | Required By | Available | Notes |
|------------|------------|-----------|-------|
| `torchvision.transforms.ColorJitter` | New transform wrapper | ✓ | Bundled with torchvision; already imported in `basic.py` |
| `torchvision.transforms.GaussianBlur` | New transform wrapper | ✓ | Bundled with torchvision; already imported in `basic.py` |
| `torchvision.transforms.RandomErasing` | Existing wrapper reference | ✓ | Bundled with torchvision; already used in `basic.py` |
| `torch.utils.tensorboard.SummaryWriter` | TensorBoard writer | ✓ | Already imported in `utils/logger.py` |

---

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest (existing tests/ directory with `tests/test_prepare_dataset.py`) |
| Config file | `pyproject.toml` (existing) |
| Quick run command | `python scripts/test_training_config.py` |
| Full suite command | `python -m pytest tests/ -x -q && python scripts/test_training_config.py` |

### Phase Requirements → Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| TRAIN-04 | Augmentation entries in train_transforms | unit | `python scripts/test_training_config.py` | ❌ Wave 0 |
| TRAIN-06 | TensorBoard block present in config | unit | `python scripts/test_training_config.py` | ❌ Wave 0 |
| EVAL-01 | val_epoch_freq=1 in assembled config | unit | `python scripts/test_training_config.py` | ❌ Wave 0 |
| EVAL-02 | iou_type=segm in assembled config | unit | `python scripts/test_training_config.py` | ❌ Wave 0 |
| TRAIN-01 | Config composes without errors | smoke | `python scripts/test_training_config.py` | ❌ Wave 0 |
| TRAIN-02 | DDP wiring (cfg.launcher.gpus_per_node) | unit | `python scripts/test_training_config.py` | ❌ Wave 0 |
| TRAIN-03 | Hyperparameter values correct | unit | `python scripts/test_config_parse.py` | ✅ exists |

### Sampling Rate
- **Per task commit:** `python scripts/test_training_config.py`
- **Per wave merge:** `python -m pytest tests/ -x -q && python scripts/test_training_config.py`
- **Phase gate:** Full suite green before `/gsd-verify-work`

### Wave 0 Gaps
- [ ] `scripts/test_training_config.py` — covers TRAIN-04, TRAIN-06, EVAL-01, EVAL-02, TRAIN-01

---

## Security Domain

> `security_enforcement` not set to false — including for completeness.

### Applicable ASVS Categories

| ASVS Category | Applies | Note |
|---------------|---------|------|
| V2 Authentication | No | Local training script, no auth surface |
| V5 Input Validation | Low | Config values come from YAML under researcher control |
| V6 Cryptography | No | No secrets or encryption in scope |

**No security-sensitive changes in Phase 3** — all work is config extension and transform wrappers on a local training tool.

---

## Sources

### Primary (HIGH confidence)
- `sam3/train/train.py` — CLI argparse definition, single_node_runner, submitit integration [VERIFIED: file inspection]
- `sam3/train/trainer.py` — LoggingConf dataclass, val_epoch_freq param (line 159), TensorBoard integration [VERIFIED: file inspection]
- `sam3/train/transforms/basic.py` — RandomErasing class (line 381) as pattern reference [VERIFIED: file inspection]
- `sam3/train/transforms/basic_for_api.py` — ComposeAPI (line 922), ToTensorAPI (line 867), NormalizeAPI (line 882), ColorJitter (line 958) [VERIFIED: file inspection]
- `sam3/train/configs/custom_finetune/base.yaml` — Complete current state; TensorBoard already present, val_epoch_freq=10 [VERIFIED: file inspection]
- `sam3/train/data/sam3_image_dataset.py` — transform calling convention `transform(datapoint, epoch=...)` [VERIFIED: file inspection]
- `scripts/test_config_parse.py` — Dry-run script pattern to follow [VERIFIED: file inspection]

### Secondary (MEDIUM confidence)
- `sam3/train/configs/odinw13/odinw_text_only_train.yaml` — Reference for transform pipeline structure [VERIFIED: file inspection]
- `sam3/train/configs/roboflow_v100/roboflow_v100_full_ft_100_images.yaml` — Confirms no augmentation in reference configs [VERIFIED: file inspection]

---

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — all dependencies verified via codebase inspection
- Architecture: HIGH — training loop, DDP, transforms all verified via code reading
- Pitfalls: HIGH — interface mismatch verified by inspecting both ComposeAPI and RandomErasing signatures
- Current base.yaml state: HIGH — read the actual file

**Research date:** 2026-05-27
**Valid until:** 2026-06-27 (stable codebase — no external dependencies change)
