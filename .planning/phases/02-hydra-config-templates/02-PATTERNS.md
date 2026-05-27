# Phase 2: Hydra Config Templates - Pattern Map

**Mapped:** 2026-05-27
**Files analyzed:** 4 new files
**Analogs found:** 4 / 4

---

## File Classification

| New/Modified File | Role | Data Flow | Closest Analog | Match Quality |
|-------------------|------|-----------|----------------|---------------|
| `sam3/train/configs/custom_finetune/base.yaml` | config | CRUD (training loop) | `sam3/train/configs/roboflow_v100/roboflow_v100_full_ft_100_images.yaml` | exact |
| `sam3/train/configs/custom_finetune/finetune_strategy/decoder_only.yaml` | config | CRUD (override delta) | `sam3/train/configs/silver_image_evals/sam3_silver_image_yt1b.yaml` | role-match |
| `sam3/train/configs/custom_finetune/finetune_strategy/full_finetune.yaml` | config | CRUD (override delta) | `sam3/train/configs/silver_image_evals/sam3_silver_image_yt1b.yaml` | role-match |
| `scripts/test_config_parse.py` | utility (smoke test) | request-response | `scripts/prepare_dataset.py` | role-match |

---

## Pattern Assignments

### `sam3/train/configs/custom_finetune/base.yaml` (config, CRUD training loop)

**Analog:** `sam3/train/configs/roboflow_v100/roboflow_v100_full_ft_100_images.yaml`

**What to copy verbatim vs what to change:**
- ✅ Copy: `# @package _global_` + `defaults: [_self_]` header (lines 1–3)
- ✅ Copy: `scratch:` section structure (matcher, pos_embed, collate_fn, scheduler)
- ✅ Copy: entire transform pipeline (train_transforms, val_transforms) as-is
- ✅ Copy: entire `trainer.optim` block (lines 339–395) — optimizer, scheduler, param_group_modifiers
- ✅ Copy: `trainer.distributed`, `trainer.model`, `trainer.checkpoint`, `trainer.logging` structure
- ⚠️ CHANGE: `paths:` section — replace with 5 new fields (`dataset_img_folder`, `train_ann_file`, `val_ann_file`, `experiment_log_dir`, `bpe_path`)
- ⚠️ CHANGE: `scratch.enable_segmentation: False` → `true` (D-13)
- ⚠️ CHANGE: loss block — uncomment the segmentation loss (Masks entry); see Research Q3
- ⚠️ CHANGE: `trainer.data.train.dataset.img_folder/ann_file` — use `paths.dataset_img_folder` and `paths.train_ann_file`
- ⚠️ CHANGE: `trainer.data.val.dataset.img_folder/ann_file` — use `paths.dataset_img_folder` and `paths.val_ann_file`
- ⚠️ CHANGE: LR/epoch values for small-dataset defaults (D-12)
- ⚠️ CHANGE: eval meters to use `custom` dict_key and `iou_type: "segm"` (from Research Q4)
- ⚠️ CHANGE: `launcher.gpus_per_node: 2` → `1`; `submitit.use_cluster: True` → `False`
- ❌ DELETE: `roboflow_train:` top-level section and `all_roboflow_supercategories:` (entire Roboflow-specific block)

---

**Header pattern** (analog lines 1–3 — copy exactly):
```yaml
# @package _global_
defaults:
  - _self_
```

**Paths pattern** (analog lines 8–11 — replace fields):
```yaml
# Analog (roboflow_v100):
paths:
  roboflow_vl_100_root: <YOUR_DATASET_DIR>
  experiment_log_dir: <YOUR EXPERIMENET LOG_DIR>
  bpe_path: <BPE_PATH> # This should be under sam3/assets/bpe_simple_vocab_16e6.txt.gz

# → base.yaml version (D-05, D-06, Research Q8):
paths:
  dataset_img_folder: null  # REQUIRED: absolute path to images directory
  train_ann_file: null      # REQUIRED: absolute path to train.json from prepare_dataset.py
  val_ann_file: null        # REQUIRED: absolute path to val.json from prepare_dataset.py
  experiment_log_dir: null  # REQUIRED: where checkpoints, TensorBoard logs, and evals are saved
  bpe_path: null            # Optional: null uses bundled vocab at sam3/assets/bpe_simple_vocab_16e6.txt.gz
```

**`scratch:` key mutations** (analog lines 159–237):
```yaml
# Analog defaults → base.yaml overrides:
#   enable_segmentation: False   →  true           (D-13)
#   lr_scale: 0.1                →  0.03           (D-10: decoder-only)
#   lr_transformer: ${times:...} →  8e-5           (D-12: explicit literal)
#   lr_vision_backbone: ...      →  2.5e-6         (D-12)
#   lr_language_backbone: ...    →  1.5e-6         (derived from pattern)
#   max_data_epochs: 20          →  40             (D-12)
#   target_epoch_size: 1500      →  500            (D-12: small dataset)
#   train_batch_size: 1          →  1              (unchanged — D-12)
#   gradient_accumulation_steps: 1 → 4            (D-12)
#   collate_fn.dict_key: all/roboflow100 → custom  (rename for custom dataset)
#   collate_fn_val.dict_key: same → custom

# Copy VERBATIM from analog:
  d_model: 256
  pos_embed: ...               # lines 163–168: copy as-is
  use_presence_eval: True
  original_box_postprocessor: ... # lines 173–177: copy as-is
  matcher: ...                 # lines 180–188: copy as-is
  scale_by_find_batch_size: True
  resolution: 1008
  consistent_transform: False
  max_ann_per_img: 200
  train_norm_mean: [0.5, 0.5, 0.5]  # lines 197–200: copy as-is (D-14)
  train_norm_std: [0.5, 0.5, 0.5]
  val_norm_mean: [0.5, 0.5, 0.5]
  val_norm_std: [0.5, 0.5, 0.5]
  num_train_workers: 10
  num_val_workers: 0
  hybrid_repeats: 1
  context_length: 2
  gather_pred_via_filesys: false
  wd: 0.1
  scheduler_timescale: 20
  scheduler_warmup: 20
  scheduler_cooldown: 20
  val_batch_size: 1
```

**Segmentation postprocessor to ADD to `scratch:`** (Research Q4):
```yaml
# Add this block in scratch: alongside original_box_postprocessor
  segm_postprocessor:
    _target_: sam3.eval.postprocessors.PostProcessImage
    max_dets_per_img: -1
    iou_type: "segm"               # enables pred_masks processing
    use_original_ids: true
    use_original_sizes_box: true
    use_original_sizes_mask: true  # resize masks to original image size
    convert_mask_to_rle: True      # required for COCO eval JSON format
    use_presence: ${scratch.use_presence_eval}
```

**Loss block — segmentation version** (analog lines 107–154 — uncommented form, Research Q3):
```yaml
# Replace the active non-segmentation loss block (lines 75–104) with this:
loss:
  _target_: sam3.train.loss.sam3_loss.Sam3LossWrapper
  matcher: ${scratch.matcher}
  o2m_weight: 2.0
  o2m_matcher:
    _target_: sam3.train.matcher.BinaryOneToManyMatcher
    alpha: 0.3
    threshold: 0.4
    topk: 4
  use_o2m_matcher_on_o2m_aux: false
  loss_fns_find:
    - _target_: sam3.train.loss.loss_fns.Boxes
      weight_dict:
        loss_bbox: 5.0
        loss_giou: 2.0
    - _target_: sam3.train.loss.loss_fns.IABCEMdetr
      weak_loss: False
      weight_dict:
        loss_ce: 20.0
        presence_loss: 20.0
      pos_weight: 10.0
      alpha: 0.25
      gamma: 2
      use_presence: True
      pos_focal: false
      pad_n_queries: 200
      pad_scale_pos: 1.0
    - _target_: sam3.train.loss.loss_fns.Masks  # segmentation mask loss
      focal_alpha: 0.25
      focal_gamma: 2.0
      weight_dict:
        loss_mask: 200.0
        loss_dice: 10.0
      compute_aux: false
  loss_fn_semantic_seg: null   # optional per-pixel semantic seg; null = disabled
  scale_by_find_batch_size: ${scratch.scale_by_find_batch_size}
```

**`trainer.data` section** (analog lines 266–312 — path substitution):
```yaml
# Analog uses ${paths.roboflow_vl_100_root}/...  → base.yaml uses direct paths.* refs:
data:
  train:
    _target_: sam3.train.data.torch_dataset.TorchDataset
    dataset:
      _target_: sam3.train.data.sam3_image_dataset.Sam3ImageDataset
      img_folder: ${paths.dataset_img_folder}   # D-05: REQUIRED
      ann_file: ${paths.train_ann_file}           # D-05: REQUIRED
      load_segmentation: ${scratch.enable_segmentation}
      transforms: ...  # train_transforms chain (copy from roboflow_train.train_transforms)
      max_ann_per_img: 500000
      multiplier: 1
      max_train_queries: 50000
      max_val_queries: 50000
      training: true
      use_caching: False
    shuffle: True
    batch_size: ${scratch.train_batch_size}
    num_workers: ${scratch.num_train_workers}
    pin_memory: True
    drop_last: True
    collate_fn: ${scratch.collate_fn}

  val:
    _target_: sam3.train.data.torch_dataset.TorchDataset
    dataset:
      _target_: sam3.train.data.sam3_image_dataset.Sam3ImageDataset
      img_folder: ${paths.dataset_img_folder}   # D-05: REQUIRED (same folder as train)
      ann_file: ${paths.val_ann_file}             # D-05: REQUIRED
      load_segmentation: ${scratch.enable_segmentation}
      coco_json_loader:
        _target_: sam3.train.data.coco_json_loaders.COCO_FROM_JSON
        include_negatives: true
        category_chunk_size: 2
        _partial_: true
      transforms: ...  # val_transforms chain (copy from roboflow_train.val_transforms)
      max_ann_per_img: 100000
      multiplier: 1
      training: false
    shuffle: False
    batch_size: ${scratch.val_batch_size}
    num_workers: ${scratch.num_val_workers}
    pin_memory: True
    drop_last: False
    collate_fn: ${scratch.collate_fn_val}
```

**`trainer.meters.val` — segmentation version** (Research Q4):
```yaml
# Replace analog's bbox meter (lines 322–336) with segm meter:
meters:
  val:
    custom:  # must match scratch.collate_fn.dict_key
      detection:
        _target_: sam3.eval.coco_writer.PredictionDumper
        iou_type: "segm"
        dump_dir: ${launcher.experiment_log_dir}/dumps/custom
        merge_predictions: True
        postprocessor: ${scratch.segm_postprocessor}
        gather_pred_via_filesys: ${scratch.gather_pred_via_filesys}
        maxdets: 100
        pred_file_evaluators:
          - _target_: sam3.eval.coco_eval_offline.CocoEvaluatorOfflineWithPredFileEvaluators
            gt_path: ${paths.val_ann_file}
            tide: False
            iou_type: "segm"
```

**`trainer.optim` block** (analog lines 339–395 — copy verbatim):
```yaml
# Copy these lines exactly from roboflow_v100 config:
optim:
  amp:
    enabled: True
    amp_dtype: bfloat16
  optimizer:
    _target_: torch.optim.AdamW
  gradient_clip:
    _target_: sam3.train.optim.optimizer.GradientClipper
    max_norm: 0.1
    norm_type: 2
  param_group_modifiers:
    - _target_: sam3.train.optim.optimizer.layer_decay_param_modifier
      _partial_: True
      layer_decay_value: ${scratch.lrd_vision_backbone}
      apply_to: 'backbone.vision_backbone.trunk'
      overrides:
        - pattern: '*pos_embed*'
          value: 1.0
  options:
    lr:
      - scheduler:
          _target_: sam3.train.optim.schedulers.InverseSquareRootParamScheduler
          base_lr: ${scratch.lr_transformer}
          timescale: ${scratch.scheduler_timescale}
          warmup_steps: ${scratch.scheduler_warmup}
          cooldown_steps: ${scratch.scheduler_cooldown}
      - scheduler:
          _target_: sam3.train.optim.schedulers.InverseSquareRootParamScheduler
          base_lr: ${scratch.lr_vision_backbone}
          timescale: ${scratch.scheduler_timescale}
          warmup_steps: ${scratch.scheduler_warmup}
          cooldown_steps: ${scratch.scheduler_cooldown}
        param_names:
          - 'backbone.vision_backbone.*'
      - scheduler:
          _target_: sam3.train.optim.schedulers.InverseSquareRootParamScheduler
          base_lr: ${scratch.lr_language_backbone}
          timescale: ${scratch.scheduler_timescale}
          warmup_steps: ${scratch.scheduler_warmup}
          cooldown_steps: ${scratch.scheduler_cooldown}
        param_names:
          - 'backbone.language_backbone.*'
    weight_decay:
      - scheduler:
          _target_: fvcore.common.param_scheduler.ConstantParamScheduler
          value: ${scratch.wd}
      - scheduler:
          _target_: fvcore.common.param_scheduler.ConstantParamScheduler
          value: 0.0
        param_names:
          - '*bias*'
        module_cls_names: ['torch.nn.LayerNorm']
```

**`trainer.checkpoint` pattern** (analog lines 397–399 — copy, add save_best_meters):
```yaml
# Copy base, add save_best_meters per Research Q10:
checkpoint:
  save_dir: ${launcher.experiment_log_dir}/checkpoints
  save_freq: 0           # 0 = only save last checkpoint (plus best-metric ones)
  save_best_meters:
    - "val_custom/detection"  # saves best-AP50 checkpoint automatically
```

**`launcher` and `submitit` sections** (analog lines 415–433 — change for local use):
```yaml
# Analog:  gpus_per_node: 2, use_cluster: True
# base.yaml safe local defaults:
launcher:
  num_nodes: 1
  gpus_per_node: 1         # change to 2+ for multi-GPU
  experiment_log_dir: ${paths.experiment_log_dir}
  multiprocessing_context: forkserver

submitit:
  account: null
  partition: null
  qos: null
  timeout_hour: 72
  use_cluster: False       # True = submit to SLURM; False = run locally
  cpus_per_task: 10
  port_range: [10000, 65000]
  constraint: null
```

---

### `sam3/train/configs/custom_finetune/finetune_strategy/decoder_only.yaml` (config, CRUD override delta)

**Analog:** `sam3/train/configs/silver_image_evals/sam3_silver_image_yt1b.yaml` (lines 1–4)

**What to copy:**
- Header pattern showing `defaults` list inheritance with absolute path

**What to write:**
- Only the `# @package _global_` + `defaults:` block and a `scratch:` delta with NO fields overridden
- This file is effectively a no-op (base.yaml already ships with decoder-only as default per D-10)
- Its purpose is documentation: explicitly labelling the decoder-only strategy

**Header + defaults inheritance pattern** (analog lines 1–4):
```yaml
# @package _global_
defaults:
  - /configs/eval_base.yaml   # analog: inherits parent
  - _self_

# → decoder_only.yaml version:
# @package _global_
defaults:
  - /configs/custom_finetune/base   # absolute path → sam3/train/configs/custom_finetune/base.yaml
  - _self_
```

**Full file content** (no analog for single-field override files — Research Q7):
```yaml
# @package _global_
defaults:
  - /configs/custom_finetune/base
  - _self_

# Decoder-only fine-tuning strategy (default in base.yaml).
# This file exists to make the strategy explicit and selectable by name.
# Usage: python sam3/train/train.py -c custom_finetune/finetune_strategy/decoder_only
#
# Effect: lr_scale=0.03 keeps backbone LR near-zero (~2.5e-6 effective),
# training only the transformer decoder and class heads.
# Recommended for <500 images where full fine-tune risks overfitting.

# No overrides needed — base.yaml already defaults to decoder-only.
```

---

### `sam3/train/configs/custom_finetune/finetune_strategy/full_finetune.yaml` (config, CRUD override delta)

**Analog:** `sam3/train/configs/silver_image_evals/sam3_silver_image_yt1b.yaml` (lines 1–4)

**Pattern:** Inherits base via `defaults:` and overrides only 2 LR fields in `scratch:`.

**Full file content** (Research Q1, Q7; analog from RESEARCH.md):
```yaml
# @package _global_
defaults:
  - /configs/custom_finetune/base
  - _self_

# Full fine-tuning strategy: trains ALL layers including ViT vision backbone.
# Recommended only when dataset >500 images.
# Usage: python sam3/train/train.py -c custom_finetune/finetune_strategy/full_finetune

scratch:
  lr_scale: 1.0          # restore backbone LR to full value (vs 0.03 in decoder-only)
  lrd_vision_backbone: 0.9  # layer-wise LR decay across ViT trunk (D-09)
  lr_vision_backbone: 2.5e-4  # full backbone LR (lr_scale=1.0 × base value)
```

**Key difference from decoder_only:** overrides `scratch.lr_scale` and `scratch.lrd_vision_backbone`.
The `lr_vision_backbone` value should also be updated to reflect the new effective rate (Research Q5).

---

### `scripts/test_config_parse.py` (utility, smoke test, request-response)

**Analog:** `scripts/prepare_dataset.py`

**Script conventions to copy from analog:**
- Copyright header
- Module-level docstring with usage example
- `argparse`-based argument parsing via `parse_args()` function
- `main(args)` + `if __name__ == "__main__": main(parse_args())` pattern
- `sys.exit(1)` on error
- `sys.path.insert(0, ".")` for project root imports

**Copyright + docstring pattern** (analog lines 1–17):
```python
# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

# pyre-unsafe

"""
Smoke test: validate that all custom_finetune Hydra configs parse without errors.

Verifies config syntax, Hydra interpolation resolution, and that all three
config variants compose successfully using the same compose API as train.py.

Usage:
    python scripts/test_config_parse.py
"""
```

**Import pattern** (analog lines 19–26 — adapted for Hydra):
```python
import sys

sys.path.insert(0, ".")

from hydra import compose, initialize_config_module
from omegaconf import OmegaConf

from sam3.train.utils.train_utils import register_omegaconf_resolvers
```

**Core smoke-test logic** (Research Q6 — no analog; follows compose API from train.py):
```python
def main():
    register_omegaconf_resolvers()
    initialize_config_module("sam3.train", version_base="1.2")

    config_names = [
        "custom_finetune/base",
        "custom_finetune/finetune_strategy/decoder_only",
        "custom_finetune/finetune_strategy/full_finetune",
    ]

    all_passed = True
    for name in config_names:
        try:
            cfg = compose(config_name=name)
            # Trigger interpolation resolution to surface unresolved references
            OmegaConf.to_container(cfg, resolve=True, throw_on_missing=False)
            print(f"✓ {name}")
        except Exception as e:
            print(f"✗ {name}: {e}", file=sys.stderr)
            all_passed = False

    if not all_passed:
        sys.exit(1)
    print("\nAll configs parsed successfully.")


if __name__ == "__main__":
    main()
```

**No argparse needed** — this script takes no arguments (unlike prepare_dataset.py).

---

## Shared Patterns

### Hydra Package Declaration
**Source:** `sam3/train/configs/roboflow_v100/roboflow_v100_full_ft_100_images.yaml` (lines 1–3)
**Apply to:** All three YAML files
```yaml
# @package _global_
defaults:
  - _self_
```
For child configs (decoder_only, full_finetune), `_self_` stays last; parent config is listed first.

---

### `scratch:` Interpolation Variable Pattern
**Source:** `sam3/train/configs/roboflow_v100/roboflow_v100_full_ft_100_images.yaml` (lines 157–237)
**Apply to:** `base.yaml` only
The `scratch:` section defines all mutable parameters first; every other section references them via `${scratch.field}`. Avoids magic numbers scattered through the file.
```yaml
scratch:
  lr_transformer: 8e-5           # define here
  ...

trainer:
  optim:
    options:
      lr:
        - scheduler:
            base_lr: ${scratch.lr_transformer}  # reference here
```

---

### `${times:A,B}` Custom Resolver
**Source:** `sam3/train/utils/train_utils.py` (verified in RESEARCH.md Q5)
**Apply to:** `base.yaml` (optional — RESEARCH recommends using literal values instead)
The `${times:A,B}` resolver multiplies all arguments. With `lr_scale=0.03`:
- `${times:8e-4,${scratch.lr_scale}}` → `2.4e-5` — NOT the target `8e-5` from D-12.
- **Recommendation:** Use literal LR values in base.yaml (do NOT use `${times:...}`) to match D-12 exactly.

---

### Copyright Header
**Source:** `scripts/prepare_dataset.py` (line 1)
**Apply to:** `scripts/test_config_parse.py`
```python
# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

# pyre-unsafe
```

---

### Error Exit Pattern
**Source:** `scripts/prepare_dataset.py` (lines 50–53)
**Apply to:** `scripts/test_config_parse.py`
```python
print(f"ERROR: ...", file=sys.stderr)
sys.exit(1)
```

---

## No Analog Found

All files have analogs. The only pattern with no prior codebase example is the **single-field config override delta** used by `decoder_only.yaml` and `full_finetune.yaml`. The `silver_image_evals/*.yaml` pattern is used for the `defaults:` inheritance block, but those configs are eval-only overlays, not training-strategy deltas. This is documented in RESEARCH.md Q1 and Q7.

---

## Critical Differences from Analogs

| File | Key Divergence from Analog | Decision Source |
|------|---------------------------|-----------------|
| `base.yaml` | `enable_segmentation: true` (analog is `False`) | D-13 |
| `base.yaml` | Loss block includes `Masks` loss fn (analog has it commented out) | D-13, Research Q3 |
| `base.yaml` | Eval meters use `iou_type: "segm"` (analog uses `"bbox"`) | Research Q4 |
| `base.yaml` | `collate_fn.dict_key: custom` (analog uses `all`/`roboflow100`) | D-05 |
| `base.yaml` | `submitit.use_cluster: False` (analog uses `True`) | D-11 |
| `base.yaml` | `launcher.gpus_per_node: 1` (analog uses `2`) | D-11 |
| `base.yaml` | LR values are literals not `${times:...}` expressions | Research Q5 |
| `base.yaml` | `paths.bpe_path: null` (analog uses explicit path) | Research Q8 |
| `full_finetune.yaml` | Only 2–3 fields in `scratch:`, all others inherited | D-03 |
| `decoder_only.yaml` | Zero field overrides (base.yaml IS decoder-only) | D-10 |
| `test_config_parse.py` | Uses Hydra compose API directly (not `@hydra.main`) | Research Q6 |

---

## Metadata

**Analog search scope:** `sam3/train/configs/`, `scripts/`
**Files scanned:** 6 (roboflow config, yt1b eval config, prepare_dataset.py, extract_roboflow_vl100_results.py, train.py review from RESEARCH.md, train_utils.py patterns)
**Pattern extraction date:** 2026-05-27
