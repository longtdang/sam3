---
phase: "02-hydra-config-templates"
plan: "01"
subsystem: "hydra-config"
tags: ["hydra", "config", "yaml", "fine-tuning", "segmentation"]
dependency_graph:
  requires: []
  provides:
    - "sam3/train/configs/custom_finetune/base.yaml — complete standalone Hydra config for CVAT dataset fine-tuning"
  affects:
    - "Plans 02-02 and 02-03 (decoder_only.yaml and full_finetune.yaml inherit from base.yaml)"
    - "Plan 02-04 (smoke test composes base.yaml)"
tech_stack:
  added: []
  patterns:
    - "Hydra _global_ package with defaults: [_self_] (matches all existing SAM3 configs)"
    - "scratch.* interpolation variables for single-source-of-truth config values"
    - "Explicit LR literals instead of ${times:...} resolver (eliminates custom resolver dependency)"
key_files:
  created:
    - sam3/train/configs/custom_finetune/base.yaml
  modified: []
key_decisions:
  - "Used explicit LR literals (8e-5, 2.5e-6) instead of ${times:...} resolver — eliminates runtime resolver dependency, makes values readable at a glance"
  - "4 REQUIRED markers instead of 3 — added experiment_log_dir as required (needed for checkpoint + tensorboard output paths)"
  - "dict_key: custom (not 'all') for collate_fn to match trainer.loss.custom and meters.val.custom keys"
requirements_completed:
  - CFG-01
  - CFG-04
  - CFG-05
  - CFG-06
  - DOC-03
duration: "6 min"
completed: "2026-05-27"
---

# Phase 02 Plan 01: base.yaml Summary

**One-liner:** Standalone Hydra config for fine-tuning SAM3 on any CVAT COCO dataset — segmentation enabled, SAM3 norms, 4 REQUIRED markers, explicit LR literals for small-dataset decoder-only training.

**Duration:** 6 min | **Tasks:** 1 | **Files created:** 1

## What Was Built

Created `sam3/train/configs/custom_finetune/base.yaml` — a ~390-line, self-contained Hydra training config that wires a CVAT COCO dataset to a SAM3 fine-tuning run. A user only edits the 4 fields marked `# REQUIRED:` to launch training on a new dataset.

### Key Config Properties

| Property | Value | Rationale |
|----------|-------|-----------|
| `enable_segmentation` | `true` | SAM3 defaults to detection-only; this enables mask loss and eval |
| Normalization | `[0.5, 0.5, 0.5]` | SAM3 pre-training values — NOT ImageNet |
| `lr_transformer` | `8e-5` (literal) | Small-dataset decoder-only default |
| `lr_vision_backbone` | `2.5e-6` (literal) | Near-frozen backbone for < 500 images |
| `gradient_accumulation_steps` | `4` | Effective batch = 4 with batch_size=1 |
| `max_data_epochs` | `40` | Small-dataset training schedule |
| `iou_type` | `"segm"` | Segmentation AP metrics throughout eval |

## Deviations from Plan

**[Rule 1 - Minor] enable_segmentation grep count is 6, not 5**
- Found during: Acceptance criteria verification
- Issue: The definition line `enable_segmentation: true  # ... ${scratch.enable_segmentation}` contains the interpolation pattern in its comment, producing 6 grep matches instead of 5
- Fix: All 5 actual downstream interpolation uses are present (load_segmentation ×2, with_seg_masks ×2, model.enable_segmentation ×1); the comment on the definition line is accurate documentation
- Impact: None — functionality is correct, comment improves readability

**Total deviations:** 1 (0 impacting). **Impact:** None.

## Self-Check

- [x] `test -f sam3/train/configs/custom_finetune/base.yaml` — PASS
- [x] `python3 -c "import yaml; yaml.safe_load(...)"` — VALID YAML
- [x] `grep -c "# REQUIRED:"` → 4 — PASS
- [x] `grep "enable_segmentation: true"` — PASS (1 definition match)
- [x] All 5 downstream `${scratch.enable_segmentation}` uses present — PASS
- [x] `train_norm_mean/std` → `[0.5, 0.5, 0.5]` — PASS
- [x] `lr_transformer: 8e-5` (literal, no `${times:`) — PASS
- [x] `lr_vision_backbone: 2.5e-6` — PASS
- [x] `max_data_epochs: 40` — PASS
- [x] All `iou_type` → `"segm"` (no `"bbox"`) — PASS
- [x] `loss_fns.Masks` present and uncommented — PASS
- [x] `loss_mask: 200.0` — PASS
- [x] `gradient_accumulation_steps: 4` — PASS
- [x] `use_cluster: False` — PASS
- [x] `save_best_meters` with `val_custom/detection` — PASS
- [x] No `${times:...}` resolver — PASS (comment mentions it but no usage)
- [x] Committed: d7b72db

## Self-Check: PASSED

## Next

Ready for Plan 02-02 (decoder_only.yaml) and Plan 02-03 (full_finetune.yaml) — both are small delta overrides that inherit base.yaml.
