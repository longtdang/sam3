---
phase: "02-hydra-config-templates"
plan: "03"
subsystem: "hydra-config"
tags: ["hydra", "config", "yaml", "fine-tuning", "full-finetune", "llrd"]
dependency_graph:
  requires: ["02-01"]
  provides:
    - "sam3/train/configs/custom_finetune/finetune_strategy/full_finetune.yaml — full backbone fine-tune strategy override"
  affects:
    - "Plan 02-04 (smoke test composes full_finetune.yaml)"
tech_stack:
  added: []
  patterns:
    - "Hydra config group delta override (inherits base, adds only LR/LLRD strategy fields)"
key_files:
  created:
    - sam3/train/configs/custom_finetune/finetune_strategy/full_finetune.yaml
  modified: []
key_decisions:
  - "3 delta fields: lr_vision_backbone (10x), lr_language_backbone (10x), lrd_vision_backbone (0.9) — matches roboflow full-FT reference pattern exactly"
requirements_completed:
  - CFG-03
duration: "1 min"
completed: "2026-05-27"
---

# Phase 02 Plan 03: full_finetune.yaml Summary

**One-liner:** Full backbone fine-tune Hydra strategy override — inherits base.yaml, raises ViT and language backbone LRs 10× and applies LLRD (0.9) for adaptive layer-wise decay.

**Duration:** 1 min | **Tasks:** 1 | **Files created:** 1

## What Was Built

Created `sam3/train/configs/custom_finetune/finetune_strategy/full_finetune.yaml` — a ~30-line delta override that enables full SAM3 fine-tuning with 3 field changes vs. decoder-only:

| Field | base.yaml value | full_finetune.yaml value |
|-------|----------------|--------------------------|
| `lr_vision_backbone` | `2.5e-6` | `2.5e-5` (+10×) |
| `lr_language_backbone` | `1.5e-6` | `1.5e-5` (+10×) |
| `lrd_vision_backbone` | `0.9` (inherited) | `0.9` (explicit) |

Switching from decoder-only to full fine-tune requires only: `+finetune_strategy=full_finetune`.

## Deviations from Plan

None — plan executed exactly as written.

**Total deviations:** 0. **Impact:** None.

## Self-Check

- [x] File exists — PASS
- [x] Inherits `/configs/custom_finetune/base` in defaults — PASS
- [x] `lrd_vision_backbone: 0.9` set — PASS
- [x] `lr_vision_backbone: 2.5e-5` set — PASS
- [x] `lr_language_backbone: 1.5e-5` set — PASS
- [x] Delta-only (≤ 15 YAML field lines) — PASS (3 fields)
- [x] Valid YAML — PASS
- [x] Committed: 203e121

## Self-Check: PASSED

## Next

Ready for Plan 02-04 (smoke test: verify all three configs compose without errors).
