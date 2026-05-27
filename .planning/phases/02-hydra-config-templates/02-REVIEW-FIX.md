---
phase: 02-hydra-config-templates
fixed_at: 2025-07-15T00:00:00Z
review_path: .planning/phases/02-hydra-config-templates/02-REVIEW.md
iteration: 1
findings_in_scope: 4
fixed: 4
skipped: 0
status: all_fixed
---

# Phase 02: Code Review Fix Report

**Fixed at:** 2025-07-15T00:00:00Z
**Source review:** `.planning/phases/02-hydra-config-templates/02-REVIEW.md`
**Iteration:** 1

**Summary:**
- Findings in scope: 4 (CR-01, CR-02, WR-01, WR-02)
- Fixed: 4
- Skipped: 0

---

## Fixed Issues

### CR-01: `lr_scale` is defined but never consumed — adjusting it silently has zero effect

**Files modified:** `sam3/train/configs/custom_finetune/base.yaml`, `sam3/train/configs/custom_finetune/finetune_strategy/decoder_only.yaml`
**Commit:** `5c0d9b6`
**Applied fix:**
- In `base.yaml`: Added a prominent `INFORMATIONAL` warning block above `lr_scale` to make clear it does **not** drive any optimizer LR. The literal `lr_transformer`, `lr_vision_backbone`, and `lr_language_backbone` fields are unchanged (they remain the actual optimizer inputs).
- In `decoder_only.yaml`: Expanded the single `lr_scale: 0.03` override to explicitly restate all three LR literals (`lr_transformer: 8e-5`, `lr_vision_backbone: 2.5e-6`, `lr_language_backbone: 1.5e-6`) with `informational` tag comments. This makes the decoder-only strategy self-documenting — a reader can see all three LRs without jumping to `base.yaml`. The `lr_scale` field is kept as informational context (not functional).

### CR-02: `decoder_only.yaml` states backbone LR is `~7.5e-6` but actual value is `2.5e-6`

**Files modified:** `sam3/train/configs/custom_finetune/finetune_strategy/decoder_only.yaml`
**Commit:** `5c0d9b6` (same commit as CR-01 — both changes were to the same two files)
**Applied fix:**
- Replaced the broken strategy comment that claimed `keeping backbone LR at ~7.5e-6 — roughly 30× lower` (the stale `${times:...}` formula result) with the accurate statement: `near-frozen via a low literal lr_vision_backbone (2.5e-6), roughly 10× lower than full fine-tune (2.5e-5)`.
- Removed the misleading `lr_scale` attribution from the strategy description.

### WR-01: `full_finetune.yaml` usage comment describes a broken invocation

**Files modified:** `sam3/train/configs/custom_finetune/finetune_strategy/full_finetune.yaml`
**Commit:** `cbb5fe9`
**Applied fix:**
- Replaced the broken `--config-name custom_finetune/base '+finetune_strategy=full_finetune'` invocation (which loads `base.yaml` twice and triggers a Hydra merge conflict) with the correct:
  ```
  python sam3/train/train.py \
      --config-name custom_finetune/finetune_strategy/full_finetune
  ```
- Added an explicit warning note explaining why the `+group=item` pattern must not be combined with `--config-name custom_finetune/base`.
- Preserved the rest of the strategy comment block (section title, recommendation).

### WR-02: Smoke test asserts the value of `lr_scale` but never validates that any actual LR was changed

**Files modified:** `scripts/test_config_parse.py`
**Commit:** `bd56e7b`
**Applied fix:**
- Kept the `lr_scale == 0.03` check (the field is still present in `decoder_only.yaml` as informational).
- Added a **meaningful** `lr_vision_backbone == 2.5e-6` assertion with updated context comment explaining this is the literal near-frozen value.
- Added cross-config assertions: decoder_only backbone LR **must be lower** than full_finetune backbone LR, and the ratio must be exactly 10×. This ensures the test would fail if the decoder-only strategy were accidentally set to the same backbone LR as full fine-tune.
- Updated the module docstring to describe the improved Test 2 coverage.

**Smoke test result:** All 3 configs passed after fixes:
```
✓ custom_finetune/base
✓ custom_finetune/finetune_strategy/decoder_only
✓ custom_finetune/finetune_strategy/full_finetune

All configs parsed successfully.
```

---

_Fixed: 2025-07-15T00:00:00Z_
_Fixer: the agent (gsd-code-fixer)_
_Iteration: 1_
