---
phase: 02-hydra-config-templates
reviewed: 2025-07-15T00:00:00Z
depth: standard
files_reviewed: 4
files_reviewed_list:
  - sam3/train/configs/custom_finetune/base.yaml
  - sam3/train/configs/custom_finetune/finetune_strategy/decoder_only.yaml
  - sam3/train/configs/custom_finetune/finetune_strategy/full_finetune.yaml
  - scripts/test_config_parse.py
findings:
  critical: 2
  warning: 2
  info: 1
  total: 5
status: issues_found
---

# Phase 02: Code Review Report

**Reviewed:** 2025-07-15T00:00:00Z
**Depth:** standard
**Files Reviewed:** 4
**Status:** issues_found

## Summary

Reviewed four Phase 2 deliverables: the Hydra `base.yaml` training config, two fine-tuning strategy override configs (`decoder_only.yaml`, `full_finetune.yaml`), and a smoke-test script (`test_config_parse.py`).

The core config architecture is sound: `@package _global_` is used correctly, inheritance via `defaults: [/configs/custom_finetune/base]` is correct, the REQUIRED-null pattern for the four user paths is appropriate, and the SAM3 norm values and segmentation flags are set correctly.

Two blockers were found that stem from a partially completed design decision: the explicit-LR-literals approach was adopted (dropping `${times:...}`) but `lr_scale` was left in the config as a dead field rather than removed or documented as non-functional. This creates a silent behavior trap: users who adjust `lr_scale` following the pattern established by every other SAM3 config (`odinw`, `roboflow_v100`, `eval_base`) will observe no change in actual optimizer learning rates. The decoder_only.yaml comment compounds this with a numerically wrong backbone-LR claim.

---

## Critical Issues

### CR-01: `lr_scale` is defined but never consumed — adjusting it silently has zero effect

**File:** `sam3/train/configs/custom_finetune/base.yaml:64`

**Issue:** `scratch.lr_scale: 0.03` is documented as controlling the backbone freeze strategy ("decoder-only strategy: backbone effectively near-frozen"), but it is **never referenced via `${scratch.lr_scale}` anywhere in the config**. All three learning rates that follow it (`lr_transformer`, `lr_vision_backbone`, `lr_language_backbone`) are hardcoded literals:

```yaml
lr_scale: 0.03           # decoder-only strategy: backbone effectively near-frozen
lr_transformer: 8e-5     # LITERAL — lr_scale plays no part
lr_vision_backbone: 2.5e-6  # LITERAL — lr_scale plays no part
lr_language_backbone: 1.5e-6  # LITERAL — lr_scale plays no part
```

Every other SAM3 training config (`roboflow_v100`, `odinw_text_only_train`, `eval_base`) derives LRs from `lr_scale` via the `${times:...}` resolver (e.g. `lr_vision_backbone: ${times:2.5e-4,${scratch.lr_scale}}`). A user following that established pattern who overrides `lr_scale` will get **no change** in optimizer behavior.

This also means `decoder_only.yaml`'s `scratch.lr_scale: 0.03` override (its only field) is a no-op: it changes a field that controls nothing.

**Fix:** Either (a) restore the `${times:...}` resolver pattern to make `lr_scale` functional, or (b) remove `lr_scale` entirely and rename the scratch fields with a comment making clear the values are pre-computed for decoder-only:

```yaml
# Option A — restore functional lr_scale (consistent with all other SAM3 configs):
lr_scale: 0.03
lr_transformer: ${times:8e-4,${scratch.lr_scale}}       # 0.03 * 8e-4 = 2.4e-5
lr_vision_backbone: ${times:2.5e-4,${scratch.lr_scale}} # 0.03 * 2.5e-4 = 7.5e-6
lr_language_backbone: ${times:5e-5,${scratch.lr_scale}} # 0.03 * 5e-5  = 1.5e-6

# Option B — remove lr_scale, document the pre-computed decoder-only literals:
# lr_transformer is 8e-4 × 0.03 (decoder-only pre-computed):
lr_transformer: 2.4e-5
# lr_vision_backbone is 2.5e-4 × 0.03 (near-frozen pre-computed):
lr_vision_backbone: 7.5e-6
# lr_language_backbone is 5e-5 × 0.03 (near-frozen pre-computed):
lr_language_backbone: 1.5e-6
```

Note: the current literal `lr_transformer: 8e-5` is inconsistent with the decoder-only scale calculation (`8e-4 × 0.03 = 2.4e-5`), suggesting the values were set ad-hoc rather than derived from `lr_scale`.

---

### CR-02: `decoder_only.yaml` states backbone LR is `~7.5e-6` but actual value is `2.5e-6`

**File:** `sam3/train/configs/custom_finetune/finetune_strategy/decoder_only.yaml:14-15`

**Issue:** The strategy description comment reads:

> "The ViT backbone is effectively near-frozen via a low lr_scale (0.03), **keeping backbone LR at ~7.5e-6** — roughly 30× lower than a full fine-tune."

The `7.5e-6` figure comes from the formula `2.5e-4 × 0.03` — the calculation that applies when `lr_scale` is used as a multiplier (as in `roboflow_v100`). But `base.yaml` uses the hardcoded literal `lr_vision_backbone: 2.5e-6` (line 66), not `7.5e-6`. Users will observe a backbone LR of `2.5e-6`, not the `7.5e-6` stated in the comment. The comment also incorrectly attributes the LR to `lr_scale` which, per CR-01, is not consumed.

**Fix:** Update the comment to state the actual literal value and remove the misleading `lr_scale` attribution:

```yaml
# Strategy: Train only the decoder (cross-attention, query heads, class head).
# The ViT backbone is near-frozen via a low literal lr_vision_backbone (2.5e-6),
# roughly 10× lower than full fine-tune (2.5e-5). Override lr_vision_backbone
# directly to adjust the backbone freeze degree.
```

---

## Warnings

### WR-01: `full_finetune.yaml` usage comment describes a broken invocation

**File:** `sam3/train/configs/custom_finetune/finetune_strategy/full_finetune.yaml:13-14`

**Issue:** The comment instructs users to run:

```bash
python sam3/train/train.py --config-name custom_finetune/base \
    '+finetune_strategy=full_finetune'
```

This invocation has two problems:

1. **`+finetune_strategy=full_finetune` appends a config group item** whose own `defaults` list includes `- /configs/custom_finetune/base`. With `--config-name custom_finetune/base`, base.yaml gets loaded twice — once as the primary config and once pulled in by `full_finetune`'s defaults — which will cause a Hydra merge conflict or unexpected key ordering.

2. The `train.py` entry point uses `compose(config_name=args.config)` (single positional config); the `+group=item` override pattern is a Hydra config-group mechanism that only works correctly when the primary config has a `defaults:` slot for that group.

The correct usage (as the smoke test correctly demonstrates) is to name the override file directly as the primary config:

```bash
python sam3/train/train.py --config-name custom_finetune/finetune_strategy/full_finetune
```

**Fix:** Replace the broken usage comment:

```yaml
# Usage: python sam3/train/train.py \
#            --config-name custom_finetune/finetune_strategy/full_finetune
#
# This config inherits base.yaml via its defaults list, then overrides the
# backbone LR fields. Do not combine with --config-name custom_finetune/base
# (+finetune_strategy syntax) — that would load base.yaml twice.
```

---

### WR-02: Smoke test asserts the value of `lr_scale` but never validates that any actual LR was changed

**File:** `scripts/test_config_parse.py:129-141`

**Issue:** Test 2 (decoder_only) asserts:

```python
assert math.isclose(cfg_decoder.scratch.lr_scale, 0.03, rel_tol=1e-6)
assert math.isclose(cfg_decoder.scratch.lr_vision_backbone, 2.5e-6, rel_tol=1e-6)
```

The first assertion validates a field that (per CR-01) has no effect on the optimizer. The second assertion passes because `lr_vision_backbone` inherits the base value unchanged — the same value would pass whether `decoder_only.yaml` was applied or not. The test gives a green signal that the decoder-only strategy is correctly applied when the strategy's distinguishing parameter (`lr_scale`) has no mechanical connection to any optimizer input.

Additionally, if CR-01 is resolved and `${times:...}` is restored, the test would need to validate the computed backbone LR against the expected product (e.g., `7.5e-6`) rather than the raw literal.

**Fix:** After resolving CR-01, update the assertion to check the computed backbone LR reflects the `lr_scale` override:

```python
# Decoder-only: backbone LR = 2.5e-4 * lr_scale (0.03) = 7.5e-6
assert math.isclose(cfg_decoder.scratch.lr_vision_backbone, 7.5e-6, rel_tol=1e-6), (
    f"Expected lr_vision_backbone=7.5e-6 (near-frozen, 0.03 × 2.5e-4), "
    f"got {cfg_decoder.scratch.lr_vision_backbone}"
)
# Also verify a different lr_scale value would produce a different LR (integration check)
```

---

## Info

### IN-01: `full_finetune.yaml` redundantly overrides `lrd_vision_backbone: 0.9` — same as base default

**File:** `sam3/train/configs/custom_finetune/finetune_strategy/full_finetune.yaml:29`

**Issue:** `base.yaml` already sets `scratch.lrd_vision_backbone: 0.9` (line 68). The `full_finetune.yaml` override sets it to the same value, creating a false impression that `lrd_vision_backbone` differs from the base default in full fine-tune mode. The comment says this value "matches roboflow full-FT reference" — but since the base is already `0.9`, the override has no effect.

**Fix:** Remove the redundant field and note the existing base default in the comment, or change the value to `1.0` if full fine-tune truly intends uniform LR across ViT layers (no layer decay). If `0.9` is intentional and meaningful for full fine-tune (as it is in the roboflow reference), keep it with a note that it happens to match the base default:

```yaml
# lrd_vision_backbone is already 0.9 in base.yaml (exponential layer-wise decay).
# No override needed unless switching to uniform LR (lrd_vision_backbone: 1.0).
```

---

_Reviewed: 2025-07-15T00:00:00Z_
_Reviewer: gsd-code-reviewer (adversarial)_
_Depth: standard_
