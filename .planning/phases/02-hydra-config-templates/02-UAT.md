---
status: complete
phase: 02-hydra-config-templates
source:
  - 02-01-SUMMARY.md
  - 02-02-SUMMARY.md
  - 02-03-SUMMARY.md
  - 02-04-SUMMARY.md
started: 2025-07-15T00:00:00Z
updated: 2025-07-15T00:00:00Z
---

## Current Test

[testing complete]

## Tests

### 1. Smoke test — all 3 configs parse
expected: |
  Running `python3 scripts/test_config_parse.py` prints exactly 3 ✓ lines and
  "All configs parsed successfully." then exits with code 0. No tracebacks or ✗ lines.
result: pass

### 2. base.yaml has 4 REQUIRED markers
expected: |
  `grep -c "# REQUIRED:" sam3/train/configs/custom_finetune/base.yaml` returns 4.
  The 4 fields are: dataset_img_folder, train_ann_file, val_ann_file, experiment_log_dir —
  all set to null so training fails loudly if the user forgets to fill them in.
result: pass

### 3. base.yaml segmentation configuration
expected: |
  base.yaml has `enable_segmentation: true` and uses SAM3 normalization values
  `[0.5, 0.5, 0.5]` (not ImageNet). Running `python3 -c "import yaml; c = yaml.safe_load(open('sam3/train/configs/custom_finetune/base.yaml')); print(c['scratch']['enable_segmentation'], c['scratch']['train_norm_mean'])"` prints `True [0.5, 0.5, 0.5]`.
result: pass

### 4. Decoder-only backbone LR is near-frozen (10× lower than full fine-tune)
expected: |
  decoder_only.yaml keeps backbone at 2.5e-6 (near-frozen), while full_finetune.yaml
  raises it to 2.5e-5 — exactly 10× higher. The smoke test cross-config assertion
  explicitly verifies this ratio. Both configs still parse with all assertions passing.
result: pass

### 5. Correct usage comment in full_finetune.yaml
expected: |
  The usage comment in full_finetune.yaml shows:
    `python sam3/train/train.py --config-name custom_finetune/finetune_strategy/full_finetune`
  It does NOT suggest `'+finetune_strategy=full_finetune'` (the broken double-load pattern).
  `grep "finetune_strategy" sam3/train/configs/custom_finetune/finetune_strategy/full_finetune.yaml`
  returns only the defaults line, not a broken +group=item override invocation.
result: pass

## Summary

total: 5
passed: 5
issues: 0
pending: 0
skipped: 0

## Gaps

[none yet]
