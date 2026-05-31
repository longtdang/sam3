---
status: fixing
trigger: "dryrun training config, fix all issues to ensure training can run (T4 15GB GPU)"
created: 2026-05-31
updated: 2026-05-31
---

## Symptoms
- expected: Training dry run completes successfully
- actual: Not yet run — proactive fix before real training
- errors: Known so far: `device: cpus` typo in base.yaml line 293 (should be `cuda`)
- timeline: New session
- reproduction: Run training script with custom_finetune/base.yaml

## Current Focus
- hypothesis: "amp_dtype: bfloat16 on T4 (sm_75, Turing) conflicts with always-enabled GradScaler and lacks native hardware support — float16 is correct for T4"
- test: "Run 1-epoch train_only with float16 AMP on 16-image dataset"
- expecting: "Training completes 1 epoch (16 steps) without crash"
- next_action: "Verify 1-epoch run: python -m sam3.train.train --config-name custom_finetune/base trainer.mode=train_only scratch.max_data_epochs=1"

## Evidence
- timestamp: 2026-05-31T11:26 observation: "base.yaml line 293 has `device: cpus` — should be `cuda`"
- timestamp: 2026-05-31T11:26 observation: "T4 GPU with 15360 MiB VRAM available"
- timestamp: 2026-05-31T11:26 observation: "840M total params, 32.7M trainable (LoRA-style), batch_size=1, bfloat16 AMP enabled"
- timestamp: 2026-05-31T11:26 observation: "gradient_accumulation_steps=1, train_batch_size=1 — already memory-optimized"
- timestamp: 2026-05-31T15:15 observation: "T4 = sm_75 (Turing) — no native bfloat16 hardware; trainer.py line 1111 always initializes GradScaler regardless of amp_dtype; GradScaler is for float16 only; bfloat16 + GradScaler = undefined behavior"
- timestamp: 2026-05-31T15:15 observation: "trainer.py default (line 66) is float16 — the config override to bfloat16 was incorrect for this GPU"
- timestamp: 2026-05-31T15:15 observation: "Dataset: 16 train images, 4 val images — tiny dataset, 1 epoch = 16 steps, OOM unlikely"
- timestamp: 2026-05-31T15:15 observation: "FIXED: amp_dtype changed from bfloat16 to float16 in custom_finetune/base.yaml"

## Eliminated
(none yet)

## Resolution
- root_cause: pending
- fix: pending
- verification: pending
- files_changed: []
