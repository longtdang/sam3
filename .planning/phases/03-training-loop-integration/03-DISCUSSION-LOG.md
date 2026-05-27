# Phase 3: Training Loop Integration - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-05-27
**Phase:** 3-training-loop-integration
**Areas discussed:** Training launcher & command, Data augmentation additions, Evaluation frequency, Phase scope (dry-run script)

---

## Training Launcher & Command

| Option | Description | Selected |
|--------|-------------|----------|
| Accept existing launcher | Document `python sam3/train/train.py` as canonical command | ✓ |
| Investigate torchrun too | Check if torchrun is also supported, document both if so | |
| Add torchrun support | Modify train.py to accept torchrun env vars | |

**User's choice:** Accept the existing launcher — document `python sam3/train/train.py` as canonical command

| GPU control option | Description | Selected |
|--------------------|-------------|----------|
| `--num_gpus N` CLI flag | e.g. `--num_gpus 2` | ✓ |
| Set in config YAML | `launcher.gpus_per_node: 2` | |
| Both (default in config, CLI override) | | |

**User's choice:** `--num_gpus N` CLI flag

**Notes:** Codebase scouting revealed `train.py` uses `submitit` + `torch.multiprocessing.start_processes` internally — `torchrun` is not the entry point. The canonical command will be `python sam3/train/train.py --config-name configs/custom_finetune/base --num_gpus 2`.

---

## Data Augmentation Additions

| Option | Description | Selected |
|--------|-------------|----------|
| Add wrapper classes to basic.py | Like RandomErasing — consistent with existing pattern | ✓ |
| Use torchvision transforms directly via `_target_` | Config-only, no code changes | |
| Skip augmentation for now | Basic ResizePad+Normalize sufficient for Phase 3 | |

**User's choice:** Add wrapper classes to basic.py (like RandomErasing)

| Augmentation scope | Selected |
|--------------------|----------|
| All three: ColorJitter + GaussianBlur + RandomErasing | ✓ |
| Just ColorJitter + GaussianBlur | |
| ColorJitter only | |

**User's choice:** All three

**Notes:** `RandomErasing` already exists in `basic.py`. Two new wrapper classes (`ColorJitter`, `GaussianBlur`) will be added. All three will appear in `base.yaml`'s `train_transforms` pipeline.

---

## Evaluation Frequency

| Option | Description | Selected |
|--------|-------------|----------|
| Every epoch | 40 eval runs — maximum monitoring | ✓ |
| Every 5 epochs | 8 eval runs — faster iteration | |
| Every 10 epochs | 4 eval runs — minimal overhead | |

**User's choice:** Every epoch (`val_epoch_freq: 1`)

**Notes:** Trainer uses `val_epoch_freq` param. Default in codebase is 1 (every epoch), which aligns with user's preference for maximum monitoring on 40-epoch runs.

---

## Phase Scope: Dry-Run Script

| Option | Description | Selected |
|--------|-------------|----------|
| Verify wiring only — no dry-run | Team has no available GPU right now | |
| Add a dry-run config-validation script | Similar to Phase 2 smoke test | ✓ |
| Leave runtime verification for Phase 4 | Phase 3 is config + code additions only | |

**User's choice:** Add a dry-run config-validation script (`scripts/test_training_config.py`)

---

## the agent's Discretion

- Exact placement of augmentation transforms in the `train_transforms` list
- Whether to add a `use_augmentation` flag in `scratch` for easy disabling
- Val transform pipeline augmentation strategy (standard: no augmentation on val)

## Deferred Ideas

- `torchrun` support — could be added for cloud/SLURM launch in future
- `use_augmentation: true/false` flag — for runbook phase if needed
