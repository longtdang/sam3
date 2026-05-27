# Phase 3: Training Loop Integration - Context

**Gathered:** 2026-05-27
**Status:** Ready for planning

<domain>
## Phase Boundary

Wire the Phase 2 Hydra configs to the existing SAM3 training stack so that a team member can launch a fine-tuning run with a single command and monitor progress in TensorBoard — without modifying any SAM3 model code.

Deliverables:
- `base.yaml` extended with TensorBoard block, augmentation transforms, eval frequency, and launcher fields
- `ColorJitter` and `GaussianBlur` wrapper classes added to `sam3/train/transforms/basic.py` (matching the `RandomErasing` pattern that already exists)
- A dry-run config-validation script similar to the Phase 2 smoke test

Not in scope: SLURM support, LoRA adapters, mixed-precision training, actual training run (Phase 4), checkpoint export (Phase 4).

</domain>

<decisions>
## Implementation Decisions

### Training Launcher
- **D-03-01:** Use the existing `python sam3/train/train.py` launcher (NOT `torchrun`). The training stack uses `torch.multiprocessing.start_processes` internally via `submitit` LocalExecutor — `torchrun` is not the entry point.
- **D-03-02:** GPU count is controlled via `--num_gpus N` CLI flag (not config YAML). The canonical 2-GPU command is:
  ```bash
  python sam3/train/train.py --config-name configs/custom_finetune/base --num_gpus 2
  ```
  Single-GPU: `--num_gpus 1`. Do NOT document `torchrun`.

### Data Augmentation
- **D-03-03:** Add `ColorJitter` and `GaussianBlur` wrapper classes to `sam3/train/transforms/basic.py`, following the exact `RandomErasing` pattern (wraps `torchvision.transforms.v2.ColorJitter` / `.GaussianBlur`).
- **D-03-04:** Add all three augmentation transforms (`ColorJitter`, `GaussianBlur`, `RandomErasing`) to `base.yaml`'s `train_transforms` pipeline. They should appear AFTER the resize/pad/normalize steps.
- **D-03-05:** Augmentation defaults for industrial defect data:
  - `ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.0)` — subtle color changes, no hue shift for industrial parts
  - `GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))` — light blur to simulate sensor noise
  - `RandomErasing(p=0.2, scale=(0.02, 0.1))` — small random occlusion patches

### Evaluation Frequency
- **D-03-06:** Set `val_epoch_freq: 1` in the trainer config block — run validation after every epoch (maximum monitoring for 40-epoch runs).

### TensorBoard Integration
- **D-03-07:** Add TensorBoard logging block to `base.yaml` using the existing pattern from `sam3/train/configs/odinw13/`:
  ```yaml
  trainer:
    logging:
      tensorboard_writer:
        _target_: sam3.train.utils.logger.make_tensorboard_logger
        log_dir: ${launcher.experiment_log_dir}/tensorboard
  ```

### Dry-Run Smoke Test
- **D-03-08:** Create `scripts/test_training_config.py` — validates that the training config assembles correctly (datasets, trainer, loss instantiate via Hydra) without launching a real training run. Pattern: `hydra.utils.instantiate` in dry-run mode or config-only validation. Similar scope to `scripts/test_config_parse.py` from Phase 2.

### the agent's Discretion
- Exact placement of augmentation transforms in the `train_transforms` list (before or after existing filter steps is the planner's call based on what makes semantic sense)
- Whether to add a `use_augmentation: true` flag in `scratch` to allow disabling augmentation without editing the transform list
- Val transform pipeline is NOT augmented (standard practice — only train pipeline gets augmentation)

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Training Entry Point & Launcher
- `sam3/train/train.py` — CLI entry point; `single_node_runner()` handles multi-GPU via `torch.multiprocessing`; `--num_gpus` CLI arg sets `cfg.launcher.gpus_per_node`

### Trainer Configuration
- `sam3/train/trainer.py` — `Trainer` class; `val_epoch_freq` param (line ~159) controls eval interval; `logging_conf.tensorboard_writer` is the TensorBoard integration point

### Reference Config (TensorBoard + transform patterns)
- `sam3/train/configs/odinw13/odinw_text_only_train.yaml` — shows `trainer.logging.tensorboard_writer` block and `train_transforms` pipeline structure to follow

### Augmentation
- `sam3/train/transforms/basic.py` — `RandomErasing` class (lines ~381+) is the exact pattern to follow for `ColorJitter` and `GaussianBlur` wrapper classes

### Phase 2 Config (baseline to extend)
- `sam3/train/configs/custom_finetune/base.yaml` — the file being extended in Phase 3

### Phase 2 Smoke Test (pattern to follow for dry-run script)
- `scripts/test_config_parse.py` — module-stub pattern, Hydra compose approach

### Requirements
- `.planning/REQUIREMENTS.md` §TRAIN-01–TRAIN-06, EVAL-01–EVAL-02

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `sam3.train.utils.logger.make_tensorboard_logger` — already wired in odinw configs; add the same block to `base.yaml`
- `sam3/train/transforms/basic.py::RandomErasing` — copy/adapt this class for `ColorJitter` and `GaussianBlur`
- `scripts/test_config_parse.py` — module-stub + Hydra compose pattern; reuse for dry-run validation script

### Established Patterns
- `@package _global_` + `defaults: [_self_]` — all SAM3 training configs use this header (Phase 2 set this correctly)
- `_target_:` instantiation via Hydra — transforms, loss, trainer all wired this way; no direct Python construction
- `${launcher.experiment_log_dir}` — the correct interpolation path for the log dir in TensorBoard config (Phase 2 base.yaml uses `scratch.experiment_log_dir` as the REQUIRED marker — planner should verify the wiring between `scratch.experiment_log_dir` and `launcher.experiment_log_dir`)

### Integration Points
- `base.yaml` `trainer:` block — extend with `val_epoch_freq`, `logging.tensorboard_writer`
- `base.yaml` `train_transforms:` — add augmentation entries after existing resize/normalize steps
- `sam3/train/transforms/basic.py` — add two new wrapper classes

</code_context>

<specifics>
## Specific Ideas

- The augmentation defaults (D-03-05) are tuned for industrial defect/parts data: subtle color changes, no hue shift, light blur for sensor noise, small random erasing for occlusion robustness
- The dry-run script should confirm that all three configs (base, decoder_only, full_finetune) can instantiate their datasets and trainers without requiring actual data files (using the REQUIRED-null guard from Phase 2)

</specifics>

<deferred>
## Deferred Ideas

- `torchrun` support — not needed given existing multi-GPU launcher; could be added if cloud/SLURM-style launch is needed in future
- `use_augmentation: true/false` flag — may be useful to disable augmentation for evaluation configs without editing transform lists; defer to runbook phase if needed

</deferred>

---

*Phase: 3-training-loop-integration*
*Context gathered: 2026-05-27*
