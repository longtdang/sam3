# Phase 3: Validation Architecture

> Derived from `03-RESEARCH.md §Validation Architecture` — reviewed and confirmed by plan-checker.

## Test Framework

| Property | Value |
|----------|-------|
| Framework | pytest (existing `tests/` directory) |
| Config file | `pyproject.toml` |
| Quick run command | `python scripts/test_training_config.py` |
| Full suite command | `python -m pytest tests/ -x -q && python scripts/test_training_config.py` |

## Phase Requirements → Test Map

| Req ID | Behavior | Test Type | Automated Command | Covered By |
|--------|----------|-----------|-------------------|------------|
| TRAIN-01 | Config composes without errors | smoke | `python scripts/test_training_config.py` | Plan 03-02 |
| TRAIN-02 | DDP wiring (cfg.launcher.gpus_per_node) | unit | `python scripts/test_training_config.py` | Plan 03-02 (implicit — Hydra compose fails if broken) |
| TRAIN-03 | Hyperparameter values correct | unit | `python scripts/test_config_parse.py` | Already exists (Phase 2) |
| TRAIN-04 | Augmentation entries in train_transforms | unit | `python scripts/test_training_config.py` | Plan 03-02 |
| TRAIN-05 | Checkpoint save_dir configured | unit | `python scripts/test_training_config.py` | Plan 03-02 |
| TRAIN-06 | TensorBoard block present in config | unit | `python scripts/test_training_config.py` | Plan 03-02 |
| EVAL-01 | val_epoch_freq=1 in assembled config | unit | `python scripts/test_training_config.py` | Plan 03-02 |
| EVAL-02 | iou_type=segm in assembled config | unit | `python scripts/test_training_config.py` | Plan 03-02 |

## Sampling Rate

- **Per task commit:** `python scripts/test_training_config.py`
- **Per wave merge:** `python -m pytest tests/ -x -q && python scripts/test_training_config.py`
- **Phase gate:** Full suite green before `/gsd-verify-work`

## Wave 0 Gaps

- [ ] `scripts/test_training_config.py` — created by Plan 03-02 (covers TRAIN-01, TRAIN-04, TRAIN-06, EVAL-01, EVAL-02)

## Notes

- TRAIN-02 (DDP) and TRAIN-05 (checkpoint) are already wired from Phase 2. No Phase 3 changes touch these blocks — Hydra compose validation provides implicit coverage.
- `test_config_parse.py` (Phase 2) already covers TRAIN-03 hyperparameters.
