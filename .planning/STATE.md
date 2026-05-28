# Project State

**Last updated:** 2026-05-28
**Current phase:** Phase 5 — Runbook Documentation (complete)

## Project Reference

See: `.planning/PROJECT.md` (updated 2026-05-26)

**Core value:** Any team member can point the pipeline at a CVAT COCO export and produce a fine-tuned SAM3 checkpoint without touching model code.
**Current focus:** Phase 1 — Dataset Preparation

## Phase Status

| Phase | Name | Status |
|-------|------|--------|
| 1 | Dataset Preparation | ✅ Complete (2/2 plans) |
| 2 | Hydra Config Templates | ✅ Complete (4/4 plans) |
| 3 | Training Loop Integration | ✅ Complete (2/2 plans) |
| 4 | Checkpoint Export & Validation | ✅ Complete |
| 5 | Runbook Documentation | ✅ Complete |

## Current Position

- **Phase:** 3 — Training Loop Integration (complete, 2/2 plans)
- **Next:** Verify Phase 4 → `/gsd-verify-work 04` or plan Phase 5 → `/gsd-plan-phase 05`

## Decisions Recorded

- D-01: Stratified-by-category split using greedy multi-label algorithm (no sklearn)
- D-06/D-07: Silent basename strip for file_name prefix repair (os.path.basename)
- D-10: Independent reindex per ID type — handles mixed 0/1-based CVAT exports
- D-13: sys.exit(1) with stderr message on missing required COCO keys (not Python traceback)
- D-14: Stats summary always printed: total images, per-split count, per-category instances
- T-02-01: copy.deepcopy() on fixtures before mutation prevents cross-test pollution
- D-P2-01: Explicit LR literals (8e-5, 2.5e-6) instead of ${times:...} resolver in base.yaml
- D-P2-02: 4 REQUIRED markers (including experiment_log_dir) not 3
- D-P2-03: Config names use configs/ prefix with initialize_config_module("sam3.train")
- D-P3-01: ColorJitter/GaussianBlur use PIL-stage datapoint API signature; RandomErasingAPI uses tensor-stage datapoint API signature
- D-P3-02: val_epoch_freq set to 1 (not 10) for maximum monitoring on 40-epoch runs
- D-P3-03: TensorBoard block was already present from Phase 2 — no change needed

## Next Step

Phase 3 complete. Run `/gsd-verify-work 03` to verify or `/gsd-plan-phase 04` to plan Phase 4: Checkpoint Export & Validation.

Phase 4 complete. Run `/gsd-verify-work 04` to verify or `/gsd-plan-phase 05` to plan Phase 5: Runbook Documentation.

## Phase 4 Decisions

- D-04-01: Fake dataset for CI; real data at `data/industrial_defect/` (train.json, val.json, images/)
- D-04-02: `save_checkpoint()` patched to export `best_checkpoint.pth` in HF format (`{"model": {"detector.<k>": tensor}}`)
- D-04-03 (revised): `test_checkpoint_compatibility.py` checks `len(model.state_dict()) > 0` instead of forward pass — `Sam3Image.forward()` requires `BatchedDatapoint`, no `SAM3ImagePredictor` exists
- D-04-04: `generate_fake_dataset.py` — 5 images, 64×64, single "defect" category
- D-04-05: VAL-01 `AP50 > 0` is manual on real data; CI asserts no crash (1-epoch dry run)

## Planning Artifacts

- `.planning/PROJECT.md` — project context and requirements
- `.planning/REQUIREMENTS.md` — 25 v1 requirements
- `.planning/ROADMAP.md` — 5-phase roadmap
- `.planning/config.json` — workflow config (mode: yolo, quality models, research+verify on)
- `.planning/codebase/` — 7 codebase analysis documents
- `.planning/research/FINETUNING_STRATEGIES.md` — fine-tuning strategy research
- `.planning/research/DATASET_INTEGRATION.md` — CVAT COCO integration research

## Phase 5 Decisions

- D-05-01: FINE_TUNING.md at repo root (alongside README.md)
- D-05-02: Inference example uses Sam3Processor (public API) — not raw BatchedDatapoint
- D-05-03 (corrected): Launch command is `python sam3/train/train.py -c ... --use-cluster 0 --num-gpus N` — NOT torchrun
