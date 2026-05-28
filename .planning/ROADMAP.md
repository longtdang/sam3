# Roadmap: SAM3 Custom Fine-Tuning Pipeline

**Created:** 2026-05-26
**Granularity:** Standard
**Mode:** YOLO

## Phases

- [x] **Phase 1: Dataset Preparation** — CVAT COCO export → SAM3-ready train/val JSON splits
- [x] **Phase 2: Hydra Config Templates** — Drop-in config files that wire a dataset to a training run
- [ ] **Phase 3: Training Loop Integration** — End-to-end fine-tuning with eval metrics and TensorBoard
- [ ] **Phase 4: Checkpoint Export & Validation** — Best checkpoint saved and verified against SAM3 API
- [ ] **Phase 5: Runbook Documentation** — Written guide that makes the pipeline self-service

---

## Phase Details

### Phase 1: Dataset Preparation

**Goal:** Any CVAT COCO export can be converted into clean, SAM3-compatible train/val JSON splits using a single CLI command.
**Depends on:** Nothing
**Requirements:** DATA-01, DATA-02, DATA-03, DATA-04

#### Plans

1. **Write `scripts/prepare_dataset.py`** — CLI script that reads a CVAT COCO JSON and image folder, fixes common CVAT quirks (0-based IDs → 1-based, `file_name` prefix normalisation, contiguous category IDs), performs a configurable stratified 80/20 train/val split, writes two output JSON files, and prints dataset statistics.
2. **Add unit tests for CVAT fixups** — Cover the three repair cases (ID reindex, prefix strip, category reindex) with minimal fixture JSON files.

**Success Criteria:**
- [x] Running `python scripts/prepare_dataset.py --ann-file instances_default.json --img-folder images/ --output data/splits/` produces `train.json` and `val.json` with no 0-based IDs and no broken `file_name` paths.
- [x] The script rejects malformed input (missing required COCO keys) with a clear error message rather than a Python traceback.
- [x] Output includes a printed summary: total images, images per split, and instance count per category.
- [x] `--split-ratio` and `--seed` CLI flags override the 80/20 default and random seed 42 default.
- [x] All categories present in the source JSON appear in both split files (stratified split preserves rare classes).

**Plans:** 2 plans

Plans:
- [x] 01-01-PLAN.md — Test scaffolding (tests/conftest.py fixtures) + scripts/prepare_dataset.py implementation (all 10 functions)
- [x] 01-02-PLAN.md — Unit tests in tests/test_prepare_dataset.py (7 test functions, all green)

---

### Phase 2: Hydra Config Templates

**Goal:** A new dataset can be wired to a fine-tuning run by editing exactly three fields in a config file — dataset path, annotation file, and class names.
**Depends on:** Phase 1
**Requirements:** CFG-01, CFG-02, CFG-03, CFG-04, CFG-05, CFG-06, DOC-03

#### Plans

1. **Create `configs/custom_finetune/base.yaml`** — Base config that includes all required fields (`enable_segmentation: true`, SAM3 normalisation `[0.5, 0.5, 0.5]`, dataset path placeholders, default small-dataset hyperparameters) with inline comments on every required field.
2. **Create `configs/custom_finetune/decoder_only.yaml`** — Override that sets `lr_scale: 0.03` and activates differential LR groups to effectively freeze the ViT backbone; intended for datasets < 500 images.
3. **Create `configs/custom_finetune/full_finetune.yaml`** — Override that enables full fine-tuning via `lrd_vision_backbone: 0.9` (LLRD across all parameter groups).
4. **Smoke-test config parsing** — Verify all three configs compose without Hydra errors using `python -m sam3.train.train --config-name custom_finetune/base --cfg job` (dry-run / config-print mode).

**Success Criteria:**
- [ ] Running `python -m sam3.train.train --config-name custom_finetune/base --cfg job` prints a composed config without errors.
- [ ] `base.yaml` contains `enable_segmentation: true` and normalisation mean/std of `[0.5, 0.5, 0.5]` (not ImageNet values).
- [ ] Switching from decoder-only to full fine-tune requires changing only the override flag (`+custom_finetune=full_finetune`), not editing the base config.
- [ ] Every field a user must change to run on a new dataset is marked with a `# REQUIRED:` inline comment.
- [ ] `decoder_only.yaml` and `full_finetune.yaml` differ only in the LR/freeze strategy fields — all other values inherit from `base.yaml`.

**Plans:** 4 plans

Plans:
- [x] 02-01-PLAN.md — Create `sam3/train/configs/custom_finetune/base.yaml` (full standalone config, segmentation enabled, SAM3 norms, small-dataset LR defaults, 4 REQUIRED markers)
- [x] 02-02-PLAN.md — Create `sam3/train/configs/custom_finetune/finetune_strategy/decoder_only.yaml` (delta override: lr_scale: 0.03)
- [x] 02-03-PLAN.md — Create `sam3/train/configs/custom_finetune/finetune_strategy/full_finetune.yaml` (delta override: lrd_vision_backbone: 0.9, lr_vision_backbone: 2.5e-5)
- [x] 02-04-PLAN.md — Create `scripts/test_config_parse.py` smoke test + run (Hydra compose API, asserts all three configs parse)

---

### Phase 3: Training Loop Integration

**Goal:** A team member can launch a fine-tuning run on a multi-GPU workstation with a single command and monitor progress in TensorBoard without modifying any model code.
**Depends on:** Phase 2
**Requirements:** TRAIN-01, TRAIN-02, TRAIN-03, TRAIN-04, TRAIN-05, TRAIN-06, EVAL-01, EVAL-02

#### Plans

1. **Wire dataset config to training data loaders** — Confirm `COCO_FROM_JSON` loads the prepared JSONs correctly via Hydra interpolation; validate `load_segmentation: true` activates mask loss.
2. **Set small-dataset defaults** — Patch `base.yaml` with validated hyperparameters: `epochs=40`, `target_epoch_size=500`, `train_batch_size=1`, `gradient_accumulation_steps=4`, transformer LR `8e-5`, backbone LR `2.5e-6`.
3. **Add augmentation config block** — Extend the data transform pipeline in config with `ColorJitter`, `GaussianBlur`, and `RandomErasing` entries on top of existing multi-scale resize/pad.
4. **Verify DDP launch path** — Confirm `torchrun` (or existing launcher) drives multi-GPU training via the unmodified `torch.distributed` infrastructure; document the exact launch command.
5. **Enable TensorBoard logging** — Ensure `trainer.logging.tensorboard: true` is set in `base.yaml` and loss curves appear under `experiment_log_dir`.
6. **Validate eval metrics** — Confirm `iou_type: "segm"` is set in the eval config and that `coco_eval_segm_AP50`, `coco_eval_segm_APs`, and `coco_eval_segm_AP` are reported after the first eval interval.

**Success Criteria:**
- [ ] `python -m sam3.train.train --config-name custom_finetune/base` starts training (loss decreasing by epoch 2) without code changes to `sam3/train/`.
- [ ] On a 2-GPU machine, `torchrun --nproc_per_node=2 -m sam3.train.train --config-name custom_finetune/base` runs DDP without hanging or crashing.
- [ ] TensorBoard at `experiment_log_dir` shows loss curves and `coco_eval_segm_AP50` after the first evaluation interval.
- [ ] Eval output in logs includes `coco_eval_segm_AP`, `coco_eval_segm_AP50`, and `coco_eval_segm_APs` (not bbox metrics).
- [ ] Checkpoints appear in the configured output directory after each evaluation interval.

**Plans:** 2 plans

Plans:
- [x] 03-01-PLAN.md — Add ColorJitter, GaussianBlur, RandomErasingAPI to basic.py + patch base.yaml (val_epoch_freq: 1 + augmentation entries)
- [x] 03-02-PLAN.md — Create scripts/test_training_config.py dry-run validation (asserts all 8 Phase 3 requirements)

---

### Phase 4: Checkpoint Export & End-to-End Validation

**Goal:** The pipeline produces a verified fine-tuned checkpoint on the industrial defect dataset that loads cleanly into existing SAM3 inference scripts.
**Depends on:** Phase 3
**Requirements:** CKPT-01, CKPT-02, VAL-01, VAL-02

#### Plans

1. **Implement best-checkpoint tracking** — Add or confirm config for saving `best_checkpoint.pth` based on peak `coco_eval_segm_AP50` across all eval intervals; existing checkpoint infra should support a `save_best_metric` key.
2. **Run pipeline on industrial defect dataset** — Execute full Phase 1 → 3 pipeline end-to-end on the first target dataset (industrial defect/parts); record `coco_eval_segm_AP50` smoke-test result.
3. **Verify checkpoint API compatibility** — Write a one-file test script: load `best_checkpoint.pth` via `sam3.build_sam3_image_model()`, run inference on one validation image, assert output tensor shapes are valid.

**Success Criteria:**
- [ ] `best_checkpoint.pth` is written to the output directory and corresponds to the epoch with highest `coco_eval_segm_AP50`.
- [ ] `sam3.build_sam3_image_model(checkpoint="best_checkpoint.pth")` loads without errors or missing-key warnings.
- [ ] Running the existing SAM3 inference script against `best_checkpoint.pth` on a validation image produces segmentation masks without runtime errors.
- [ ] `coco_eval_segm_AP50 > 0` on the industrial defect dataset (smoke test — confirms the model is learning, not collapsing).

**Plans:** 3 plans

Plans:
- [ ] 04-01-PLAN.md — Patch trainer.py::save_checkpoint() to export best_checkpoint.pth in HuggingFace inference format (CKPT-01, CKPT-02)
- [ ] 04-02-PLAN.md — Create scripts/generate_fake_dataset.py (synthetic COCO, 5 images) + document 1-epoch training smoke test command (VAL-01)
- [ ] 04-03-PLAN.md — Create scripts/test_checkpoint_compatibility.py (load best_checkpoint.pth via build_sam3_image_model, assert non-zero params) (CKPT-02, VAL-02)

---

### Phase 5: Runbook Documentation

**Goal:** A new team member with no prior SAM3 knowledge can complete their first fine-tuning run on a new dataset using only `FINE_TUNING.md` — no Slack questions required.
**Depends on:** Phase 4
**Requirements:** DOC-01, DOC-02

#### Plans

1. **Write `FINE_TUNING.md` runbook** — Full workflow documentation: environment setup → CVAT export → `prepare_dataset.py` usage → config editing (the three required fields) → launch command → reading TensorBoard → locating best checkpoint → inference verification. Include copy-pasteable commands for every step.
2. **Write troubleshooting section** — Cover the top 5 gotchas confirmed during implementation: (1) `enable_segmentation` off by default, (2) normalisation mismatch (ImageNet vs SAM3 `[0.5, 0.5, 0.5]`), (3) 0-based ID reindexing, (4) `file_name` path prefix collision with `img_folder`, (5) mask loss block commented out in base configs.

**Success Criteria:**
- [ ] `FINE_TUNING.md` covers all seven workflow stages: install, prepare data, configure, train, evaluate, export checkpoint, run inference.
- [ ] Troubleshooting section addresses at minimum the five gotchas identified in research with symptom → cause → fix format.
- [ ] Every shell command in the runbook is copy-pasteable (no placeholder ellipses inside command blocks).
- [ ] A team member following the runbook cold can produce a non-zero `coco_eval_segm_AP50` result on a fresh CVAT export without additional guidance.

**Plans:** TBD
**UI hint**: no

---

## Progress Table

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Dataset Preparation | 2/2 | ✅ Complete | 2026-05-27 |
| 2. Hydra Config Templates | 0/4 | In progress | - |
| 3. Training Loop Integration | 2/2 | ✅ Complete | 2026-05-28 |
| 4. Checkpoint Export & Validation | 0/3 | Not started | - |
| 5. Runbook Documentation | 0/2 | Not started | - |

---

## Milestone: v1.0 — Fine-Tuning Pipeline Complete

**Phases:** 1–5
**Definition of Done:**
- [ ] CVAT COCO export → fine-tuned `.pth` checkpoint in < 30 minutes of setup
- [ ] Pipeline tested on industrial defect dataset (`coco_eval_segm_AP50 > 0`)
- [ ] Runbook complete and verified against a cold run

---

## Coverage

| Requirement | Phase | Status |
|-------------|-------|--------|
| DATA-01 | Phase 1 | ✅ Complete |
| DATA-02 | Phase 1 | ✅ Complete |
| DATA-03 | Phase 1 | ✅ Complete |
| DATA-04 | Phase 1 | ✅ Complete |
| CFG-01 | Phase 2 | Pending |
| CFG-02 | Phase 2 | Pending |
| CFG-03 | Phase 2 | Pending |
| CFG-04 | Phase 2 | Pending |
| CFG-05 | Phase 2 | Pending |
| CFG-06 | Phase 2 | Pending |
| DOC-03 | Phase 2 | Pending |
| TRAIN-01 | Phase 3 | Pending |
| TRAIN-02 | Phase 3 | Pending |
| TRAIN-03 | Phase 3 | Pending |
| TRAIN-04 | Phase 3 | Pending |
| TRAIN-05 | Phase 3 | Pending |
| TRAIN-06 | Phase 3 | Pending |
| EVAL-01 | Phase 3 | Pending |
| EVAL-02 | Phase 3 | Pending |
| CKPT-01 | Phase 4 | Pending |
| CKPT-02 | Phase 4 | Pending |
| VAL-01 | Phase 4 | Pending |
| VAL-02 | Phase 4 | Pending |
| DOC-01 | Phase 5 | Pending |
| DOC-02 | Phase 5 | Pending |

**v1 requirements mapped: 25/25 ✓**
