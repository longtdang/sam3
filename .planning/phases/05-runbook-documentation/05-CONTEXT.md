# Phase 5: Runbook Documentation — Context

## Phase Goal
Write a FINE_TUNING.md runbook so a new team member can complete their first fine-tuning run using only that document.

## Decisions

### D-05-01: FINE_TUNING.md location
**Decision:** Repo root (alongside README.md)
**Rationale:** Maximum discoverability; same level as README.md so it's the first doc a user sees.

### D-05-02: Inference section depth
**Decision:** Full inference example — load model + run on a sample image + show output masks
**Rationale:** A load-only check is insufficient for users who need to understand what the model actually produces. Include:
- Load `best_checkpoint.pth` via `build_sam3_image_model(load_from_HF=False, device="cpu")`
- Preprocess an image into a `BatchedDatapoint`
- Call `model.forward()` and interpret the output masks
- Code snippet that users can copy-paste

### D-05-03: Multi-GPU coverage
**Decision:** Cover both single-GPU and multi-GPU (--nproc_per_node=N with NCCL/CUDA notes)
**Rationale:** Production workloads require multi-GPU. Include:
- Single-GPU launch: `torchrun --nproc_per_node=1 -m sam3.train.train --config-name custom_finetune/base`
- Multi-GPU launch: `torchrun --nproc_per_node=N ...` with `--master_addr` / `--master_port` guidance
- NCCL backend requirement (CUDA)
- Note about `batch_size` per GPU (effective batch = batch_size × N_GPUs)

## Scope

### In scope (FINE_TUNING.md)
1. **Prerequisites** — Python env setup, model weights download, CVAT COCO export
2. **Dataset preparation** — COCO JSON format requirements (1-based IDs, `enable_segmentation=true`), directory layout
3. **Config setup** — Fill in the 3 required null fields in `base.yaml`; explain key hyperparameters
4. **Training launch** — Single-GPU and multi-GPU commands; what to watch in TensorBoard
5. **Checkpoint output** — Where `best_checkpoint.pth` lands; format explanation
6. **Inference example** — Full code snippet: load model → preprocess image → forward → output masks
7. **Troubleshooting** — Top 5 gotchas (see below)

### In scope (Troubleshooting section)
Top 5 gotchas identified during development:
1. `enable_segmentation=True` must be set in CVAT export (missing → empty `segmentation` fields)
2. Normalization mismatch — SAM3 expects `mean=[123.675, 116.28, 103.53]` / `std=[58.395, 57.12, 57.375]` (ImageNet RGB); OpenCV reads BGR → must convert
3. 0-based annotation ID reindexing — COCO requires 1-based IDs; re-export or patch if CVAT outputs 0-based
4. `file_name` prefix collision — if image `file_name` in COCO JSON includes a path prefix that doesn't match the actual file layout, loader fails silently
5. Mask loss commented out — `sam3/train/trainer.py` has mask loss disabled by default; mention this is intentional and how to re-enable

### Out of scope
- Phase 1–4 reimplementation (already done)
- Video/interactive predictor usage (different model variant)
- Deployment / serving infrastructure

## Files to create
- `FINE_TUNING.md` — repo root

## Files to update
- `.planning/STATE.md` — current phase → 5
- `.planning/ROADMAP.md` — Phase 5 tasks
