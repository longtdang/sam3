# Phase 5: Runbook Documentation — Context

## Phase Goal
Write a FINE_TUNING.md runbook so a new team member can complete their first fine-tuning run using only that document.

## Decisions

### D-05-01: FINE_TUNING.md location
**Decision:** Repo root (alongside README.md)
**Rationale:** Maximum discoverability; same level as README.md so it's the first doc a user sees.

### D-05-02: Inference section depth
**Decision:** Full inference example using `Sam3Processor` (the documented public API)
**Rationale:** `Sam3Processor` handles `BatchedDatapoint` construction internally (5 nested dataclasses); raw construction is too complex for a runbook. `Sam3Processor` is documented in `README.md` "Basic Usage". Include:
- Load `best_checkpoint.pth` via `build_sam3_image_model(load_from_HF=False, device="cpu")`
- Use `Sam3Processor(model)` + `set_image(image)` + `set_text_prompt(state, prompt=...)`
- Show `output["masks"]`, `output["boxes"]`, `output["scores"]` interpretation
- Full Python code snippet users can copy-paste
**Updated:** Research confirmed `Sam3Processor` is the public API; user approved 2026-05-28

### D-05-03: Multi-GPU coverage
**Decision:** Cover both single-GPU and multi-GPU using `python sam3/train/train.py` (NOT torchrun)
**Rationale:** `torchrun` is incompatible with `sam3/train/train.py` — it uses an internal `single_node_runner` + `torch.multiprocessing.start_processes` which double-spawns workers with `torchrun`. Verified from `sam3/train/train.py` argparser. Include:
- Single-GPU: `python sam3/train/train.py -c custom_finetune/base --use-cluster 0 --num-gpus 1`
- Multi-GPU: `python sam3/train/train.py -c custom_finetune/base --use-cluster 0 --num-gpus N`
- NCCL backend requirement (CUDA) for multi-GPU
- Note about `batch_size` per GPU (effective batch = batch_size × N_GPUs)
- Include `⚠️ Warning: Do NOT use torchrun — it double-spawns workers with this launcher`
**Updated:** Research found torchrun incompatible; user approved 2026-05-28

## Scope

### In scope (FINE_TUNING.md)
1. **Prerequisites** — Python env setup, model weights download, CVAT COCO export
2. **Dataset preparation** — COCO JSON format requirements (1-based IDs, `enable_segmentation=true`), directory layout
3. **Config setup** — Fill in the 4 required null fields in `base.yaml` (`dataset_img_folder`, `train_ann_file`, `val_ann_file`, `experiment_log_dir`); explain key hyperparameters
4. **Training launch** — Single-GPU and multi-GPU commands; what to watch in TensorBoard
5. **Checkpoint output** — Where `best_checkpoint.pth` lands; format explanation
6. **Inference example** — Full code snippet: load model → preprocess image → forward → output masks
7. **Troubleshooting** — Top 5 gotchas (see below)

### In scope (Troubleshooting section)
Top 5 gotchas identified during development:
1. `enable_segmentation=True` must be set in CVAT export (missing → empty `segmentation` fields)
2. Normalization mismatch — SAM3 uses `mean=[0.5, 0.5, 0.5]` / `std=[0.5, 0.5, 0.5]` (NOT ImageNet values); using ImageNet normalization causes poor convergence
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
