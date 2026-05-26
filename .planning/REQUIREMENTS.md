# Requirements: SAM3 Custom Fine-Tuning Pipeline

**Defined:** 2026-05-26
**Core Value:** Any team member can point the pipeline at a CVAT COCO export and produce a fine-tuned SAM3 checkpoint without touching model code.

## v1 Requirements

### Dataset Preparation

- [ ] **DATA-01**: A script converts a CVAT COCO export into SAM3-compatible train/val JSON splits (80/20 random split, stratified by category)
- [ ] **DATA-02**: The script validates and fixes common CVAT quirks: 0-based IDs → 1-based reindex, `file_name` prefix normalization, contiguous category IDs
- [ ] **DATA-03**: Script is configurable via CLI args: input annotation file, image folder, output directory, split ratio, random seed
- [ ] **DATA-04**: Script reports dataset statistics after preparation: total images, images per split, instances per category

### Hydra Config

- [ ] **CFG-01**: A base Hydra config template (`configs/custom_finetune/base.yaml`) defines all required fields for a fine-tuning run
- [ ] **CFG-02**: A `decoder_only.yaml` override freezes the ViT backbone (default for < 500 images) — sets differential LRs via `lr_scale: 0.03`
- [ ] **CFG-03**: A `full_finetune.yaml` override enables full model fine-tuning via LLRD (`lrd_vision_backbone: 0.9`)
- [ ] **CFG-04**: Dataset path, annotation file, and class names are the only fields a user must change to run on a new dataset
- [ ] **CFG-05**: `enable_segmentation: true` is set in all fine-tuning configs (critical — off by default in SAM3)
- [ ] **CFG-06**: Normalization uses SAM3's expected values (`[0.5, 0.5, 0.5]`), not ImageNet defaults

### Training

- [ ] **TRAIN-01**: Fine-tuning runs end-to-end with `python -m sam3.train.train --config-name custom_finetune/base` on a multi-GPU workstation
- [ ] **TRAIN-02**: Multi-GPU DDP training works automatically using existing `torch.distributed` infrastructure
- [ ] **TRAIN-03**: Default hyperparameters are set for small datasets: epochs=40, target_epoch_size=500, batch=1 + gradient_accumulation×4, transformer LR=8e-5, backbone LR=2.5e-6
- [ ] **TRAIN-04**: Data augmentation config includes ColorJitter, GaussianBlur, and RandomErasing on top of existing multi-scale resize/pad pipeline
- [ ] **TRAIN-05**: Training checkpoints save to a user-configurable output directory
- [ ] **TRAIN-06**: TensorBoard logging is enabled for loss curves and evaluation metrics

### Evaluation

- [ ] **EVAL-01**: Training loop evaluates on the validation split at configurable intervals and reports `coco_eval_segm_AP50`, `coco_eval_segm_APs`, and `coco_eval_segm_AP`
- [ ] **EVAL-02**: `iou_type: "segm"` is set in the evaluation config (not `"bbox"`)

### Checkpoint Export

- [ ] **CKPT-01**: The best-performing checkpoint (by `coco_eval_segm_AP50`) is saved as `best_checkpoint.pth`
- [ ] **CKPT-02**: The exported checkpoint loads cleanly with the existing `sam3.build_sam3_image_model()` API without modifications

### Documentation & Runbook

- [ ] **DOC-01**: A `FINE_TUNING.md` runbook documents the full workflow: install → prepare data → configure → train → evaluate → export
- [ ] **DOC-02**: Runbook includes a troubleshooting section covering the top 5 gotchas from research (e.g., `enable_segmentation`, normalization values, ID reindexing)
- [ ] **DOC-03**: Config templates include inline comments explaining every required field

### Validation

- [ ] **VAL-01**: The pipeline produces a fine-tuned checkpoint on the industrial defect dataset (first target domain) with `coco_eval_segm_AP50 > 0` (smoke test)
- [ ] **VAL-02**: The exported checkpoint loads and runs inference via existing SAM3 scripts without errors

## v2 Requirements

### Extended Format Support

- **FMT-01**: Datumaro format converter (CVAT → COCO conversion as preprocessing step)
- **FMT-02**: Support for CVAT video exports (for future video fine-tuning)

### Training Improvements

- **TRAIN-07**: LoRA adapter support for parameter-efficient fine-tuning (< 100 images use case)
- **TRAIN-08**: Early stopping based on validation AP plateau
- **TRAIN-09**: Mixed-precision (bfloat16) training support for memory efficiency

### Infrastructure

- **INFRA-01**: SLURM/submitit config template for cluster training
- **INFRA-02**: Automated hyperparameter sweep via Hydra multirun

## Out of Scope

| Feature | Reason |
|---------|--------|
| Video fine-tuning | Image-only for v1; video adds significant complexity (temporal consistency, tracking labels) |
| Architecture modifications | No new model modules — fine-tune existing weights only |
| ONNX / TensorRT export | Inference optimization is a separate concern from the training pipeline |
| Real-time inference integration | Out of scope for this milestone |
| CPU-only training | Hardware constraint: CUDA required |
| Automated hyperparameter search | Manual config is sufficient for v1 |

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| DATA-01 | Phase 1 | Pending |
| DATA-02 | Phase 1 | Pending |
| DATA-03 | Phase 1 | Pending |
| DATA-04 | Phase 1 | Pending |
| CFG-01 | Phase 2 | Pending |
| CFG-02 | Phase 2 | Pending |
| CFG-03 | Phase 2 | Pending |
| CFG-04 | Phase 2 | Pending |
| CFG-05 | Phase 2 | Pending |
| CFG-06 | Phase 2 | Pending |
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
| DOC-01 | Phase 5 | Pending |
| DOC-02 | Phase 5 | Pending |
| DOC-03 | Phase 2 | Pending |
| VAL-01 | Phase 4 | Pending |
| VAL-02 | Phase 4 | Pending |

**Coverage:**
- v1 requirements: 25 total
- Mapped to phases: 25
- Unmapped: 0 ✓

---
*Requirements defined: 2026-05-26*
*Last updated: 2026-05-26 after initialization*
