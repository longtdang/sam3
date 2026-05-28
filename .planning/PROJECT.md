# SAM3 Custom Fine-Tuning Pipeline

## What This Is

SAM3 is Meta's Segment Anything Model 3 — a vision-language model for open-vocabulary image and video segmentation with object tracking. This project extends SAM3 with a **reusable, config-driven fine-tuning pipeline** for custom COCO-format datasets (exported from CVAT), starting with an industrial defect/parts segmentation use case. The goal is a reproducible workflow for adapting SAM3 to any domain with instance segmentation masks.

## Core Value

Any team member can point the pipeline at a CVAT COCO export and produce a fine-tuned SAM3 checkpoint without touching model code.

## Requirements

### Validated

<!-- Existing capabilities inferred from codebase map -->
- ✓ Image segmentation inference (`sam3/model/sam3_image.py`) — existing
- ✓ Video segmentation + object tracking inference (`sam3/model/sam3_video_predictor.py`) — existing
- ✓ Distributed multi-GPU training via `torch.distributed` — existing
- ✓ Hydra-based config system for training experiments — existing
- ✓ COCO annotation format loaders (`sam3/train/data/coco_json_loaders.py`) — existing
- ✓ Image dataset loader (`sam3/train/data/sam3_image_dataset.py`) — existing
- ✓ Hungarian matcher + focal loss training stack — existing
- ✓ Checkpoint save/load infrastructure — existing

### Active

<!-- New capabilities being built -->
- [ ] CVAT COCO export can be loaded by the training pipeline with minimal config
- [ ] A single Hydra config file fully specifies a fine-tuning run (dataset path, classes, hyperparams)
- [ ] Fine-tuning strategy defaults to frozen ViT backbone + fine-tuned decoder/segmentation head (optimal for < 500 images)
- [ ] Config template supports swapping to full fine-tune or head-only fine-tune via one flag
- [ ] Train/validation split logic handles small datasets (< 500 images) gracefully
- [ ] Multi-GPU training (DDP) works out of the box on the same machine
- [ ] Fine-tuned checkpoint exported as `.pth` compatible with existing SAM3 inference scripts
- [ ] A README / runbook documents the end-to-end workflow for new datasets
- [ ] Pipeline is validated on the industrial defect dataset (first target domain)

### Out of Scope

- Video fine-tuning — image-only for now; can be added later with the same config structure
- Datumaro format support — COCO first; Datumaro converter can be added as a preprocessing step
- SLURM / cloud training — multi-GPU workstation only for this milestone
- Model architecture changes — only fine-tune existing weights, no new modules
- Real-time inference optimization (TorchScript, ONNX, TensorRT) — out of scope for training pipeline

## Context

- **Existing training infrastructure**: `sam3/train/` has a complete training stack (Trainer, loss, optimizer, schedulers, distributed utils). Fine-tuning adds dataset adapters and config templates on top — it does NOT rewrite training logic.
- **COCO loader exists**: `sam3/train/data/coco_json_loaders.py` already parses COCO JSON. The gap is wiring a CVAT-exported COCO structure into the Hydra config.
- **Small dataset strategy**: For < 500 images, freezing the ViT backbone (1B parameters) and fine-tuning only the decoder + segmentation head is standard practice. Full fine-tune risks overfitting without augmentation. This should be the default with a flag to override.
- **Hydra configs live in**: `sam3/configs/` — new dataset configs follow the same pattern.
- **Known codebase risks**: Large god-class model files (3,000+ lines), CUDA-only (no CPU path), near-zero test coverage. Fine-tuning work stays in `sam3/train/` and `sam3/configs/` to minimize exposure to fragile model code.
- **GPU environment**: CUDA 12.6+, PyTorch ≥ 2.7, multi-GPU workstation.

## Constraints

- **Tech Stack**: Python 3.12+, PyTorch ≥ 2.7, Hydra config system — no new frameworks
- **Backward Compatibility**: Existing SAM3 inference scripts must continue to work unchanged
- **Dataset Format**: COCO JSON (CVAT default export) — primary supported format
- **Hardware**: Multi-GPU workstation (CUDA); no CPU-only or SLURM requirements for this milestone
- **Model**: Fine-tune existing SAM3 weights only — no architectural modifications

## Current State

**Version:** v1.0 (shipped 2026-05-28)
**Status:** Pipeline complete — 5 phases, 13 plans, all verified

Key artifacts:
- `FINE_TUNING.md` — self-service runbook at repo root
- `scripts/prepare_dataset.py` — CVAT → SAM3-ready splits
- `sam3/train/configs/custom_finetune/` — drop-in Hydra configs
- `sam3/train/trainer.py` — patched to export `best_checkpoint.pth`
- `scripts/test_checkpoint_compatibility.py` — smoke test for checkpoint

**Deferred to v1.1:** Real training run on industrial defect dataset (AP50 > 0 smoke test)

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Freeze ViT backbone by default for small datasets | < 500 images is insufficient to update 1B ViT params without overfitting | `decoder_only.yaml` is default |
| Use COCO format as primary input | CVAT's default export; existing `coco_json_loaders.py` reduces integration effort | Phase 1 complete |
| Build on existing Hydra config system | Avoids parallel config systems; users stay in the same workflow | Phase 2 complete |
| Image-only for first milestone | User's dataset is images; keeps scope contained | Phase 1–5 complete |
| `python sam3/train/train.py` (not torchrun) | train.py uses internal process spawning; torchrun would double-spawn and crash | Documented in FINE_TUNING.md |
| Sam3Processor for inference | Public API that handles BatchedDatapoint internally | Phase 5 runbook |
| SAM3 norms `[0.5, 0.5, 0.5]` | Not ImageNet; must match Sam3Processor internal normalization | base.yaml + gotcha 2 |
| best_checkpoint.pth in HuggingFace format | Satisfies `_load_checkpoint` weights_only + "model" key requirements | Phase 4 |

## Evolution

This document evolves at phase transitions and milestone boundaries.

**After each phase transition** (via `/gsd-transition`):
1. Requirements invalidated? → Move to Out of Scope with reason
2. Requirements validated? → Move to Validated with phase reference
3. New requirements emerged? → Add to Active
4. Decisions to log? → Add to Key Decisions
5. "What This Is" still accurate? → Update if drifted

**After each milestone** (via `/gsd-complete-milestone`):
1. Full review of all sections
2. Core Value check — still the right priority?
3. Audit Out of Scope — reasons still valid?
4. Update Context with current state

---
*Last updated: 2026-05-26 after initialization*
