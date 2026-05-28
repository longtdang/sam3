# Roadmap: SAM3 Custom Fine-Tuning Pipeline

**Created:** 2026-05-26
**Mode:** YOLO

## Milestones

### ✅ v1.0 — Fine-Tuning Pipeline Complete (SHIPPED 2026-05-28)

Full details: [.planning/milestones/v1.0-ROADMAP.md](milestones/v1.0-ROADMAP.md)

<details>
<summary>v1.0 Phase Summary (5 phases · 13 plans)</summary>

- [x] **Phase 1: Dataset Preparation** — `scripts/prepare_dataset.py` converts CVAT COCO exports to SAM3-ready train/val splits
- [x] **Phase 2: Hydra Config Templates** — `sam3/train/configs/custom_finetune/` with base + decoder_only + full_finetune configs
- [x] **Phase 3: Training Loop Integration** — Augmentation transforms, small-dataset hyperparameters, TensorBoard, segm eval metrics
- [x] **Phase 4: Checkpoint Export & Validation** — `best_checkpoint.pth` in HuggingFace format; smoke test scripts
- [x] **Phase 5: Runbook Documentation** — `FINE_TUNING.md` runbook (380 lines, 7 sections, 5 gotchas); UAT 7/7 ✅

**Known Deferred:**
- Real training run on industrial defect dataset (`coco_eval_segm_AP50 > 0`) — v1.1

</details>

---

## Next Milestone: v1.1

_Not yet planned. Run `/gsd-new-milestone` to start._

Candidate items from v2 requirements:
- **TRAIN-07**: LoRA adapter support for parameter-efficient fine-tuning
- **VAL-01 (full)**: End-to-end validation on real industrial CVAT export
- **INFRA-01**: SLURM/submitit config template for cluster training
- **TRAIN-09**: Mixed-precision (bfloat16) training support

## Backlog

See `.planning/milestones/v1.0-REQUIREMENTS.md` for v2 requirements list.
