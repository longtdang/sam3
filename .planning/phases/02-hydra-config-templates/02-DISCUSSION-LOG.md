# Phase 2: Hydra Config Templates - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-05-27
**Phase:** 2-hydra-config-templates
**Areas discussed:** Config location, Composition pattern, The three REQUIRED fields, Backbone freeze strategy, Config scope

---

## Config Location

| Option | Description | Selected |
|--------|-------------|----------|
| `sam3/train/configs/custom_finetune/` | Follow existing project pattern | ✓ |
| `configs/custom_finetune/` | Top-level, visible outside the package | |

**User's choice:** `sam3/train/configs/custom_finetune/` — follow existing project pattern
**Notes:** Existing configs (roboflow_v100, odinw13) all live under `sam3/train/configs/`. Hydra discovers configs relative to `sam3/train/`. Following this pattern avoids Hydra config discovery issues.

---

## Composition Pattern

| Option | Description | Selected |
|--------|-------------|----------|
| Overlay/compose | decoder_only.yaml and full_finetune.yaml contain only the LR/freeze delta | ✓ |
| Full standalone | Each config is self-contained (like roboflow_v100) | |

**User's choice:** Overlay/compose (agent-recommended)
**Notes:** User asked for recommendation. Agent recommended overlay because ROADMAP success criteria explicitly state "decoder_only and full_finetune differ only in LR/freeze strategy fields." This avoids ~200 lines of duplication per override file.

---

## The Three REQUIRED Fields

| Option | Description | Selected |
|--------|-------------|----------|
| 3 path fields | dataset_img_folder + train_ann_file + val_ann_file | ✓ |
| 2 path fields | dataset_root + annotation_subpath (train/val inferred) | |
| With class_names | Keep a class_names field for documentation/clarity | |

**User's choice:** 3 path fields: `dataset_img_folder` + `train_ann_file` + `val_ann_file`

**Inline comments decision:**
| Option | Description | Selected |
|--------|-------------|----------|
| Full comments | # REQUIRED: on 3 required fields + inline comments on every other field | ✓ |
| Minimal | Only mark the 3 required fields | |

**User's choice:** Full inline comments on all fields
**Notes:** SAM3 reads class names from COCO JSON categories — no separate `class_names` config field needed. The three paths map directly to the output of `prepare_dataset.py`.

---

## Backbone Freeze Strategy

| Option | Description | Selected |
|--------|-------------|----------|
| `lr_scale: 0.03` | Near-zero backbone LR, follows ROADMAP spec, no code changes | ✓ |
| `lrd_vision_backbone: 0.0` | Mathematically cleaner "true freeze" via decay | |
| You decide | Match existing SAM3 fine-tuning patterns | |

**User's choice:** `lr_scale: 0.03` for decoder_only — matches ROADMAP specification

**Full finetune backbone decay:**
| Option | Description | Selected |
|--------|-------------|----------|
| `lrd_vision_backbone: 0.9` | Follow existing roboflow full-finetune pattern | ✓ |
| `lrd_vision_backbone: 1.0` | Uniform LR across all backbone layers | |

**User's choice:** `lrd_vision_backbone: 0.9` — matches `roboflow_v100_full_ft_100_images.yaml`

---

## Config Scope

| Option | Description | Selected |
|--------|-------------|----------|
| Full config | base.yaml includes paths, data, model, optimizer, scheduler, eval | ✓ |
| Minimal config | Only dataset wiring + segmentation flag; inherit training loop elsewhere | |

**User's choice:** Full standalone config (like roboflow config)

**Hyperparameter defaults:**
| Option | Description | Selected |
|--------|-------------|----------|
| Small-dataset defaults | epochs=40, batch=1, grad_accum=4, transformer_lr=8e-5, backbone_lr=2.5e-6 | ✓ |
| Mirror roboflow defaults | Let user tune from roboflow baseline | |

**User's choice:** Small-dataset defaults from ROADMAP specification

---

## Agent's Discretion

None — all areas had clear user selections.

## Deferred Ideas

None — discussion stayed within phase scope.
