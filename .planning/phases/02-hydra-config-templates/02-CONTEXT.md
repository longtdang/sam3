# Phase 2: Hydra Config Templates - Context

**Gathered:** 2026-05-27
**Status:** Ready for planning

<domain>
## Phase Boundary

Deliver three Hydra YAML config files under `sam3/train/configs/custom_finetune/` — `base.yaml`, `decoder_only.yaml`, and `full_finetune.yaml` — plus a smoke test that verifies all three compose without Hydra errors. A user with a CVAT COCO dataset prepared by `scripts/prepare_dataset.py` should be able to launch a fine-tuning run by editing exactly three fields in `base.yaml`.

</domain>

<decisions>
## Implementation Decisions

### Config Location
- **D-01:** New configs live at `sam3/train/configs/custom_finetune/` — follows the existing project convention where all training configs live under `sam3/train/configs/`.
- **D-02:** `base.yaml` uses `# @package _global_` and `defaults: [_self_]` — matching all existing configs in this codebase.

### Composition Pattern
- **D-03:** `decoder_only.yaml` and `full_finetune.yaml` are Hydra config group members that contain only the LR/freeze strategy delta fields (2–3 fields each). They do NOT duplicate base.yaml — all other values inherit from base.yaml via Hydra compose.
- **D-04:** Users switch strategy with: `python sam3/train/train.py --config-name custom_finetune/base '+finetune_strategy=decoder_only'` — no editing of base.yaml required.

### The Three REQUIRED Fields
- **D-05:** The three fields a user MUST change to wire a new dataset are:
  1. `paths.dataset_img_folder` — path to the images directory (same for train and val)
  2. `paths.train_ann_file` — path to `train.json` produced by `prepare_dataset.py`
  3. `paths.val_ann_file` — path to `val.json` produced by `prepare_dataset.py`
- **D-06:** All three are marked with `# REQUIRED:` inline comments in base.yaml. Every other field in the config also gets an inline comment explaining what it does.
- **D-07:** Class names are NOT a separate config field — SAM3 reads category names directly from the COCO JSON `categories` list. No user-maintained class list needed.

### Backbone Freeze Strategy
- **D-08 (decoder_only.yaml delta):** `lr_scale: 0.03` — keeps backbone LR near-zero (~7.5e-6 effective) without modifying trainer code. This is the ROADMAP-specified approach and requires no code changes.
- **D-09 (full_finetune.yaml delta):** `lrd_vision_backbone: 0.9` — layer-wise LR decay across all ViT backbone layers, matching the existing `roboflow_v100_full_ft_100_images.yaml` pattern.
- **D-10:** The base.yaml defaults to decoder-only strategy (`lr_scale: 0.03`) since the target use case is < 500 images where full fine-tune risks overfitting.

### Config Scope (base.yaml)
- **D-11:** `base.yaml` is a complete, self-contained training config — includes paths, transforms, dataset wiring (train + val), model config, optimizer, scheduler, and eval sections. Users never need to reference any other config file to run training.
- **D-12:** Small-dataset hyperparameter defaults:
  - `epochs: 40`
  - `train_batch_size: 1`
  - `gradient_accumulation_steps: 4`
  - `lr_transformer: 8e-5` (equivalent to `lr_scale × base_lr`)
  - `lr_vision_backbone: 2.5e-6`
- **D-13:** `enable_segmentation: true` is set in `scratch` (not the default in existing configs, which is `False`). The data loaders also have `load_segmentation: true`. This is a critical difference from existing configs and must be clearly commented.
- **D-14:** Normalization uses `[0.5, 0.5, 0.5]` for mean and std (SAM3 standard — NOT ImageNet values). Follows the existing roboflow config pattern.

### Smoke Test
- **D-15:** The smoke test plan verifies all three configs compose without Hydra errors using `python sam3/train/train.py --config-name custom_finetune/base --cfg job` (Hydra's config-print-only mode — no actual training).
- **D-16:** Smoke test is a standalone test plan (Plan 4 per ROADMAP) — not a pytest file. It verifies the configs are syntactically valid and Hydra can resolve all interpolations.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Closest Existing Config Analog
- `sam3/train/configs/roboflow_v100/roboflow_v100_full_ft_100_images.yaml` — The primary reference for how a custom dataset is wired into SAM3 training. Shows `paths`, `scratch`, `data.train`, `data.val`, `model`, `optimizer`, `eval` structure. MUST read before writing base.yaml.

### Data Loading Infrastructure
- `sam3/train/data/sam3_image_dataset.py` — Dataset class; understand `img_folder`, `ann_file`, `load_segmentation`, `coco_json_loader` parameters
- `sam3/train/data/coco_json_loaders.py` — `COCO_FROM_JSON` loader used for val dataset; understand `include_negatives`, `category_chunk_size`

### Training Entry Point
- `sam3/train/train.py` — Hydra entry point; check how `--config-name` and config discovery work
- `sam3/train/trainer.py` — Trainer class; check how `enable_segmentation` flows through, and where `lrd_vision_backbone` layer decay is applied

### Phase 1 Context (integration point)
- `.planning/phases/01-dataset-preparation/01-CONTEXT.md` — Output format decisions; `file_name` is filename-only, `img_folder` is set separately in config

### Planning Reference
- `.planning/ROADMAP.md` §Phase 2 — Success criteria and four plan descriptions
- `.planning/REQUIREMENTS.md` — CFG-01 through CFG-06, DOC-03 requirements

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `sam3/train/configs/roboflow_v100/roboflow_v100_full_ft_100_images.yaml` — The ~400-line template to adapt for base.yaml. Key sections to copy and modify: `paths`, `scratch` (enable_segmentation, norm values, LR, epochs), `data.train`, `data.val`, `model`, `optimizer` (with layer-decay block), `eval`.
- Transform pipeline (`train_transforms`, `val_transforms`) — Copy the existing ResizeAPI + PadToSizeAPI + ToTensorAPI + NormalizeAPI chain from roboflow config; adjust norm values.

### Established Patterns
- **`# @package _global_` + `defaults: [_self_]`** — All existing configs use this header. base.yaml must follow the same convention.
- **`scratch` section** — Variables are defined under `scratch:` and referenced via `${scratch.field}` throughout the config. New config follows this pattern.
- **`paths` section** — User-editable path variables live at the top under `paths:`. Three REQUIRED fields go here.
- **`img_folder` + `ann_file` pattern** — `Sam3ImageDataset` takes `img_folder` (directory) and `ann_file` (JSON path) separately. `file_name` in JSON is filename-only (ensured by Phase 1 script). This maps directly to D-05's three REQUIRED fields.
- **`enable_segmentation`** — Must be `true` in both `scratch` and `model.enable_segmentation`. The default in all existing configs is `False` (they're detection-only). Our config sets it to `True` — comment this prominently.
- **Layer decay optimizer block** — `sam3/train/train.py` (or trainer) applies LLRD to `backbone.vision_backbone.trunk` when `lrd_vision_backbone` is set. The optimizer config block in roboflow config (around line 350+) shows the exact structure.

### Integration Points
- `prepare_dataset.py` output → `paths.train_ann_file` and `paths.val_ann_file` in base.yaml
- `paths.dataset_img_folder` → the images directory used in both train and val `Sam3ImageDataset` instances
- `enable_segmentation: true` activates mask loss in `Sam3LossWrapper` and mask decoding in the model

</code_context>

<specifics>
## Specific Ideas

- Decoder-only is the DEFAULT strategy (base.yaml ships with `lr_scale: 0.03`) — full fine-tune is opt-in via override flag. This reflects the < 500 image target use case where decoder-only is safer.
- The Hydra compose override naming: `'+finetune_strategy=full_finetune'` (the config group is `finetune_strategy/`, members are `decoder_only.yaml` and `full_finetune.yaml`).
- The smoke test uses Hydra's `--cfg job` flag to print the composed config without launching training — this validates config syntax and Hydra interpolation without GPU.

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope.

</deferred>

---

*Phase: 2-Hydra Config Templates*
*Context gathered: 2026-05-27*
