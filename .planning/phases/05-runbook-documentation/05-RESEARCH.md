# Phase 5 Research: Runbook Documentation

**Researched:** 2026-05-28
**Domain:** Technical writing / SAM3 fine-tuning pipeline runbook
**Confidence:** HIGH — all findings verified from source code in this repo

---

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

**D-05-01: FINE_TUNING.md location**
Repo root (alongside README.md). Maximum discoverability; first doc a user sees.

**D-05-02: Inference section depth**
Full inference example — load model + run on a sample image + show output masks. Include:
- Load `best_checkpoint.pth` via `build_sam3_image_model(load_from_HF=False, device="cpu")`
- Preprocess an image into a `BatchedDatapoint`
- Call `model.forward()` and interpret the output masks
- Code snippet that users can copy-paste

**D-05-03: Multi-GPU coverage**
Cover both single-GPU and multi-GPU (--nproc_per_node=N with NCCL/CUDA notes). Include:
- Single-GPU launch: `torchrun --nproc_per_node=1 -m sam3.train.train --config-name custom_finetune/base`
- Multi-GPU launch: `torchrun --nproc_per_node=N ...` with `--master_addr` / `--master_port` guidance
- NCCL backend requirement (CUDA)
- Note about `batch_size` per GPU (effective batch = batch_size × N_GPUs)

### the agent's Discretion
None specified.

### Deferred Ideas (OUT OF SCOPE)
- Phase 1–4 reimplementation (already done)
- Video/interactive predictor usage (different model variant)
- Deployment / serving infrastructure
</user_constraints>

---

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| DOC-01 | `FINE_TUNING.md` runbook documents full workflow: install → prepare data → configure → train → evaluate → export checkpoint → run inference | All 7 stages mapped to verified commands/APIs below |
| DOC-02 | Runbook includes troubleshooting section covering top 5 gotchas (symptom → cause → fix) | 5 gotchas verified against codebase with exact line references |
</phase_requirements>

---

## Summary

Phase 5 produces a single Markdown file (`FINE_TUNING.md`) at the repo root. The research task is to verify every command, API call, and code example that will appear in that file — no invention, no approximation.

All findings below are verified against source files in this repo. Two significant corrections are flagged vs. CONTEXT.md locked decisions: (1) the launch command is **not** `torchrun` but `python sam3/train/train.py -c ...`; and (2) the inference path uses `Sam3Processor` (the official high-level API) rather than hand-constructing `BatchedDatapoint` (which `test_checkpoint_compatibility.py` explicitly describes as "complex to construct").

**Primary recommendation:** Write FINE_TUNING.md as a linear, copy-paste-first runbook. Use `Sam3Processor` for the inference example — it is the supported public API and is demonstrated in README.md. Document the 4 (not 3) required config fields. Use the `python sam3/train/train.py -c ...` launch form, not `torchrun`.

---

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| Dataset preparation | CLI script | Python stdlib | `scripts/prepare_dataset.py` is a standalone script; no framework |
| Hydra config editing | Config file | — | User edits 4 null fields in `base.yaml`; Hydra handles composition |
| Training launch | CLI / process manager | submitit (optional) | `train.py` spawns GPU workers internally via `torch.multiprocessing` |
| Checkpoint storage | Filesystem | — | `{experiment_log_dir}/checkpoints/best_checkpoint.pth` |
| Inference / verification | Python API | Sam3Processor | `build_sam3_image_model()` + `Sam3Processor` high-level wrapper |

---

## API Inventory

### `build_sam3_image_model()` — verified from `sam3/model_builder.py:573`

```python
def build_sam3_image_model(
    bpe_path=None,                # None → uses bundled sam3/assets/bpe_simple_vocab_16e6.txt.gz
    device="cuda",                # "cuda" (default if CUDA available) or "cpu"
    eval_mode=True,               # puts model in eval mode (inference)
    checkpoint_path=None,         # path to .pth checkpoint; None + load_from_HF=True → download from HF
    load_from_HF=True,            # True = download sam3.pt from HF Hub; set False for custom checkpoints
    enable_segmentation=True,     # must be True to use the mask decoder head
    enable_inst_interactivity=False,
    compile=False,
) -> Sam3Image
```

**Returns:** `Sam3Image` instance (a `torch.nn.Module`).

**For fine-tuned checkpoint loading:**
```python
model = build_sam3_image_model(
    checkpoint_path="/path/to/best_checkpoint.pth",
    load_from_HF=False,   # prevent accidental HuggingFace download
    enable_segmentation=True,
    device="cpu",         # cpu for compatibility testing; cuda for production
    eval_mode=True,
)
```
[VERIFIED: `sam3/model_builder.py:573-654`, `scripts/test_checkpoint_compatibility.py:46-52`]

---

### `Sam3Processor` — the recommended inference wrapper

**Why this over raw `BatchedDatapoint`:** `Sam3Image.forward()` takes a `BatchedDatapoint`
(a complex multi-field dataclass — see `sam3/model/data_misc.py:217`). Constructing it
requires populating `find_inputs: List[FindStage]`, `find_targets: List[BatchedFindTarget]`,
`find_metadatas: List[BatchedInferenceMetadata]`, and several tensor fields. The test script
`scripts/test_checkpoint_compatibility.py:17-18` explicitly states: *"Sam3Image.forward()
requires a structured BatchedDatapoint input that is complex to construct."* `Sam3Processor`
encapsulates all of this; it is the documented public API (README.md "Basic Usage").

```python
from PIL import Image
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

# 1. Load fine-tuned model
model = build_sam3_image_model(
    checkpoint_path="/path/to/best_checkpoint.pth",
    load_from_HF=False,
    enable_segmentation=True,
    device="cpu",
    eval_mode=True,
)

# 2. Create processor (handles image resizing, normalization, and BatchedDatapoint construction internally)
processor = Sam3Processor(model)

# 3. Set image (accepts PIL.Image, np.ndarray, or torch.Tensor)
image = Image.open("/path/to/image.jpg")
state = processor.set_image(image)

# 4. Query with text prompt
output = processor.set_text_prompt("defect", state)

# 5. Interpret output
masks  = output["masks"]         # bool tensor, shape (N, 1, H, W)  — N detected instances
boxes  = output["boxes"]         # float tensor, shape (N, 4)       — xyxy format, pixel coords
scores = output["scores"]        # float tensor, shape (N,)          — confidence scores [0,1]
```

[VERIFIED: `sam3/model/sam3_image_processor.py:42-221`, README.md "Basic Usage"]

---

### `Sam3Image.forward()` — low-level output structure (for reference)

```python
# forward() returns SAM3Output (a list subclass — see sam3/model/model_misc.py:954)
# Access the last stage output dict:
out = model.forward(batched_datapoint)   # returns SAM3Output
stage_dict = out[-1][-1]                  # last stage, last step

# stage_dict keys (relevant to inference):
stage_dict["pred_logits"]   # (batch, num_queries, 1) — raw classification logits
stage_dict["pred_boxes"]    # (batch, num_queries, 4) — cxcywh, normalized [0,1]
stage_dict["pred_masks"]    # (batch, num_queries, H, W) — mask logits (NOT binary)
```

`Sam3Processor._forward_grounding()` handles the postprocessing:
- thresholds on `pred_logits` to get presence scores
- converts `pred_boxes` from cxcywh normalized → xyxy pixel coords
- thresholds `pred_masks` at 0.5 → binary masks
[VERIFIED: `sam3/model/sam3_image.py:555-601`, `sam3/model/sam3_image_processor.py:191-221`]

---

## Config Fields Reference

**Critical:** STATE.md decision D-P2-02 states: **"4 REQUIRED markers (including experiment_log_dir) not 3"** — the CONTEXT.md says "3 required null fields" but the actual `base.yaml` has 4.

### All 4 Required Null Fields (`sam3/train/configs/custom_finetune/base.yaml`)

| Field (yaml path) | Type | Example value | Purpose |
|-------------------|------|---------------|---------|
| `paths.dataset_img_folder` | `str` (absolute path) | `/data/my_dataset/images` | Directory containing ALL images (train + val share one dir when using `prepare_dataset.py`) |
| `paths.train_ann_file` | `str` (absolute path) | `/data/splits/train.json` | Output of `scripts/prepare_dataset.py` |
| `paths.val_ann_file` | `str` (absolute path) | `/data/splits/val.json` | Output of `scripts/prepare_dataset.py` |
| `paths.experiment_log_dir` | `str` (absolute path) | `/runs/my_experiment` | Root for checkpoints, TensorBoard, eval dumps, logs |

**Optional field:**
- `paths.bpe_path`: `null` = use bundled `sam3/assets/bpe_simple_vocab_16e6.txt.gz` (recommended)

[VERIFIED: `sam3/train/configs/custom_finetune/base.yaml:9-14`, STATE.md decision D-P2-02]

### Key Hyperparameters (non-null defaults users may want to tune)

| Field | Default | Notes |
|-------|---------|-------|
| `scratch.max_data_epochs` | 40 | Increase for larger datasets |
| `scratch.train_batch_size` | 1 | Per-GPU batch size |
| `scratch.gradient_accumulation_steps` | 4 | Effective batch = 1 × 4 = 4 |
| `scratch.lr_transformer` | 8e-5 | Main LR for transformer/attention heads |
| `scratch.lr_vision_backbone` | 2.5e-6 | ViT trunk LR (low = near-frozen) |
| `trainer.val_epoch_freq` | 1 | Evaluate every epoch |
| `launcher.gpus_per_node` | 1 | Override with `--num-gpus N` on CLI |

[VERIFIED: `sam3/train/configs/custom_finetune/base.yaml:54-79`]

---

## Checkpoint Output Path

### Location

```
{experiment_log_dir}/checkpoints/best_checkpoint.pth
```

Configured in `base.yaml`:
```yaml
trainer:
  checkpoint:
    save_dir: ${launcher.experiment_log_dir}/checkpoints
    save_best_meters:
      - "val_custom/detection"   # triggers best_checkpoint.pth when AP improves
```

`${launcher.experiment_log_dir}` resolves to `${paths.experiment_log_dir}`.

[VERIFIED: `sam3/train/configs/custom_finetune/base.yaml:369-374`]

### Checkpoint Format (HuggingFace inference format)

```python
# best_checkpoint.pth structure (written by trainer.py patch — CKPT-01)
{
    "model": {
        "detector.<layer_name>": <torch.Tensor>,
        # ...one key per model parameter, prefixed with "detector."
    }
}
```

This format is loaded by `_load_checkpoint()` which:
1. Loads with `weights_only=True`
2. Extracts `ckpt["model"]`
3. Strips the `"detector."` prefix
4. Loads into `Sam3Image.state_dict()`

**Other checkpoint files** (`checkpoint.pt`, `checkpoint_N.pt`): full training checkpoints containing optimizer state, epoch, scaler, etc. — NOT suitable for inference loading via `build_sam3_image_model()`.

[VERIFIED: `sam3/train/trainer.py:377-399`]

---

## Launch Commands

### ⚠️ IMPORTANT CORRECTION vs. CONTEXT.md D-05-03

CONTEXT.md D-05-03 specifies `torchrun --nproc_per_node=N -m sam3.train.train --config-name ...`.
**This is incorrect.** Verified from `sam3/train/train.py`:

- `train.py` uses its **own** process management (`torch.multiprocessing.start_processes` via `single_node_runner`)
- The argparser uses `-c`/`--config` (not `--config-name`)
- `torchrun` + `single_node_runner` would double-spawn processes (each torchrun process would try to spawn N more)
- Multi-GPU count is set via `--num-gpus N` (CLI) or `launcher.gpus_per_node` (yaml)

### Verified Launch Commands

**Single GPU (local):**
```bash
python sam3/train/train.py \
    -c custom_finetune/base \
    --use-cluster 0 \
    --num-gpus 1
```

**Multi-GPU (local, N GPUs on one machine):**
```bash
python sam3/train/train.py \
    -c custom_finetune/base \
    --use-cluster 0 \
    --num-gpus N
```

**With finetune strategy override (decoder-only):**
```bash
python sam3/train/train.py \
    -c custom_finetune/base \
    +custom_finetune/finetune_strategy=decoder_only \
    --use-cluster 0 \
    --num-gpus 1
```

**Config-only dry run (prints resolved config, no training):**
```bash
python scripts/test_config_parse.py
```

[VERIFIED: `sam3/train/train.py:60-77, 312-338`, README_TRAIN.md]

### NCCL Backend

Configured in `base.yaml`:
```yaml
trainer:
  distributed:
    backend: nccl           # set in YAML — not inferred
    find_unused_parameters: True
    gradient_as_bucket_view: True
```

- Requires CUDA — CPU training is not supported (project requirement: `CUDA required` per REQUIREMENTS.md Out of Scope)
- `find_unused_parameters: True` is required — SAM3 has conditional graph branches

**Effective batch size with multi-GPU:**
`effective_batch = train_batch_size × gradient_accumulation_steps × N_GPUs`  
Default: 1 × 4 × N

[VERIFIED: `sam3/train/configs/custom_finetune/base.yaml:230-234`]

---

## Inference Code Sketch

Full verified copy-paste snippet for `FINE_TUNING.md`:

```python
from PIL import Image
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

# Step 1: Load fine-tuned checkpoint
model = build_sam3_image_model(
    checkpoint_path="/path/to/best_checkpoint.pth",
    load_from_HF=False,        # critical: prevents HF download
    enable_segmentation=True,  # must match training config
    device="cpu",              # use "cuda" for GPU inference
    eval_mode=True,
)

# Step 2: Create processor (wraps BatchedDatapoint construction)
processor = Sam3Processor(model)

# Step 3: Load and set image (PIL.Image is the standard input)
image = Image.open("/path/to/validation_image.jpg")
state = processor.set_image(image)

# Step 4: Query with the class name used during training
output = processor.set_text_prompt("defect", state)  # replace with your class name

# Step 5: Inspect outputs
masks  = output["masks"]   # shape (N, 1, H, W), dtype=bool — binary segmentation masks
boxes  = output["boxes"]   # shape (N, 4), dtype=float — bounding boxes in xyxy pixel coords
scores = output["scores"]  # shape (N,), dtype=float — confidence scores in [0, 1]

print(f"Detected {len(scores)} instance(s)")
for i, (mask, box, score) in enumerate(zip(masks, boxes, scores)):
    print(f"  Instance {i}: score={score:.3f}, box={box.tolist()}")
    # mask[0] is a 2D bool tensor of shape (H, W)
    print(f"    Mask pixels: {mask[0].sum().item()} / {mask[0].numel()}")
```

**Notes for FINE_TUNING.md:**
- `set_image()` accepts `PIL.Image`, `np.ndarray` (HWC), or `torch.Tensor`
- The text prompt should exactly match the category name used in training (e.g., "defect")
- If using OpenCV to load images, convert BGR→RGB first:
  `image = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)`
  `image = Image.fromarray(image)`

[VERIFIED: `sam3/model/sam3_image_processor.py:42-221`, `sam3/model_builder.py:573-654`]

---

## Existing Documentation

The planner must know what FINE_TUNING.md must NOT duplicate:

| Document | Location | Content | FINE_TUNING.md relationship |
|----------|----------|---------|---------------------------|
| `README.md` | repo root | Installation (conda, PyTorch, pip install -e .), basic inference via `Sam3Processor`, HF checkpoint download | **Do not repeat** install steps verbatim — cross-link to README.md |
| `README_TRAIN.md` | repo root | Training with `python sam3/train/train.py -c ...`, SLURM cluster config, config file structure | **Complements** — FINE_TUNING.md focuses on custom dataset workflow, not the general training framework |
| `sam3/train/configs/custom_finetune/base.yaml` | embedded | Inline comments on every field (`# REQUIRED:`, `# CFG-05:`, `# CFG-06:`) | **Cross-link** — runbook sends users to the YAML for field reference |
| `CONTRIBUTING.md` | repo root | Development guidelines | Irrelevant to runbook |
| `RELEASE_SAM3p1.md` | repo root | SAM 3.1 release notes | Mention SAM 3.1 is available but fine-tuning targets SAM 3 |

[VERIFIED: repo root listing, `README.md`, `README_TRAIN.md`]

---

## Top 5 Gotchas — Verified Details

### Gotcha 1: `enable_segmentation` Off by Default

**Symptom:** Training runs without error but only bounding-box metrics appear; no `coco_eval_segm_AP` in logs; masks are always empty.

**Cause:** `scratch.enable_segmentation` defaults to `false` in upstream SAM3 configs (detection-only mode). Our `base.yaml` overrides this to `true` via `scratch.enable_segmentation: true`, which cascades to 5 downstream fields via `${scratch.enable_segmentation}` interpolation: `collate_fn.with_seg_masks`, `collate_fn_val.with_seg_masks`, `trainer.model.enable_segmentation`, `dataset.load_segmentation`, and the `Masks` loss entry.

**Fix:** Verify `scratch.enable_segmentation: true` is set (the top of `base.yaml`). Do NOT override it to `false`.

**Code evidence:** `sam3/train/configs/custom_finetune/base.yaml:20` — `enable_segmentation: true  # CFG-05:`
[VERIFIED: `base.yaml:20, 101, 108, 247, 270, 205`]

---

### Gotcha 2: Normalization Mismatch

**Symptom:** Inference produces poor or random masks even after training converges on the training set; precision/recall degrades significantly at test time.

**Cause:** SAM3's pre-training used `mean=[0.5, 0.5, 0.5]`, `std=[0.5, 0.5, 0.5]` — NOT ImageNet values (`mean≈[0.485, 0.456, 0.406]` = `[123.7/255, 116.3/255, 103.5/255]`). If custom preprocessing uses ImageNet norms (common default in torchvision), the input distribution shifts relative to what the model was trained on.

**Fix for training config:** `base.yaml` already sets SAM3 norms — do not override `scratch.train_norm_mean` / `scratch.val_norm_mean`. **Fix for inference preprocessing:** `Sam3Processor` handles normalization internally — do not normalize manually before calling `set_image()`.

**Additional sub-case:** OpenCV loads images as BGR; PIL loads as RGB. Always convert BGR → RGB before passing to `Sam3Processor.set_image()`.

**Code evidence:** `sam3/train/configs/custom_finetune/base.yaml:49-52`
```yaml
train_norm_mean: [0.5, 0.5, 0.5]  # SAM3 standard; changing breaks pretrained weight compatibility
train_norm_std: [0.5, 0.5, 0.5]
```
[VERIFIED: `base.yaml:49-52`, `sam3/model/sam3_image_processor.py` (processor applies normalization internally)]

---

### Gotcha 3: 0-Based Annotation IDs (CVAT Export)

**Symptom:** Training starts but the COCO evaluator throws a KeyError or produces AP=0 even with visually correct masks; OR `prepare_dataset.py` reports no images in a split.

**Cause:** CVAT sometimes exports annotation/image IDs starting at 0. COCO format requires 1-based IDs. SAM3's `COCO_FROM_JSON` loader fails silently (or crashes) when IDs are 0-based.

**Fix:** Always pipe CVAT exports through `scripts/prepare_dataset.py` — it applies `repair_ids()` automatically. If supplying a manually curated JSON, verify `min(img["id"] for img in data["images"]) >= 1`.

**Code evidence:** `scripts/prepare_dataset.py:63-96` — `repair_ids()` function.
[VERIFIED: `scripts/prepare_dataset.py:63-96`]

---

### Gotcha 4: `file_name` Path Prefix Collision

**Symptom:** Training crashes immediately with `FileNotFoundError` on the first image, OR silently skips all images, yielding a dataset of 0 samples.

**Cause:** CVAT exports sometimes include a path prefix in `file_name` (e.g., `"images/frame_001.jpg"` instead of `"frame_001.jpg"`). SAM3's dataset loader concatenates `img_folder + "/" + file_name`, so a prefixed name becomes `img_folder/images/frame_001.jpg` — a path that doesn't exist.

**Fix:** `scripts/prepare_dataset.py` strips all prefixes via `repair_filenames()` using `os.path.basename()`. Always use the script's output JSONs. Set `paths.dataset_img_folder` to the directory that **directly** contains the image files (the output of `prepare_dataset.py` prints a reminder: `"Set img_folder ... to the directory directly containing the image files"`).

**Code evidence:** `scripts/prepare_dataset.py:56-60` — `repair_filenames()`, `scripts/prepare_dataset.py:238-240` — the printed reminder.
[VERIFIED: `scripts/prepare_dataset.py:56-60, 238-240`]

---

### Gotcha 5: Mask Loss Not Enabled in Upstream Reference Configs

**Symptom:** If starting from a non-project YAML (e.g., copied from `sam3/train/configs/roboflow_v100/`), training produces bounding-box metrics but no mask AP; `coco_eval_segm_AP50` is never logged.

**Cause:** The upstream roboflow reference configs comment out the `Masks` entry in `loss_fns_find`. Our `base.yaml` has it uncommented (enabled). Users who accidentally mix upstream configs with our fine-tuning workflow will miss mask training.

**Fix:** Check that the `Masks` loss entry is present and uncommented in the config:
```yaml
loss_fns_find:
  - _target_: sam3.train.loss.loss_fns.Masks   # ← must be present and uncommented
    focal_alpha: 0.25
    focal_gamma: 2.0
    weight_dict:
      loss_mask: 200.0
      loss_dice: 10.0
```
Also verify `scratch.enable_segmentation: true` (Gotcha 1 overlap).

**Code evidence:** `sam3/train/configs/custom_finetune/base.yaml:205-210` — `Masks` loss uncommented with comment `# mask supervision (UNCOMMENTED — required for segmentation training)`.
[VERIFIED: `base.yaml:205-210`]

---

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest (existing, `tests/`) |
| Config file | none — inferred from `tests/` directory |
| Quick run command | `pytest tests/ -x -q` |
| Full suite command | `pytest tests/ -v` |

### FINE_TUNING.md Completeness Tests

FINE_TUNING.md is a documentation file, not code — automated tests check structural properties:

| Check | Test Method | Pass Condition |
|-------|-------------|----------------|
| All 7 workflow stages present | `grep` for stage headings | Find: install, prepare data, configure, train, evaluate, export checkpoint, run inference |
| No ellipsis in code blocks | `grep -n "\.\.\." FINE_TUNING.md` | Zero matches inside fenced code blocks |
| Troubleshooting has 5+ entries | Count `###` headings under Troubleshooting section | ≥ 5 entries |
| Symptom/Cause/Fix format | `grep -c "Symptom\|Cause\|Fix"` | ≥ 15 matches (3 per gotcha × 5) |
| Required fields documented | `grep "dataset_img_folder\|train_ann_file\|val_ann_file\|experiment_log_dir"` | 4 matches |
| Inference example has `Sam3Processor` | `grep "Sam3Processor"` | ≥ 1 match |
| Launch command present | `grep "sam3/train/train.py"` | ≥ 1 match |

### Wave 0 Gaps
None — existing test infrastructure (`pytest tests/`) is sufficient. The FINE_TUNING.md validation is a post-write grep check, not a pytest test.

*(No new test files required for Phase 5.)*

---

## Environment Availability

> Phase 5 is documentation-only (write one Markdown file). No external tool dependencies.

**Step 2.6: SKIPPED** — Phase 5 creates `FINE_TUNING.md` via text editing only. No CLI tools, databases, or external services are required at write time. The commands documented in `FINE_TUNING.md` have pre-validated dependencies from Phases 1–4.

---

## Security Domain

> Phase 5 is documentation-only. No code is executed, no credentials are handled, no user input is processed. ASVS categories V2–V6 are not applicable.

---

## Common Pitfalls

### Pitfall 1: Using `torchrun` (from CONTEXT.md D-05-03)

**What goes wrong:** A runbook that says `torchrun --nproc_per_node=N -m sam3.train.train --config-name custom_finetune/base` will fail — `train.py`'s argparser expects `-c`/`--config`, and `torchrun` conflicts with `single_node_runner`'s own `torch.multiprocessing.start_processes`.

**Why it happens:** CONTEXT.md D-05-03 was written before verifying `train.py`'s actual interface.

**How to avoid:** Use the verified command: `python sam3/train/train.py -c custom_finetune/base --use-cluster 0 --num-gpus N`

### Pitfall 2: Documenting 3 Required Fields (vs. 4)

**What goes wrong:** User sets only 3 fields, leaves `experiment_log_dir: null`, and gets a Hydra interpolation error at training start (not a clear config-missing error).

**Why it happens:** CONTEXT.md says "3 required null fields" but STATE.md decision D-P2-02 records "4 REQUIRED markers (including experiment_log_dir)".

**How to avoid:** Document all 4 fields. `base.yaml:9-13` shows all 4 with `# REQUIRED:` comments.

### Pitfall 3: Direct `BatchedDatapoint` Construction

**What goes wrong:** Following D-05-02 literally ("Preprocess an image into a BatchedDatapoint") leads to a complex dataclass construction exercise. `BatchedDatapoint` has 7 fields including `find_inputs: List[FindStage]` and `find_metadatas: List[BatchedInferenceMetadata]` — each a nested dataclass.

**Why it happens:** D-05-02 uses internal terminology. The practical implementation uses `Sam3Processor` which handles `BatchedDatapoint` construction internally.

**How to avoid:** Use `Sam3Processor` in the inference example. It IS the preprocessing step. The runbook can note that `Sam3Processor` handles image-to-`BatchedDatapoint` conversion internally.

### Pitfall 4: Commands with Placeholder Ellipses

**What goes wrong:** A code block like `python sam3/train/train.py -c ... --num-gpus 2` is not copy-pasteable — users paste the literal `...` and get a Python error.

**Why it happens:** Temptation to use `...` as shorthand for "other args".

**How to avoid:** Every shell command in a fenced code block must be fully formed. Variability goes in comments (`# replace N with your GPU count`), not in the command itself.

### Pitfall 5: Normalization Values Confusion

**What goes wrong:** The comment in CONTEXT.md (Gotcha 2) lists `mean=[123.675, 116.28, 103.53]` and describes these as "SAM3 expects" values — but these are ImageNet values that SAM3 does NOT use. SAM3 uses `[0.5, 0.5, 0.5]` (verified in `base.yaml:49-52`).

**Why it happens:** The CONTEXT.md description of Gotcha 2 appears to have the example values reversed — the point of the gotcha is to NOT use ImageNet norms.

**How to avoid:** The troubleshooting entry should say: "SAM3 uses `[0.5, 0.5, 0.5]`; do not substitute ImageNet norms `[0.485, 0.456, 0.406]`."

---

## Code Examples

All verified patterns from source files:

### Dataset Preparation (verified CLI)
```bash
python scripts/prepare_dataset.py \
    --ann-file /path/to/instances_default.json \
    --img-folder /path/to/images \
    --output /path/to/data/splits \
    --split-ratio 0.8 \
    --seed 42
# Produces: /path/to/data/splits/train.json and val.json
```
[VERIFIED: `scripts/prepare_dataset.py:30-42`]

### Config Dry Run (verified)
```bash
python scripts/test_config_parse.py
# Verifies all 3 custom_finetune configs parse without Hydra errors
```
[VERIFIED: `scripts/test_config_parse.py`]

### Checkpoint Compatibility Check (verified)
```bash
python scripts/test_checkpoint_compatibility.py \
    --checkpoint /path/to/best_checkpoint.pth
# Exit 0: [OK] Loaded checkpoint ... / [OK] Model params: N
# Exit 1: [FAIL] <error>
```
[VERIFIED: `scripts/test_checkpoint_compatibility.py:29-75`]

### CI Fake Dataset Generation (verified)
```bash
OUT=$(python scripts/generate_fake_dataset.py --out /tmp/fake_defect_data)
echo "Dataset at: $OUT"
# Creates: $OUT/images/fake_0001.png ... fake_0005.png + train.json + val.json
```
[VERIFIED: `scripts/generate_fake_dataset.py:91-119`]

---

## State of the Art

| Old Approach | Current Approach | Impact |
|--------------|------------------|--------|
| ImageNet normalization | SAM3 normalization `[0.5, 0.5, 0.5]` | Critical for pretrained weight compatibility |
| Full fine-tune by default | Decoder-only (low backbone LR 2.5e-6) for < 500 images | Prevents overfitting on small datasets |
| Single checkpoint | Separate training checkpoints (`checkpoint.pt`) + inference checkpoint (`best_checkpoint.pth`) | `best_checkpoint.pth` is inference-ready; `checkpoint.pt` is for resuming training |

---

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | `Sam3Processor.set_text_prompt()` is decorated with `@torch.inference_mode()` (or equivalent) — so no explicit `torch.no_grad()` needed in the code example | Inference Code Sketch | Low — worst case: user adds redundant `torch.no_grad()` wrapper |
| A2 | FINE_TUNING.md can cross-link README.md for installation steps without duplicating them | Existing Documentation | Low — decision confirmed by D-05-01 (both files at repo root, adjacent) |

All other claims in this document are verified against source files in this repo.

---

## Open Questions

1. **Torchrun vs. python launcher for FINE_TUNING.md**
   - What we know: `train.py` uses its own process management; `torchrun` is incompatible with `single_node_runner`; D-05-03 specifies `torchrun` commands
   - What's unclear: Whether the planner should follow D-05-03 literally (document broken commands) or document the verified working commands
   - Recommendation: **Document the verified `python sam3/train/train.py` commands.** A runbook with broken commands defeats its own purpose. Note in FINE_TUNING.md that training uses internal DDP management (not torchrun).

2. **D-05-02 "BatchedDatapoint" wording**
   - What we know: D-05-02 says "Preprocess an image into a BatchedDatapoint"; in practice `Sam3Processor` handles this
   - What's unclear: Whether the user wants to see the BatchedDatapoint construction explicitly
   - Recommendation: **Use `Sam3Processor` in the code example** (the public API), and add a comment: "Sam3Processor handles BatchedDatapoint construction internally." This satisfies the intent of D-05-02 (full inference example) without requiring users to construct internal data structures.

---

## Sources

### Primary (HIGH confidence — verified from source files)
- `sam3/model_builder.py:573-654` — `build_sam3_image_model()` signature and behavior
- `sam3/model/sam3_image_processor.py:42-221` — `Sam3Processor` API (`set_image`, `set_text_prompt`, output keys)
- `sam3/train/configs/custom_finetune/base.yaml` — all 4 required fields, normalization values, NCCL config, checkpoint paths
- `sam3/train/trainer.py:333-399` — `save_checkpoint()` patch, `best_checkpoint.pth` format and path
- `sam3/train/train.py:44-338` — actual launch interface (`-c` flag, `--num-gpus`, `single_node_runner`)
- `sam3/model/data_misc.py:217-225` — `BatchedDatapoint` dataclass fields
- `sam3/model/sam3_image.py:555-601` — `Sam3Image.forward()` signature and return type
- `scripts/prepare_dataset.py` — CLI interface and all repair functions
- `scripts/test_checkpoint_compatibility.py` — verified `build_sam3_image_model()` call pattern
- `scripts/generate_fake_dataset.py` — CI fake dataset structure
- `.planning/STATE.md` — decisions D-P2-02 (4 required fields), D-04-02 (checkpoint format)

### Secondary (MEDIUM confidence)
- `README.md` — "Basic Usage" section: confirmed `Sam3Processor` is the documented inference API
- `README_TRAIN.md` — confirmed `python sam3/train/train.py -c ...` is the documented training interface

---

## Metadata

**Confidence breakdown:**
- API inventory: HIGH — verified directly from source files
- Config fields: HIGH — verified from base.yaml + STATE.md decisions
- Checkpoint path/format: HIGH — verified from trainer.py:397-398
- Launch commands: HIGH — verified from train.py argparser; CORRECTION documented
- Gotchas: HIGH — each verified with file:line references
- Inference code: HIGH — verified from sam3_image_processor.py

**Research date:** 2026-05-28
**Valid until:** Stable (documentation phase; no external dependencies to drift)
