# Phase 5: Runbook Documentation — Pattern Map

**Mapped:** 2026-05-28
**Files analyzed:** 3 (1 new, 2 updates)
**Analogs found:** 3 / 3

---

## File Classification

| New/Modified File | Role | Data Flow | Closest Analog | Match Quality |
|-------------------|------|-----------|----------------|---------------|
| `FINE_TUNING.md` | documentation (runbook) | N/A (prose + code blocks) | `README.md` (repo root) | exact — same repo-root Markdown doc with bash + Python blocks |
| `.planning/STATE.md` | planning state record | N/A (key-value + table) | `.planning/STATE.md` itself (update in-place) | self-analog |
| `.planning/ROADMAP.md` | planning roadmap | N/A (progress table + task lists) | `.planning/ROADMAP.md` itself (update in-place) | self-analog |

---

## Pattern Assignments

### `FINE_TUNING.md` (documentation, runbook)

**Analog:** `README.md` (repo root)
**Secondary analog:** `README_TRAIN.md` (repo root) — complements, do not duplicate

---

#### Document Structure Pattern

`README.md` uses this top-level heading pattern (lines 1–65):

```markdown
# SAM 3: Segment Anything with Concepts

## Latest updates
**DATE -- Short description.**

## Installation
### Prerequisites
- Bullet list of requirements

1. **Step title:**
```bash
command
```

## Getting Started
### Basic Usage
```python
code snippet
```

## Section Title
[prose]
```

**Apply to `FINE_TUNING.md`:** Use the same `##` / `###` heading hierarchy. Section titles are sentence-case ("Dataset preparation", not "DATASET PREPARATION"). Numbered lists for sequential steps; bullet lists for options/notes.

---

#### Code Block Style Pattern

`README.md` uses fenced code blocks with language specifiers (lines 75–103):

```markdown
```bash
conda create -n sam3 python=3.12
conda deactivate
conda activate sam3
```

```python
from PIL import Image
from sam3.model_builder import build_sam3_image_model
```
```

**Rules extracted:**
- Always specify language: ` ```bash ` or ` ```python ` or ` ```yaml `
- No trailing `\` line-continuation in bash blocks — use one command per line or `\` with real newline
- Multi-step bash: each step in its own block preceded by a numbered list item
- Python snippets: inline comments on every non-obvious line (see README.md lines 122–138)

---

#### Cross-Link Pattern

`README.md` cross-links to other docs rather than duplicating (lines 62–63, 165–168):

```markdown
See [`RELEASE_SAM3p1.md`](RELEASE_SAM3p1.md) for full details.

- [`sam3_image_predictor_example.ipynb`](examples/sam3_image_predictor_example.ipynb)
  : Demonstrates how to prompt SAM 3 with text and visual box prompts on images.
```

**Apply to `FINE_TUNING.md`:**
- Cross-link installation steps → `README.md#installation` (do NOT repeat conda/pip steps verbatim)
- Cross-link general training reference → `README_TRAIN.md`
- Cross-link config field reference → `sam3/train/configs/custom_finetune/base.yaml` inline

---

#### Inline Warnings / Notes Pattern

`README.md` uses the `⚠️` emoji prefix for critical warnings (lines 113–117):

```markdown
⚠️ Before using SAM 3, please request access to the checkpoints on the SAM 3
Hugging Face [repo](https://huggingface.co/facebook/sam3). Once accepted, you
need to be authenticated to download the checkpoints.
```

**Apply to `FINE_TUNING.md`:** Use `> **⚠️ Warning:**` blockquotes for gotcha callouts inline in sections, and dedicate a full `## Troubleshooting` section for the 5 gotchas.

---

#### Basic Inference Pattern (verified from README.md lines 122–138 + RESEARCH.md)

This is the exact copy-paste template confirmed by `Sam3Processor` API and README.md "Basic Usage":

```python
from PIL import Image
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

# Load the model
model = build_sam3_image_model()
processor = Sam3Processor(model)
# Load an image
image = Image.open("<YOUR_IMAGE_PATH.jpg>")
inference_state = processor.set_image(image)
# Prompt the model with text
output = processor.set_text_prompt(state=inference_state, prompt="<YOUR_TEXT_PROMPT>")

# Get the masks, bounding boxes, and scores
masks, boxes, scores = output["masks"], output["boxes"], output["scores"]
```

**For `FINE_TUNING.md` inference section**, extend this with fine-tuned checkpoint loading
(verified from `sam3/model_builder.py:573-654`, `scripts/test_checkpoint_compatibility.py:46-52`):

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
    print(f"    Mask pixels: {mask[0].sum().item()} / {mask[0].numel()}")
```

[SOURCE: `sam3/model/sam3_image_processor.py:42-221`, `sam3/model_builder.py:573-654`]

---

#### Launch Command Pattern (verified from README_TRAIN.md lines 19–79)

`README_TRAIN.md` documents the correct launch form (lines 19–24, 72–76):

```bash
# Example: Train on Roboflow dataset
python sam3/train/train.py -c configs/roboflow_v100/roboflow_v100_full_ft_100_images.yaml

# Single GPU training
python sam3/train/train.py -c configs/roboflow_v100/... --use-cluster 0 --num-gpus 1

# Multi-GPU training on a single node
python sam3/train/train.py -c configs/roboflow_v100/... --use-cluster 0 --num-gpus 4
```

**For `FINE_TUNING.md`**, use the verified custom-finetune form:

```bash
# Single GPU
python sam3/train/train.py \
    -c custom_finetune/base \
    --use-cluster 0 \
    --num-gpus 1

# Multi-GPU (N GPUs on one machine)
python sam3/train/train.py \
    -c custom_finetune/base \
    --use-cluster 0 \
    --num-gpus N

# Decoder-only strategy override
python sam3/train/train.py \
    -c custom_finetune/base \
    +custom_finetune/finetune_strategy=decoder_only \
    --use-cluster 0 \
    --num-gpus 1
```

> **⚠️ Do NOT use `torchrun`** — `train.py` manages its own process spawning via
> `torch.multiprocessing.start_processes`. Using `torchrun` would double-spawn workers.
> Use `--num-gpus N` instead.

[SOURCE: `sam3/train/train.py:60-77, 312-338`, `README_TRAIN.md:19-79`]

---

#### Troubleshooting Section Pattern

`README_TRAIN.md` and `RELEASE_SAM3p1.md` use `###` sub-headings with bold inline labels.
For `FINE_TUNING.md` troubleshooting, use Symptom / Cause / Fix triple per gotcha:

```markdown
## Troubleshooting

### 1. Segmentation metrics missing — `enable_segmentation` off

**Symptom:** Training runs without error but only bounding-box metrics appear in logs;
`coco_eval_segm_AP` is never logged; masks are always empty.

**Cause:** ...

**Fix:** ...
```

Apply this triple (`**Symptom:**` / `**Cause:**` / `**Fix:**`) to all 5 gotchas.

---

### `.planning/STATE.md` (planning state, update)

**Self-analog:** Current content at `.planning/STATE.md` lines 1–6.

Current header to update:
```markdown
# Project State

**Last updated:** 2026-05-28
**Current phase:** Phase 4 — Checkpoint Export & End-to-End Validation (complete)
```

**Change to:**
```markdown
**Last updated:** 2026-05-28
**Current phase:** Phase 5 — Runbook Documentation (in progress)
```

Phase status table (lines 15–22) — change Phase 5 row:
```markdown
| 5 | Runbook Documentation | ⬜ Not started |
```
→
```markdown
| 5 | Runbook Documentation | 🔄 In progress |
```

---

### `.planning/ROADMAP.md` (planning roadmap, update)

**Self-analog:** Current content at `.planning/ROADMAP.md` lines 155–162 (Progress Table).

Current Phase 5 row:
```markdown
| 5. Runbook Documentation | 0/2 | Not started | - |
```

After Phase 5 plans are written, update to reflect plan count and status. Pattern from completed phases (lines 157–160):
```markdown
| 1. Dataset Preparation | 2/2 | ✅ Complete | 2026-05-27 |
```

---

## Shared Patterns

### Markdown Heading Hierarchy
**Source:** `README.md` (entire file), `README_TRAIN.md` (entire file)
**Apply to:** `FINE_TUNING.md` all sections

```markdown
# Document Title            ← H1: one per file, document title only
## Major Section            ← H2: workflow stages, top-level groupings
### Sub-section             ← H3: steps within a stage, individual gotchas
#### Detail (optional)      ← H4: only when H3 needs further breakdown
```

### Warning / Critical Note Callout
**Source:** `README.md` lines 113–117
**Apply to:** `FINE_TUNING.md` — any step where omission would silently corrupt training

```markdown
⚠️ [inline warning for critical single-sentence notes]

> **⚠️ Warning:** [blockquote for multi-sentence warnings that need visual separation]
```

### Code Block Language Tags
**Source:** `README.md` lines 75–109, `README_TRAIN.md` lines 19–79
**Apply to:** All code blocks in `FINE_TUNING.md`

- Shell commands: ` ```bash `
- Python scripts: ` ```python `
- YAML config snippets: ` ```yaml `
- Directory trees / output: ` ```text `

### Cross-Reference Style
**Source:** `README.md` lines 62–63
**Apply to:** All references to other project files in `FINE_TUNING.md`

```markdown
[`filename.ext`](relative/path/to/filename.ext)
```

---

## Content Contract: What FINE_TUNING.md Must NOT Duplicate

| Already documented in | Content | Action in FINE_TUNING.md |
|-----------------------|---------|--------------------------|
| `README.md` lines 66–103 | Conda env creation, PyTorch install, `pip install -e .` | Cross-link only: "See [Installation](README.md#installation)" |
| `README.md` lines 119–160 | Basic `Sam3Processor` usage with HF checkpoint | Extend with fine-tuned checkpoint loading, do not repeat verbatim |
| `README_TRAIN.md` lines 1–80 | General `train.py` CLI arguments | Cross-link; only document custom-finetune-specific args |
| `sam3/train/configs/custom_finetune/base.yaml` | All config fields with inline `# REQUIRED:` comments | Reference by path; show only the 4 required fields inline |

---

## FINE_TUNING.md Section Map (for planner)

The planner should produce FINE_TUNING.md with this exact section structure:

```
# Fine-Tuning SAM 3 on a Custom Dataset

## Prerequisites
## 1. Prepare Your Dataset
## 2. Configure the Training Run
## 3. Launch Training
## 4. Monitor Training (TensorBoard)
## 5. Checkpoint Output
## 6. Run Inference on Your Fine-Tuned Model
## Troubleshooting
### 1. Segmentation metrics missing — `enable_segmentation` off
### 2. Normalization mismatch (ImageNet vs SAM3 norms)
### 3. 0-based annotation IDs (CVAT export)
### 4. `file_name` path prefix collision
### 5. Mask loss not enabled — starting from upstream config
```

Each of the 7 workflow sections maps to a CONTEXT.md in-scope item:
- §Prerequisites → D-05-01 environment setup
- §1 → "Dataset preparation" scope item
- §2 → "Config setup" scope item (4 required fields, verified from D-P2-02)
- §3 → "Training launch" scope item (D-05-03 corrected: `python train.py`, not `torchrun`)
- §4 → "TensorBoard" scope item
- §5 → "Checkpoint output" scope item
- §6 → "Inference example" scope item (D-05-02: full `Sam3Processor` snippet)

---

## No Analog Found

All three files have clear analogs. No files are without pattern guidance.

---

## Metadata

**Analog search scope:** repo root (README.md, README_TRAIN.md, RELEASE_SAM3p1.md), `.planning/` (STATE.md, ROADMAP.md)
**Files scanned:** 5
**Pattern extraction date:** 2026-05-28
**Key correction (RESEARCH.md verified):** Launch command is `python sam3/train/train.py -c custom_finetune/base --use-cluster 0 --num-gpus N` — NOT `torchrun`. Config has **4** required null fields — NOT 3.
