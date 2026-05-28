# Fine-Tuning SAM 3 on a Custom Dataset

This runbook walks you through every step to fine-tune SAM 3 on your own CVAT COCO dataset — from raw annotation export to running inference on the best checkpoint.

---

## Prerequisites

- **Python environment with SAM 3 installed** — Follow the [Installation](README.md#installation) steps in `README.md` to create the `sam3` conda environment and install all dependencies.

- **SAM 3 model weights** — Request HuggingFace access at [facebook/sam3](https://huggingface.co/facebook/sam3) and authenticate with `huggingface-cli login`. The pretrained weights (`sam3.pt`) are **automatically downloaded** to your HuggingFace cache (`~/.cache/huggingface/hub/`) on first training run — you do not need to place the file manually. If you have already downloaded `sam3.pt` locally and want to skip the download, add these two lines under the `model:` block in `base.yaml`:

  ```yaml
      load_from_HF: false
      checkpoint_path: /absolute/path/to/sam3.pt
  ```

- **CVAT export with segmentation enabled** — When exporting from CVAT, select **COCO 1.0** format and check **Save images** → **Enable Segmentation masks**.

  > **⚠️ Warning:** If segmentation is not enabled, the exported JSON will have empty `segmentation` fields and training will fail silently.

- **CUDA GPU** — CUDA is required. CPU training is not supported.

---

## 1. Prepare Your Dataset

### 1.1 Expected CVAT export layout

After exporting from CVAT, your dataset directory should look like:

```text
my_dataset/
├── images/
│   ├── frame_0001.jpg
│   ├── frame_0002.jpg
│   └── frame_0003.jpg        (and any additional frames)
└── instances_default.json   ← CVAT COCO annotation file
```

### 1.2 Run prepare_dataset.py

Run the preparation script to fix CVAT quirks (0-based IDs, `file_name` prefixes) and produce stratified train/val splits:

```bash
python scripts/prepare_dataset.py \
    --ann-file my_dataset/instances_default.json \
    --img-folder my_dataset/images \
    --output data/splits
```

The script prints a summary and creates two files:

```text
Dataset preparation complete.
  Total images : 120
  Train images : 96
  Val images   : 24
  Category 'defect': 203 instances (train), 51 instances (val)

data/splits/
├── train.json
└── val.json
```

To customize the split ratio or seed:

```bash
python scripts/prepare_dataset.py \
    --ann-file my_dataset/instances_default.json \
    --img-folder my_dataset/images \
    --output data/splits \
    --split-ratio 0.8 \
    --seed 42
```

The script automatically: fixes 0-based annotation IDs → 1-based (COCO requirement), strips path prefixes from `file_name` fields, and ensures all categories appear in both splits.

---

## 2. Configure the Training Run

Open [`sam3/train/configs/custom_finetune/base.yaml`](sam3/train/configs/custom_finetune/base.yaml). Fill in the four required fields at the top of the file:

```yaml
paths:
  dataset_img_folder: /absolute/path/to/my_dataset/images  # REQUIRED
  train_ann_file: /absolute/path/to/data/splits/train.json # REQUIRED
  val_ann_file: /absolute/path/to/data/splits/val.json     # REQUIRED
  experiment_log_dir: /absolute/path/to/runs/my_experiment # REQUIRED
```

> **⚠️ Warning:** All four paths must be **absolute**. Relative paths will fail silently because Hydra resolves them from the repo root, not the working directory.

### Key hyperparameters

The defaults in `base.yaml` are tuned for small datasets (< 500 images). You may want to adjust:

| Config field | Default | When to change |
|---|---|---|
| `scratch.max_data_epochs` | 40 | Increase to 100+ for datasets with > 500 images |
| `scratch.train_batch_size` | 1 | Increase if GPU VRAM > 24 GB |
| `scratch.gradient_accumulation_steps` | 4 | Effective batch = batch_size × accumulation steps × N_GPUs |
| `scratch.lr_transformer` | 8e-5 | Reduce if loss oscillates |
| `scratch.lr_vision_backbone` | 2.5e-6 | ViT trunk LR — keep low for decoder-only strategy |

### Fine-tuning strategy

Two strategies are available as config overrides:

- **Decoder-only (default, recommended for < 500 images):** The ViT backbone is effectively frozen by setting a very low LR (`lr_vision_backbone: 2.5e-6`). Only the transformer heads and mask decoder are trained aggressively. This is the default in `base.yaml`.

- **Full fine-tuning (recommended for > 500 images):** Override with `+custom_finetune/finetune_strategy=full_finetune` to enable LLRD (`lrd_vision_backbone: 0.9`) across all ViT trunk layers. See [`sam3/train/configs/custom_finetune/finetune_strategy/full_finetune.yaml`](sam3/train/configs/custom_finetune/finetune_strategy/full_finetune.yaml).

---

## 3. Launch Training

From the repo root, launch training with:

### Single GPU

```bash
python sam3/train/train.py \
    -c custom_finetune/base \
    --use-cluster 0 \
    --num-gpus 1
```

### Multiple GPUs (single machine)

Replace `N` with the number of GPUs available (e.g., 2, 4, 8):

```bash
python sam3/train/train.py \
    -c custom_finetune/base \
    --use-cluster 0 \
    --num-gpus N
```

> **⚠️ Warning:** Do NOT use `torchrun`. `sam3/train/train.py` manages its own process spawning via `torch.multiprocessing.start_processes`. Calling `torchrun` would double-spawn GPU workers and hang or crash immediately.

Effective batch size with multi-GPU DDP:
`effective_batch = train_batch_size × gradient_accumulation_steps × N_GPUs`
Default: 1 × 4 × N. The NCCL backend is configured automatically (`trainer.distributed.backend: nccl`).

### Full fine-tuning strategy override

```bash
python sam3/train/train.py \
    -c custom_finetune/base \
    "+custom_finetune/finetune_strategy=full_finetune" \
    --use-cluster 0 \
    --num-gpus 1
```

### Dry run (verify config only, no training)

```bash
python scripts/test_config_parse.py
```

This prints the fully-resolved Hydra config and exits without launching any training process.

---

## 4. Monitor Training (TensorBoard)

TensorBoard logs are written to `{experiment_log_dir}/tensorboard/`. Launch TensorBoard with:

```bash
tensorboard --logdir /path/to/experiment_log_dir/tensorboard
```

Open http://localhost:6006 in a browser.

**What to look for:**

- `custom/loss` — overall training loss (should decrease each epoch)
- `custom/loss_mask` — mask supervision loss (should decrease after epoch 5-10)
- `custom/loss_bbox` — bounding box regression loss
- `val_custom/detection/coco_eval_segm_AP50` — primary eval metric; appears every epoch (val_epoch_freq: 1); must be > 0 by epoch 10 on clean datasets
- `val_custom/detection/coco_eval_segm_AP` — segmentation AP at IoU 0.50:0.95
- `val_custom/detection/coco_eval_segm_APs` — AP for small objects

Checkpoints are saved when `coco_eval_segm_AP50` improves (see §5 below).

---

## 5. Checkpoint Output

After training completes, your experiment directory contains:

```text
{experiment_log_dir}/
├── checkpoints/
│   ├── best_checkpoint.pth   ← best epoch by coco_eval_segm_AP50 (use this for inference)
│   ├── checkpoint.pt         ← most recent epoch (full training state — optimizer, scaler, etc.)
│   └── checkpoint_N.pt       ← periodic save at epoch N
├── tensorboard/              ← TensorBoard event files
├── dumps/                    ← COCO prediction JSON files (one per eval run)
└── logs/                     ← training log text files
```

`best_checkpoint.pth` is saved in HuggingFace-compatible inference format:

```python
# best_checkpoint.pth internal structure
{
    "model": {
        "detector.<layer_name>": "<torch.Tensor>",
        # one key per model parameter, prefixed with "detector."
    }
}
```

> **⚠️ Warning:** `checkpoint.pt` and `checkpoint_N.pt` contain full training state (optimizer, scaler, epoch counter) and are **NOT** loadable via `build_sam3_image_model()`. Always use `best_checkpoint.pth` for inference.

---

## 6. Run Inference on Your Fine-Tuned Model

Use [`Sam3Processor`](sam3/model/sam3_image_processor.py) — the public API that handles image preprocessing and `BatchedDatapoint` construction internally. See also the [Basic Usage](README.md#basic-usage) section in `README.md` for the standard (non-fine-tuned) inference workflow.

```python
from PIL import Image
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

# Step 1: Load fine-tuned checkpoint
model = build_sam3_image_model(
    checkpoint_path="/path/to/best_checkpoint.pth",
    load_from_HF=False,        # critical: prevents accidental HuggingFace download
    enable_segmentation=True,  # must match training config (scratch.enable_segmentation)
    device="cpu",              # use "cuda" for GPU inference
    eval_mode=True,
)

# Step 2: Create processor (handles image resizing, normalization, and
#         BatchedDatapoint construction internally)
processor = Sam3Processor(model)

# Step 3: Load and set image
image = Image.open("/path/to/validation_image.jpg")  # PIL.Image is the standard input
state = processor.set_image(image)

# Note: if loading with OpenCV, convert BGR -> RGB first:
# import cv2
# img_bgr = cv2.imread("/path/to/image.jpg")
# image = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))

# Step 4: Query with the class name used during training
output = processor.set_text_prompt("defect", state)  # replace with your category name

# Step 5: Inspect outputs
masks  = output["masks"]   # shape (N, 1, H, W), dtype=bool — binary segmentation masks
boxes  = output["boxes"]   # shape (N, 4), dtype=float — bounding boxes in xyxy pixel coords
scores = output["scores"]  # shape (N,), dtype=float — confidence scores in [0, 1]

print(f"Detected {len(scores)} instance(s)")
for i, (mask, box, score) in enumerate(zip(masks, boxes, scores)):
    print(f"  Instance {i}: score={score:.3f}, box={box.tolist()}")
    print(f"    Mask pixels: {mask[0].sum().item()} / {mask[0].numel()}")
```

**Output fields:**

- `masks` — boolean tensor of shape `(N, 1, H, W)` — one binary mask per detected instance
- `boxes` — float tensor of shape `(N, 4)` — bounding boxes in `[x1, y1, x2, y2]` pixel coordinates
- `scores` — float tensor of shape `(N,)` — confidence scores in `[0, 1]`

See [`scripts/test_checkpoint_compatibility.py`](scripts/test_checkpoint_compatibility.py) for a minimal load-and-verify script that checks the checkpoint loads without missing-key warnings.

---

## Troubleshooting

### 1. Segmentation metrics missing — `enable_segmentation` off

**Symptom:** Training runs without error but only bounding-box metrics appear in logs (`loss_bbox`, `loss_ce`). `coco_eval_segm_AP50` is never logged. Validation masks are always empty.

**Cause:** `scratch.enable_segmentation` defaults to `false` in upstream SAM3 detection configs. Our `base.yaml` overrides this to `true`, which cascades to five downstream fields via Hydra interpolation: `collate_fn.with_seg_masks`, `collate_fn_val.with_seg_masks`, `trainer.model.enable_segmentation`, `dataset.load_segmentation`, and the `Masks` loss entry. If any override sets this field back to `false`, or if you start from an upstream config that lacks this field, mask training is silently disabled.

**Fix:** Verify the top of `base.yaml` contains:

```yaml
scratch:
  enable_segmentation: true  # CFG-05: drives 5 downstream fields
```

Do NOT override this field to `false` in any Hydra override. If you copied a config from `sam3/train/configs/roboflow_v100/` or another upstream source, add this field explicitly.

---

### 2. Normalization mismatch (ImageNet vs SAM3 norms)

**Symptom:** Training loss decreases on the training set but inference on unseen images produces poor or random masks. Validation AP50 stays near zero even after 20+ epochs.

**Cause:** SAM3's pre-training used `mean=[0.5, 0.5, 0.5]`, `std=[0.5, 0.5, 0.5]` — NOT ImageNet values (`mean≈[0.485, 0.456, 0.406]`, `std≈[0.229, 0.224, 0.225]`). If custom preprocessing applies ImageNet normalization (common torchvision default), the input pixel distribution is shifted relative to what the pretrained weights expect, degrading all pretrained representations.

**Fix:** `base.yaml` already sets the correct values — do not override them:

```yaml
scratch:
  train_norm_mean: [0.5, 0.5, 0.5]  # SAM3 standard; changing breaks pretrained weight compatibility
  train_norm_std:  [0.5, 0.5, 0.5]
  val_norm_mean:   [0.5, 0.5, 0.5]
  val_norm_std:    [0.5, 0.5, 0.5]
```

For inference, `Sam3Processor.set_image()` applies normalization internally — do NOT normalize the image before passing it to `set_image()`. If loading images with OpenCV, convert BGR → RGB before calling `set_image()` (OpenCV returns BGR; SAM3 expects RGB):

```python
import cv2
from PIL import Image
img = Image.fromarray(cv2.cvtColor(cv2.imread("/path/to/image.jpg"), cv2.COLOR_BGR2RGB))
state = processor.set_image(img)
```

---

### 3. 0-based annotation IDs (CVAT export)

**Symptom:** `prepare_dataset.py` runs without error but a split contains 0 images. OR: training starts but the COCO evaluator throws a `KeyError` during eval, or reports AP=0 even when masks look correct in debug outputs.

**Cause:** CVAT sometimes exports annotation and image IDs starting from 0. COCO format requires 1-based IDs for all images, annotations, and categories. SAM3's `COCO_FROM_JSON` loader does not reindex IDs — it loads them as-is. A category ID of 0 can collide with background or cause lookup failures.

**Fix:** Always pipe CVAT exports through `scripts/prepare_dataset.py` — it applies `repair_ids()` automatically, reindexing all image, annotation, and category IDs to start from 1. If you are supplying a manually curated COCO JSON, verify:

```bash
python3 -c "
import json
data = json.load(open('your_annotations.json'))
min_img_id = min(img['id'] for img in data['images'])
min_ann_id = min(ann['id'] for ann in data['annotations'])
min_cat_id = min(cat['id'] for cat in data['categories'])
print(f'Min IDs -- image: {min_img_id}, annotation: {min_ann_id}, category: {min_cat_id}')
print('OK' if all(v >= 1 for v in [min_img_id, min_ann_id, min_cat_id]) else 'FAIL: 0-based IDs detected')
"
```

---

### 4. `file_name` path prefix collision

**Symptom:** Training crashes immediately with `FileNotFoundError` on the first image. OR: dataset loader initializes without error but reports 0 training samples, and loss is `nan` from the first step.

**Cause:** CVAT exports sometimes include a relative path prefix in the `file_name` field of each image record (e.g., `"images/frame_001.jpg"` instead of `"frame_001.jpg"`). SAM3's dataset loader constructs the full image path as `os.path.join(img_folder, file_name)`. A prefixed name becomes `img_folder/images/frame_001.jpg` — a path that does not exist, causing silent skips or a crash at the first batch.

**Fix:** `scripts/prepare_dataset.py` strips all path prefixes via `repair_filenames()` using `os.path.basename()`. Always use the script's output JSONs, not the raw CVAT export. Set `paths.dataset_img_folder` to the directory that **directly** contains the image files (not a parent directory). The script prints a reminder:

```text
Set img_folder in your config to the directory directly containing the image files
(e.g., /path/to/my_dataset/images -- NOT /path/to/my_dataset)
```

---

### 5. Mask loss not enabled — starting from an upstream config

**Symptom:** No `loss_mask` or `loss_dice` terms appear in TensorBoard. Training produces bounding-box and classification metrics only. `coco_eval_segm_AP50` stays at 0.

**Cause:** The upstream SAM3 reference configs (e.g., `sam3/train/configs/roboflow_v100/`) have the `Masks` loss entry commented out in `loss_fns_find`, because those configs target detection-only runs. Our `base.yaml` has it uncommented. If a user copies an upstream config and edits it instead of starting from `base.yaml`, the mask loss will be absent.

**Fix:** Verify the `Masks` entry is present and uncommented in your config's `loss_fns_find`:

```yaml
loss_fns_find:
  - _target_: sam3.train.loss.loss_fns.Boxes
    # ... (other loss entries)
  - _target_: sam3.train.loss.loss_fns.Masks   # must be PRESENT and UNCOMMENTED
    focal_alpha: 0.25
    focal_gamma: 2.0
    weight_dict:
      loss_mask: 200.0
      loss_dice: 10.0
    compute_aux: false
```

Also confirm `scratch.enable_segmentation: true` is set (see Gotcha 1 — the two are independent: `enable_segmentation` controls mask loading and model head activation; the `Masks` loss entry controls whether mask gradients are computed during training).
