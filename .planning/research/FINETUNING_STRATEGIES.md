# Fine-Tuning Strategy Research

**Domain:** SAM3 fine-tuning on small custom industrial/defect datasets
**Researched:** 2025
**Sources:** SAM3 codebase (direct), official SAM3 training configs, SAM2/SAM fine-tuning literature

---

## Summary

SAM3 ships with a complete, production-tested fine-tuning pipeline (`sam3/train/train.py` + Hydra configs) that already demonstrates 100-image fine-tuning on Roboflow-V100 categories (`roboflow_v100_full_ft_100_images.yaml`). The official configs use a **soft-freeze** strategy: all model components train, but differential learning rates—0.1× for the transformer, 0.025× for the vision backbone, and 0.005× for the language backbone—effectively treat the pretrained backbone as near-frozen while letting heads adapt. For <500 industrial defect images, the primary levers are (1) lowering `lr_scale` to 0.05–0.01 relative to the defaults, (2) aggressive use of the `multiplier` field in `Sam3ImageDataset` to oversample rare classes, (3) enabling `gradient_accumulation_steps` to simulate larger batches at batch_size=1, and (4) adding color/noise augmentation layers not in the default pipeline. Evaluation follows COCO mAP semantics (`iou_type: segm` for masks, `bbox` for detection), with `AP@IoU=0.5:0.95`, `AP50`, and `AP75` as the primary metrics.

---

## Fine-Tuning Strategy for Small Datasets (< 500 images)

### Which components to fine-tune?

SAM3's official fine-tuning strategy (from `roboflow_v100_full_ft_100_images.yaml` and `odinw_text_only_train.yaml`) uses **differential learning rates across three parameter groups**, not hard freezing:

| Component | Effective LR (with lr_scale=0.1) | Notes |
|-----------|----------------------------------|-------|
| Transformer / decoder / heads | 8e-5 | Primary adaptation target |
| Vision backbone (`backbone.vision_backbone.*`) | 2.5e-5 | LLRD (layer_decay=0.9) applied |
| Language backbone (`backbone.language_backbone.*`) | 5e-6 | Near-frozen |

The trainer's `construct_optimizer` maps these via unix-glob `param_names` patterns. Layer-wise learning rate decay (`lrd_vision_backbone: 0.9`) further reduces LR for earlier ViT blocks.

**For <500 images, two options (pick based on domain shift):**

**Option A — Decoder-only (recommended when domain is very narrow / high visual similarity to pretrain data):**
Freeze the backbone by setting `requires_grad_(False)` on `backbone.vision_backbone` and `backbone.language_backbone`, then remove those param groups from the optimizer config. Skip saving frozen params with `checkpoint.skip_saving_parameters`. This halves VRAM and prevents overfitting dramatically.

```yaml
# In trainer.checkpoint:
checkpoint:
  skip_saving_parameters:
    - 'backbone.vision_backbone.*'
    - 'backbone.language_backbone.*'
```

Corresponding in the model, call `param.requires_grad_(False)` in a model subclass or pre-training init hook.

**Option B — Full fine-tuning with very low lr_scale (recommended for significant domain shift):**
Keep everything trainable but reduce `lr_scale` from 0.1 → 0.02–0.05. This is the approach the official SAM3 configs already use and is safe for 100–500 images with sufficient regularization.

**Recommendation:** Start with Option B (`lr_scale: 0.03`) and monitor val loss at every epoch. Switch to Option A if validation AP plateaus then degrades (classic overfitting signature).

### Dataset size concerns

With 500 images at batch_size=1:
- `target_epoch_size: 1500` (default) means each "epoch" samples each image ~3× per epoch
- `multiplier` in `Sam3ImageDataset` multiplies repeat factors: set to `5–10` for <100 images, `2–4` for 100–500
- Use `limit_ids: null` (do not restrict) unless explicitly held-out splits are needed

---

## Recommended Hyperparameters

All values below are grounded in the official SAM3 configs and adjusted for small-dataset fine-tuning:

### Learning Rate

```yaml
scratch:
  lr_scale: 0.03          # Reduced from default 0.1 for small datasets
  lr_transformer: ${times:8e-4,${scratch.lr_scale}}   # → 2.4e-5
  lr_vision_backbone: ${times:2.5e-4,${scratch.lr_scale}}  # → 7.5e-6
  lr_language_backbone: ${times:5e-5,${scratch.lr_scale}}  # → 1.5e-6
  lrd_vision_backbone: 0.9   # Layer-wise decay, keep default
  wd: 0.1                 # AdamW weight decay — keep high for regularization
```

For decoder-only fine-tuning (Option A), raise `lr_scale` back to 0.1 since only the decoder/head parameters train.

### Batch Size & Gradient Accumulation

```yaml
scratch:
  train_batch_size: 1               # SAM3 native default
  gradient_accumulation_steps: 4    # Effective batch = 4 (tune up on multi-GPU)
```

With 4 GPUs and batch_size=1 + accumulation=4, effective batch ≈ 16. Larger effective batch stabilizes the Hungarian matcher loss.

### Epochs & Target Epoch Size

```yaml
scratch:
  max_data_epochs: 40               # More epochs to compensate for small data
  target_epoch_size: 500            # Set to ~dataset_size for small data
                                    # (default 1500 makes sense for 500 images → 3× oversample)
```

For datasets ≤100 images, set `target_epoch_size: 200` and `max_data_epochs: 80` with early stopping (monitor `AP@50` on val split).

### Scheduler (exact class from codebase)

```yaml
# InverseSquareRootParamScheduler — defined in sam3/train/optim/schedulers.py
# Warmup → inverse sqrt decay → cooldown
scheduler_timescale: 20
scheduler_warmup: 20     # Steps, not epochs. With target_epoch_size=500 →
                         # warmup covers first ~4% of training
scheduler_cooldown: 20
```

The scheduler formula: `lr = base_lr / sqrt((step + shift) / timescale)`. With short warm-up for small datasets, consider `scheduler_warmup: 10` to reach peak LR faster.

### Optimizer

```yaml
optim:
  amp:
    enabled: True
    amp_dtype: bfloat16   # Requires CUDA 12+, matches your hardware
  optimizer:
    _target_: torch.optim.AdamW
  gradient_clip:
    _target_: sam3.train.optim.optimizer.GradientClipper
    max_norm: 0.1          # Critical — prevents loss spikes, keep this
    norm_type: 2
```

### Resolution

```yaml
scratch:
  resolution: 1008   # Default, keep unless memory-constrained
                     # For memory savings: 768 or 640 (must be divisible by 32)
```

---

## Data Augmentation

### Default SAM3 pipeline (from configs)

The official transform pipeline provides:
1. **FilterCrowds** — removes crowd-annotated instances from training
2. **RandomizeInputBbox** — adds noise to query bounding boxes (`box_noise_std: 0.1`, `box_noise_max: 20px`) — simulates imperfect user prompts
3. **DecodeRle** — decodes RLE masks to binary tensors
4. **RandomResize (multi-scale)** — scales from 480 to 1008 in 32-px steps (`get_random_resize_scales(size=1008, min_size=480, rounded=False)`)
5. **PadToSize** — square-pads with randomized offset (left/top randomized, preserving bbox)
6. **Normalize** — `mean=[0.5,0.5,0.5]`, `std=[0.5,0.5,0.5]` (NOT ImageNet stats — important!)

### What to add for industrial/defect segmentation

The default pipeline is good but **lacks color/texture augmentation** which is critical for industrial defect data (surface anomalies, rust, cracks) where lighting and surface appearance vary:

```yaml
# Insert after DecodeRle, before RandomResizeAPI in train_transforms:
- _target_: sam3.train.transforms.basic_for_api.ComposeAPI
  transforms:
    - _target_: torchvision.transforms.ColorJitter
      brightness: 0.3
      contrast: 0.3
      saturation: 0.2
      hue: 0.05
    - _target_: torchvision.transforms.RandomGrayscale
      p: 0.1
```

**Additional augmentations to consider (implement as custom transform classes in `sam3/train/transforms/`):**

| Augmentation | Rationale for industrial defects | Implementation |
|---|---|---|
| **Horizontal + Vertical flip** | Defects are orientation-invariant | `RandomHorizontalFlip(p=0.5)` exists in `basic.py`; add vertical flip |
| **Random rotation (±15°)** | Surface images often captured off-axis | Custom `RandomRotation` wrapper |
| **Gaussian blur / sharpness jitter** | Camera focus variation | `torchvision.transforms.GaussianBlur(kernel_size=5, sigma=(0.1,2.0))` |
| **Random erasing / cutout** | Simulates occlusion; forces global context learning | `RandomErasing` exists in `basic.py` |
| **Grid distortion / elastic** | Non-rigid surface deformation | `albumentations.ElasticTransform` (requires custom wrapper) |
| **CLAHE / histogram equalization** | Enhance subtle defect texture under uneven lighting | OpenCV-based custom transform |

### Augmentation priority for <500 images

1. **Must have:** Multi-scale resize, horizontal flip, box noise (already default)
2. **High value:** ColorJitter, GaussianBlur (add immediately)
3. **Medium value:** Random rotation, random erasing
4. **Low/risky:** Elastic distortion (can break small defect masks)

### Key constraint: All transforms must handle `target` dict (boxes + masks)

SAM3 transforms receive `(image, target)` pairs. The `target` dict contains `"boxes"` (XYXY), `"masks"` (binary tensors), `"labels"`, etc. Any augmentation touching geometry **must update boxes and masks accordingly**. The existing transforms in `sam3/train/transforms/basic.py` (`crop`, `hflip`, `resize`, `pad`) already do this correctly. Use those as templates for custom transforms.

---

## Hydra Config Structure

### Recommended config layout for a custom fine-tuning experiment

```
sam3/train/configs/
├── defect_finetune/
│   ├── defect_base.yaml          # Core shared config (paths, scratch, trainer skeleton)
│   ├── defect_full_ft.yaml       # Full fine-tuning (extends base, lower lr_scale)
│   ├── defect_decoder_only.yaml  # Decoder-only (extends base, backbone frozen)
│   └── defect_eval.yaml          # Eval-only mode (trainer.mode: val)
```

### Base config template (`defect_base.yaml`)

```yaml
# @package _global_
defaults:
  - _self_

paths:
  defect_root: /path/to/defect/dataset
  experiment_log_dir: /path/to/logs
  bpe_path: /path/to/sam3/assets/bpe_simple_vocab_16e6.txt.gz

scratch:
  enable_segmentation: True       # CRITICAL: set True for instance masks
  resolution: 1008
  train_batch_size: 1
  gradient_accumulation_steps: 4
  num_train_workers: 4
  num_val_workers: 0
  max_data_epochs: 40
  target_epoch_size: 500          # ~= dataset size
  consistent_transform: False
  max_ann_per_img: 200
  train_norm_mean: [0.5, 0.5, 0.5]   # Match SAM3 pretraining stats
  train_norm_std: [0.5, 0.5, 0.5]
  val_norm_mean: [0.5, 0.5, 0.5]
  val_norm_std: [0.5, 0.5, 0.5]
  lr_scale: 0.03
  lr_transformer: ${times:8e-4,${scratch.lr_scale}}
  lr_vision_backbone: ${times:2.5e-4,${scratch.lr_scale}}
  lr_language_backbone: ${times:5e-5,${scratch.lr_scale}}
  lrd_vision_backbone: 0.9
  wd: 0.1
  scheduler_timescale: 20
  scheduler_warmup: 10
  scheduler_cooldown: 20

trainer:
  _target_: sam3.train.trainer.Trainer
  skip_saving_ckpts: false
  empty_gpu_mem_cache_after_eval: True
  skip_first_val: False
  max_epochs: ${scratch.max_data_epochs}
  accelerator: cuda
  seed_value: 42
  val_epoch_freq: 5             # Validate every 5 epochs
  mode: train
  gradient_accumulation_steps: ${scratch.gradient_accumulation_steps}

  distributed:
    backend: nccl
    find_unused_parameters: True
    gradient_as_bucket_view: True

  checkpoint:
    save_dir: ${launcher.experiment_log_dir}/checkpoints
    save_freq: 5
    save_best_meters:
      - "val_defect/detection/AP50"  # Save best checkpoint by AP50

  # ... data, model, optim, loss, meters (see roboflow config for complete example)
```

### Segmentation loss config (critical — omitted in bbox-only Roboflow example)

```yaml
# Enable this in scratch and in the loss block when using instance masks:
scratch:
  enable_segmentation: True

# Loss block must include Masks loss:
loss_fns_find:
  - _target_: sam3.train.loss.loss_fns.Boxes
    weight_dict:
      loss_bbox: 5.0
      loss_giou: 2.0
  - _target_: sam3.train.loss.loss_fns.IABCEMdetr
    weak_loss: False
    weight_dict:
      loss_ce: 20.0
      presence_loss: 20.0
    pos_weight: 10.0
    alpha: 0.25
    gamma: 2
    use_presence: True
    pos_focal: false
    pad_n_queries: 200
    pad_scale_pos: 1.0
  - _target_: sam3.train.loss.loss_fns.Masks
    focal_alpha: 0.25
    focal_gamma: 2.0
    weight_dict:
      loss_mask: 200.0
      loss_dice: 10.0
    compute_aux: false
```

### Config override via CLI (Hydra feature — no file editing needed)

```bash
# Override lr_scale for a quick experiment:
python sam3/train/train.py -c configs/defect_finetune/defect_full_ft \
  --use-cluster 0 --num-gpus 4 \
  scratch.lr_scale=0.05 \
  scratch.max_data_epochs=60

# Override to decoder-only mode:
python sam3/train/train.py -c configs/defect_finetune/defect_full_ft \
  --use-cluster 0 --num-gpus 4 \
  scratch.lr_scale=0.1 \
  trainer.checkpoint.skip_saving_parameters="['backbone.vision_backbone.*','backbone.language_backbone.*']"
```

---

## Evaluation Metrics

### Standard COCO metrics (used by SAM3's `CocoEvaluatorOfflineWithPredFileEvaluators`)

The eval pipeline runs `pycocotools` evaluation with both `"bbox"` and `"segm"` iou_types. Logged keys:

| Metric Key | Description | Priority |
|---|---|---|
| `coco_eval_segm_AP` | Mask AP @ IoU 0.5:0.95 (primary segmentation metric) | **Primary** |
| `coco_eval_segm_AP50` | Mask AP @ IoU 0.50 | **Primary** |
| `coco_eval_segm_AP75` | Mask AP @ IoU 0.75 | Secondary |
| `coco_eval_bbox_AP` | Box AP @ IoU 0.5:0.95 | Monitor |
| `coco_eval_bbox_AP50` | Box AP @ IoU 0.50 | Monitor |
| `coco_eval_segm_APs` | AP for small instances (<32² px) | Defect-relevant |
| `coco_eval_segm_APm` | AP for medium instances | Defect-relevant |
| `coco_eval_segm_APl` | AP for large instances | Defect-relevant |

**For industrial defect segmentation, prioritize:**
1. **`coco_eval_segm_AP50`** — lenient IoU threshold, useful when defect boundaries are ambiguous
2. **`coco_eval_segm_APs`** — small defects are the hard case
3. **`coco_eval_segm_AP`** (standard average) — overall quality

### Enabling segmentation evaluation in config

```yaml
meters:
  val:
    defect:
      detection:
        _target_: sam3.eval.coco_writer.PredictionDumper
        iou_type: "segm"           # Use "segm" not "bbox" for mask eval
        dump_dir: ${launcher.experiment_log_dir}/dumps/defect
        merge_predictions: True
        postprocessor: ${scratch.original_box_postprocessor}
        maxdets: 100
        pred_file_evaluators:
          - _target_: sam3.eval.coco_eval_offline.CocoEvaluatorOfflineWithPredFileEvaluators
            gt_path: /path/to/defect/test/_annotations.coco.json
            tide: False
            iou_type: "segm"
```

### Additional metrics for defect detection

- **Per-defect-class AP** — COCO evaluator computes per-category stats. Log these to identify which defect types are hardest.
- **cgF1 / class-agnostic F1** — SAM3 has a `cgf1_eval.py` evaluator that ignores category labels. Useful when your CVAT annotation categories may not perfectly align with model query categories.
- **Mean IoU (mIoU) per instance** — complementary to mAP, more intuitive for segmentation quality.

---

## Gotchas & Pitfalls

### 1. `enable_segmentation: False` by default — masks silently ignored

**What goes wrong:** The Roboflow example config and the base trainer both default `enable_segmentation: False`. This means mask losses are disabled and masks are NOT loaded, even if your COCO JSON contains them. You get bbox-only fine-tuning.

**Prevention:** Set `enable_segmentation: True` in `scratch`, AND add the `Masks` loss component, AND set `with_seg_masks: True` in the collator (`collate_fn` and `collate_fn_val`).

**Detection:** Model trains/evals without errors but `coco_eval_segm_*` metrics will be absent or wrong.

### 2. Normalization mismatch — SAM3 uses [0.5, 0.5, 0.5] NOT ImageNet stats

**What goes wrong:** If you inherit a transform from standard torchvision examples or SAM (original), you might use `mean=[0.485, 0.456, 0.406]` (ImageNet). SAM3's pretrained weights expect `[0.5, 0.5, 0.5]`.

**Prevention:** Always use `train_norm_mean: [0.5, 0.5, 0.5]` and `train_norm_std: [0.5, 0.5, 0.5]` as defined in the official configs.

### 3. `find_unused_parameters: True` required for partial training

**What goes wrong:** When freezing the backbone or training decoder-only, frozen parameters create DDP errors unless `find_unused_parameters: True`. However, this adds ~15% communication overhead.

**Prevention:** Keep `distributed.find_unused_parameters: True` any time you have frozen components. If all parameters train (Option B), you can set to `False` for slight speedup.

### 4. Hungarian matcher instability with tiny batch sizes

**What goes wrong:** With effective batch size < 4, the Hungarian matcher receives very few positive examples per step, causing high gradient variance. Loss may oscillate or diverge in early epochs (especially focal classification loss with `loss_ce: 20.0`).

**Prevention:** Use `gradient_accumulation_steps: 4–8` to increase effective batch size. If loss still diverges, reduce `loss_ce` weight from 20.0 to 5.0–10.0, or reduce `pos_weight` from 10.0 to 5.0.

**Detection:** Watch for `core_loss` spiking >100× baseline within first 50 steps. The ODinW docs note: *"a small number of jobs may diverge during training, in which case we just use the last checkpoint's result before it diverges."*

### 5. Overfitting with <200 images

**What goes wrong:** With insufficient augmentation, the model memorizes training images. Val AP climbs then crashes while train loss keeps decreasing.

**Prevention:**
- Use `val_epoch_freq: 2–5` to monitor val AP frequently
- Enable `save_best_meters` with val AP to save best checkpoint, not last
- Use `ColorJitter` + `GaussianBlur` augmentation
- Increase `wd: 0.1` (already high, don't reduce)
- Use `dropout: 0.1` (already default in encoder/decoder layers)

### 6. `target_epoch_size` and small dataset interaction

**What goes wrong:** Default `target_epoch_size: 1500` with 500 images means each "epoch" samples each image 3× randomly. With 100 images it's 15× per "epoch" — the number of "epochs" in config no longer reflects passes over data intuitively.

**Prevention:** Set `target_epoch_size` to approximately your dataset size. This makes `max_data_epochs: 40` mean "40 full passes" rather than "40 × 3 passes over the data."

### 7. `skip_saving_ckpts: true` in example config

**What goes wrong:** The Roboflow example sets `skip_saving_ckpts: true` (designed for SLURM job arrays where hundreds of models run in parallel). For a development loop, this means no checkpoint is ever saved — retraining from scratch on every run.

**Prevention:** Set `skip_saving_ckpts: false` in your custom config.

### 8. COCO JSON category IDs must be contiguous starting from 1

**What goes wrong:** CVAT exports may produce non-contiguous category IDs (e.g., IDs 1, 5, 12). The `coco_reindex.reindex_coco_to_temp` utility in `sam3/eval/coco_reindex.py` handles this for eval, but training data also needs clean IDs.

**Prevention:** Verify your COCO JSON's category IDs are contiguous (1, 2, 3, …N) before use. If not, use the reindex utility or fix in post-processing.

### 9. Multi-GPU memory spikes during validation

**What goes wrong:** Default `val_batch_size: 1` and `num_val_workers: 0` are intentional — validation processes many queries per image and can OOM if workers pre-fetch aggressively.

**Prevention:** Keep `num_val_workers: 0`. If OOM during val, reduce `category_chunk_size` in `COCO_FROM_JSON` (default 2).

### 10. `consistent_transform: False` is correct for training

**What goes wrong:** Setting `consistent_transform: True` applies the same spatial transform to all frames in a clip — correct for video but wrong for single-image training. It can cause issues where the random crop is applied identically across augmentation variants.

**Prevention:** Keep `consistent_transform: False` for image fine-tuning.

---

## References

### Official SAM3 code (primary source, HIGH confidence)
- `sam3/train/configs/roboflow_v100/roboflow_v100_full_ft_100_images.yaml` — Production 100-image fine-tuning config
- `sam3/train/configs/odinw13/odinw_text_only_train.yaml` — 10-shot few-shot training config
- `sam3/train/optim/schedulers.py` — `InverseSquareRootParamScheduler` implementation
- `sam3/train/optim/optimizer.py` — `layer_decay_param_modifier`, `construct_optimizer`
- `sam3/train/transforms/basic.py` — All geometry-preserving augmentations
- `sam3/train/trainer.py` — Full training loop, checkpoint, DDP setup
- `sam3/eval/coco_eval.py` — COCO metric computation

### Official SAM3 documentation (HIGH confidence)
- `README_TRAIN.md` — Training setup, local/cluster execution, job arrays
- `RELEASE_SAM3p1.md` — Release notes (model capabilities)

### Related work (MEDIUM confidence — training data knowledge, unverified against latest)
- SAM2 fine-tuning guide (Meta): https://github.com/facebookresearch/sam2/tree/main/training
  - Uses identical training infrastructure; SAM3 configs are direct descendents
- "Segment Anything in Medical Images" (MedSAM) — demonstrates decoder-only fine-tuning strategy for domain-shifted small datasets
- OdinW benchmark (GLIP) — 10-shot fine-tuning setup closely mirrors SAM3's `odinw_text_only_train.yaml`
- Roboflow-100-VL — 100-image benchmark that validates SAM3's small-data fine-tuning approach works

### Key hyperparameter grounding

All hyperparameters documented in "Recommended Hyperparameters" are directly read from production SAM3 configs in this repository. Modifications (reducing `lr_scale`, adjusting `target_epoch_size`, enabling `enable_segmentation`) are logical extensions following the same config patterns.
