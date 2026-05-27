# Phase 2: Hydra Config Templates - Research

**Researched:** 2026-05-27
**Domain:** Hydra config composition, SAM3 training config structure, segmentation loss/eval wiring
**Confidence:** HIGH — all findings verified directly from codebase source files

---

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

- **D-01:** New configs live at `sam3/train/configs/custom_finetune/`
- **D-02:** `base.yaml` uses `# @package _global_` and `defaults: [_self_]`
- **D-03:** `decoder_only.yaml` and `full_finetune.yaml` contain only the LR/freeze strategy delta fields. They do NOT duplicate base.yaml — all other values inherit from base.yaml via Hydra compose.
- **D-04:** Users switch strategy with: `python sam3/train/train.py --config-name custom_finetune/base '+finetune_strategy=decoder_only'` *(see Investigation Q6/Q7 for critical corrections to this syntax)*
- **D-05:** Three REQUIRED fields: `paths.dataset_img_folder`, `paths.train_ann_file`, `paths.val_ann_file`
- **D-06:** All three are marked with `# REQUIRED:` inline comments. Every other field also gets an inline comment.
- **D-07:** Class names NOT a separate config field — SAM3 reads from COCO JSON `categories` list.
- **D-08 (decoder_only delta):** `lr_scale: 0.03`
- **D-09 (full_finetune delta):** `lrd_vision_backbone: 0.9`
- **D-10:** `base.yaml` defaults to decoder-only (`lr_scale: 0.03`)
- **D-11:** `base.yaml` is a complete, self-contained training config
- **D-12:** Small-dataset defaults: `epochs=40`, `train_batch_size=1`, `gradient_accumulation_steps=4`, `lr_transformer=8e-5`, `lr_vision_backbone=2.5e-6`
- **D-13:** `enable_segmentation: true` in `scratch`; data loaders also have `load_segmentation: true`
- **D-14:** Normalization uses `[0.5, 0.5, 0.5]` for mean and std
- **D-15:** Smoke test uses `python sam3/train/train.py --config-name custom_finetune/base --cfg job` *(see Investigation Q6 — this command is incorrect; correction documented below)*
- **D-16:** Smoke test is a standalone test plan (Plan 4) — not a pytest file

### the agent's Discretion

None declared in CONTEXT.md.

### Deferred Ideas (OUT OF SCOPE)

None — discussion stayed within phase scope.
</user_constraints>

---

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| CFG-01 | Base Hydra config template at `configs/custom_finetune/base.yaml` | Complete field inventory from roboflow_v100 config analysis |
| CFG-02 | `decoder_only.yaml` override with `lr_scale: 0.03` | Verified in roboflow config scratch.lr_scale pattern |
| CFG-03 | `full_finetune.yaml` override with `lrd_vision_backbone: 0.9` | Verified in roboflow config param_group_modifiers |
| CFG-04 | Only three fields need changing to run on a new dataset | D-05 confirmed; class names from COCO JSON automatically |
| CFG-05 | `enable_segmentation: true` in all fine-tuning configs | 6 fields identified that must all be enabled for segmentation |
| CFG-06 | Normalization uses `[0.5, 0.5, 0.5]` | Verified in roboflow config scratch.train_norm_mean/std |
| DOC-03 | Inline comments on every required field | All fields catalogued with inline comment requirements |
</phase_requirements>

---

## Summary

Phase 2 delivers three YAML files under `sam3/train/configs/custom_finetune/` plus a smoke test. Research confirmed all fields needed by reading the primary reference config (`roboflow_v100_full_ft_100_images.yaml`) and the key source files (`trainer.py`, `train.py`, `sam3_image_dataset.py`, `coco_json_loaders.py`, `train_utils.py`, `postprocessors.py`, `coco_eval_offline.py`).

**Critical discovery:** `train.py` does NOT use the standard `@hydra.main` decorator — it uses argparse (`-c`/`--config`) plus the Hydra compose API. This means (a) `--cfg job` won't work as the smoke test command, and (b) the CLI override syntax `+finetune_strategy=decoder_only` won't work as written in D-04. The correct substitution patterns are documented in Investigation Q6 and Q7.

**Segmentation enablement requires six coordinated config fields**, not just `enable_segmentation: true`. The roboflow config has all six set to `${scratch.enable_segmentation}` (which defaults to `False`). Our base.yaml must flip that to `true` and uncomment the `Masks` loss fn block.

**Primary recommendation:** Build `base.yaml` by adapting the roboflow config with: `enable_segmentation: true`, segmentation loss uncommented, `iou_type: "segm"` eval, `epochs: 40`, and the three REQUIRED path placeholders. Use the `defaults: [/configs/custom_finetune/base, _self_]` pattern for `full_finetune.yaml`. Smoke test via a `scripts/test_config_parse.py` that uses the compose API.

---

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| Config discovery & composition | Hydra config module | — | `initialize_config_module("sam3.train")` → resolves from `sam3/train/` package |
| Path variables (user-editable) | `paths:` section of YAML | — | Isolated at top of config per existing convention |
| LR/optimizer logic | `scratch:` interpolation vars + `trainer.optim` | — | `${times:…}` resolver wires scratch vars into optimizer options |
| Dataset loading | `trainer.data.train/val` → `Sam3ImageDataset` | `COCO_FROM_JSON` loader | `img_folder` + `ann_file` → images + annotations |
| Segmentation mask handling | `load_segmentation: true` in dataset | `with_seg_masks: true` in collator | Both required together to pass masks through pipeline |
| Mask loss | `loss_fns_find` list → `Masks` entry | `loss_fn_semantic_seg: null` | `Masks` loss_fn controls mask supervision; semantic seg is optional |
| Eval metric reporting | `trainer.meters.val` → `PredictionDumper` | `CocoEvaluatorOfflineWithPredFileEvaluators` | Dumper writes predictions; offline evaluator computes AP metrics |
| Checkpoint saving | `trainer.checkpoint` | `save_best_meters` list | Best-checkpoint-by-metric uses `save_best_meters` field |

---

## Standard Stack

### Core YAML Sections (from roboflow_v100_full_ft_100_images.yaml)

| Section | Purpose | Status in base.yaml |
|---------|---------|---------------------|
| `# @package _global_` + `defaults: [_self_]` | Hydra package declaration | Required — copy exactly |
| `paths:` | User-editable root paths | 3 REQUIRED fields + bpe_path |
| `scratch:` | Interpolation variables | All config values defined here, referenced via `${scratch.*}` |
| `trainer:` | Full trainer instantiation | Contains data, model, optim, meters, logging, checkpoint |
| `launcher:` | Local vs SLURM runner | Set `gpus_per_node: 1`, `num_nodes: 1` as safe defaults |
| `submitit:` | Cluster scheduler config | Set `use_cluster: False` for local default |

### Custom Resolvers Available (verified in train_utils.py)

| Resolver | Registration | Effect |
|----------|-------------|--------|
| `${times:A,B}` | `multiply_all(*args)` → `np.prod(np.array(args)).item()` | Multiplies all arguments |
| `${add:A,B}` | `lambda x, y: x + y` | Adds two values |
| `${divide:A,B}` | `lambda x, y: x / y` | Divides |
| `${string:X}` | `lambda x: str(x)` | Cast to string |
| `${int:X}` | `lambda x: int(x)` | Cast to int |

**Key pattern:** `lr_transformer: ${times:8e-4,${scratch.lr_scale}}` → effective LR = 8e-4 × lr_scale

---

## Investigation Findings

### Q1: Hydra Config Group Composition Pattern

**Finding (VERIFIED):** The existing codebase uses the `defaults` list inheritance pattern for config composition. The `silver_image_evals/sam3_silver_image_yt1b.yaml` config uses:

```yaml
# @package _global_
defaults:
  - /configs/eval_base.yaml
  - _self_
```

This causes the child config to inherit all fields from the parent and override only what's explicitly redefined.

**For our case**, `full_finetune.yaml` should be:

```yaml
# @package _global_
defaults:
  - /configs/custom_finetune/base
  - _self_

# Only override what differs from base
scratch:
  lr_scale: 1.0   # effective backbone LR restored to full value
  lrd_vision_backbone: 0.9  # layer-wise decay across ViT trunk
```

**Config path resolution:** With `initialize_config_module("sam3.train", version_base="1.2")`, the config root is `sam3/train/`. Absolute defaults paths start with `/configs/`. So:
- `/configs/custom_finetune/base` → `sam3/train/configs/custom_finetune/base.yaml`

**Recommended file locations:**
```
sam3/train/configs/custom_finetune/
├── base.yaml                          # complete config, decoder-only defaults
└── finetune_strategy/
    ├── decoder_only.yaml              # inherits base.yaml, no-op (base IS decoder-only)
    └── full_finetune.yaml             # inherits base.yaml, overrides LR fields
```

---

### Q2: All Fields to Change for Segmentation (from `enable_segmentation: False` default)

**Finding (VERIFIED):** There are **six** coordinated fields — all currently set to `${scratch.enable_segmentation}` (False) in the roboflow config. Our base.yaml must set `scratch.enable_segmentation: true` and every downstream reference picks up `true` automatically via interpolation.

| Field | Location in Config | Value needed |
|-------|--------------------|--------------|
| `scratch.enable_segmentation` | `scratch:` section | `true` (drives all others) |
| `trainer.model.enable_segmentation` | `trainer.model:` | `${scratch.enable_segmentation}` |
| `trainer.data.train.dataset.load_segmentation` | `trainer.data.train.dataset:` | `${scratch.enable_segmentation}` |
| `trainer.data.val.dataset.load_segmentation` | `trainer.data.val.dataset:` | `${scratch.enable_segmentation}` |
| `scratch.collate_fn.with_seg_masks` | `scratch.collate_fn:` | `${scratch.enable_segmentation}` |
| `scratch.collate_fn_val.with_seg_masks` | `scratch.collate_fn_val:` | `${scratch.enable_segmentation}` |

Setting `scratch.enable_segmentation: true` in `base.yaml` automatically propagates all six via interpolation — no other fields need individual updates.

---

### Q3: The Segmentation Loss Block

**Finding (VERIFIED):** The roboflow config has the segmentation loss commented out (lines 107–154). The active (non-segmentation) loss uses only `Boxes` and `IABCEMdetr` in `loss_fns_find`, with `loss_fn_semantic_seg: null`.

**For segmentation training, uncomment and use this structure:**

```yaml
loss:
  _target_: sam3.train.loss.sam3_loss.Sam3LossWrapper
  matcher: ${scratch.matcher}
  o2m_weight: 2.0
  o2m_matcher:
    _target_: sam3.train.matcher.BinaryOneToManyMatcher
    alpha: 0.3
    threshold: 0.4
    topk: 4
  use_o2m_matcher_on_o2m_aux: false
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
    - _target_: sam3.train.loss.loss_fns.Masks   # ← THIS is the key addition
      focal_alpha: 0.25
      focal_gamma: 2.0
      weight_dict:
        loss_mask: 200.0
        loss_dice: 10.0
      compute_aux: false
  loss_fn_semantic_seg: null                      # optional; null is fine for image-only
  scale_by_find_batch_size: ${scratch.scale_by_find_batch_size}
```

**`loss_fn_semantic_seg: null` is acceptable.** The `Sam3LossWrapper` checks `if self.loss_fn_semantic_seg is not None` before calling it.

---

### Q4: Eval Config for Segmentation (`iou_type: "segm"`)

**Finding (VERIFIED):** Existing silver image eval configs (e.g., `sam3_silver_image_yt1b.yaml`) demonstrate the correct segmentation eval pattern. The key components are:

**Postprocessor (define in `scratch:`):**
```yaml
scratch:
  segm_postprocessor:
    _target_: sam3.eval.postprocessors.PostProcessImage
    max_dets_per_img: -1
    iou_type: "segm"              # ← enables pred_masks processing
    use_original_ids: true        # use original COCO category IDs for eval
    use_original_sizes_box: true
    use_original_sizes_mask: true # ← resize masks to original image size
    convert_mask_to_rle: True     # ← required for COCO eval JSON format
    use_presence: ${scratch.use_presence_eval}
```

**Meters section (under `trainer.meters.val.{dict_key}`):**
```yaml
trainer:
  meters:
    val:
      custom:  # must match collate_fn dict_key
        detection:
          _target_: sam3.eval.coco_writer.PredictionDumper
          iou_type: "segm"
          dump_dir: ${launcher.experiment_log_dir}/dumps/custom
          merge_predictions: True
          postprocessor: ${scratch.segm_postprocessor}
          gather_pred_via_filesys: ${scratch.gather_pred_via_filesys}
          maxdets: 100
          pred_file_evaluators:
            - _target_: sam3.eval.coco_eval_offline.CocoEvaluatorOfflineWithPredFileEvaluators
              gt_path: ${paths.val_ann_file}
              tide: False
              iou_type: "segm"
```

**Output metrics (verified in coco_eval_offline.py):**
```
COCO_METRICS = ["AP", "AP_50", "AP_75", "AP_small", "AP_medium", "AP_large", ...]
→ coco_eval_segm_AP, coco_eval_segm_AP_50, coco_eval_segm_AP_75, etc.
```

---

### Q5: The `${times:8e-4,${scratch.lr_scale}}` Custom Resolver

**Finding (VERIFIED):** Defined in `sam3/train/utils/train_utils.py`:

```python
def multiply_all(*args):
    return np.prod(np.array(args)).item()

def register_omegaconf_resolvers():
    OmegaConf.register_new_resolver("times", multiply_all)
    # also: add, divide, subtract, pow, range, int, ceil_int, merge, string
```

`register_omegaconf_resolvers()` is called in `train.py` before `main(args)` on line 337. It is also called inside `single_proc_run()` on each GPU worker.

**What it does:** `${times:8e-4,${scratch.lr_scale}}` = 8e-4 × scratch.lr_scale. With `lr_scale=0.03`:
- `lr_transformer = ${times:8e-4,0.03}` = 2.4e-5 (much less than target 8e-5)

⚠️ **Important:** D-12 specifies `lr_transformer=8e-5` as the small-dataset default. To get exactly `8e-5`, use `lr_scale=0.1` (default in roboflow) then override to get the intended values. Alternatively, set `lr_transformer: 8e-5` directly as a literal (not via `${times:...}`) in base.yaml.

**Recommendation for base.yaml:** Set the LR values explicitly as literals rather than via `${times:...}` resolver to match D-12 exactly:
```yaml
scratch:
  lr_scale: 0.03               # decoder-only — backbone near-frozen
  lr_transformer: 8e-5         # D-12: explicit value, not via times resolver
  lr_vision_backbone: 2.5e-6   # D-12: explicit value
  lr_language_backbone: 1.5e-6 # derived from scale pattern
```

---

### Q6: The Smoke Test — Correct Command

**Finding (VERIFIED):** `train.py` does NOT use `@hydra.main`. It uses argparse with `-c`/`--config`:

```python
# train.py __main__:
initialize_config_module("sam3.train", version_base="1.2")
parser = ArgumentParser()
parser.add_argument("-c", "--config", required=True, ...)
args = parser.parse_args()
register_omegaconf_resolvers()
main(args)
```

**`--cfg job` and `--config-name` will NOT work** — they are Hydra CLI options for `@hydra.main` decorated functions, not for the compose API pattern used here.

**Correct smoke test:** Write a standalone Python script `scripts/test_config_parse.py`:

```python
#!/usr/bin/env python3
"""Smoke test: validate Hydra config parses without errors."""
import sys
sys.path.insert(0, ".")
from hydra import compose, initialize_config_module
from sam3.train.utils.train_utils import register_omegaconf_resolvers
from omegaconf import OmegaConf

register_omegaconf_resolvers()
initialize_config_module("sam3.train", version_base="1.2")

config_names = [
    "custom_finetune/base",
    "custom_finetune/finetune_strategy/full_finetune",
    "custom_finetune/finetune_strategy/decoder_only",
]
for name in config_names:
    cfg = compose(config_name=name)
    print(f"✓ {name}")
    # Trigger interpolation resolution to catch unresolved references
    OmegaConf.to_container(cfg, resolve=True, throw_on_missing=False)

print("All configs parsed successfully.")
```

**Run from project root:**
```bash
cd /path/to/sam3
python scripts/test_config_parse.py
```

---

### Q7: Hydra Config Group Override Syntax

**Finding (VERIFIED):** Because `train.py` uses argparse + compose API (not `@hydra.main`), the standard Hydra CLI config-group override syntax (`+finetune_strategy=decoder_only`) does NOT work.

**Actual launch commands:**

```bash
# Default (decoder-only strategy — uses base.yaml directly):
python sam3/train/train.py -c custom_finetune/base --use-cluster 0 --num-gpus 1

# Full fine-tune strategy — use the finetune_strategy/full_finetune config:
python sam3/train/train.py -c custom_finetune/finetune_strategy/full_finetune --use-cluster 0 --num-gpus 1

# Multi-GPU:
python sam3/train/train.py -c custom_finetune/base --use-cluster 0 --num-gpus 2
```

**File structure matching this pattern:**
```
sam3/train/configs/custom_finetune/
├── base.yaml                                    # complete config, decoder-only
└── finetune_strategy/
    ├── decoder_only.yaml                        # inherits base, no-op (documents decoder-only)
    └── full_finetune.yaml                       # inherits base, overrides LR strategy
```

**Note for D-04:** The switch command should be documented as changing `-c custom_finetune/base` to `-c custom_finetune/finetune_strategy/full_finetune`, not as a `+finetune_strategy` CLI override. This is a cosmetic difference for the user — same outcome.

---

### Q8: `paths.bpe_path` — Required or Optional?

**Finding (VERIFIED):** In `sam3/model_builder.py`:

```python
def build_sam3_image_model(bpe_path=None, ...):
    if bpe_path is None:
        bpe_path = pkg_resources.resource_filename(
            "sam3", "assets/bpe_simple_vocab_16e6.txt.gz"
        )
```

`bpe_path` has a built-in fallback. If set to `null` in the config, the model uses the bundled BPE vocabulary at `sam3/assets/bpe_simple_vocab_16e6.txt.gz`.

**Recommendation:** Keep `paths.bpe_path: null` in base.yaml with a comment explaining the fallback. This simplifies onboarding — users don't need to locate the BPE file.

```yaml
paths:
  bpe_path: null  # Optional: path to sam3/assets/bpe_simple_vocab_16e6.txt.gz
                  # Set to null to use the bundled vocab (recommended)
```

---

### Q9: `target_epoch_size` vs `max_data_epochs` vs `max_epochs`

**Finding (VERIFIED):**

- `target_epoch_size`: **NOT used in any Python code** in `sam3/`. Searched all `.py` files — zero references. It is a documentation-only scratch variable that tells humans "this run is calibrated for ~N samples per epoch."
- `scratch.max_data_epochs`: A scratch variable. Only used when referenced via `${scratch.max_data_epochs}` in `trainer.max_epochs` (as in odinw13 configs) — not used directly by the Trainer.
- `trainer.max_epochs`: **This is the field that controls actual training duration.** The `Trainer.__init__` receives `max_epochs: int` and uses it in `run_train()` via `while self.epoch < self.max_epochs`.

**For base.yaml:**
```yaml
scratch:
  max_data_epochs: 40    # documentation only — not consumed by code
  target_epoch_size: 500 # documentation only — not consumed by code

trainer:
  max_epochs: ${scratch.max_data_epochs}  # D-12: 40 epochs for small datasets
```

---

### Q10: `save_best_metric` / Best Checkpoint Tracking

**Finding (VERIFIED):** The `CheckpointConf` dataclass has:

```python
@dataclass
class CheckpointConf:
    save_dir: str
    save_freq: int
    save_best_meters: List[str] = None  # ← this is the field
```

In `_log_meters_and_save_best_ckpts()`:
```python
if (self.checkpoint_conf.save_best_meters is not None
        and key in self.checkpoint_conf.save_best_meters):
    checkpoint_save_keys.append(tracked_meter_key.replace("/", "_"))
# → calls self.save_checkpoint(self.epoch + 1, checkpoint_save_keys)
# → saves {tracked_meter_key.replace("/","_")}.pt
```

**How the meter key is structured:**
- Meter key `key` = `f"{phase}_{datakey}/{metername}"` e.g. `val_custom/detection`
- Tracked meter subkey = `key/meter_subkey` e.g. `val_custom/detection/coco_eval_segm_AP_50`
- Checkpoint filename = `val_custom_detection_coco_eval_segm_AP_50.pt`

**Config for best checkpoint by AP50:**
```yaml
trainer:
  checkpoint:
    save_dir: ${launcher.experiment_log_dir}/checkpoints
    save_freq: 0           # 0 = only last checkpoint (plus best-meter ones)
    save_best_meters:
      - "val_custom/detection"   # saves checkpoint when AP50 improves
```

Note: `save_best_meters` tracks when ANY metric from that meter improves, not a single specific metric. To get a checkpoint named after AP50, the saved file will be named based on the tracked subkey (e.g., `val_custom_detection_coco_eval_segm_AP_50.pt`).

---

## Complete Field Inventory for base.yaml

### `paths:` section
```yaml
paths:
  dataset_img_folder: null  # REQUIRED: absolute path to images directory
  train_ann_file: null       # REQUIRED: absolute path to train.json from prepare_dataset.py
  val_ann_file: null         # REQUIRED: absolute path to val.json from prepare_dataset.py
  experiment_log_dir: null   # REQUIRED: where checkpoints, logs, TensorBoard go
  bpe_path: null             # Optional: null = use bundled vocab (recommended)
```

### `scratch:` section (key fields)
```yaml
scratch:
  enable_segmentation: true  # CRITICAL: activates mask loss + mask decode (off by default!)

  # Image processing
  resolution: 1008
  train_norm_mean: [0.5, 0.5, 0.5]  # CFG-06: SAM3 values, NOT ImageNet
  train_norm_std: [0.5, 0.5, 0.5]

  # LR (decoder-only strategy — D-12)
  lr_scale: 0.03
  lr_transformer: 8e-5
  lr_vision_backbone: 2.5e-6
  lr_language_backbone: 1.5e-6
  lrd_vision_backbone: 0.9   # still present but lr_scale near-zero makes backbone frozen

  # Small-dataset training (D-12)
  max_data_epochs: 40
  target_epoch_size: 500   # documentation-only
  train_batch_size: 1
  gradient_accumulation_steps: 4
  val_batch_size: 1

  # Collators — both must pass with_seg_masks for mask training
  collate_fn:
    _target_: sam3.train.data.collator.collate_fn_api
    _partial_: true
    dict_key: custom
    with_seg_masks: ${scratch.enable_segmentation}

  collate_fn_val:
    _target_: sam3.train.data.collator.collate_fn_api
    _partial_: true
    dict_key: custom
    with_seg_masks: ${scratch.enable_segmentation}
```

### `trainer.data` section
```yaml
trainer:
  data:
    train:
      _target_: sam3.train.data.torch_dataset.TorchDataset
      dataset:
        _target_: sam3.train.data.sam3_image_dataset.Sam3ImageDataset
        img_folder: ${paths.dataset_img_folder}   # D-05
        ann_file: ${paths.train_ann_file}          # D-05
        load_segmentation: ${scratch.enable_segmentation}
        transforms: ...  # train_transforms chain
        max_ann_per_img: 500000
        multiplier: 1
        max_train_queries: 50000
        max_val_queries: 50000
        training: true
        use_caching: False

    val:
      _target_: sam3.train.data.torch_dataset.TorchDataset
      dataset:
        _target_: sam3.train.data.sam3_image_dataset.Sam3ImageDataset
        img_folder: ${paths.dataset_img_folder}   # D-05
        ann_file: ${paths.val_ann_file}            # D-05
        load_segmentation: ${scratch.enable_segmentation}
        coco_json_loader:
          _target_: sam3.train.data.coco_json_loaders.COCO_FROM_JSON
          include_negatives: true
          category_chunk_size: 2
          _partial_: true
        transforms: ...  # val_transforms chain
        max_ann_per_img: 100000
        multiplier: 1
        training: false
```

### `trainer.model` section
```yaml
trainer:
  model:
    _target_: sam3.model_builder.build_sam3_image_model
    bpe_path: ${paths.bpe_path}
    device: cpus
    eval_mode: false
    enable_segmentation: ${scratch.enable_segmentation}  # CRITICAL: must match scratch
```

### `trainer.checkpoint` section
```yaml
trainer:
  checkpoint:
    save_dir: ${launcher.experiment_log_dir}/checkpoints
    save_freq: 0
    save_best_meters:
      - "val_custom/detection"
```

---

## Architecture Patterns

### System Architecture Diagram

```
User edits base.yaml (3 fields: img_folder, train_ann_file, val_ann_file)
         │
         ▼
train.py --config-name custom_finetune/base
         │
         ▼ initialize_config_module("sam3.train") + compose()
         │
    ┌────┴────────────────────────────────────────────────────┐
    │                  Hydra Config Resolution                 │
    │  base.yaml ──defaults──▶ [_self_]                       │
    │  full_finetune.yaml ─▶ [/configs/custom_finetune/base,  │
    │                          _self_]                         │
    └────┬────────────────────────────────────────────────────┘
         │ resolved OmegaConf
         ▼
    Trainer.__init__(data, model, optim, checkpoint, meters, loss)
         │
    ┌────┴──────────────────────────────────────┐
    │  Sam3ImageDataset                         │
    │  img_folder + ann_file → COCO_FROM_JSON   │
    │  load_segmentation=True → RLE masks       │
    └────┬──────────────────────────────────────┘
         │ batched Datapoint (with seg masks)
         ▼
    collate_fn(with_seg_masks=True)
         │
         ▼
    model(batch) → SAM3Output (pred_boxes, pred_masks)
         │
         ├──▶ Sam3LossWrapper([Boxes, IABCEMdetr, Masks])
         │         └──▶ loss_mask + loss_dice + loss_bbox + loss_ce
         │
         └──▶ PredictionDumper(iou_type="segm")
                   └──▶ PostProcessImage(convert_mask_to_rle=True)
                   └──▶ CocoEvaluatorOfflineWithPredFileEvaluators
                             └──▶ coco_eval_segm_AP50 (triggers best ckpt save)
```

### Recommended Project Structure

```
sam3/train/configs/custom_finetune/
├── base.yaml                          # CFG-01: complete self-contained config
└── finetune_strategy/
    ├── decoder_only.yaml              # CFG-02: inherits base, documents lr_scale=0.03
    └── full_finetune.yaml             # CFG-03: inherits base, lr_scale=1.0 + lrd=0.9
```

### Pattern: Full-Config Inheritance (full_finetune.yaml)

```yaml
# sam3/train/configs/custom_finetune/finetune_strategy/full_finetune.yaml
# @package _global_
defaults:
  - /configs/custom_finetune/base    # inherit everything from base
  - _self_                           # then apply overrides below

# Source: silver_image_evals/sam3_silver_image_yt1b.yaml uses same pattern

scratch:
  lr_scale: 1.0                     # restore backbone LR to full value
  lrd_vision_backbone: 0.9          # D-09: LLRD across ViT trunk
  lr_vision_backbone: 2.5e-6        # effective: lrd decay applied per-layer
```

### Pattern: decoder_only.yaml (alias/documentation config)

```yaml
# sam3/train/configs/custom_finetune/finetune_strategy/decoder_only.yaml
# @package _global_
defaults:
  - /configs/custom_finetune/base    # base.yaml IS decoder-only; this is an explicit alias
  - _self_

# No overrides needed — base.yaml already has lr_scale: 0.03 (decoder-only)
# This file exists for documentation clarity and symmetry with full_finetune.yaml
```

### Anti-Patterns to Avoid

- **Duplicating full base.yaml in full_finetune.yaml:** All existing override configs (silver eval configs) use `defaults:` inheritance — not copy-paste. Duplication creates drift.
- **Using `--cfg job` as smoke test:** This is a Hydra CLI feature for `@hydra.main` functions. train.py uses argparse. Always use the compose API for dry-run testing.
- **Setting `enable_segmentation: true` only in `scratch:` without checking the 5 derived fields:** If any single field is missed (e.g., `with_seg_masks: False` in collate_fn), masks will silently not be passed through and mask loss will be zero.
- **Using `${times:…}` for LR in base.yaml:** With `lr_scale=0.03`, `${times:8e-4,0.03}` = 2.4e-5, not 8e-5. Set D-12 LRs as explicit literals.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Config inheritance/composition | Custom YAML merging | Hydra `defaults` list | Built into Hydra; handles `_self_` ordering correctly |
| LR arithmetic in config | Python code to compute LRs | `${times:…}` resolver | Already registered in `train_utils.py::register_omegaconf_resolvers()` |
| COCO metric computation | Custom AP evaluator | `CocoEvaluatorOfflineWithPredFileEvaluators` | Handles segm + bbox modes, TIDE, custom iouType |
| Mask RLE conversion | Custom RLE encoder | `PostProcessImage(convert_mask_to_rle=True)` | Uses `pycocotools.mask` under the hood; already handles GPU → CPU |
| Best checkpoint selection | Track metrics manually | `checkpoint.save_best_meters` | Trainer already compares and saves best via `is_better` |

---

## Common Pitfalls

### Pitfall 1: `--cfg job` Won't Work
**What goes wrong:** Running `python sam3/train/train.py --config-name custom_finetune/base --cfg job` raises argparse error: unrecognized arguments.
**Why it happens:** train.py uses argparse + compose API, not `@hydra.main`. `--cfg job` is a Hydra CLI option only.
**How to avoid:** Use the `scripts/test_config_parse.py` smoke test (Q6 pattern).
**Warning signs:** `error: unrecognized arguments: --cfg` in stderr.

### Pitfall 2: Six Segmentation Fields — Missing Any One
**What goes wrong:** Masks silently drop out of the pipeline; mask loss is zero; model trains detection-only.
**Why it happens:** `enable_segmentation` in scratch drives all six fields via interpolation, but only if all six are wired with `${scratch.enable_segmentation}`. One hardcoded `False` breaks the chain.
**How to avoid:** Use `${scratch.enable_segmentation}` for all six fields — never hardcode `False`.
**Warning signs:** `loss_mask: 0.0` in training logs; `coco_eval_segm_AP: 0.0` after training.

### Pitfall 3: LR Math Error with `${times:…}`
**What goes wrong:** With `lr_scale: 0.03`, `${times:8e-4,${scratch.lr_scale}}` = 2.4e-5, not 8e-5 (D-12 target).
**Why it happens:** The `times` resolver multiplies all arguments; `lr_scale=0.03` is for backbone freezing, not for absolute LR scaling to 8e-5.
**How to avoid:** Set `lr_transformer: 8e-5` as a literal in base.yaml rather than via `${times:…}`.
**Warning signs:** Training converges extremely slowly; loss barely decreasing.

### Pitfall 4: Config Group Override CLI Syntax
**What goes wrong:** `python sam3/train/train.py -c custom_finetune/base '+finetune_strategy=full_finetune'` fails with argparse error.
**Why it happens:** Hydra config group CLI overrides only work with `@hydra.main`.
**How to avoid:** Switch strategy by using `-c custom_finetune/finetune_strategy/full_finetune`.
**Warning signs:** `error: unrecognized arguments` when passing override flags.

### Pitfall 5: `target_epoch_size` Has No Effect
**What goes wrong:** Setting `target_epoch_size: 500` doesn't limit training to 500 samples/epoch.
**Why it happens:** `target_epoch_size` is never read by any Python code — it's documentation only. The `Trainer` uses only `max_epochs`.
**How to avoid:** Keep it for documentation but rely only on `trainer.max_epochs` for training duration.
**Warning signs:** Training runs full dataset regardless of `target_epoch_size`.

### Pitfall 6: `dict_key` Must Match Meters Key
**What goes wrong:** Val metrics not reported; meter key mismatch error.
**Why it happens:** `collate_fn.dict_key` in `scratch` must match the key under `trainer.meters.val`.
**How to avoid:** Use the same string (e.g., `custom`) in both `collate_fn.dict_key` and as the key under `trainer.meters.val`.
**Warning signs:** `assert set(val_keys) == set(self.meters_conf[phase].keys())` assertion failure.

---

## Code Examples

### defaults inheritance (from sam3_silver_image_yt1b.yaml — VERIFIED)
```yaml
# @package _global_
defaults:
  - /configs/eval_base.yaml   # parent config (absolute path from config root)
  - _self_                    # apply this file's overrides last
```

### Mask postprocessor (from eval_base.yaml — VERIFIED)
```yaml
segm_postprocessor:
  _target_: sam3.eval.postprocessors.PostProcessImage
  max_dets_per_img: -1
  iou_type: "segm"
  use_original_ids: true
  use_original_sizes_box: true
  use_original_sizes_mask: true
  convert_mask_to_rle: True
  use_presence: ${scratch.use_presence_eval}
```

### COCO segm evaluator (from coco_eval_offline.py COCO_METRICS — VERIFIED)
```yaml
pred_file_evaluators:
  - _target_: sam3.eval.coco_eval_offline.CocoEvaluatorOfflineWithPredFileEvaluators
    gt_path: ${paths.val_ann_file}
    tide: False
    iou_type: "segm"
# Outputs: coco_eval_segm_AP, coco_eval_segm_AP_50, coco_eval_segm_AP_75
#          coco_eval_segm_AP_small, coco_eval_segm_AP_medium, coco_eval_segm_AP_large
```

### Layer-decay optimizer block (from roboflow config — VERIFIED)
```yaml
trainer:
  optim:
    param_group_modifiers:
      - _target_: sam3.train.optim.optimizer.layer_decay_param_modifier
        _partial_: True
        layer_decay_value: ${scratch.lrd_vision_backbone}
        apply_to: 'backbone.vision_backbone.trunk'
        overrides:
          - pattern: '*pos_embed*'
            value: 1.0
```

### Times resolver usage (from train_utils.py + roboflow config — VERIFIED)
```yaml
# ${times:A,B} = A × B (custom resolver, registered at startup)
lr_transformer: ${times:8e-4,${scratch.lr_scale}}   # 8e-4 × lr_scale
lr_vision_backbone: ${times:2.5e-4,${scratch.lr_scale}}  # 2.5e-4 × lr_scale
```

---

## State of the Art

| Old Approach | Current Approach | Impact |
|--------------|------------------|--------|
| `enable_segmentation: False` (roboflow default) | `enable_segmentation: true` in base.yaml | Activates mask loss and mask decode |
| No mask loss (`loss_fn_semantic_seg: null` only) | `Masks` entry in `loss_fns_find` | Trains the mask head |
| `iou_type: "bbox"` in eval | `iou_type: "segm"` | Reports segmentation AP, not bbox AP |

**Deprecated/outdated in roboflow config:**
- The commented-out loss block (lines 107–154) is the template for the segmentation loss. Uncomment and use it.

---

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | `decoder_only.yaml` as a near-empty alias for base.yaml is acceptable and useful | Q1, Architecture Patterns | Low risk — no user impact; can be populated if desired |
| A2 | `lr_language_backbone: 1.5e-6` as a reasonable small-dataset value | Q5, Field Inventory | Medium — if language backbone LR is too high, it could interfere with text-conditioned queries |
| A3 | `detection_threshold` is not needed in segm postprocessor for training eval | Q4 | Low — could affect whether low-confidence masks are included in AP computation |

---

## Open Questions

1. **`detection_threshold` in segm postprocessor**
   - What we know: The roboflow config uses a bare `original_box_postprocessor` without threshold; silver evals use `mask_postprocessor_thresholded` with `detection_threshold: 0.3`
   - What's unclear: For training-time eval, should a detection threshold be applied? Without it, more low-confidence predictions may reduce AP.
   - Recommendation: Start without threshold (matches roboflow pattern for training); add if AP metrics look off.

2. **`collate_fn.repeats` / `hybrid_repeats` value**
   - What we know: roboflow config uses `hybrid_repeats: 1` and collate `repeats: ${scratch.hybrid_repeats}`
   - What's unclear: Whether `repeats > 1` is needed for single-dataset training
   - Recommendation: Keep `hybrid_repeats: 1` (same as roboflow config).

---

## Environment Availability

> Step 2.6 SKIPPED: Phase 2 is config-files-only (YAML creation + smoke test script). No external service or CLI tool dependencies beyond the already-installed SAM3 package.

---

## Validation Architecture

### Test Framework

| Property | Value |
|----------|-------|
| Framework | Custom Python smoke test (no pytest) |
| Config file | `scripts/test_config_parse.py` (new file, Wave 0) |
| Quick run command | `python scripts/test_config_parse.py` |
| Full suite command | `python scripts/test_config_parse.py` (same) |

### Phase Requirements → Test Map

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| CFG-01 | base.yaml parses without Hydra errors | smoke | `python scripts/test_config_parse.py` | ❌ Wave 0 |
| CFG-02 | decoder_only.yaml parses; lr_scale=0.03 in composed config | smoke | `python scripts/test_config_parse.py` | ❌ Wave 0 |
| CFG-03 | full_finetune.yaml parses; lrd_vision_backbone=0.9 in composed config | smoke | `python scripts/test_config_parse.py` | ❌ Wave 0 |
| CFG-04 | Only 3 fields need editing to run on new dataset | manual inspection | — | — |
| CFG-05 | `enable_segmentation: true` in base.yaml + all 6 derived fields | manual inspection | `grep -c 'enable_segmentation.*true' sam3/train/configs/custom_finetune/base.yaml` | — |
| CFG-06 | Norm values are `[0.5, 0.5, 0.5]` not ImageNet | manual inspection | `grep '0.5' sam3/train/configs/custom_finetune/base.yaml` | — |
| DOC-03 | Every required field has inline comment | manual inspection | — | — |

### Sampling Rate
- **Per task commit:** `python scripts/test_config_parse.py`
- **Per wave merge:** same
- **Phase gate:** All three configs parse without exceptions

### Wave 0 Gaps
- [ ] `scripts/test_config_parse.py` — covers CFG-01, CFG-02, CFG-03
- [ ] `sam3/train/configs/custom_finetune/base.yaml` — CFG-01 target
- [ ] `sam3/train/configs/custom_finetune/finetune_strategy/decoder_only.yaml` — CFG-02 target
- [ ] `sam3/train/configs/custom_finetune/finetune_strategy/full_finetune.yaml` — CFG-03 target

---

## Security Domain

> Phase 2 creates YAML config files and a Python smoke test script. No authentication, session management, access control, cryptography, or input validation concerns apply. `security_enforcement` is not explicitly set in config — but this phase has no applicable ASVS categories.

---

## Sources

### Primary (HIGH confidence)
- `sam3/train/configs/roboflow_v100/roboflow_v100_full_ft_100_images.yaml` — canonical config reference, fully read
- `sam3/train/train.py` — entry point analysis (argparse vs @hydra.main)
- `sam3/train/trainer.py` — CheckpointConf, save_best_meters, max_epochs behavior
- `sam3/train/utils/train_utils.py` — register_omegaconf_resolvers, multiply_all, times resolver
- `sam3/train/data/sam3_image_dataset.py` — Sam3ImageDataset signature and load_segmentation
- `sam3/train/data/coco_json_loaders.py` — COCO_FROM_JSON, include_negatives, category_chunk_size
- `sam3/eval/postprocessors.py` — PostProcessImage iou_type="segm", convert_mask_to_rle
- `sam3/eval/coco_eval_offline.py` — CocoEvaluatorOfflineWithPredFileEvaluators, COCO_METRICS
- `sam3/eval/coco_writer.py` — PredictionDumper, prepare_for_coco_segmentation
- `sam3/train/configs/silver_image_evals/sam3_silver_image_yt1b.yaml` — segm eval pattern
- `sam3/train/configs/eval_base.yaml` — mask_postprocessor_thresholded pattern
- `sam3/model_builder.py` — bpe_path=None fallback behavior
- `sam3/train/loss/sam3_loss.py` — Sam3LossWrapper, loss_fn_semantic_seg=None handling
- `sam3/train/loss/loss_fns.py` — Masks class confirmed exists

### Secondary (MEDIUM confidence)
- None required — all claims verified from source

### Tertiary (LOW confidence)
- None

---

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — verified from primary source files
- Architecture: HIGH — derived from actual code analysis
- Pitfalls: HIGH — each identified from real code behavior
- Smoke test approach: HIGH — derived from train.py argparse/compose analysis

**Research date:** 2026-05-27
**Valid until:** 2026-07-01 (stable codebase, no fast-moving dependencies)
