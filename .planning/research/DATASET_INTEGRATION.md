# Dataset Integration Research

## Summary

SAM3 consumes standard COCO-format JSON annotation files via the `COCO_FROM_JSON` loader in `sam3/train/data/coco_json_loaders.py`. The closest built-in fine-tuning analogue is the Roboflow 100-VL workflow (`roboflow_v100_full_ft_100_images.yaml`), which organises data as `<dataset_root>/<split>/` images plus `<split>/_annotations.coco.json`. CVAT's COCO export is structurally compatible with this pattern but has two common pitfalls: (1) it may use 0-based IDs that must be reindexed, and (2) its `file_name` field may include a directory prefix that must match the `img_folder` root. For a small industrial-defect dataset (< 500 images) the simplest integration is a manual random split, two separate JSON files, and a custom YAML config modelled on the Roboflow training config, with `load_segmentation: true` and the mask loss block uncommented.

---

## CVAT COCO Export Structure

### Directory Layout

When you export a CVAT task/project using **"COCO 1.0"** format, the zip contains:

```
<export>.zip
├── annotations/
│   └── instances_default.json   # single-file export (task-level)
│   # OR, for project-level with multiple tasks:
│   ├── instances_train.json
│   └── instances_val.json       # only if you pre-split in CVAT
└── images/
    ├── <image1>.jpg
    ├── <image2>.jpg
    └── ...
```

> **Note:** For a project-level export CVAT writes one JSON per subset (train/val/test) defined inside the CVAT UI. For a single task without subsets it writes `instances_default.json`.

### JSON Schema

```json
{
  "info":        { "year": 2024, "version": "1.0", "description": "...", ... },
  "licenses":    [],
  "categories": [
    { "id": 1, "name": "scratch",  "supercategory": "" },
    { "id": 2, "name": "dent",     "supercategory": "" }
  ],
  "images": [
    {
      "id": 1,
      "file_name": "images/frame_0001.jpg",   // ← includes "images/" prefix
      "width": 1920,
      "height": 1080,
      "date_captured": "",
      "license": 0,
      "coco_url": "",
      "flickr_url": ""
    }
  ],
  "annotations": [
    {
      "id": 1,
      "image_id": 1,
      "category_id": 1,
      "segmentation": [[x1,y1, x2,y2, ...]],  // polygon(s)
      "area": 3456.0,
      "bbox": [x, y, w, h],                    // XYWH (absolute pixels)
      "iscrowd": 0,
      "attributes": { "occluded": false }
    }
  ]
}
```

### Key CVAT-specific behaviours

| Aspect | Default CVAT behaviour |
|--------|----------------------|
| `file_name` | Includes `"images/"` directory prefix → `"images/foo.jpg"` |
| ID indexing | **1-based** by default in recent CVAT (≥ 2.x), but older exports and some plugins may produce **0-based** IDs |
| Segmentation | Exported as **polygon lists** (not RLE) for instance segmentation tasks |
| `iscrowd` | Always `0` for manually drawn annotations |
| Supercategory | Empty string `""` unless you explicitly set it in CVAT |
| Attributes | Non-COCO `"attributes"` dict appended to each annotation; SAM3 ignores it |

---

## SAM3 COCO Loader Requirements

### `load_coco_and_group_by_image` (used internally by `COCO_FROM_JSON`)

Reads three top-level keys from the JSON:

| Key | Required fields |
|-----|----------------|
| `images` | `id`, `file_name`, `width`, `height` |
| `annotations` | `id`, `image_id`, `category_id`, `bbox` (XYWH absolute), `segmentation`, `iscrowd`, `area` |
| `categories` | `id`, `name` |

### `COCO_FROM_JSON.__init__` parameters

```python
COCO_FROM_JSON(
    annotation_file,          # path to JSON — required
    prompts=None,             # optional: override category names with custom text prompts
    include_negatives=True,   # include images that have zero instances of a category
    category_chunk_size=None, # split categories into chunks (GPU memory tuning)
)
```

### How `file_name` is resolved

In `CustomCocoDetectionAPI._load_images`:

```python
path = os.path.join(self.root, current_meta["file_name"])
```

`self.root` = `img_folder` from the Hydra config.

So if `img_folder = "/data/defects/train"` and `file_name = "images/frame_001.jpg"`, the loader
will look for `/data/defects/train/images/frame_001.jpg`.

There is also a `fix_fname=True` option that strips everything before the last `/`, effectively
using only the bare filename.

### Segmentation handling

- The loader converts polygons → RLE on the fly via `ann_to_rle()` (calls `pycocotools.mask.frPyObjects`).
- Training with mask loss requires `load_segmentation: true` in the dataset config AND the mask loss block enabled in the YAML.
- If `segmentation` is `None` or `[]`, the annotation is loaded as box-only (no mask loss contribution).

### ID indexing requirement

SAM3 does **not** require 1-based IDs per se, but `coco_reindex.py` (`reindex_coco_to_temp`) is
available and automatically upgrades any 0-based IDs to 1-based. It is used in the ODinW training
config. You should apply it as a pre-processing step or via the Hydra `ann_file` instantiation:

```yaml
ann_file:
  _target_: sam3.eval.coco_reindex.reindex_coco_to_temp
  input_json_path: /path/to/annotations.json
```

---

## Compatibility Analysis

| Issue | CVAT Default | SAM3 Expectation | Bridge |
|-------|-------------|-----------------|--------|
| `file_name` path prefix | `"images/foo.jpg"` | Relative to `img_folder` root | Set `img_folder` to the dir **containing** `images/`, OR strip prefix with `fix_fname=True`, OR rename files |
| 0-based IDs | Possible in older CVAT | Handled by `coco_reindex.py` | Add `reindex_coco_to_temp` to Hydra `ann_file` block |
| Polygon segmentation | ✔ polygon list | ✔ polygon list → RLE via pycocotools | No change needed |
| `iscrowd` | Always `0` | Accepted | No change needed |
| Extra `attributes` key | Present | Ignored by loader | No change needed |
| `area` field | CVAT computes it | Required in annotation | No change needed; CVAT fills it |
| `bbox` format | XYWH absolute | XYWH absolute | No change needed |
| Category `supercategory` | Empty string | Ignored | No change needed |
| Single-file export | `instances_default.json` | Any filename | Rename or point config to file |

### Critical check: `file_name` vs `img_folder`

The safest layout after unzipping CVAT export:

```
/data/defects/
├── train/
│   ├── images/              ← actual image files
│   └── train.json           ← annotation file with file_name = "images/frame_001.jpg"
└── val/
    ├── images/
    └── val.json
```

Then in config:
```yaml
img_folder: /data/defects/train
ann_file: /data/defects/train/train.json
```

The loader resolves `os.path.join("/data/defects/train", "images/frame_001.jpg")` → correct.

---

## Train/Val Split Strategy

**Recommendation for < 500 images: random split into two separate JSON files (80/20).**

### Why not a single JSON with a split field?

COCO format has no standard "split" field. SAM3 reads the entire JSON as one dataset. There is no
built-in way to select a subset by field value — you would need a custom loader. Keep it simple.

### Why not split via `limit_ids` config?

`limit_ids` randomly subsamples training images but cannot produce a held-out validation set from
the same JSON without risk of leakage.

### Recommended procedure

```python
import json, random, copy

with open("instances_default.json") as f:
    coco = json.load(f)

random.seed(42)
image_ids = [img["id"] for img in coco["images"]]
random.shuffle(image_ids)

split = int(0.8 * len(image_ids))
train_ids = set(image_ids[:split])
val_ids   = set(image_ids[split:])

def filter_split(coco, keep_ids):
    out = copy.deepcopy(coco)
    out["images"] = [img for img in coco["images"] if img["id"] in keep_ids]
    out["annotations"] = [ann for ann in coco["annotations"] if ann["image_id"] in keep_ids]
    return out

train_coco = filter_split(coco, train_ids)
val_coco   = filter_split(coco, val_ids)

with open("train.json", "w") as f:
    json.dump(train_coco, f)
with open("val.json", "w") as f:
    json.dump(val_coco, f)
```

### Considerations for small datasets

- **Stratified split**: If some defect categories are rare (< 10 instances), consider stratified
  sampling by category to ensure both splits contain all categories, otherwise SAM3 training will
  error on categories with zero instances unless `include_negatives=True`.
- **k-fold** is overkill here given SAM3's fine-tuning paradigm and epoch budget.
- **Augmentation multiplier**: The config `multiplier` key in `Sam3ImageDataset` repeats the
  dataset N times per epoch — use `multiplier: 5` or higher for datasets < 200 images.
- `target_epoch_size: 1500` in the Roboflow config means SAM3 will loop through the small dataset
  multiple times per epoch automatically; this is fine.

---

## Hydra Config Parameters

All dataset-relevant keys live under `trainer.data.train` and `trainer.data.val`.

### Key parameters for dataset integration

| Config key | Purpose | Example value |
|-----------|---------|--------------|
| `trainer.data.train.dataset.img_folder` | Root dir for image file resolution | `/data/defects/train` |
| `trainer.data.train.dataset.ann_file` | Path to train annotation JSON | `/data/defects/train/train.json` |
| `trainer.data.val.dataset.img_folder` | Root dir for val images | `/data/defects/val` |
| `trainer.data.val.dataset.ann_file` | Path to val annotation JSON | `/data/defects/val/val.json` |
| `trainer.data.train.dataset.load_segmentation` | Enable mask training | `true` |
| `trainer.data.train.dataset.multiplier` | Repeat dataset N× per epoch | `5` (small datasets) |
| `trainer.data.train.dataset.limit_ids` | Cap number of training images | `null` (use all) |
| `trainer.data.train.dataset.training` | Flag for train vs. val mode | `true` |
| `scratch.enable_segmentation` | Controls mask loss + collation | `true` |
| `paths.bpe_path` | Path to BPE tokenizer file | `/path/to/bpe_simple_vocab_16e6.txt.gz` |
| `launcher.num_nodes` | Nodes for distributed training | `1` |
| `launcher.gpus_per_node` | GPUs per node | `1` or `2` |

### Custom prompts (optional but recommended for industrial defects)

The `prompts` parameter on `COCO_FROM_JSON` allows overriding category names with richer text:

```yaml
coco_json_loader:
  _target_: sam3.train.data.coco_json_loaders.COCO_FROM_JSON
  prompts: "[{'id': 1, 'name': 'surface scratch defect'}, {'id': 2, 'name': 'dent damage'}]"
  _partial_: true
```

The `prompts` string is `eval()`-ed by Python, so it must be a valid Python literal string
representing a list of `{id, name}` dicts. The number of entries must exactly match the number of
categories in the JSON.

### Category selection (which categories to fine-tune on)

SAM3 fine-tunes on **all categories present in the annotation JSON**. There is no filter-by-category
config key. To fine-tune on a subset of CVAT categories:

1. Filter the JSON `categories` array to only the desired categories.
2. Remove annotations for excluded categories.
3. SAM3 will create one query per remaining category per image.

Use `category_chunk_size` to control GPU memory pressure when many categories exist:

```yaml
coco_json_loader:
  _target_: sam3.train.data.coco_json_loaders.COCO_FROM_JSON
  category_chunk_size: 4   # process 4 categories at a time
  _partial_: true
```

---

## Integration Steps

### Step 0: Export from CVAT

1. In CVAT, go to **Task → Export Dataset → COCO 1.0** (select "Instance Segmentation").
2. Unzip the archive.
3. Verify structure: `images/` directory + `annotations/instances_default.json`.

### Step 1: Reindex IDs (if needed)

Check whether image/annotation IDs start at 0:

```bash
python -c "
import json
with open('annotations/instances_default.json') as f:
    d = json.load(f)
min_img_id = min(img['id'] for img in d['images'])
min_ann_id = min(ann['id'] for ann in d['annotations'])
min_cat_id = min(cat['id'] for cat in d['categories'])
print(f'min image_id={min_img_id}, ann_id={min_ann_id}, cat_id={min_cat_id}')
"
```

If any minimum is 0, either:
- Use `sam3.eval.coco_reindex.reindex_coco_to_temp` in the Hydra config (recommended), or
- Run the reindex script manually as a one-time pre-processing step.

### Step 2: Train/Val split

Run the Python split script from [Train/Val Split Strategy](#trainval-split-strategy) above.

Recommended final layout:

```
/data/defects/
├── images/          ← all images (shared; paths in JSON are relative)
├── train.json       ← ~80% images + their annotations
└── val.json         ← ~20% images + their annotations
```

### Step 3: Verify `file_name` resolution

```python
import json, os
with open("train.json") as f:
    coco = json.load(f)
sample_fname = coco["images"][0]["file_name"]
img_folder = "/data/defects"                 # ← your img_folder value
full_path = os.path.join(img_folder, sample_fname)
print(full_path, "→ exists:", os.path.exists(full_path))
```

Adjust `img_folder` (or strip the prefix in JSON) until this resolves correctly.

### Step 4: Create Hydra config

Copy `sam3/train/configs/roboflow_v100/roboflow_v100_full_ft_100_images.yaml` to, e.g.,
`sam3/train/configs/defects/defects_finetune.yaml`.

Minimal changes to make:

```yaml
paths:
  defects_root: /data/defects
  experiment_log_dir: /output/defects_run
  bpe_path: /path/to/sam3/assets/bpe_simple_vocab_16e6.txt.gz

scratch:
  enable_segmentation: true   # ← turn on mask loss

trainer:
  data:
    train:
      dataset:
        img_folder: ${paths.defects_root}
        ann_file: ${paths.defects_root}/train.json
        load_segmentation: ${scratch.enable_segmentation}
        multiplier: 5          # ← repeat dataset for small sets
        limit_ids: null

    val:
      dataset:
        img_folder: ${paths.defects_root}
        ann_file: ${paths.defects_root}/val.json
        load_segmentation: ${scratch.enable_segmentation}

  loss:
    all:
      # Uncomment the mask loss block from the config comments
      # (the commented-out section starting at line ~107 of the Roboflow config)
      ...
```

Also update the evaluator's `gt_path` to `${paths.defects_root}/val.json`.

### Step 5: Enable segmentation loss

The Roboflow config has the mask loss **commented out** by default. Uncomment the block under
`# NOTE: Loss to be used for training in case of segmentation` and set
`scratch.enable_segmentation: true`. The full block to uncomment is in
`roboflow_v100_full_ft_100_images.yaml` lines ~108–154.

### Step 6: Launch training (single GPU local)

```bash
cd /home/longtdang/KMS/sam3
python sam3/train/train.py \
    -c configs/defects/defects_finetune.yaml \
    --use-cluster 0 \
    --num-gpus 1
```

### Step 7: Verify data loading before full run

Add a quick sanity check before training:

```python
from sam3.train.data.coco_json_loaders import COCO_FROM_JSON
loader = COCO_FROM_JSON("/data/defects/train.json")
ids = loader.getDatapointIds()
print(f"{len(ids)} datapoints, {len(loader._sorted_cat_ids)} categories")
q, a = loader.loadQueriesAndAnnotationsFromDatapoint(ids[0])
print("queries:", [qq['query_text'] for qq in q])
print("annotations:", len(a))
```

---

## References

### Code paths in this repo

| File | Purpose |
|------|---------|
| `sam3/train/data/coco_json_loaders.py` | Core COCO loader — `COCO_FROM_JSON`, `ann_to_rle` |
| `sam3/train/data/sam3_image_dataset.py` | `Sam3ImageDataset`, `CustomCocoDetectionAPI` |
| `sam3/eval/coco_reindex.py` | 0→1 index fixer: `reindex_coco_to_temp` |
| `sam3/train/configs/roboflow_v100/roboflow_v100_full_ft_100_images.yaml` | **Closest template** for custom COCO fine-tuning |
| `sam3/train/configs/odinw13/odinw_text_only_train.yaml` | Shows `prompts` override + `reindex_coco_to_temp` usage |
| `sam3/train/configs/eval_base.yaml` | Base config with all path/launcher defaults |
| `README_TRAIN.md` | Official training documentation |

### External references

- **CVAT COCO export docs:** https://docs.cvat.ai/docs/manual/advanced/formats/format-coco/
- **COCO format spec:** https://cocodataset.org/#format-data
- **pycocotools (mask RLE):** https://github.com/cocodataset/cocoapi/tree/master/PythonAPI/pycocotools
- **Roboflow 100-VL (analogous dataset):** https://github.com/roboflow/rf100-vl/
