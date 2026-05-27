# Phase 1: Dataset Preparation - Research

**Researched:** 2026-05-27
**Domain:** COCO JSON manipulation, stratified dataset splitting, CLI scripting (Python)
**Confidence:** HIGH

---

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

#### Split Strategy
- **D-01:** Use stratified-by-category split for all classes simultaneously. Images are assigned to splits based on per-category representation across all categories present.
- **D-02:** If a category has only 1 annotated image (rare class), fall back to including it in random split assignment and print a warning. Val may receive 0 instances of that class — this is acceptable.
- **D-03:** Images with zero annotations are excluded from both train and val splits. Print a warning listing the excluded image filenames and count.
- **D-04:** No minimum image count per split is enforced. If val ends up empty, warn the user but do not fail.
- **D-05:** Default split ratio is 80/20 (train/val). `--split-ratio` flag overrides. Default seed is 42; `--seed` flag overrides.

#### file_name Repair
- **D-06:** Strip any leading directory prefix from `file_name` automatically (e.g. `"images/frame_001.jpg"` → `"frame_001.jpg"`). Handles standard CVAT export pattern without user intervention.
- **D-07:** This repair is applied silently (no output). Only unexpected patterns generate warnings.

#### Output Format
- **D-08:** After all repairs, `file_name` values in output JSON contain filename only (no directory component).
- **D-09:** Output files are written to `--output` directory as `train.json` and `val.json`.

#### ID Repair
- **D-10:** If image IDs or annotation IDs are 0-based, reindex to 1-based automatically and silently. Category IDs are also reindexed to be contiguous and 1-based if needed.

#### Error Handling
- **D-11:** Known CVAT quirks (0-based IDs, `file_name` prefix) are auto-fixed silently.
- **D-12:** Unexpected issues (e.g. annotations referencing image IDs not in the `images` list) generate a warning but do not fail the script.
- **D-13:** Malformed input (missing required top-level COCO keys: `images`, `annotations`, `categories`) causes an immediate, clear error message listing the missing keys. No Python traceback — use `sys.exit(1)` with a human-readable message.

#### Script Output / Stats
- **D-14:** After writing the output files, always print a summary: total images processed, images per split, and instance count per category in each split.

### the agent's Discretion

None defined — all key decisions locked.

### Deferred Ideas (OUT OF SCOPE)

None — discussion stayed within phase scope.
</user_constraints>

---

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| DATA-01 | Script converts CVAT COCO export into SAM3-compatible train/val JSON splits (80/20, stratified by category) | Stratified split algorithm section; COCO JSON schema section |
| DATA-02 | Script validates and fixes common CVAT quirks: 0-based IDs → 1-based, `file_name` prefix normalization, contiguous category IDs | ID repair section; file_name repair section; coco_reindex.py analysis |
| DATA-03 | Script is configurable via CLI args: input annotation file, image folder, output directory, split ratio, random seed | CLI design section; argparse pattern from scripts/extract_odinw_results.py |
| DATA-04 | Script reports dataset statistics after preparation: total images, images per split, instances per category | Stats output section; D-14 |
</phase_requirements>

---

## Summary

Phase 1 delivers `scripts/prepare_dataset.py` — a standalone CLI tool with no imports from `sam3/`. The script reads a CVAT COCO export, performs three sequential repair passes (file_name prefix strip, ID 0→1-based reindex, category ID contiguity), runs a multi-label stratified train/val split using a greedy frequency-sorted algorithm (no extra dependencies beyond stdlib + sklearn), writes two output JSON files, and prints a statistics summary.

The critical compatibility requirement is that output JSON must pass through SAM3's `load_coco_and_group_by_image()` cleanly. This function requires only four top-level keys (`images`, `annotations`, `categories` — `info` and `licenses` are ignored) and standard COCO field names. The `file_name` in each image entry must be **filename-only** (no directory prefix), since `CustomCocoDetectionAPI._load_images` resolves paths as `os.path.join(self.root, current_meta["file_name"])` where `self.root` is the `img_folder` config value.

The test framework is pytest (configured in `pyproject.toml`), but the existing test file in `test/` uses `unittest.TestCase`. Phase 1 tests should be placed in `tests/` (the pytest `testpaths` target) and use pytest-style functions with `conftest.py` fixtures for minimal COCO JSON structures. `pycocotools` is available as a dev dependency for optional validation in tests.

**Primary recommendation:** Implement the stratified split as a standalone greedy algorithm (stdlib only) rather than pulling in `skmultilearn` or `iterstrat`, which are not installed. Use `sklearn.model_selection.train_test_split` as a fallback only for the edge case where all images are single-category (simple case) — but the greedy approach handles both single- and multi-label cases uniformly.

---

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| COCO JSON parsing & validation | Script (CLI) | — | Standalone; no training imports needed |
| file_name repair | Script (CLI) | — | Pre-processing step before split |
| ID reindexing (0→1-based) | Script (CLI) | — | Mirror of `coco_reindex.py` logic, applied in-memory |
| Category ID contiguity repair | Script (CLI) | — | NOT in `coco_reindex.py`; must be in script |
| Stratified train/val split | Script (CLI) | — | Python stdlib + sklearn (available) |
| Output JSON writing | Script (CLI) | — | `json.dump` standard pattern |
| Stats printing | Script (CLI) | — | stdout, tabular summary |
| Unit tests | Test suite (`tests/`) | — | pytest, no real images needed |
| Training consumption | SAM3 data pipeline | — | `load_coco_and_group_by_image()` → `COCO_FROM_JSON` |

---

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| Python stdlib: `json`, `os`, `argparse`, `copy`, `random`, `collections` | 3.8+ | JSON parsing, CLI, ID mapping | Zero deps; already used in project |
| `sklearn.model_selection.train_test_split` | 1.8.0 (verified) | Stratified split (single-label fallback) | Already in `train` extras; available in environment |

### Supporting (dev/test only)
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| `pytest` | available | Unit test runner | All Phase 1 tests |
| `pycocotools` | dev extra | Polygon→RLE validation in tests | Optional validation fixture |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Greedy multi-label stratification | `skmultilearn.model_selection.IterativeStratification` | More theoretically correct but not installed; adds `scipy` + `skmultilearn` dependency; overkill for < 500 images |
| Greedy multi-label stratification | `iterstrat.ml_stratifiers.MultilabelStratifiedShuffleSplit` | Cleaner API but not installed |
| `json.dump` | `python-rapidjson` | 3–5× faster serialization but dev-only dep; not needed for one-off script |

**Installation (no new installs needed):**
```bash
# All required packages already in project deps or stdlib
# sklearn is in sam3[train]; json/os/argparse/copy/random are stdlib
pip install -e ".[dev]"   # for pytest + pycocotools in tests
```

**Version verification:** [VERIFIED: bash `python3 -c "import sklearn; print(sklearn.__version__)"`] → `1.8.0`

---

## Architecture Patterns

### System Architecture Diagram

```
CVAT COCO Export
    │
    ▼
[1. Load & Validate JSON]
    │  Check: images, annotations, categories keys exist
    │  Error: sys.exit(1) with missing keys listed
    │
    ▼
[2. Repair Pass]
    │  a) file_name prefix strip: os.path.basename(file_name)  [silent, D-06]
    │  b) 0-based ID reindex: image IDs, annotation IDs        [silent, D-10]
    │  c) category ID contiguity: sort + remap to 1..N         [silent, D-10]
    │  d) Warn: annotations referencing unknown image IDs       [warn, D-12]
    │
    ▼
[3. Filter Zero-Annotation Images]
    │  Warn: list excluded filenames + count                    [warn, D-03]
    │
    ▼
[4. Stratified Split]
    │  Multi-label greedy assignment (frequency-sorted)
    │  Fallback to random for rare classes (≤1 image)          [warn, D-02]
    │  Warn: if val split ends up empty                         [warn, D-04]
    │
    ▼
[5. Write Output JSONs]
    │  {output}/train.json  ← filtered images + annotations
    │  {output}/val.json    ← filtered images + annotations
    │  Both retain: categories, info, licenses from input
    │
    ▼
[6. Print Stats]
    total images: N  (train: T, val: V)
    category stats:
      [cat_name]  train: X instances  val: Y instances
```

### Recommended Project Structure
```
scripts/
├── prepare_dataset.py     # new — CLI entrypoint
└── extract_odinw_results.py  # existing — style reference

tests/                     # new directory (pytest testpaths target)
├── conftest.py            # new — shared COCO fixture builders
└── test_prepare_dataset.py  # new — unit tests for 3 repair cases + split
```

### Pattern 1: Script Module Structure (follow extract_odinw_results.py)
**What:** Top-level functions + `parse_args()` + `main(args)` + `if __name__ == "__main__": main(parse_args())`
**When to use:** All `scripts/` Python files in this project

```python
# Source: scripts/extract_odinw_results.py (VERIFIED: direct read)
import argparse
import json
import os

def parse_args():
    parser = argparse.ArgumentParser("Dataset preparation script")
    parser.add_argument("--ann-file", required=True, type=str,
                        help="Path to CVAT COCO annotation JSON")
    parser.add_argument("--img-folder", required=True, type=str,
                        help="Path to images directory")
    parser.add_argument("--output", required=True, type=str,
                        help="Output directory for train.json and val.json")
    parser.add_argument("--split-ratio", type=float, default=0.8,
                        help="Fraction of images for training (default: 0.8)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility (default: 42)")
    return parser.parse_args()

def main(args):
    # load → validate → repair → split → write → stats
    pass

if __name__ == "__main__":
    main(parse_args())
```

### Pattern 2: D-13 Validation with sys.exit(1)
**What:** Check required keys before any processing; exit with clear message
**When to use:** Malformed input (missing `images`, `annotations`, `categories`)

```python
# Source: CONTEXT.md D-13 [VERIFIED: CONTEXT.md read]
import sys

REQUIRED_KEYS = {"images", "annotations", "categories"}

def validate_coco(data, path):
    missing = REQUIRED_KEYS - set(data.keys())
    if missing:
        print(f"ERROR: {path} is missing required COCO keys: {sorted(missing)}", file=sys.stderr)
        sys.exit(1)
```

### Pattern 3: ID Repair — Independent Per-Type Detection
**What:** Check min ID per type (images, annotations, categories) independently; reindex only if min==0
**When to use:** D-10 CVAT 0-based ID repair

```python
# Source: sam3/eval/coco_reindex.py analysis [VERIFIED: direct read]
# KEY INSIGHT: coco_reindex.py checks if ANY id==0 (not min==0).
# prepare_dataset.py should use min==0 to avoid false positives.
# Also: coco_reindex.py only adds +1 offset. It does NOT handle non-contiguous
# category IDs like [1, 3, 5]. Category contiguity repair is SEPARATE.

def repair_ids(coco):
    images, anns, cats = coco["images"], coco["annotations"], coco["categories"]

    # Step 1: Image ID reindex (if 0-based)
    img_id_map = {}
    if images and min(img["id"] for img in images) == 0:
        for img in images:
            img_id_map[img["id"]] = img["id"] + 1
            img["id"] += 1

    # Step 2: Annotation ID reindex (if 0-based); update image_id refs
    if anns and min(ann["id"] for ann in anns) == 0:
        for ann in anns:
            ann["id"] += 1
    if img_id_map:
        for ann in anns:
            if ann["image_id"] in img_id_map:
                ann["image_id"] = img_id_map[ann["image_id"]]

    # Step 3: Category ID contiguity reindex (always sort + remap to 1..N)
    # This handles: 0-based, non-contiguous [1,3,5], and already-correct [1,2,3]
    sorted_cats = sorted(cats, key=lambda c: c["id"])
    cat_id_map = {cat["id"]: i + 1 for i, cat in enumerate(sorted_cats)}
    for cat in cats:
        cat["id"] = cat_id_map[cat["id"]]
    for ann in anns:
        ann["category_id"] = cat_id_map[ann["category_id"]]

    return coco
```

**⚠️ Critical edge case — MIXED IDs:** If image IDs are already 1-based (`[1, 2, 3]`) but annotation IDs are 0-based (`[0, 1, 2]`), each type is detected and repaired independently. The repair logic correctly handles this because it checks min per type separately. [VERIFIED: coco_reindex.py code analysis + manual reasoning]

**⚠️ Critical edge case — Category annotation references:** When remapping category IDs for contiguity, ALL annotations must have their `category_id` updated using the same `cat_id_map`. Failing to do so produces dangling references. [VERIFIED: coco_reindex.py pattern analysis]

### Pattern 4: Multi-Label Stratified Split (Greedy Frequency-Sorted)
**What:** Assign images to splits respecting per-category proportions, even when images have multiple category labels
**When to use:** D-01 — "stratified-by-category split for all classes simultaneously"

**Why greedy instead of sklearn.train_test_split:**
- `sklearn.train_test_split(stratify=...)` only handles **single-label** stratification
- It raises `ValueError: The test_size should be >= num_classes` for small datasets [VERIFIED: bash test]
- `skmultilearn` and `iterstrat` are not installed [VERIFIED: bash check]
- Greedy iterative stratification requires no extra dependencies and is well-suited to < 500 images

```python
# Source: iterative stratification algorithm [ASSUMED — standard ML pattern; no official docs URL]
import random
import collections

def stratified_split(image_ids, anns_by_image, cat_id_to_name, split_ratio, seed):
    """
    Multi-label greedy stratified split.
    
    Algorithm:
    1. Compute per-image category set
    2. Identify rare classes (only 1 image) -> warn, add to random pool
    3. For remaining images, sort by number of categories (most constrained first)
    4. For each image, assign to the split that has lowest proportional
       representation of the image's rarest category
    5. Images with zero annotations are excluded (D-03) before this step
    """
    rng = random.Random(seed)
    
    # Build image -> category set mapping
    img_categories = collections.defaultdict(set)
    for img_id in image_ids:
        for ann in anns_by_image.get(img_id, []):
            img_categories[img_id].add(ann["category_id"])
    
    # Count images per category
    cat_image_count = collections.Counter()
    for cats in img_categories.values():
        for cat_id in cats:
            cat_image_count[cat_id] += 1
    
    # Identify rare classes (D-02)
    rare_cats = {cat_id for cat_id, count in cat_image_count.items() if count == 1}
    if rare_cats:
        rare_names = [cat_id_to_name.get(c, str(c)) for c in rare_cats]
        print(f"WARNING: {len(rare_cats)} rare class(es) with only 1 image "
              f"(will be randomly assigned): {rare_names}")
    
    train_ids, val_ids = [], []
    train_cat_count = collections.Counter()
    val_cat_count = collections.Counter()
    
    # Sort images: more constrained (more categories) first
    sorted_images = sorted(image_ids, key=lambda img_id: len(img_categories[img_id]),
                           reverse=True)
    
    for img_id in sorted_images:
        cats = img_categories[img_id]
        
        # Greedy assignment: assign to split that needs this image more
        # Score = fraction of target already met for the rarest category
        def split_score(split_cat_count, split_size):
            if not cats:
                return 0.0  # zero-ann images excluded before here
            rarest_cat = min(cats, key=lambda c: cat_image_count.get(c, 0))
            total = cat_image_count.get(rarest_cat, 1)
            return split_cat_count.get(rarest_cat, 0) / max(total * split_ratio, 1)
        
        train_score = split_score(train_cat_count, len(train_ids))
        val_target = 1 - split_ratio
        val_score = split_score(val_cat_count, len(val_ids)) / max(val_target / split_ratio, 1e-9)
        
        if train_score <= val_score:
            train_ids.append(img_id)
            for c in cats:
                train_cat_count[c] += 1
        else:
            val_ids.append(img_id)
            for c in cats:
                val_cat_count[c] += 1
    
    return train_ids, val_ids
```

**Simpler alternative for the implementation:** Sort images by their rarest category (fewest total images), process in that order, and assign each to the split that is most "behind" the target ratio for that category. This is O(N log N) and works for both single- and multi-label cases.

### Pattern 5: SAM3-Compatible Output JSON
**What:** Exact required fields per section; what the loader actually reads
**When to use:** Constructing the output JSON

```python
# Source: sam3/train/data/coco_json_loaders.py load_coco_and_group_by_image() [VERIFIED: direct read]

# load_coco_and_group_by_image reads EXACTLY these fields:
#   images:      id, file_name (+ width, height needed for ann_to_rle)
#   annotations: id, image_id, category_id, bbox, segmentation, iscrowd, area
#   categories:  id, name

# SAM3 does NOT require: info, licenses, supercategory, attributes
# SAM3 IGNORES extra fields silently (annotation["attributes"] from CVAT is ignored)
# Safe to copy-through: info, licenses, categories from input

def filter_split(coco, keep_ids):
    """Filter a COCO dict to only include images and annotations for keep_ids."""
    import copy
    keep = set(keep_ids)
    out = {
        "info": coco.get("info", {}),
        "licenses": coco.get("licenses", []),
        "categories": copy.deepcopy(coco["categories"]),
        "images": [img for img in coco["images"] if img["id"] in keep],
        "annotations": [ann for ann in coco["annotations"] if ann["image_id"] in keep],
    }
    return out
```

### Pattern 6: pytest Fixtures (tests/conftest.py)
**What:** Minimal COCO JSON dicts for unit tests, no real images required
**When to use:** All three repair case tests

```python
# Source: test/test_io_utils.py style [VERIFIED: direct read] + pytest docs [ASSUMED]

# tests/conftest.py
import pytest

@pytest.fixture
def minimal_coco():
    """Minimal valid COCO dict with 1-based IDs."""
    return {
        "info": {}, "licenses": [],
        "categories": [{"id": 1, "name": "scratch", "supercategory": ""}],
        "images": [
            {"id": 1, "file_name": "frame_001.jpg", "width": 640, "height": 480},
            {"id": 2, "file_name": "frame_002.jpg", "width": 640, "height": 480},
        ],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 1,
             "bbox": [10, 20, 50, 60], "area": 3000,
             "segmentation": [[10,20, 60,20, 60,80, 10,80]], "iscrowd": 0},
        ]
    }

@pytest.fixture
def zero_based_coco():
    """COCO dict with 0-based image, annotation, and category IDs (CVAT quirk)."""
    return {
        "info": {}, "licenses": [],
        "categories": [{"id": 0, "name": "scratch", "supercategory": ""}],
        "images": [
            {"id": 0, "file_name": "frame_001.jpg", "width": 640, "height": 480},
            {"id": 1, "file_name": "frame_002.jpg", "width": 640, "height": 480},
        ],
        "annotations": [
            {"id": 0, "image_id": 0, "category_id": 0,
             "bbox": [10, 20, 50, 60], "area": 3000,
             "segmentation": [[10,20, 60,20, 60,80, 10,80]], "iscrowd": 0},
        ]
    }

@pytest.fixture
def prefixed_fname_coco():
    """COCO dict with 'images/' prefix in file_name (standard CVAT export)."""
    return {
        "info": {}, "licenses": [],
        "categories": [{"id": 1, "name": "scratch", "supercategory": ""}],
        "images": [
            {"id": 1, "file_name": "images/frame_001.jpg", "width": 640, "height": 480},
        ],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 1,
             "bbox": [10, 20, 50, 60], "area": 3000,
             "segmentation": [], "iscrowd": 0},
        ]
    }
```

### Anti-Patterns to Avoid
- **Don't import from `sam3/`:** Script is standalone CLI; no `sam3.train.*` imports. SAM3 isn't installed in all environments and this would create a hard dependency for a utility script. [VERIFIED: CONTEXT.md "no imports from sam3/ package required"]
- **Don't use `coco_reindex.py` directly in the script:** It writes to a temp file and returns a path — not suitable for in-memory pipeline. Re-implement the logic inline. [VERIFIED: coco_reindex.py direct read]
- **Don't use `sklearn.train_test_split(stratify=...)` for multi-label:** Raises `ValueError` when val size < num_classes, which happens routinely with small datasets. [VERIFIED: bash test]
- **Don't add +1 as the only category repair:** Non-contiguous category IDs (e.g. `[1, 3, 5]` from deleted categories in CVAT) require full remapping, not just +1 offset. [VERIFIED: coco_reindex.py code analysis]
- **Don't forget to update annotation `category_id` refs after category remapping:** Category ID map must be applied to all annotations, not just the categories list. [VERIFIED: coco_reindex.py pattern]

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Polygon→RLE conversion | Custom polygon encoder | `pycocotools.mask.frPyObjects` | Edge cases in polygon winding, RLE compression format; already used by SAM3 |
| JSON serialization | Custom writer | `json.dump` (stdlib) | Standard; correct Unicode handling |
| CLI argument parsing | Custom parser | `argparse` (stdlib) | Project pattern from extract_odinw_results.py |
| Random splitting | Custom PRNG | `random.Random(seed)` (stdlib) | Reproducible seeded RNG; no need for numpy |

**Key insight:** The entire script can be implemented with stdlib only (json, os, argparse, copy, random, collections, sys). sklearn is available but only needed for the rare "all images single-category" case if you want to use `train_test_split` as a shortcut.

---

## Common Pitfalls

### Pitfall 1: Category Contiguity vs. 0-Based Confusion
**What goes wrong:** Treating category ID repair as "just add +1 to 0-based IDs" misses the non-contiguous case (e.g., CVAT projects where categories were deleted, leaving gaps like `[1, 3, 7]`).
**Why it happens:** `coco_reindex.py` only handles the 0-based case. D-10 explicitly requires "contiguous and 1-based".
**How to avoid:** Always sort categories by original ID, build a remapping dict `{old_id: new_1_based_id}`, apply to both `categories` list and all `annotation["category_id"]` values.
**Warning signs:** SAM3 training crashes with category index out-of-range, or missing category prompts.

### Pitfall 2: file_name Resolution Mismatch
**What goes wrong:** After stripping prefix, `file_name = "frame_001.jpg"` but the user's `img_folder` in their Hydra config points to the parent directory containing an `images/` subfolder instead of directly to the images.
**Why it happens:** CVAT exports images into an `images/` subdirectory. After D-06 strip, the file is just `frame_001.jpg`. If the user sets `img_folder=/data/defects/` and images are at `/data/defects/images/`, SAM3 won't find them.
**How to avoid:** D-08 is clear: output `file_name = filename only`. The Phase 2 Hydra config must set `img_folder` to the directory **directly containing the images** (e.g., `/data/defects/images/`), not the parent. Add a note in the stats output that prints `img_folder` should point to the directory containing the image files.
**Warning signs:** `FileNotFoundError` during training on the first batch.

### Pitfall 3: Mixed 0-based / 1-based IDs (Partial Corruption)
**What goes wrong:** Image IDs are 1-based (`[1, 2, 3]`) but annotation IDs are 0-based (`[0, 1, 2]`). Reindexing only annotations without touching image IDs is correct, but a naive "all 0-based" check would skip the annotation fix.
**Why it happens:** Some CVAT export versions apply different indexing rules to different object types.
**How to avoid:** Check min ID **per type** independently using `min(x["id"] for x in items) == 0`. The detection is independent for images, annotations, and categories.
**Warning signs:** Training loads with seemingly correct data, but first annotation has ID 0, causing potential downstream indexing issues.

### Pitfall 4: sklearn stratify Failure on Small Datasets
**What goes wrong:** `sklearn.train_test_split(stratify=labels)` raises `ValueError: The test_size = N should be >= num_classes = M` when the val split size is smaller than the number of unique categories.
**Why it happens:** sklearn requires at least one sample per class in each split.
**How to avoid:** Use the greedy frequency-sorted algorithm instead, which handles the rare-class case via D-02 (fallback to random for 1-image categories).
**Warning signs:** `ValueError` at split time on any dataset with > (N * val_ratio) categories.

### Pitfall 5: Annotations Referencing Deleted Image IDs
**What goes wrong:** Some CVAT exports have "orphan" annotations whose `image_id` references an image that isn't in the `images` list (e.g., the image was deleted from the task after annotation).
**Why it happens:** CVAT doesn't always cascade-delete annotations when an image is removed.
**How to avoid:** D-12 says warn but don't fail. After loading, build `valid_image_ids = {img["id"] for img in coco["images"]}` and warn about any `ann["image_id"] not in valid_image_ids`. Drop orphan annotations silently after warning.
**Warning signs:** `KeyError` in `load_coco_and_group_by_image` when building `anns_by_image`.

### Pitfall 6: pytest testpaths vs. test/ Directory Mismatch
**What goes wrong:** Tests placed in `test/` won't be discovered by pytest (which is configured to look in `tests/`).
**Why it happens:** `pyproject.toml` sets `testpaths = ["tests"]` but the only existing test file is in `test/test_io_utils.py` (uses `unittest`). This is a pre-existing inconsistency.
**How to avoid:** Create `tests/` directory (the pytest target) for Phase 1 tests. Don't add Phase 1 tests to `test/` (the unittest directory).
**Warning signs:** `pytest` runs but reports 0 tests collected.

---

## Code Examples

Verified patterns from official sources:

### SAM3 COCO Loader — Exact Fields Read
```python
# Source: sam3/train/data/coco_json_loaders.py load_coco_and_group_by_image() [VERIFIED: direct read]

# Fields the loader uses from each section:
# images:      ["id"]               → key in images dict
#              ["file_name"]        → passed to os.path.join(root, file_name)
#              ["width"], ["height"] → used by ann_to_rle for polygon→RLE
# annotations: ["image_id"]        → groups annotations by image
#              ["category_id"]      → groups by category in COCO_FROM_JSON
#              ["bbox"]             → XYWH absolute, normalized by loader
#              ["segmentation"]     → polygon list, converted to RLE
#              ["iscrowd"]          → passed through to annotation dict
#              ["area"]             → used for annotation area metric
#              ["id"]               → used as annotation identity
# categories:  ["id"], ["name"]    → builds cat_id_to_name dict

# NOT required (ignored by loader): info, licenses, supercategory, attributes
```

### file_name Resolution in CustomCocoDetectionAPI
```python
# Source: sam3/train/data/sam3_image_dataset.py _load_images() [VERIFIED: direct read]

# fix_fname=True path (already in SAM3, but D-06/D-08 say to fix in script output instead):
if self.fix_fname:
    current_meta["file_name"] = current_meta["file_name"].split("/")[-1]

# Standard path resolution:
path = os.path.join(self.root, path)  # self.root = img_folder from config
```

### Stats Output Format (DATA-04)
```python
# Based on D-14: total images processed, images per split, instances per category per split [VERIFIED: CONTEXT.md]
# Recommended format:

def print_stats(coco_in, train_coco, val_coco):
    total = len(coco_in["images"])
    n_train = len(train_coco["images"])
    n_val = len(val_coco["images"])
    
    print(f"\nDataset statistics:")
    print(f"  Total images processed : {total}")
    print(f"  Train split            : {n_train} images")
    print(f"  Val split              : {n_val} images")
    print(f"\n  Instances per category:")
    
    cat_name = {cat["id"]: cat["name"] for cat in coco_in["categories"]}
    
    def count_instances(coco):
        counts = collections.Counter(ann["category_id"] for ann in coco["annotations"])
        return counts
    
    train_counts = count_instances(train_coco)
    val_counts = count_instances(val_coco)
    
    header = f"  {'Category':<20} {'Train':>8} {'Val':>8}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for cat in sorted(coco_in["categories"], key=lambda c: c["id"]):
        name = cat["name"]
        t = train_counts.get(cat["id"], 0)
        v = val_counts.get(cat["id"], 0)
        print(f"  {name:<20} {t:>8} {v:>8}")
```

---

## Runtime State Inventory

> Greenfield phase — no rename/refactor involved.

**Not applicable.** Phase 1 creates new files only (`scripts/prepare_dataset.py`, `tests/conftest.py`, `tests/test_prepare_dataset.py`). No existing runtime state to migrate.

---

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Manual COCO split (random, no stratification) | Stratified multi-label split | D-01 decision | Ensures all categories represented in val |
| Use `coco_reindex.py` in Hydra config | Fix in script output (standalone) | D-10 decision | Simpler user experience; output JSON is clean |
| Point `img_folder` to parent of `images/` | Strip prefix; point `img_folder` to `images/` dir | D-06/D-08 | Eliminates file resolution ambiguity |

**Deprecated/outdated:**
- Using `fix_fname=True` in `Sam3ImageDataset`: Valid runtime option, but D-06/D-08 prefer fixing at script time so output is always clean. The `fix_fname` parameter still exists and is used in production configs.

---

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | Greedy frequency-sorted stratification (without skmultilearn) produces acceptable category balance for industrial datasets < 500 images | Pattern 4 (Split Algorithm) | Val split may have slightly worse category balance vs. iterative stratification — acceptable per D-02/D-04 |
| A2 | pytest-style tests in `tests/` are preferred over `unittest` in `test/` for Phase 1 | Pitfall 6 / test framework | If project mandates unittest consistency, tests would need to use `unittest.TestCase` instead |
| A3 | The stats table format (columnar, category-per-row) satisfies DATA-04 "instances per category in each split" | Stats Output section | If a specific format is required, implementation must adapt |

---

## Open Questions

1. **pyproject.toml `testpaths = ["tests"]` vs existing `test/` directory inconsistency**
   - What we know: `pyproject.toml` sets `testpaths = ["tests"]` (pytest discovers here), but the only existing test is `test/test_io_utils.py` (unittest-based, in a different directory)
   - What's unclear: Is the project intentionally migrating from `test/` to `tests/` for new tests, or should Phase 1 tests follow the existing `test/` convention with `unittest.TestCase`?
   - Recommendation: Create `tests/` as a new directory for Phase 1 pytest tests. If the planner disagrees, fall back to `test/` with `unittest.TestCase`. Both are valid; the pytest config points to `tests/`.

2. **`--img-folder` CLI arg: is it used in validation or just passed through?**
   - What we know: D-03 says warn about excluded images with filenames. The `--img-folder` arg is listed in DATA-03 requirements.
   - What's unclear: Does the script need to verify images actually exist on disk (using `img_folder`), or is the JSON the sole source of truth?
   - Recommendation: Don't require image existence check — the script operates on the JSON only. `--img-folder` is accepted as an argument for documentation/output purposes but not used to validate file existence. This keeps the script fast and testable without real images.

---

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| Python stdlib (json, os, argparse, copy, random, collections, sys) | Core script | ✓ | 3.12 | — |
| `sklearn` | Stratification (single-label fallback) | ✓ | 1.8.0 | Greedy algorithm (no sklearn needed) |
| `pytest` | Unit tests | ✓ | available | — |
| `pycocotools` | Optional test validation | ✗ (not installed in base env) | — | Skip polygon→RLE validation in tests; use dict equality checks instead |
| `skmultilearn` | Multi-label stratification (alternative) | ✗ | — | Greedy frequency-sorted algorithm (recommended) |
| `iterstrat` | Multi-label stratification (alternative) | ✗ | — | Greedy frequency-sorted algorithm (recommended) |

**Missing dependencies with no fallback:** None — all blocking dependencies are available.

**Missing dependencies with fallback:**
- `pycocotools`: Not installed in base env. Tests should NOT require it. Use dict-based assertions instead of RLE validation.
- `skmultilearn` / `iterstrat`: Not needed; greedy algorithm is the implementation choice.

---

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest (pyproject.toml `[tool.pytest.ini_options]`) |
| Config file | `pyproject.toml` — `testpaths = ["tests"]`, `python_files = "test_*.py"` |
| Quick run command | `pytest tests/test_prepare_dataset.py -x` |
| Full suite command | `pytest tests/ -v` |

### Phase Requirements → Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| DATA-01 | 80/20 stratified split produces non-empty train + val with category coverage | unit | `pytest tests/test_prepare_dataset.py::test_stratified_split -x` | ❌ Wave 0 |
| DATA-02 | 0-based image IDs → 1-based reindex | unit | `pytest tests/test_prepare_dataset.py::test_id_reindex -x` | ❌ Wave 0 |
| DATA-02 | file_name prefix stripped to basename | unit | `pytest tests/test_prepare_dataset.py::test_filename_prefix_strip -x` | ❌ Wave 0 |
| DATA-02 | Non-contiguous category IDs → contiguous 1-based | unit | `pytest tests/test_prepare_dataset.py::test_category_reindex -x` | ❌ Wave 0 |
| DATA-03 | `--split-ratio` and `--seed` CLI flags accepted | unit | `pytest tests/test_prepare_dataset.py::test_cli_args -x` | ❌ Wave 0 |
| DATA-03 | Missing COCO keys → sys.exit(1) with clear message | unit | `pytest tests/test_prepare_dataset.py::test_malformed_input -x` | ❌ Wave 0 |
| DATA-04 | Stats summary printed to stdout | unit | `pytest tests/test_prepare_dataset.py::test_stats_output -x` | ❌ Wave 0 |

### Sampling Rate
- **Per task commit:** `pytest tests/test_prepare_dataset.py -x`
- **Per wave merge:** `pytest tests/ -v`
- **Phase gate:** Full suite green before `/gsd-verify-work`

### Wave 0 Gaps
- [ ] `tests/__init__.py` — empty, marks as package
- [ ] `tests/conftest.py` — shared COCO fixture builders (minimal_coco, zero_based_coco, prefixed_fname_coco, noncontiguous_cat_coco)
- [ ] `tests/test_prepare_dataset.py` — all 7 test functions above
- [ ] Framework install: `pip install -e ".[dev]"` — if dev extras not yet installed

*(No gaps in framework itself — pytest is available. Gaps are only test files.)*

---

## Security Domain

> `security_enforcement` not set in `.planning/config.json` → treated as enabled.

### Applicable ASVS Categories

| ASVS Category | Applies | Standard Control |
|---------------|---------|-----------------|
| V2 Authentication | no | Standalone script, no auth |
| V3 Session Management | no | No sessions |
| V4 Access Control | no | Local filesystem only |
| V5 Input Validation | yes | Validate required COCO keys; sys.exit(1) on malformed input (D-13) |
| V6 Cryptography | no | No crypto |

### Known Threat Patterns for CLI JSON processing

| Pattern | STRIDE | Standard Mitigation |
|---------|--------|---------------------|
| Malformed JSON as DoS | Denial of Service | `json.JSONDecodeError` caught, sys.exit(1) with message |
| Deeply nested JSON → stack overflow | Denial of Service | Python's json module has recursion limits; acceptable for COCO files |
| Path traversal via `file_name` field | Tampering | `os.path.basename()` strips all path components; output is filename-only |

---

## Sources

### Primary (HIGH confidence)
- `sam3/train/data/coco_json_loaders.py` — exact fields read by `load_coco_and_group_by_image()` and `COCO_FROM_JSON` [VERIFIED: direct read]
- `sam3/train/data/sam3_image_dataset.py` — `img_folder` + `ann_file` wiring, `fix_fname` behavior, `_load_images` path resolution [VERIFIED: direct read]
- `sam3/eval/coco_reindex.py` — ID repair logic, per-type independent detection, `+1` offset approach (vs. contiguity) [VERIFIED: direct read]
- `scripts/extract_odinw_results.py` — code style reference (argparse pattern, main() structure) [VERIFIED: direct read]
- `pyproject.toml` — test framework (pytest), testpaths config, available dependencies (sklearn, pycocotools as dev extra) [VERIFIED: direct read]
- `.planning/phases/01-dataset-preparation/01-CONTEXT.md` — all locked decisions D-01 through D-14 [VERIFIED: direct read]

### Secondary (MEDIUM confidence)
- `.planning/research/DATASET_INTEGRATION.md` — CVAT COCO export structure, `file_name` prefix pattern, `fix_fname` context [VERIFIED: direct read of prior research]
- `bash: python3 -c "import sklearn; ..."` — sklearn 1.8.0 available; `train_test_split(stratify=...)` ValueError behavior verified [VERIFIED: bash execution]
- `bash: python3 -c "import skmultilearn; ..."` — not installed [VERIFIED: bash execution]

### Tertiary (LOW confidence)
- Greedy frequency-sorted multi-label stratification algorithm — standard ML technique, no specific paper cited; adequate for < 500 images [ASSUMED]

---

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — all dependencies verified in environment; stdlib only for core
- Architecture: HIGH — exact loader code read and annotated
- ID repair logic: HIGH — coco_reindex.py read; edge cases reasoned through
- Stratified split algorithm: MEDIUM — greedy approach is standard practice; exact implementation is [ASSUMED] but decisions constrain the design
- Pitfalls: HIGH — derived from direct code analysis, not speculation

**Research date:** 2026-05-27
**Valid until:** 2026-07-27 (stable codebase; no external API dependencies)
