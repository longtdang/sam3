# Phase 1: Dataset Preparation - Pattern Map

**Mapped:** 2026-05-27
**Files analyzed:** 4 (new/modified)
**Analogs found:** 4 / 4

---

## File Classification

| New/Modified File | Role | Data Flow | Closest Analog | Match Quality |
|-------------------|------|-----------|----------------|---------------|
| `scripts/prepare_dataset.py` | utility / CLI script | file-I/O + transform | `scripts/extract_odinw_results.py` | role-match (same project tier, same CLI pattern) |
| `tests/__init__.py` | config | — | (empty file, no analog needed) | n/a |
| `tests/conftest.py` | test fixture provider | — | `test/test_io_utils.py` (style) | partial-match (pytest fixture style vs unittest) |
| `tests/test_prepare_dataset.py` | test | request-response / transform | `test/test_io_utils.py` | role-match (same project test style) |

---

## Pattern Assignments

### `scripts/prepare_dataset.py` (utility/CLI, file-I/O + transform)

**Analog:** `scripts/extract_odinw_results.py`

**Imports pattern** (lines 1–18 of analog):
```python
# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

# pyre-unsafe

"""This script prepares a CVAT COCO export for SAM3 fine-tuning."""

"""
python3 scripts/prepare_dataset.py \
    --ann-file /path/to/annotations.json \
    --img-folder /path/to/images \
    --output /path/to/output \
    [--split-ratio 0.8] [--seed 42]
"""
import argparse
import collections
import copy
import json
import os
import random
import sys
```

**argparse / parse_args() pattern** (lines 39–49 of analog):
```python
# Exact pattern from scripts/extract_odinw_results.py lines 39-49
def parse_args():
    parser = argparse.ArgumentParser("Dataset preparation script")
    parser.add_argument(
        "--ann-file",
        required=True,
        type=str,
        help="Path to CVAT COCO annotation JSON",
    )
    parser.add_argument(
        "--img-folder",
        required=True,
        type=str,
        help="Path to directory containing the images",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=str,
        help="Output directory for train.json and val.json",
    )
    parser.add_argument(
        "--split-ratio",
        type=float,
        default=0.8,
        help="Fraction of images for training (default: 0.8)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    return parser.parse_args()
```

**main() + entrypoint pattern** (lines 52–97 of analog):
```python
# Exact pattern from scripts/extract_odinw_results.py lines 52-97
def main(args):
    # load → validate → repair → split → write → stats
    pass

if __name__ == "__main__":
    main(parse_args())
```

**Warning pattern** (lines 62–65 of analog — `print(f"Warning: ...")`):
```python
# Project uses bare print() for warnings (not logging module).
# From scripts/extract_odinw_results.py line 64:
print(f"Warning: {val_stats_path} not found, skipping {subset}")
# For D-13 fatal errors, use stderr + sys.exit(1):
print(f"ERROR: {path} is missing required COCO keys: {sorted(missing)}", file=sys.stderr)
sys.exit(1)
```

**Error handling pattern** (lines 66–77 of analog):
```python
# From scripts/extract_odinw_results.py lines 66-77
try:
    res = json.load(open(val_stats_path))
    # ... process
except (json.JSONDecodeError, IOError) as e:
    print(f"Error reading {val_stats_path}: {e}")
    continue
```

**JSON read/write pattern** (lines 67–68 of analog):
```python
# Read (analog pattern):
res = json.load(open(val_stats_path))

# Write (standard project extension of same style):
os.makedirs(args.output, exist_ok=True)
with open(os.path.join(args.output, "train.json"), "w") as f:
    json.dump(train_coco, f)
with open(os.path.join(args.output, "val.json"), "w") as f:
    json.dump(val_coco, f)
```

**SAM3 loader compatibility — required fields**
(Source: `sam3/train/data/coco_json_loaders.py` `load_coco_and_group_by_image()` lines 37–68):
```python
# load_coco_and_group_by_image() reads EXACTLY these fields:
#   coco["images"]      → each entry needs: id, file_name (+ width, height for RLE)
#   coco["annotations"] → each entry needs: id, image_id, category_id, bbox,
#                         segmentation, iscrowd, area
#   coco["categories"]  → each entry needs: id, name
#
# coco["info"] and coco["licenses"] are ignored but safe to copy through.
# extra fields (e.g. CVAT "attributes") are also silently ignored.
#
# CRITICAL: file_name must be FILENAME ONLY — no directory prefix.
# SAM3 resolves: os.path.join(self.root, current_meta["file_name"])
# where self.root = img_folder from the Hydra config.

def filter_split(coco: dict, keep_ids: set) -> dict:
    """Filter a repaired COCO dict to only images/annotations for keep_ids."""
    return {
        "info": coco.get("info", {}),
        "licenses": coco.get("licenses", []),
        "categories": copy.deepcopy(coco["categories"]),
        "images": [img for img in coco["images"] if img["id"] in keep_ids],
        "annotations": [ann for ann in coco["annotations"] if ann["image_id"] in keep_ids],
    }
```

**D-13 validation (sys.exit, no traceback) pattern:**
```python
# Source: CONTEXT.md D-13 / RESEARCH.md Pattern 2
REQUIRED_KEYS = {"images", "annotations", "categories"}

def validate_coco(data: dict, path: str) -> None:
    missing = REQUIRED_KEYS - set(data.keys())
    if missing:
        print(
            f"ERROR: {path} is missing required COCO keys: {sorted(missing)}",
            file=sys.stderr,
        )
        sys.exit(1)
```

**D-10 ID repair pattern:**
```python
# Source: RESEARCH.md Pattern 3 (derived from sam3/eval/coco_reindex.py analysis)
# Check min ID per type INDEPENDENTLY — handles mixed 0-based/1-based (Pitfall 3)
def repair_ids(coco: dict) -> dict:
    images = coco["images"]
    anns   = coco["annotations"]
    cats   = coco["categories"]

    # Step 1: Image IDs — reindex if 0-based
    img_id_map: dict = {}
    if images and min(img["id"] for img in images) == 0:
        for img in images:
            img_id_map[img["id"]] = img["id"] + 1
            img["id"] += 1

    # Step 2: Annotation IDs — reindex if 0-based; patch image_id refs
    if anns and min(ann["id"] for ann in anns) == 0:
        for ann in anns:
            ann["id"] += 1
    if img_id_map:
        for ann in anns:
            if ann["image_id"] in img_id_map:
                ann["image_id"] = img_id_map[ann["image_id"]]

    # Step 3: Category IDs — always sort + remap to 1..N (handles gaps too)
    sorted_cats = sorted(cats, key=lambda c: c["id"])
    cat_id_map  = {cat["id"]: i + 1 for i, cat in enumerate(sorted_cats)}
    for cat in cats:
        cat["id"] = cat_id_map[cat["id"]]
    for ann in anns:
        ann["category_id"] = cat_id_map[ann["category_id"]]

    return coco
```

**D-06 file_name prefix strip pattern (silent):**
```python
# Source: CONTEXT.md D-06/D-07; os.path.basename is stdlib
def repair_filenames(coco: dict) -> dict:
    for img in coco["images"]:
        img["file_name"] = os.path.basename(img["file_name"])
    return coco
```

**D-12 orphan annotation warning pattern:**
```python
# Source: CONTEXT.md D-12; RESEARCH.md Pitfall 5
def warn_orphan_annotations(coco: dict) -> None:
    valid_ids = {img["id"] for img in coco["images"]}
    orphans = [ann for ann in coco["annotations"] if ann["image_id"] not in valid_ids]
    if orphans:
        print(
            f"Warning: {len(orphans)} annotation(s) reference unknown image IDs "
            f"and will be skipped: {sorted({a['image_id'] for a in orphans})}"
        )
    # Drop orphans in-place
    coco["annotations"] = [a for a in coco["annotations"] if a["image_id"] in valid_ids]
```

**D-03 zero-annotation image exclusion + warning:**
```python
# Source: CONTEXT.md D-03
def exclude_zero_annotation_images(coco: dict) -> tuple:
    """Returns (annotated_image_ids, excluded_filenames)."""
    annotated_ids = {ann["image_id"] for ann in coco["annotations"]}
    excluded = [img["file_name"] for img in coco["images"] if img["id"] not in annotated_ids]
    if excluded:
        print(
            f"Warning: {len(excluded)} image(s) with zero annotations excluded: {excluded}"
        )
    return annotated_ids, excluded
```

**Multi-label stratified split pattern:**
```python
# Source: RESEARCH.md Pattern 4 (greedy frequency-sorted; avoids sklearn multi-label failure)
def stratified_split(
    image_ids: list,
    anns_by_image: dict,
    cat_id_to_name: dict,
    split_ratio: float,
    seed: int,
) -> tuple:
    rng = random.Random(seed)

    # Build per-image category sets
    img_categories: dict = collections.defaultdict(set)
    for img_id in image_ids:
        for ann in anns_by_image.get(img_id, []):
            img_categories[img_id].add(ann["category_id"])

    # Count images per category for scoring
    cat_image_count: dict = collections.Counter()
    for cats in img_categories.values():
        for cat_id in cats:
            cat_image_count[cat_id] += 1

    # Warn on rare classes (D-02)
    rare_cats = {cat_id for cat_id, count in cat_image_count.items() if count == 1}
    if rare_cats:
        rare_names = [cat_id_to_name.get(c, str(c)) for c in sorted(rare_cats)]
        print(
            f"Warning: {len(rare_cats)} rare class(es) with only 1 annotated image "
            f"(will be randomly assigned to a split): {rare_names}"
        )

    train_ids: list = []
    val_ids: list   = []
    train_cat_count: dict = collections.Counter()
    val_cat_count: dict   = collections.Counter()

    # Most-constrained images first (most categories)
    sorted_images = sorted(
        image_ids, key=lambda img_id: len(img_categories[img_id]), reverse=True
    )

    for img_id in sorted_images:
        cats = img_categories[img_id]
        if not cats:
            continue  # zero-ann images filtered before this step

        # Assign to split that is most "behind" target for this image's rarest category
        rarest = min(cats, key=lambda c: cat_image_count.get(c, 0))
        total  = max(cat_image_count.get(rarest, 1), 1)
        train_fill = train_cat_count.get(rarest, 0) / (total * split_ratio + 1e-9)
        val_fill   = val_cat_count.get(rarest, 0)   / (total * (1 - split_ratio) + 1e-9)

        if train_fill <= val_fill:
            train_ids.append(img_id)
            for c in cats:
                train_cat_count[c] += 1
        else:
            val_ids.append(img_id)
            for c in cats:
                val_cat_count[c] += 1

    # Warn if val is empty (D-04)
    if not val_ids:
        print("Warning: val split is empty. Consider increasing dataset size or adjusting --split-ratio.")

    return train_ids, val_ids
```

**D-14 stats summary pattern:**
```python
# Source: CONTEXT.md D-14 / RESEARCH.md §Stats output
# Uses print() — same as extract_odinw_results.py lines 80-91
def print_stats(coco: dict, train_ids: set, val_ids: set) -> None:
    cat_id_to_name = {cat["id"]: cat["name"] for cat in coco["categories"]}

    train_counts: dict = collections.Counter()
    val_counts:   dict = collections.Counter()
    for ann in coco["annotations"]:
        if ann["image_id"] in train_ids:
            train_counts[ann["category_id"]] += 1
        elif ann["image_id"] in val_ids:
            val_counts[ann["category_id"]] += 1

    total = len(train_ids) + len(val_ids)
    print(f"\nDataset summary:")
    print(f"  Total images processed : {total}  (train: {len(train_ids)}, val: {len(val_ids)})")
    print(f"  Category instance counts:")
    for cat_id, name in sorted(cat_id_to_name.items()):
        print(f"    {name:30s}  train: {train_counts.get(cat_id, 0):5d}  val: {val_counts.get(cat_id, 0):5d}")
    print(f"\nNOTE: Set img_folder in your Hydra config to the directory DIRECTLY containing the image files.")
```

---

### `tests/__init__.py` (config, no data flow)

**Analog:** None needed — this is an empty file.

```python
# Empty file — required to make tests/ a Python package for pytest discovery.
# Pattern: all existing Python packages in this project use empty __init__.py files.
```

---

### `tests/conftest.py` (test fixture provider)

**Analog:** `test/test_io_utils.py` (style reference — note: existing tests use `unittest.TestCase`;
Phase 1 tests use pytest-style per `pyproject.toml [tool.pytest.ini_options]` config)

**Copyright / file header pattern** (lines 1–3 of analog):
```python
# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

"""Shared pytest fixtures for prepare_dataset tests."""
```

**pytest fixture pattern** (no analog in `test/` — uses unittest; pattern from RESEARCH.md §Pattern 6):
```python
import copy
import pytest

@pytest.fixture
def minimal_coco():
    """Minimal valid COCO dict with 1-based IDs and one annotated image."""
    return {
        "info": {},
        "licenses": [],
        "categories": [{"id": 1, "name": "scratch", "supercategory": ""}],
        "images": [
            {"id": 1, "file_name": "frame_001.jpg", "width": 640, "height": 480},
            {"id": 2, "file_name": "frame_002.jpg", "width": 640, "height": 480},
        ],
        "annotations": [
            {
                "id": 1, "image_id": 1, "category_id": 1,
                "bbox": [10, 20, 50, 60], "area": 3000,
                "segmentation": [[10, 20, 60, 20, 60, 80, 10, 80]], "iscrowd": 0,
            },
        ],
    }

@pytest.fixture
def zero_based_coco():
    """COCO dict with 0-based image, annotation, and category IDs (CVAT quirk)."""
    return {
        "info": {},
        "licenses": [],
        "categories": [{"id": 0, "name": "scratch", "supercategory": ""}],
        "images": [
            {"id": 0, "file_name": "frame_001.jpg", "width": 640, "height": 480},
            {"id": 1, "file_name": "frame_002.jpg", "width": 640, "height": 480},
        ],
        "annotations": [
            {
                "id": 0, "image_id": 0, "category_id": 0,
                "bbox": [10, 20, 50, 60], "area": 3000,
                "segmentation": [[10, 20, 60, 20, 60, 80, 10, 80]], "iscrowd": 0,
            },
        ],
    }

@pytest.fixture
def prefixed_fname_coco():
    """COCO dict with 'images/' prefix in file_name (standard CVAT export quirk)."""
    return {
        "info": {},
        "licenses": [],
        "categories": [{"id": 1, "name": "scratch", "supercategory": ""}],
        "images": [
            {"id": 1, "file_name": "images/frame_001.jpg", "width": 640, "height": 480},
        ],
        "annotations": [
            {
                "id": 1, "image_id": 1, "category_id": 1,
                "bbox": [10, 20, 50, 60], "area": 3000,
                "segmentation": [], "iscrowd": 0,
            },
        ],
    }

@pytest.fixture
def noncontiguous_cat_coco():
    """COCO dict with non-contiguous category IDs [1, 3, 7] (CVAT deleted-category quirk)."""
    return {
        "info": {},
        "licenses": [],
        "categories": [
            {"id": 1, "name": "cat_a", "supercategory": ""},
            {"id": 3, "name": "cat_b", "supercategory": ""},
            {"id": 7, "name": "cat_c", "supercategory": ""},
        ],
        "images": [
            {"id": 1, "file_name": "frame_001.jpg", "width": 640, "height": 480},
        ],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 1, "bbox": [0, 0, 10, 10], "area": 100, "segmentation": [], "iscrowd": 0},
            {"id": 2, "image_id": 1, "category_id": 3, "bbox": [0, 0, 10, 10], "area": 100, "segmentation": [], "iscrowd": 0},
            {"id": 3, "image_id": 1, "category_id": 7, "bbox": [0, 0, 10, 10], "area": 100, "segmentation": [], "iscrowd": 0},
        ],
    }
```

---

### `tests/test_prepare_dataset.py` (test, transform)

**Analog:** `test/test_io_utils.py`

**Copyright / file header pattern** (lines 1–3 of analog):
```python
# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

"""Unit tests for scripts/prepare_dataset.py repair logic and split behaviour."""
```

**Import pattern** — note: Phase 1 uses pytest-style (not `unittest.TestCase`) per `pyproject.toml`:
```python
# pyproject.toml [tool.pytest.ini_options]:
#   testpaths = ["tests"]
#   python_files = "test_*.py"
#   python_functions = "test_*"
# → use bare pytest functions, not TestCase classes

import copy
import sys
import os
import pytest

# Script under test lives in scripts/ (not a package) — import via sys.path manipulation:
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
import prepare_dataset  # noqa: E402
```

**Pytest function test pattern** (derived from analog's TestCase methods, translated to pytest):
```python
# Analog (test/test_io_utils.py lines 16-27) uses:
#   class TestFoo(unittest.TestCase):
#       def test_bar(self):
#           self.assertEqual(result, expected)
#
# Phase 1 equivalent (pytest-style per pyproject.toml config):
def test_repair_ids_zero_based(zero_based_coco):
    """Image, annotation, and category IDs are reindexed from 0-based to 1-based."""
    coco = copy.deepcopy(zero_based_coco)
    repaired = prepare_dataset.repair_ids(coco)
    assert all(img["id"] >= 1 for img in repaired["images"])
    assert all(ann["id"] >= 1 for ann in repaired["annotations"])
    assert all(cat["id"] >= 1 for cat in repaired["categories"])
```

**Assertion style** — `assert` (pytest) instead of `self.assert*` (unittest):
```python
# unittest style (existing analog):
self.assertEqual(result, ("frames", 480, 640))
self.assertIn("failed to load", str(ctx.exception))

# pytest equivalent (Phase 1 style):
assert result == ("frames", 480, 640)
assert "failed to load" in str(exc_info.value)
```

**Warning/print capture pattern** (pytest capsys — no analog in project; standard pytest):
```python
def test_warns_on_zero_annotation_images(minimal_coco, capsys):
    """Images with no annotations should trigger a warning."""
    coco = copy.deepcopy(minimal_coco)
    # Add an unannotated image
    coco["images"].append({"id": 99, "file_name": "empty.jpg", "width": 640, "height": 480})
    prepare_dataset.exclude_zero_annotation_images(coco)
    captured = capsys.readouterr()
    assert "Warning" in captured.out
    assert "empty.jpg" in captured.out
```

**sys.exit() testing pattern** (pytest raises):
```python
def test_validate_coco_exits_on_missing_keys(capsys):
    """Malformed input missing required keys should call sys.exit(1)."""
    bad_data = {"images": [], "annotations": []}  # missing "categories"
    with pytest.raises(SystemExit) as exc_info:
        prepare_dataset.validate_coco(bad_data, "test.json")
    assert exc_info.value.code == 1
    captured = capsys.readouterr()
    assert "categories" in captured.err
```

---

## Shared Patterns

### Copyright Header
**Source:** Every file in the project (`scripts/extract_odinw_results.py` line 1, `test/test_io_utils.py` line 1, `sam3/train/data/coco_json_loaders.py` line 1)
**Apply to:** All four new files
```python
# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved
```

### pyre-unsafe pragma
**Source:** `scripts/extract_odinw_results.py` line 3, `sam3/train/data/coco_json_loaders.py` line 3
**Apply to:** `scripts/prepare_dataset.py` (scripts-tier files use this pragma; test files do NOT — `test/test_io_utils.py` has no pyre pragma)
```python
# pyre-unsafe
```

### Warning via bare print()
**Source:** `scripts/extract_odinw_results.py` lines 64, 76–77
**Apply to:** All `print(f"Warning: ...")` calls in `scripts/prepare_dataset.py`
```python
print(f"Warning: ...")        # stdout — informational warnings
print(f"ERROR: ...", file=sys.stderr)  # stderr — fatal errors before sys.exit(1)
```

### json.load() read pattern
**Source:** `scripts/extract_odinw_results.py` line 68; `sam3/train/data/coco_json_loaders.py` lines 49–50
**Apply to:** `scripts/prepare_dataset.py` JSON loading
```python
# Both analogs use:
with open(json_path, "r") as f:
    data = json.load(f)
# (extract_odinw_results.py uses the shorter open() form — both are acceptable)
```

### No imports from `sam3/` in scripts/
**Source:** CONTEXT.md "no imports from sam3/ package required"; RESEARCH.md Anti-Patterns
**Apply to:** `scripts/prepare_dataset.py`
```
# scripts/ tier is standalone. Never import sam3.* in scripts/prepare_dataset.py.
# sam3 may not be installed in all environments where this script is used.
```

---

## No Analog Found

| File | Role | Data Flow | Reason |
|------|------|-----------|--------|
| `tests/__init__.py` | config | — | Empty file; no analog needed |
| `tests/conftest.py` | fixture provider | — | No `conftest.py` exists anywhere in the project; existing `test/` uses `unittest.TestCase`, not pytest fixtures. Pattern from RESEARCH.md §Pattern 6 + pytest docs applies instead. |

---

## Metadata

**Analog search scope:** `scripts/`, `test/`, `sam3/train/data/`, `pyproject.toml`
**Files scanned:** 5 (`extract_odinw_results.py`, `test_io_utils.py`, `coco_json_loaders.py`, `sam3_image_dataset.py` path noted, `pyproject.toml`)
**Pattern extraction date:** 2026-05-27
**Key constraint confirmed:** `pyproject.toml` `[tool.pytest.ini_options]` sets `testpaths = ["tests"]` — new tests must live in `tests/` (not `test/`), and use pytest-style functions (not `unittest.TestCase`), matching `python_functions = "test_*"`.
