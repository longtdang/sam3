---
phase: "01-dataset-preparation"
plan: "01"
subsystem: "dataset-preparation"
tags: ["coco", "cvat", "dataset", "cli", "pytest", "fixtures"]
dependency_graph:
  requires: []
  provides:
    - "scripts/prepare_dataset.py — CLI entrypoint for CVAT COCO → SAM3 train/val split"
    - "tests/conftest.py — shared pytest fixtures for all Phase 1 tests"
  affects:
    - "Phase 2 Hydra config templates (consume train.json / val.json from --output dir)"
    - "Plan 02 tests (import conftest fixtures)"
tech_stack:
  added:
    - "scripts/prepare_dataset.py: stdlib only (argparse, collections, copy, json, os, random, sys)"
  patterns:
    - "Greedy multi-label stratified split (no sklearn dependency — handles small datasets)"
    - "ID remapping via dict (handles both 0-based reindex and non-contiguous gaps in one pass)"
    - "os.path.basename() for silent file_name prefix strip"
key_files:
  created:
    - "scripts/prepare_dataset.py (285 lines)"
    - "tests/__init__.py (1 line)"
    - "tests/conftest.py (91 lines)"
  modified: []
decisions:
  - "D-01: Stratified-by-category split using greedy multi-label algorithm"
  - "D-06/D-07: Silent basename strip for file_name prefix repair"
  - "D-10: Independent reindex per ID type — handles mixed 0/1-based CVAT exports"
  - "D-13: sys.exit(1) with stderr message on missing required COCO keys"
metrics:
  duration_seconds: 127
  completed_date: "2026-05-27"
  tasks_completed: 2
  tasks_total: 2
  files_created: 3
  files_modified: 0
---

# Phase 1 Plan 01: Dataset Preparation Script and Test Infrastructure Summary

**One-liner:** Standalone CVAT COCO → SAM3-ready train/val split CLI with stratified greedy split, 4 ID/filename repair functions, and 4 pytest fixtures for Phase 1 tests.

## Tasks Completed

| Task | Name | Commit | Files |
|------|------|--------|-------|
| 1 | Create test infrastructure | 09be481 | tests/__init__.py, tests/conftest.py |
| 2 | Write scripts/prepare_dataset.py | 19be012 | scripts/prepare_dataset.py |

## Files Created

| File | Lines | Purpose |
|------|-------|---------|
| `scripts/prepare_dataset.py` | 285 | CLI entrypoint — load, validate, repair, split, write, stats |
| `tests/__init__.py` | 1 | Empty package marker for pytest discovery |
| `tests/conftest.py` | 91 | Shared pytest fixtures for all Phase 1 tests |

## Functions Implemented (scripts/prepare_dataset.py)

| Function | Purpose | Decision |
|----------|---------|----------|
| `parse_args()` | CLI flags: `--ann-file`, `--img-folder`, `--output`, `--split-ratio`, `--seed` | D-05 |
| `validate_coco(data, path)` | `sys.exit(1)` with stderr message on missing COCO keys | D-13 |
| `repair_filenames(coco)` | Silent `os.path.basename()` strip of `file_name` directory prefix | D-06/D-07 |
| `repair_ids(coco)` | Reindex 0-based → 1-based; remap category IDs to contiguous 1..N | D-10 |
| `warn_orphan_annotations(coco)` | Warn + drop annotations referencing unknown `image_id` | D-12 |
| `exclude_zero_annotation_images(coco)` | Warn + return set of annotated image IDs | D-03 |
| `stratified_split(...)` | Greedy multi-label stratified split; warns on rare classes | D-01/D-02/D-04 |
| `filter_split(coco, keep_ids)` | Build output COCO dict for one split (bare filenames already in place) | D-08/D-09 |
| `print_stats(coco, train_ids, val_ids)` | Print total/per-split counts + per-category instance counts | D-14 |
| `main(args)` | Full pipeline orchestration; wraps file I/O in try/except | T-01-01 |

## Fixtures Implemented (tests/conftest.py)

| Fixture | Purpose |
|---------|---------|
| `minimal_coco` | Valid 1-based COCO dict; 2 images, 1 annotated — tests zero-annotation exclusion |
| `zero_based_coco` | All IDs start at 0 — tests CVAT 0-based ID repair |
| `prefixed_fname_coco` | `file_name` has `"images/"` prefix — tests filename repair |
| `noncontiguous_cat_coco` | Category IDs `[1, 3, 7]` — tests category contiguity repair |

## Threat Mitigations Applied

| Threat | Mitigation |
|--------|-----------|
| T-01-01: Malformed JSON | `try/except (OSError, json.JSONDecodeError)` + `sys.exit(1)` in `main()` |
| T-01-02: Output path safety | `os.makedirs(args.output, exist_ok=True)`; writes only `train.json`/`val.json` |

## Deviations from Plan

None — plan executed exactly as written. All function signatures, docstrings, and implementation patterns match the plan specification.

## Verification Commands Run

```bash
# All functions present
python3 -c "import sys; sys.path.insert(0,'scripts'); import prepare_dataset; ..."
# → All functions present

# CLI help
python3 scripts/prepare_dataset.py --help
# → shows --ann-file, --img-folder, --output, --split-ratio, --seed

# No sam3 imports
grep -v '^#' scripts/prepare_dataset.py | grep 'import sam3' | wc -l
# → 0

# sys.exit(1) present (D-13)
grep -c 'sys.exit(1)' scripts/prepare_dataset.py
# → 2  (in validate_coco and main)

# cat_id_map applied to both cats and annotations
grep -c 'cat_id_map' scripts/prepare_dataset.py
# → 3  (definition + cats list + annotations loop)
```

## Self-Check: PASSED
