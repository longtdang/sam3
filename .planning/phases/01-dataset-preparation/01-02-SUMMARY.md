---
phase: 01-dataset-preparation
plan: "02"
subsystem: dataset-preparation
tags: [testing, pytest, prepare-dataset, coco, unit-tests]
dependency-graph:
  requires: [01-01]
  provides: [DATA-01-test, DATA-02-test, DATA-03-test, DATA-04-test]
  affects: []
tech-stack:
  added: []
  patterns: [pytest-fixtures, capsys, monkeypatch, copy.deepcopy, sys.path-import]
key-files:
  created:
    - tests/test_prepare_dataset.py
  modified: []
decisions:
  - "Used copy.deepcopy() on all fixtures before passing to functions under test (T-02-01 threat mitigation)"
  - "Used monkeypatch.setattr(sys, 'argv', ...) for CLI test — no subprocess"
  - "Used capsys.readouterr() for stdout/stderr capture in test_malformed_input and test_stats_output"
metrics:
  duration: "53s"
  completed: "2026-05-27T04:08:09Z"
  tasks-completed: 1
  tasks-total: 1
  files-created: 1
  files-modified: 0
---

# Phase 1 Plan 02: Unit Tests for prepare_dataset.py Summary

**One-liner:** 7 pytest unit tests covering all DATA-01–DATA-04 requirements for `scripts/prepare_dataset.py` — ID reindex, filename prefix strip, category reindex, stratified split, CLI args, malformed input, and stats output.

## Tasks Completed

| # | Task | Commit | Files |
|---|------|--------|-------|
| 1 | Write tests/test_prepare_dataset.py — all 7 test functions | 213583f | tests/test_prepare_dataset.py |

## Test Results

```
================================================= test session starts ==================================================
platform linux -- Python 3.12.3, pytest-9.0.3, pluggy-1.6.0
collected 7 items

tests/test_prepare_dataset.py::test_id_reindex PASSED                 [ 14%]
tests/test_prepare_dataset.py::test_filename_prefix_strip PASSED      [ 28%]
tests/test_prepare_dataset.py::test_category_reindex PASSED           [ 42%]
tests/test_prepare_dataset.py::test_stratified_split PASSED           [ 57%]
tests/test_prepare_dataset.py::test_cli_args PASSED                   [ 71%]
tests/test_prepare_dataset.py::test_malformed_input PASSED            [ 85%]
tests/test_prepare_dataset.py::test_stats_output PASSED               [100%]

================================================== 7 passed in 0.03s ===================================================
```

**Full suite:** `pytest tests/ -v` → 7 passed, 0 failed, 0 errors.

## Requirement Coverage

| Test | Requirement | Verified Behavior |
|------|-------------|-------------------|
| test_id_reindex | DATA-02 | repair_ids() converts 0-based IDs to 1-based; annotation image_id refs updated |
| test_filename_prefix_strip | DATA-02 | repair_filenames() strips 'images/' prefix → bare filename |
| test_category_reindex | DATA-02 | repair_ids() remaps [1,3,7] → [1,2,3]; all annotation category_ids updated |
| test_stratified_split | DATA-01 | stratified_split() produces non-empty train+val; disjoint; all images assigned |
| test_cli_args | DATA-03 | parse_args() accepts --split-ratio=0.7 --seed=99; correct types |
| test_malformed_input | DATA-03 | validate_coco() raises SystemExit(1) with 'categories' in stderr |
| test_stats_output | DATA-04 | print_stats() outputs 'Total images processed' and category name 'scratch' |

## Deviations from Plan

None — plan executed exactly as written. All 7 test functions created verbatim from the plan specification.

## Known Stubs

None.

## Threat Flags

None — test file introduces no new network endpoints, auth paths, file access patterns, or schema changes.

## Self-Check: PASSED

- `tests/test_prepare_dataset.py` exists ✓
- Commit 213583f exists ✓
- 7 tests collected and all passing ✓
- `pytest tests/ -v` exits 0 ✓
