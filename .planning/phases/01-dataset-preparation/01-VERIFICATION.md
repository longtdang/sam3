---
phase: 01-dataset-preparation
verified: 2026-05-27T05:00:00Z
status: gaps_found
score: 4/5 must-haves verified
overrides_applied: 0
gaps:
  - truth: "--split-ratio and --seed CLI flags override defaults"
    status: partial
    reason: "--split-ratio correctly changes the split ratio (verified). --seed is accepted by the CLI and stored in args.seed but the rng = random.Random(seed) object is created and immediately discarded (# noqa: F841 — never used). Changing --seed from 42 to 99 produces identical output. ROADMAP SC-4 says the flag 'overrides the random seed default' but it has no behavioral effect."
    artifacts:
      - path: "scripts/prepare_dataset.py"
        issue: "Line 136: rng = random.Random(seed)  # noqa: F841 — created but never used in stratified_split(). The greedy sort is deterministic; rng is never called."
    missing:
      - "Either use rng for tie-breaking in stratified_split() so --seed produces different outputs, OR document that the split is deterministic by algorithm (making --seed a no-op by design) and update ROADMAP SC-4 wording to reflect this."
---

# Phase 1: Dataset Preparation Verification Report

**Phase Goal:** Any CVAT COCO export can be converted into clean, SAM3-compatible train/val JSON splits using a single CLI command.
**Verified:** 2026-05-27T05:00:00Z
**Status:** GAPS FOUND
**Re-verification:** No — initial verification

---

## Goal Achievement

### Observable Truths (ROADMAP Success Criteria)

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| SC-1 | Running the CLI produces `train.json` and `val.json` with no 0-based IDs and no broken `file_name` paths | ✓ VERIFIED | End-to-end smoke test: 5-image dataset → train: 3, val: 2; all IDs ≥ 1; no `/` in file_name |
| SC-2 | Script rejects malformed input with a clear error message (not a Python traceback) | ✓ VERIFIED | `validate_coco()` calls `sys.exit(1)` with stderr; `test_malformed_input` confirms; `grep 'sys.exit(1)' scripts/prepare_dataset.py` → 2 occurrences |
| SC-3 | Output includes a printed summary: total images, per-split count, per-category instance count | ✓ VERIFIED | `print_stats()` prints "Total images processed", per-split count, per-category train/val counts; `test_stats_output` confirms; smoke test output observed |
| SC-4 | `--split-ratio` and `--seed` CLI flags override the 80/20 default and random seed 42 default | ✗ PARTIAL | `--split-ratio` WORKS: 0.8 → 8 train/2 val; 0.7 → 7 train/3 val ✓. `--seed` INERT: `rng = random.Random(seed)` is created at line 136 but never used (`# noqa: F841`). Changing `--seed` from 42 to 99 on a 20-image dataset produces identical split output. |
| SC-5 | All categories present in source JSON appear in both split files (stratified split preserves rare classes) | ✓ VERIFIED | `filter_split()` uses `copy.deepcopy(coco["categories"])` — all categories copied to both splits regardless of instance count; smoke test shows 2 categories in both train/val splits |

**Score: 4/5 ROADMAP success criteria fully verified (SC-4 partial)**

---

## Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `scripts/prepare_dataset.py` | CLI entrypoint — all 10 functions | ✓ VERIFIED | 285 lines; all 10 functions present: `parse_args`, `validate_coco`, `repair_filenames`, `repair_ids`, `warn_orphan_annotations`, `exclude_zero_annotation_images`, `stratified_split`, `filter_split`, `print_stats`, `main` |
| `tests/__init__.py` | Empty package marker | ✓ VERIFIED | Exists; 1-line copyright + pyre-unsafe header |
| `tests/conftest.py` | 4 shared pytest fixtures | ✓ VERIFIED | 91 lines; all 4 fixtures: `minimal_coco`, `zero_based_coco`, `prefixed_fname_coco`, `noncontiguous_cat_coco` |
| `tests/test_prepare_dataset.py` | 7 unit tests, all passing | ✓ VERIFIED | 7 tests collected, 7 passed in 0.02s (pytest-9.0.3, Python 3.12.3) |

---

## Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `repair_ids()` image reindex | `annotations[].image_id` | `img_id_map` dict | ✓ WIRED | Line 86: `ann["image_id"] = img_id_map[ann["image_id"]]` — refs updated after image ID bump |
| `repair_ids()` category reindex | `annotations[].category_id` | `cat_id_map` dict | ✓ WIRED | Line 94: `ann["category_id"] = cat_id_map[ann["category_id"]]` — refs updated after sort+remap |
| `stratified_split()` output | `filter_split()` | `train_ids`, `val_ids` sets | ✓ WIRED | Lines 267–273: train/val ID sets passed directly to `filter_split(coco, set(train_ids))` |
| `tests/test_prepare_dataset.py` | `scripts/prepare_dataset.py` | `sys.path.insert` | ✓ WIRED | Line 15: `sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))` + `import prepare_dataset` |

---

## Data-Flow Trace (Level 4)

This phase produces a CLI tool, not a rendering component. Data flow verified via behavioral smoke test:

| Pipeline Step | Input | Transforms | Output | Status |
|--------------|-------|------------|--------|--------|
| Load → validate | COCO JSON file | `json.load()` + `validate_coco()` | in-memory dict or sys.exit(1) | ✓ FLOWING |
| Repair | Raw COCO dict | `repair_filenames()`, `repair_ids()` | Repaired dict (1-based IDs, bare filenames) | ✓ FLOWING |
| Split | Repaired image IDs | `stratified_split()` | `train_ids`, `val_ids` | ✓ FLOWING |
| Write | Filtered COCO dicts | `json.dump()` | `train.json`, `val.json` | ✓ FLOWING |

---

## Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| 7 tests pass | `python3 -m pytest tests/test_prepare_dataset.py -v` | 7 passed in 0.02s, exit 0 | ✓ PASS |
| End-to-end: produces train+val JSON | Smoke test with 5-image 0-based CVAT fixture | train: 3 images, val: 2 images | ✓ PASS |
| No 0-based IDs in output | Smoke test assertion: `all(i >= 1 for i in all_ids)` | True | ✓ PASS |
| No path prefix in file_name | Smoke test assertion: `all('/' not in img['file_name'] ...)` | True | ✓ PASS |
| Categories in both splits | Smoke test: `len(train['categories']) == 2, len(val['categories']) == 2` | True | ✓ PASS |
| `--split-ratio` changes output | 0.8 → 8 train/2 val; 0.7 → 7 train/3 val | Different outputs | ✓ PASS |
| `--seed` changes output | seed=42 vs seed=99 on 20-image dataset | **Identical outputs** | ✗ FAIL |
| Malformed input → exit(1) | `validate_coco({"images":[],"annotations":[]}, "x")` | SystemExit(1), "categories" in stderr | ✓ PASS |

---

## Requirements Coverage

| Requirement | Description | Test(s) | Status |
|-------------|-------------|---------|--------|
| DATA-01 | Stratified train/val split that preserves rare classes | `test_stratified_split`, SC-5 smoke test | ✓ SATISFIED |
| DATA-02 | CVAT quirk repair: 0-based IDs, file_name prefix, non-contiguous category IDs | `test_id_reindex`, `test_filename_prefix_strip`, `test_category_reindex` | ✓ SATISFIED |
| DATA-03 | CLI flags (`--ann-file`, `--img-folder`, `--output`, `--split-ratio`, `--seed`) + malformed input rejection | `test_cli_args`, `test_malformed_input` | ⚠️ PARTIAL (`--seed` accepted but inert) |
| DATA-04 | Dataset statistics summary printed to stdout | `test_stats_output` | ✓ SATISFIED |

---

## Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| `scripts/prepare_dataset.py` | 136 | `rng = random.Random(seed)  # noqa: F841` — created but never called | ⚠️ Warning | `--seed` CLI flag has no behavioral effect; ROADMAP SC-4 partially unmet |

No TODOs, FIXMEs, placeholder comments, or stub return values found in any Phase 1 files.

---

## Human Verification Required

None — all behaviors verifiable programmatically for this CLI-only phase.

---

## Gaps Summary

**1 gap blocking full ROADMAP success criteria: SC-4 (`--seed` inert)**

The `--split-ratio` flag works correctly and changes the split ratio as expected. However, the `--seed` flag is accepted by the CLI and stored in `args.seed` but is not used in the split algorithm.

In `stratified_split()` (line 136):
```python
rng = random.Random(seed)  # noqa: F841 — available for future tie-breaking use
```

The `rng` object is created but never called. The greedy multi-label algorithm sorts images deterministically (by category count) and assigns them greedily — there is no random sampling step where `rng` would be invoked. Running with `--seed 42` and `--seed 99` on the same dataset produces byte-for-byte identical output.

**Root cause:** The greedy algorithm (Decision D-01) is inherently deterministic and doesn't need a seed. The `--seed` parameter was included in the CLI specification anticipating random tie-breaking, but the current implementation doesn't exercise it.

**Impact:** Low for phase goal (splits are reproducible via deterministic algorithm). Medium for ROADMAP contract (SC-4 explicitly states `--seed` overrides the default seed of 42).

**Fix options:**
1. Use `rng.shuffle()` on `sorted_images` before the greedy pass so `--seed` produces visibly different splits
2. Document that `--seed` is reserved for a future randomized variant and update ROADMAP SC-4 to reflect the deterministic approach

---

## ROADMAP Flag Name Note

The ROADMAP example command (SC-1) shows `--input` and `--images`, but the implementation uses `--ann-file` and `--img-folder`. The PLAN frontmatter (01-01-PLAN.md) correctly specifies `--ann-file` and `--img-folder`. This is stale wording in ROADMAP.md only — not a code defect. Recommend updating the ROADMAP example command to match the actual CLI interface.

---

## Verdict

**PHASE NEARLY COMPLETE — 1 gap requiring closure before marking done.**

The core deliverable — a working CVAT COCO → SAM3 train/val split CLI — is **fully functional**. All 10 required functions are implemented and substantive. All 7 unit tests pass. End-to-end behavior is verified. Four of five ROADMAP success criteria are fully met.

The single gap is the `--seed` flag having no behavioral effect on split output, contrary to ROADMAP SC-4. This is a low-risk gap (splits are reproducible via deterministic algorithm) but represents an unfulfilled ROADMAP contract.

**Recommendation:** Apply a 2-line fix — add `rng.shuffle(sorted_images)` (or `sorted_images = rng.sample(sorted_images, len(sorted_images))`) before the greedy loop so `--seed` produces verifiably different orderings. This closes SC-4 without changing the greedy strategy for any given seed.

---

_Verified: 2026-05-27T05:00:00Z_
_Verifier: the agent (gsd-verifier)_
