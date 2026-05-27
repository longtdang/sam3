---
phase: 1
slug: dataset-preparation
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-05-27
---

# Phase 1 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest |
| **Config file** | `pyproject.toml` — `testpaths = ["tests"]`, `python_files = "test_*.py"` |
| **Quick run command** | `pytest tests/test_prepare_dataset.py -x` |
| **Full suite command** | `pytest tests/ -v` |
| **Estimated runtime** | ~5 seconds |

---

## Sampling Rate

- **After every task commit:** Run `pytest tests/test_prepare_dataset.py -x`
- **After every plan wave:** Run `pytest tests/ -v`
- **Before `/gsd-verify-work`:** Full suite must be green
- **Max feedback latency:** ~5 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------------|-----------|-------------------|-------------|--------|
| 1-01-01 | 01 | 1 | DATA-02 | ID repair idempotent — no data corruption | unit | `pytest tests/test_prepare_dataset.py::test_id_reindex -x` | ❌ W0 | ⬜ pending |
| 1-01-02 | 01 | 1 | DATA-02 | file_name prefix stripped to basename | unit | `pytest tests/test_prepare_dataset.py::test_filename_prefix_strip -x` | ❌ W0 | ⬜ pending |
| 1-01-03 | 01 | 1 | DATA-02 | Non-contiguous category IDs → contiguous 1-based | unit | `pytest tests/test_prepare_dataset.py::test_category_reindex -x` | ❌ W0 | ⬜ pending |
| 1-01-04 | 01 | 1 | DATA-01 | Stratified split produces non-empty train + val | unit | `pytest tests/test_prepare_dataset.py::test_stratified_split -x` | ❌ W0 | ⬜ pending |
| 1-01-05 | 01 | 1 | DATA-03 | `--split-ratio` and `--seed` CLI flags accepted | unit | `pytest tests/test_prepare_dataset.py::test_cli_args -x` | ❌ W0 | ⬜ pending |
| 1-01-06 | 01 | 1 | DATA-03 | Missing COCO keys → sys.exit(1) with clear message | unit | `pytest tests/test_prepare_dataset.py::test_malformed_input -x` | ❌ W0 | ⬜ pending |
| 1-02-01 | 02 | 1 | DATA-04 | Stats summary printed to stdout | unit | `pytest tests/test_prepare_dataset.py::test_stats_output -x` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `tests/__init__.py` — empty, marks as package
- [ ] `tests/conftest.py` — shared COCO fixture builders: `minimal_coco`, `zero_based_coco`, `prefixed_fname_coco`, `noncontiguous_cat_coco`
- [ ] `tests/test_prepare_dataset.py` — 7 test function stubs (red before implementation)

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Zero-annotation images excluded with printed warning | DATA-01 | Requires stdout inspection | Run script with fixture containing unannotated image; verify warning printed and image absent from output JSON |
| Rare class warning printed when only 1 image for a category | DATA-01 | Requires stdout inspection | Run script with 1-image category; verify warning printed |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 10s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
