---
phase: 2
slug: hydra-config-templates
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-05-27
---

# Phase 2 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | Custom Python smoke test (no pytest) |
| **Config file** | `scripts/test_config_parse.py` (new file — Wave 0) |
| **Quick run command** | `python scripts/test_config_parse.py` |
| **Full suite command** | `python scripts/test_config_parse.py` |
| **Estimated runtime** | ~5 seconds |

---

## Sampling Rate

- **After every task commit:** Run `python scripts/test_config_parse.py`
- **After every plan wave:** Run `python scripts/test_config_parse.py`
- **Before `/gsd-verify-work`:** Full suite must be green
- **Max feedback latency:** ~5 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 02-01-01 | 01 | 1 | CFG-01 | — | N/A | smoke | `python scripts/test_config_parse.py` | ❌ W0 | ⬜ pending |
| 02-02-01 | 02 | 1 | CFG-02 | — | N/A | smoke | `python scripts/test_config_parse.py` | ❌ W0 | ⬜ pending |
| 02-03-01 | 03 | 1 | CFG-03 | — | N/A | smoke | `python scripts/test_config_parse.py` | ❌ W0 | ⬜ pending |
| 02-04-01 | 04 | 2 | CFG-01,02,03 | — | N/A | smoke | `python scripts/test_config_parse.py` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `scripts/test_config_parse.py` — smoke test that composes all three configs using Hydra's compose API; verifies no exceptions, checks `lr_scale=0.03` and `lrd_vision_backbone=0.9` in composed output
- [ ] `sam3/train/configs/custom_finetune/base.yaml` — CFG-01 target (created in Plan 1)
- [ ] `sam3/train/configs/custom_finetune/finetune_strategy/decoder_only.yaml` — CFG-02 target (Plan 2)
- [ ] `sam3/train/configs/custom_finetune/finetune_strategy/full_finetune.yaml` — CFG-03 target (Plan 3)

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Only 3 fields need editing to run on a new dataset | CFG-04 | Structure/UX property, not runtime behavior | Read `base.yaml` — count fields without `# REQUIRED:` comment that a user must change |
| Every required field has inline comment | DOC-03 | Comment completeness — not machine-verifiable | Read `base.yaml` — verify every non-trivial field has an inline `#` comment |
| `enable_segmentation: true` drives all 6 dependent fields | CFG-05 | Cross-field consistency | `grep 'enable_segmentation' sam3/train/configs/custom_finetune/base.yaml` → should appear once in scratch and 5× as `${scratch.enable_segmentation}` |
| Norm values are `[0.5, 0.5, 0.5]` (not ImageNet) | CFG-06 | Visual inspection | `grep '0\.5' sam3/train/configs/custom_finetune/base.yaml` → 4 lines (train_norm_mean/std, val_norm_mean/std) |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 10s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
