# Plan 05-02 Summary: Troubleshooting Section + State Updates

**Status:** Complete
**Completed:** 2026-05-28
**Commit:** 0df6c80

## Files Modified

- `FINE_TUNING.md` — Troubleshooting section appended (108 additional lines; total: 380 lines)
- `.planning/STATE.md` — Phase 5 marked complete, D-05-xx decisions appended
- `.planning/ROADMAP.md` — Phase 5 progress 2/2 ✅, DOC-01/DOC-02 closed

## Troubleshooting Gotchas Written

| # | Topic | Evidence |
|---|-------|----------|
| 1 | `enable_segmentation` off → missing segmentation metrics | base.yaml:20 + 5 downstream fields |
| 2 | ImageNet norms vs SAM3 `[0.5, 0.5, 0.5]` | base.yaml:49-52, Sam3Processor internals |
| 3 | 0-based CVAT IDs | scripts/prepare_dataset.py:63-96 `repair_ids()` |
| 4 | `file_name` path prefix collision | scripts/prepare_dataset.py:56-60 `repair_filenames()` |
| 5 | Masks loss commented out in upstream configs | base.yaml:205-212 `Masks` entry |

## Verification Results

- `## Troubleshooting` section present ✅
- Exactly 5 `###` gotcha headings ✅
- `**Symptom:** / **Cause:** / **Fix:**` labels: 15 (≥ 15) ✅
- All 5 topics covered ✅
- No ellipses in code blocks ✅
- STATE.md Phase 5 complete ✅
- ROADMAP.md DOC-01/DOC-02 closed ✅

## Deviations

None — all content written exactly as specified in plan.
