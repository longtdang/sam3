# Phase 1: Dataset Preparation - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-05-27
**Phase:** 1-Dataset Preparation
**Areas discussed:** Split strategy, file_name repair, Output format alignment, Error handling style

---

## Split Strategy

**Pre-discussion clarification:** User asked whether fine-tuning targets one category or all categories. Answer provided: SAM3 fine-tuning is multi-class — all categories in the CVAT export are trained simultaneously in a single run.

| Option | Description | Selected |
|--------|-------------|----------|
| Fallback to random split for rare classes, print warning | Val may get 0 of that class | ✓ |
| Put sole instance in train only, warn | Simpler; val intentionally gets 0 | |
| Fail with clear error | Forces user to fix before splitting | |

**User's choice:** Fallback to random split for that rare class, print a warning — val may get 0 of that class.

| Option | Description | Selected |
|--------|-------------|----------|
| Exclude unannotated images with warning | Clean output, user knows | ✓ |
| Include in train only as hard negatives | | |
| Include in both splits proportionally | | |

**User's choice:** Exclude images with zero annotations from both splits, print a warning.

| Option | Description | Selected |
|--------|-------------|----------|
| No minimum per split, warn if val is empty | Simpler | ✓ |
| Enforce minimum 1 image per split, fail if too small | | |
| Let --split-ratio control it | | |

**User's choice:** No minimum — if val ends up with 0 images, just warn.

**Notes:** Decisions favour least-friction with clear feedback over strict validation.

---

## file_name Repair

| Option | Description | Selected |
|--------|-------------|----------|
| Strip any leading directory prefix automatically | Handles common CVAT "images/" prefix and others | ✓ |
| Strip only the exact "images/" prefix | More conservative | |
| Leave file_name untouched | User responsible for matching img_folder | |

**User's choice:** Strip any leading directory prefix automatically — safe for common CVAT case.
**Notes:** This repair is applied silently.

---

## Output Format Alignment

| Option | Description | Selected |
|--------|-------------|----------|
| Filename only (e.g. 'frame_001.jpg') | Matches SAM3/Roboflow pattern | ✓ |
| Relative path from --output dir | More portable but unusual | |
| Absolute path | Easiest to debug but not portable | |

**User's choice:** Filename only — matches the SAM3/Roboflow pattern where img_folder is set separately in config.

---

## Error Handling Style

| Option | Description | Selected |
|--------|-------------|----------|
| Auto-fix silently for known CVAT quirks, warn for unexpected | Least friction | ✓ |
| Always print what was fixed | Explicit audit trail | |
| Fail on any detected issue | Forces pre-cleaning | |

**User's choice:** Auto-fix silently for well-understood CVAT quirks (0-based IDs, file_name prefix), warn for unexpected.

| Option | Description | Selected |
|--------|-------------|----------|
| Fail with clear error listing missing required keys | | ✓ |
| Warn and skip malformed entry | | |
| Try to infer missing fields | | |

**User's choice:** Fail with a clear error message listing the missing required COCO keys.

---

## the agent's Discretion

None — all areas had explicit user decisions.

## Deferred Ideas

None — discussion stayed within phase scope.
