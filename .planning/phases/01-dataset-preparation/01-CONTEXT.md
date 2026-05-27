# Phase 1: Dataset Preparation - Context

**Gathered:** 2026-05-27
**Status:** Ready for planning

<domain>
## Phase Boundary

Deliver a single CLI script (`scripts/prepare_dataset.py`) that reads a CVAT COCO export (one JSON file + an images folder), repairs known CVAT quirks, performs a stratified train/val split, and writes two SAM3-compatible output JSON files. All categories in the dataset are preserved and processed together — this pipeline is multi-class, not per-class.

</domain>

<decisions>
## Implementation Decisions

### Split Strategy
- **D-01:** Use stratified-by-category split for all classes simultaneously. Images are assigned to splits based on per-category representation across all categories present.
- **D-02:** If a category has only 1 annotated image (rare class), fall back to including it in random split assignment and print a warning. Val may receive 0 instances of that class — this is acceptable.
- **D-03:** Images with zero annotations are excluded from both train and val splits. Print a warning listing the excluded image filenames and count.
- **D-04:** No minimum image count per split is enforced. If val ends up empty, warn the user but do not fail.
- **D-05:** Default split ratio is 80/20 (train/val). `--split-ratio` flag overrides. Default seed is 42; `--seed` flag overrides.

### file_name Repair
- **D-06:** Strip any leading directory prefix from `file_name` automatically (e.g. `"images/frame_001.jpg"` → `"frame_001.jpg"`). This handles the standard CVAT export pattern without requiring user intervention.
- **D-07:** This repair is applied silently (no output) as it is a well-understood CVAT quirk. Only unexpected patterns generate warnings.

### Output Format
- **D-08:** After all repairs, `file_name` values in output JSON contain filename only (no directory component). This matches the SAM3/Roboflow pattern where the `img_folder` is set separately in the Hydra config.
- **D-09:** Output files are written to `--output` directory as `train.json` and `val.json`.

### ID Repair
- **D-10:** If image IDs or annotation IDs are 0-based, reindex to 1-based automatically and silently (known CVAT quirk). Category IDs are also reindexed to be contiguous and 1-based if needed.

### Error Handling
- **D-11:** Known CVAT quirks (0-based IDs, `file_name` prefix) are auto-fixed silently.
- **D-12:** Unexpected issues (e.g. annotations referencing image IDs not in the `images` list) generate a warning but do not fail the script.
- **D-13:** Malformed input (missing required top-level COCO keys: `images`, `annotations`, `categories`) causes an immediate, clear error message listing the missing keys. No Python traceback — use `sys.exit(1)` with a human-readable message.

### Script Output / Stats
- **D-14:** After writing the output files, always print a summary: total images processed, images per split, and instance count per category in each split.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Existing COCO Data Infrastructure
- `sam3/train/data/coco_json_loaders.py` — existing COCO loader the training pipeline uses; output JSON must be compatible with `load_coco_and_group_by_image()` and `COCO_FROM_JSON` loader
- `sam3/train/data/sam3_image_dataset.py` — dataset class that consumes the JSON; understand `img_folder` + `json_file` config pattern

### Research Findings
- `.planning/research/DATASET_INTEGRATION.md` — CVAT COCO export structure, SAM3 loader requirements, known quirks, Roboflow config analogue
- `.planning/research/FINETUNING_STRATEGIES.md` — fine-tuning strategy context (decoder-only vs full fine-tune)

### Planning Reference
- `.planning/ROADMAP.md` §Phase 1 — success criteria and plan descriptions
- `.planning/REQUIREMENTS.md` — DATA-01 through DATA-04 requirements

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `sam3/train/data/coco_json_loaders.py` — `load_coco_and_group_by_image()` shows exactly what fields SAM3 expects; the output JSONs from `prepare_dataset.py` must produce files that pass through this loader cleanly
- `scripts/extract_odinw_results.py` — example of an existing scripts-level Python file; follow the same code style

### Established Patterns
- **img_folder + json_file pattern:** SAM3 training configs separate the image directory from the annotation file. `file_name` in JSON should be filename-only; `img_folder` in YAML config points to the images directory. The script's output must follow this convention.
- **pycocotools:** Already a project dependency (`sam3/eval/coco_eval.py` uses it). Can be used in tests for validation.

### Integration Points
- Output `train.json` / `val.json` flow directly into Phase 2 Hydra configs via the `img_file` / `img_folder` config fields.
- The `--output` directory becomes the `dataset_root` in Phase 2 config templates.

</code_context>

<specifics>
## Specific Ideas

- Multi-class pipeline: all categories in the CVAT export are processed together in a single run. There is no per-class mode.
- The script is a standalone CLI tool in `scripts/prepare_dataset.py` — no imports from `sam3/` package required.
- Unit tests cover the three repair cases: ID reindex, prefix strip, category reindex — using minimal fixture JSON files (not real images).

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope.

</deferred>

---

*Phase: 1-Dataset Preparation*
*Context gathered: 2026-05-27*
