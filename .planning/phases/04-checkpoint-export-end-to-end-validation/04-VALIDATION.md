---
phase: 04
phase-slug: checkpoint-export-end-to-end-validation
date: 2025-05-28
---

# Phase 04 Validation Strategy

## Requirement Coverage

| Req ID | Description | Test Type | Verification Command | Wave |
|--------|-------------|-----------|----------------------|------|
| CKPT-01 | `best_checkpoint.pth` is written when AP50 improves | structural (grep + ast) | `grep -n "best_checkpoint.pth" sam3/train/trainer.py` and `python -c "import ast, pathlib; ast.parse(pathlib.Path('sam3/train/trainer.py').read_text())"` | 1 |
| CKPT-02 | `best_checkpoint.pth` loads via `build_sam3_image_model` | integration (script) | `python scripts/test_checkpoint_compatibility.py --checkpoint <path>` | 2 |
| VAL-01 | 1-epoch training on fake dataset completes without crash | smoke (script) | `python scripts/generate_fake_dataset.py --out /tmp/fake && python -m sam3.train.train ...` | 2 |
| VAL-02 | Same as CKPT-02 | integration (script) | Same script as CKPT-02 | 2 |

## Validation Notes

### CKPT-01 verification approach
CKPT-01 is verified structurally (grep + ast.parse) rather than via a pytest unit test. The `tests/test_checkpoint_export.py` unit test suggested in RESEARCH.md § Validation Architecture is intentionally deferred — it would require a mock training run to produce a real checkpoint file. The grep-based check that `best_checkpoint.pth` write logic is present in `trainer.py` is consistent with the project's existing verification pattern (see Phase 2 and Phase 3 script-based checks).

This gap is accepted. If a pytest unit test for the trainer patch is desired, it can be added in a future phase as an isolated test fixture task.

### CKPT-02 + VAL-02 verification approach
`scripts/test_checkpoint_compatibility.py` exercises the full `_load_checkpoint` code path:
- `weights_only=True` compatibility (export contains only tensors)
- `"model"` top-level key extraction
- `"detector."` prefix filter
- `not model.training` (eval mode)
- `len(model.state_dict()) > 0` (non-zero parameters loaded)

No forward pass: `Sam3Image.forward()` requires `BatchedDatapoint`; user approved substitution (D-04-03 revised 2025-05-28).

### VAL-01 verification scope
Per D-04-05: CI smoke test = 1-epoch dry run on fake data exits 0. `AP50 > 0` is a manual step on the real industrial defect dataset. The VALIDATION.md test for VAL-01 asserts no crash, not AP50 > 0.

## Test Gap Register

| Gap | Severity | Resolution |
|-----|----------|------------|
| `tests/test_checkpoint_export.py` unit test for CKPT-01 trainer patch | LOW | Intentionally deferred — grep+ast.parse structural check is sufficient for CI. Add in future phase if needed. |
| Real `AP50 > 0` assertion for VAL-01 | N/A | Manual step on real data (D-04-05 explicit). Not a CI gap. |
