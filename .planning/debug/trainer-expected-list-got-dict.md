---
status: resolved
slug: trainer-expected-list-got-dict
trigger: "AssertionError: Expected a list of batches, got <class 'dict'> in trainer.py:943 _run_step"
created: 2026-05-30
updated: 2026-05-30
---

## Symptoms

- **Expected:** Training runs normally
- **Actual:** Crashes with `AssertionError: Expected a list of batches, got <class 'dict'>`
- **Error location:** `trainer.py:943` in `_run_step`
- **Context:** Colab, single GPU, `gradient_accumulation_steps: 4`

## Root Cause

`trainer.py:943` asserts `isinstance(batch, list)` **only when `gradient_accumulation_steps > 1`**.
When `gradient_accumulation_steps=4`, the trainer expects the dataloader to yield a **list of 4 dicts** (one per accumulation step), produced by `collate_fn_api_with_chunking`.

However, the config uses `collate_fn_api` (plain, no chunking), which returns a **single dict**.
The chunking collator `collate_fn_api_with_chunking` is the one that splits a batch into `num_chunks` sub-batches and returns a list — matching what `_run_step` expects.

## Fix

**Option A — Recommended:** Change `gradient_accumulation_steps` to `1` in `base.yaml` so the assertion is skipped (the `else` branch wraps the single dict in a list itself):

```yaml
scratch:
  gradient_accumulation_steps: 1   # was 4
```

**Option B:** Switch the collate_fn to `collate_fn_api_with_chunking` with `num_chunks` matching `gradient_accumulation_steps`:

```yaml
scratch:
  collate_fn:
    _target_: sam3.train.data.collator.collate_fn_api_with_chunking
    _partial_: true
    num_chunks: ${scratch.gradient_accumulation_steps}   # 4
    dict_key: custom
    with_seg_masks: ${scratch.enable_segmentation}
    repeats: ${scratch.hybrid_repeats}
```

Option A is simpler for a small dataset (84+16 samples). Gradient accumulation is only needed when you need a larger effective batch size on memory-constrained hardware.

## Files Changed

- `sam3/train/configs/custom_finetune/base.yaml` — change `gradient_accumulation_steps`
