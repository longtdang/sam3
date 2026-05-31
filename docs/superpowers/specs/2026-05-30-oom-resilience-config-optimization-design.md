# SAM3 Training: OOM Resilience + Config Optimization

**Date:** 2026-05-30  
**Scope:** `sam3/train/trainer.py` + `sam3/train/configs/custom_finetune/base.yaml`  
**Target hardware:** 16 GB VRAM (RTX 3080 / V100 class)

---

## Problem

1. **Crash on CUDA OOM:** The training loop only catches `FloatingPointError`. A `RuntimeError: CUDA out of memory` is unhandled and terminates the entire training process, losing all unsaved progress.
2. **Config not hardened for 16 GB:** Several defaults leave unnecessary memory headroom on the table or expose the server to OOM on heavy batches.

---

## Goals

- Training continues after a CUDA OOM event (batch is skipped, cache is cleared, warning is logged).
- `base.yaml` is tuned conservatively for 16 GB VRAM without changing training semantics.
- No silent swallowing of unrelated `RuntimeError`s.

---

## Architecture

### Change 1 — `trainer.py`: OOM Recovery in `train_epoch()`

**Location:** `train_epoch()`, line ~895, inside the existing `try/except` block.

Add an `except RuntimeError` clause after `except FloatingPointError`:

```python
except RuntimeError as e:
    if "out of memory" in str(e).lower():
        logging.warning(
            f"[OOM] Epoch {self.epoch}, step {data_iter}: CUDA out of memory. "
            f"Skipping batch. Details: {e}"
        )
        self.optim.optimizer.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        gc.collect()
        continue
    raise e
```

**Key properties:**
- `set_to_none=True` — releases gradient tensor memory, not just zeroes it.
- `empty_cache()` returns CUDA cached blocks to the allocator.
- `gc.collect()` ensures Python-side tensor references are swept.
- Non-OOM `RuntimeError`s are re-raised immediately so bugs aren't silenced.
- `continue` skips the optimizer step, scheduler update, and logging for the failed batch.

### Change 2 — `base.yaml`: Config Hardening

| Field | Before | After | Rationale |
|---|---|---|---|
| `scratch.max_ann_per_img` | 200 | 100 | Halves per-step annotation tensor peak; 100 is generous for most custom datasets |
| `val.coco_json_loader.category_chunk_size` | 2 | 1 | Safer 16 GB default during eval; user can raise to 2–4 if GPU has headroom |
| `trainer.cuda` block | absent (code default) | Added with `cudnn_benchmark: true` | Makes GPU tuning visible and explicit in config |
| `use_caching` comment | vague | Annotated with RAM threshold guidance | Helps users decide when to enable |

---

## Error Handling

| Scenario | Behaviour |
|---|---|
| CUDA OOM during training step | Log warning, clear cache, skip batch, continue epoch |
| Non-OOM `RuntimeError` | Re-raise immediately (no silent swallow) |
| `FloatingPointError` (NaN/Inf) | Existing behaviour unchanged (re-raise) |
| OOM during eval (`val_epoch`) | Not in scope — eval already uses `val_batch_size: 1` and `empty_gpu_mem_cache_after_eval` |

---

## Testing

- Verify: inject a mock OOM `RuntimeError` in `_run_step` and assert training continues.
- Verify: non-OOM `RuntimeError` still propagates.
- Verify: config fields load without Hydra validation errors after changes.

---

## Out of Scope

- OOM recovery during `val_epoch` (val batch size is already 1; risk is low).
- Gradient checkpointing (separate trade-off, separate change).
- Dynamic batch-size reduction on OOM.
