# Coding Conventions

**Analysis Date:** 2025-05-26

## Naming Patterns

**Files:**
- Module files: `snake_case.py` (e.g., `sam3_video_predictor.py`, `model_misc.py`, `io_utils.py`)
- Test files: `test_<module>.py` (e.g., `test_io_utils.py`)
- Private/helper modules: prefixed with underscore via convention (e.g., `_base_dataset.py`, `_base_metric.py`, `_timing.py`)

**Functions:**
- All functions use `snake_case` (e.g., `get_logger`, `build_sam3_video_model`, `load_video_frames`)
- Private helpers prefixed with `_` (e.g., `_get_session`, `_extend_expiration_time`, `_find_free_port`)
- Static model utility functions named descriptively: `get_sdpa_settings`, `inverse_sigmoid`, `gen_sineembed_for_position`

**Variables:**
- All local and instance variables use `snake_case` (e.g., `session_id`, `inference_state`, `obj_id`)
- Module-level env-derived constants use `UPPER_SNAKE_CASE` (e.g., `IS_MAIN_PROCESS`, `RANK`, `IMAGE_EXTS`, `VIDEO_EXTS`)
- Module-level private constants prefixed with `_` (e.g., `_CLEAR_CACHE_THRESHOLD = 80`, `_PADDING_NUM = -1`, `_REMOVED_NUM = -1116`)

**Classes:**
- All classes use `PascalCase` (e.g., `Sam3VideoPredictor`, `TransformerDecoderLayer`, `ColoredFormatter`, `GradientClipper`)
- PyTorch `nn.Module` subclasses: `PascalCase` matching their functional role (e.g., `Sam3Image`, `MLP`, `RoPEAttention`)
- Dataclasses for configuration: `PascalCase` with `Conf` or descriptive suffix (e.g., `OptimAMPConf`, `OptimConf`, `BatchedDatapoint`)
- Abstract/base classes: `_BaseDataset`, `_BaseMetric` (underscore prefix + Base suffix)
- Test classes: `TestPascalCase` (e.g., `TestMasksToBoxes`, `TestLoadVideoFramesRouting`)

**Constants (Class-level):**
- Class-level constants use `UPPER_SNAKE_CASE` within classes:
  ```python
  # sam3/model/sam3_image.py
  class Sam3Image(torch.nn.Module):
      TEXT_ID_FOR_TEXT = 0
      TEXT_ID_FOR_VISUAL = 1
      TEXT_ID_FOR_GEOMETRIC = 2
  ```
  ```python
  # sam3/model/sam3_tracker_base.py
  NO_OBJ_SCORE = -1024.0
  ```

## Code Style

**Formatter:**
- `black` v24.2.0 with `line-length = 88`
- Target versions: Python 3.8–3.12
- Config in `pyproject.toml` under `[tool.black]`

**Import Sorter:**
- `usort` v1.0.2 (wrapper `ufmt` v2.8.0)
- Config in `pyproject.toml` under `[tool.usort]`
- `first_party_detection = false` — sam3 imports are treated as third-party by usort

**Linting:**
- `ruff-api` v0.1.0 via `ufmt`
- No standalone ruff config or `.eslintrc` equivalent present

**Type Checker:**
- `mypy` configured in `pyproject.toml` under `[tool.mypy]`
- Python version target: 3.12
- `warn_return_any = true`, `warn_unused_configs = true`
- `disallow_untyped_defs = true`, `disallow_incomplete_defs = true`
- However: virtually all source files carry `# pyre-unsafe` (Meta's Pyre type checker suppressor), meaning Pyre type-checking is disabled for all files. Mypy is enforced separately.

**CI enforcement:**
- `ufmt` formatting check runs on all PRs to `main` via `.github/workflows/format.yml`
- Checks `sam3/` and `scripts/` paths

## File Header Pattern

Every source file begins with:
```python
# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

# pyre-unsafe
```

The `# pyre-unsafe` marker appears on line 3 in virtually all `sam3/` source files, suppressing Meta's Pyre static type checker. The `test/test_io_utils.py` is the exception — it does **not** carry `# pyre-unsafe`.

## Import Organization

**Order (enforced by usort):**
1. Standard library imports (alphabetical): `import contextlib`, `import gc`, `import os`, `import time`
2. `from` stdlib imports: `from typing import ...`, `from collections import ...`
3. Third-party imports (alphabetical): `import numpy as np`, `import torch`, `from PIL import Image`
4. First-party `sam3.*` absolute imports: `from sam3.logger import get_logger`, `from sam3.model.model_misc import SAM3Output`
5. Relative imports: `from .act_ckpt_utils import ...`, `from .model_misc import (...)`

**Example from `sam3/model/decoder.py`:**
```python
import math
from functools import partial
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
import torch.nn.functional as torchF
from sam3.sam.rope import apply_rotary_enc, ...
from sam3.sam.transformer import RoPEAttention
from torch import nn, Tensor
from torch.nn.attention import sdpa_kernel, SDPBackend
from torchvision.ops.roi_align import RoIAlign

from .act_ckpt_utils import activation_ckpt_wrapper
from .box_ops import box_cxcywh_to_xyxy
from .model_misc import (
    gen_sineembed_for_position,
    ...
)
```

**Deferred imports:**
- Heavy or optional imports placed inside functions/methods (e.g., `from sam3.perflib.fa3 import flash_attn_func` inside a method body in `sam3/model/model_misc.py`, `from sam3.model_builder import build_sam3_video_model` inside `__init__` in `sam3/model/sam3_video_predictor.py`).

**Optional imports guarded with try/except:**
```python
# sam3/model/model_misc.py
try:
    import xformers
except ImportError:
    xformers = None
```

## Type Annotations

**Typing module used:** `from typing import Dict, List, Optional, Tuple, Union` (old-style compatible with Python 3.8)

**Return type annotations:**
- Public API methods in `sam3/model/sam3_base_predictor.py` are partially annotated (some methods have full signatures, others do not)
- `nn.Module` subclass `forward` methods often omit return annotations due to `# pyre-unsafe`
- Dataclasses in `sam3/train/trainer.py` and `sam3/train/data/sam3_image_dataset.py` are fully annotated

**Dataclass pattern:**
```python
# sam3/train/trainer.py
@dataclass
class OptimAMPConf:
    enabled: bool = False
    amp_dtype: str = "float16"
```

## Comment and Documentation Patterns

**Module-level docstrings:**
- Present on files with significant standalone purpose:
  ```python
  # sam3/model/decoder.py
  """
  Transformer decoder.
  Inspired from Pytorch's version, adds the pre-norm variant
  """
  ```
  ```python
  # sam3/model/sam3_base_predictor.py
  """
  Base predictor class shared by SAM3 and SAM3.1 (multiplex) video predictors.
  ...
  """
  ```

**Class docstrings:**
- Present on most public classes with a description of purpose:
  ```python
  # sam3/model/encoder.py
  class TransformerEncoderLayer(nn.Module):
      """
      Transformer encoder layer that performs self-attention followed by cross-attention.
      ...
      """
  ```

**Method docstrings:**
- Single-line docstrings for simple helpers: `"""Update last-use time for session expiration tracking."""`
- Multi-line Google-style with `Args:` block for complex public methods:
  ```python
  def __init__(
      self,
      activation: str,
      ...
  ):
      """
      Initialize a transformer encoder layer.

      Args:
          activation: Activation function to use in the feedforward network
          ...
      """
  ```

**Inline comments:**
- Used to explain non-obvious logic, especially CUDA/memory management:
  ```python
  # torch.cuda.empty_cache() forces a CUDA synchronization that stalls all
  # streams in the process. Calling it on every close_session produces visible
  # compute-utilization gaps when many sessions are active concurrently.
  ```
- Section separators with ASCII banners:
  ```python
  # ── Request dispatch ──────────────────────────────────────────────
  # ── Session management ────────────────────────────────────────────
  ```
- Reference to internal task IDs: `# D99228861 fix`, `# FIXME` (rare)

## Error Handling Patterns

**Raise patterns:**
- `RuntimeError` for invalid state or configuration: `raise RuntimeError(f"invalid request type: {request_type}")`
- `ValueError` for invalid argument values: `raise ValueError(f"...")`
- `NotImplementedError` for unsupported paths: `raise NotImplementedError(f"failed to load ...")`
- `TypeError` for unexpected types: `raise TypeError(f"Unknown source type: {type(source)}.")`
- `IndexError` for out-of-bounds: `raise IndexError(...)`

**Exception re-raising pattern:**
```python
# sam3/model/io_utils.py
try:
    ...
except Exception as e:
    raise NotImplementedError(
        f"extensionless path {video_path!r} failed to load as video ..."
    ) from e
```

**Assert-based validation (ML code):**
```python
assert all(isinstance(img_pil, Image.Image) for img_pil in resource_path)
assert img_np.dtype == np.uint8, "np.uint8 is expected for JPEG images"
assert len(gpus_to_use) > 0 and all(isinstance(i, int) for i in gpus_to_use)
```

**Warning pattern:**
```python
# sam3/model/model_misc.py
warnings.warn(
    "Flash Attention is disabled as it requires a GPU with Ampere (8.0) CUDA capability.",
    category=UserWarning,
    stacklevel=2,
)
```

## Logging Pattern

**Logger instantiation (module-level):**
```python
from sam3.logger import get_logger
logger = get_logger(__name__)
```

Used consistently at top of every module file that logs. The `get_logger` function in `sam3/logger.py` returns a colored console logger configurable via `LOG_LEVEL` env var.

**Log call pattern:**
```python
logger.info(f"started new session {session_id}")
logger.info(f"loading model on {self.rank_str} -- this could take a while ...")
```

All log messages use f-strings. No `%`-style formatting.

## Module Design / Exports

**Package `__init__.py` patterns:**
- Top-level `sam3/__init__.py`: exports only `build_sam3_image_model` and `build_sam3_predictor` from `model_builder`, plus `__version__ = "0.1.0"`
- Sub-package inits range from empty (just copyright/pyre-unsafe) to selective re-exports:
  ```python
  # sam3/sam/__init__.py
  from .mask_decoder import MaskDecoder
  from .prompt_encoder import PromptEncoder
  from .transformer import TwoWayTransformer
  ```
- `sam3/model/__init__.py`: empty (no re-exports); consumers import directly from submodules

**No barrel files** for the model or train subpackages — consumers do explicit deep imports like `from sam3.model.sam3_video_predictor import Sam3VideoPredictor`.

## Observed Anti-Patterns / Inconsistencies

**`# pyre-unsafe` on virtually all files:**
- Every file in `sam3/train/`, `sam3/perflib/`, and most of `sam3/model/` carries `# pyre-unsafe`, suppressing Pyre type-checking. This is a Meta-internal annotation not meaningful outside of Meta's Pyre infra — it is cargo-culted into the open-source release.

**Pytest config path mismatch:**
- `pyproject.toml` configures `testpaths = ["tests"]` but the actual test directory is `test/` (no `s`). Running `pytest` without explicit path argument will find no tests.

**Mixed test frameworks:**
- `test/test_io_utils.py` uses `unittest.TestCase`
- `sam3/perflib/tests/tests.py` uses pytest-style classes (inherits nothing) with `assert` statements and `torch.testing.assert_close`
- No consistent choice between unittest and pytest styles.

**Inline test function in non-test module:**
- `sam3/eval/coco_reindex.py` contains `def test_reindex_function()` called via `if __name__ == "__main__"` — a script-style test, not picked up by pytest.

**Duplicate import alias:**
- `sam3/visualization_utils.py` (referenced in scripts) uses `from PIL import Image as PIL_Image, ImageDraw` — `ImageDraw` is imported without alias while `Image` gets one, appearing in duplicate import lines in `scripts/qualitative_test.py`.

---

*Convention analysis: 2025-05-26*
