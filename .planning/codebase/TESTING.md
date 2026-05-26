# Testing Patterns

**Analysis Date:** 2025-05-26

## Test Framework

**Runner:**
- `pytest` (version unpinned; `pytest-cov` included in dev deps)
- Config: `pyproject.toml` under `[tool.pytest.ini_options]`

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]       # ⚠️ BUG: actual test dir is "test/" (no 's')
python_files = "test_*.py"
python_classes = "Test*"
python_functions = "test_*"
```

**Assertion Libraries:**
- `unittest.TestCase` assertions: `self.assertEqual`, `self.assertRaises`, `self.assertIn` (used in `test/test_io_utils.py`)
- `torch.testing.assert_close` for tensor comparisons (used in `sam3/perflib/tests/tests.py`)
- Plain `assert` for simple checks (pytest-style in `sam3/perflib/tests/tests.py`)

**Run Commands:**
```bash
# Run all tests (note: requires explicit path due to config bug)
pytest test/

# Run with coverage
pytest test/ --cov=sam3 --cov-report=html

# Run a specific test file
pytest test/test_io_utils.py

# Run perflib tests (not in standard test dir)
pytest sam3/perflib/tests/tests.py

# Run via installed dev deps
pip install -e ".[dev]"
pytest test/
```

## Test File Organization

**Location:**
- Primary test directory: `test/` (project root level)
- Embedded test: `sam3/perflib/tests/tests.py` (co-located with implementation)
- Script-style inline test: `sam3/eval/coco_reindex.py:test_reindex_function()` (not pytest-discoverable)

**⚠️ Critical Config Mismatch:**
The `pyproject.toml` specifies `testpaths = ["tests"]` (plural), but the actual directory is `test/` (singular). Running `pytest` with no arguments from the project root discovers **zero tests**. Always invoke `pytest test/` explicitly.

**Naming:**
- Test files: `test_<module_name>.py`
- Test classes: `Test<DescriptiveName>` (e.g., `TestLoadVideoFramesRouting`, `TestMasksToBoxes`)
- Test methods: `test_<behavior_description>` (e.g., `test_mp4_extension_routes_to_video_loader`, `test_extensionless_path_raises_on_decode_failure`)

**Directory structure:**
```
sam3/                          # package root
├── perflib/
│   └── tests/
│       └── tests.py           # pytest-style class, no unittest base
test/                          # primary test directory (root-level)
└── test_io_utils.py           # unittest.TestCase style
```

## Test Structure

**`test/test_io_utils.py` — unittest.TestCase style:**
```python
import unittest
from unittest.mock import MagicMock, patch
from sam3.model.io_utils import load_video_frames

class TestLoadVideoFramesRouting(unittest.TestCase):
    """Test that load_video_frames routes paths correctly based on extension."""

    @patch("sam3.model.io_utils.load_video_frames_from_video_file")
    def test_mp4_extension_routes_to_video_loader(
        self, mock_load_video: MagicMock
    ) -> None:
        """Paths with .mp4 extension should route to load_video_frames_from_video_file."""
        mock_load_video.return_value = ("frames", 480, 640)
        result = load_video_frames(
            video_path="/tmp/test_video.mp4",
            image_size=256,
            offload_video_to_cpu=True,
        )
        mock_load_video.assert_called_once()
        self.assertEqual(result, ("frames", 480, 640))
```

**`sam3/perflib/tests/tests.py` — pytest class style:**
```python
import pytest
import torch
from sam3.perflib.masks_ops import masks_to_boxes

class TestMasksToBoxes:
    def test_masks_box(self):
        def masks_box_check(masks, expected, atol=1e-4):
            out = masks_to_boxes(masks, [1 for _ in range(masks.shape[0])])
            assert out.dtype == torch.float
            torch.testing.assert_close(
                out, expected, rtol=0.0, check_dtype=True, atol=atol
            )
        ...
```

**Patterns:**
- No shared `setUp`/`tearDown` beyond individual test methods
- `tempfile.TemporaryDirectory()` used as context manager for filesystem tests
- No pytest fixtures (`@pytest.fixture`) used in either test file

## Mocking

**Framework:** `unittest.mock` (`patch`, `MagicMock`)

**Pattern (from `test/test_io_utils.py`):**
```python
from unittest.mock import MagicMock, patch

@patch("sam3.model.io_utils.load_video_frames_from_video_file")
def test_mp4_extension_routes_to_video_loader(
    self, mock_load_video: MagicMock
) -> None:
    mock_load_video.return_value = ("frames", 480, 640)
    result = load_video_frames(...)
    mock_load_video.assert_called_once()
    self.assertEqual(result, ("frames", 480, 640))
```

**Error-path mocking:**
```python
@patch("sam3.model.io_utils.load_video_frames_from_video_file")
def test_extensionless_path_raises_on_decode_failure(
    self, mock_load_video: MagicMock
) -> None:
    mock_load_video.side_effect = RuntimeError("Could not decode video")
    with self.assertRaises(NotImplementedError) as ctx:
        load_video_frames(video_path="oil://fb_permanent/corrupted_file", ...)
    self.assertIn("failed to load", str(ctx.exception))
    self.assertIn("oil://fb_permanent/corrupted_file", str(ctx.exception))
```

**What is mocked:**
- External I/O functions (`load_video_frames_from_video_file`, `load_video_frames_from_image_folder`)
- No GPU/CUDA mocking — tests that require torch tensors use CPU

**What is NOT mocked:**
- Actual model inference (no tests cover model forward passes)
- Filesystem for directory routing (uses real `tempfile.TemporaryDirectory`)

## Fixtures and Factories

**Test Data:**
- `sam3/perflib/tests/tests.py` loads fixture data from a sibling `assets/` directory:
  ```python
  assets_directory = os.path.join(
      os.path.dirname(os.path.abspath(__file__)), "assets"
  )
  mask_path = os.path.join(assets_directory, "masks.tiff")
  ```
- `test/test_io_utils.py` uses a special pattern `"<load-dummy-video-5>"` to generate dummy frames without real files:
  ```python
  frames, h, w = load_video_frames(
      video_path="<load-dummy-video-5>",
      image_size=64,
      offload_video_to_cpu=True,
  )
  self.assertEqual(frames.shape[0], 5)
  ```

**Location:**
- `sam3/perflib/tests/assets/masks.tiff` — fixture TIFF for mask-to-box tests

## Coverage

**Requirements:** Not explicitly enforced (no coverage threshold configured)

**`pytest-cov` is available** via dev dependencies but not configured in `pyproject.toml`:
```bash
pytest test/ --cov=sam3 --cov-report=html
pytest test/ --cov=sam3 --cov-report=term-missing
```

## Test Types

**Unit Tests:**
- `test/test_io_utils.py`: 8 unit tests for `load_video_frames` routing logic in `sam3/model/io_utils.py`
- `sam3/perflib/tests/tests.py`: 1 test method covering `masks_to_boxes` across 3 float dtypes

**Integration Tests:**
- None present

**E2E Tests:**
- `scripts/qualitative_test.py`: not a pytest test; a standalone script that downloads a real checkpoint from HuggingFace and runs end-to-end inference on a synthetic video. Must be run manually.

**Script-style Tests (not pytest-discoverable):**
- `sam3/eval/coco_reindex.py`: `test_reindex_function()` called via `if __name__ == "__main__"`

## Common Patterns

**Async Testing:**
- Not present in test suite

**Error Testing:**
```python
# unittest.TestCase style (test/test_io_utils.py)
with self.assertRaises(NotImplementedError) as ctx:
    load_video_frames(video_path="oil://corrupted", ...)
self.assertIn("failed to load", str(ctx.exception))
```

**Tensor Comparison:**
```python
# sam3/perflib/tests/tests.py
torch.testing.assert_close(out, expected, rtol=0.0, check_dtype=True, atol=1e-4)
```

**Dtype Parameterization (manual loop, no pytest.mark.parametrize):**
```python
for dtype in [torch.float16, torch.float32, torch.float64]:
    masks = torch.zeros((n_frames, height, width), dtype=dtype)
    masks_box_check(masks, expected)
```

## Test Quality Assessment

**Coverage is extremely sparse:**
- Total test lines: ~183 (61 in perflib tests + 122 in test_io_utils)
- Only 2 pytest-discoverable test files across the entire codebase
- No tests for: model forward passes, training loop, eval pipeline, agent/predictor integration, video propagation, or multiplex tracking

**Test infrastructure issues:**
1. **`testpaths = ["tests"]` in pyproject.toml is wrong** — directory is `test/` (no `s`). `pytest` discovers zero tests by default.
2. **Mixed test styles** — one file uses `unittest.TestCase`, the other uses bare pytest classes. No enforced convention.
3. **`sam3/perflib/tests/tests.py` is not in the configured testpaths**, so it only runs if explicitly invoked.

**Good test practices observed:**
- Tests for both happy path and error paths in `test_io_utils.py`
- Descriptive test method names explaining the expected behavior
- Correct use of `@patch` for isolating I/O
- Type annotations on test method parameters (`mock_load_video: MagicMock`, `-> None`)

---

*Testing analysis: 2025-05-26*
