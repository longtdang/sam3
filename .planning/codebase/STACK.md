# Technology Stack

**Analysis Date:** 2025-07-14

## Languages

**Primary:**
- Python 3.12+ — entire codebase (model, training, evaluation, agent)

**Minimum Supported:**
- Python 3.8 (per `pyproject.toml` classifiers), but README requires Python 3.12+

## Runtime

**Environment:**
- CUDA-capable GPU required (CUDA 12.6+)
- Conda environment recommended (`sam3` conda env, Python 3.12)

**Package Manager:**
- pip / setuptools
- Build backend: `setuptools>=61` with `wheel` (`pyproject.toml` line 1–3)
- No lockfile present (no `requirements.txt` or `pip.lock`)

## Frameworks

**Core Deep Learning:**
- **PyTorch** `>=2.7` (README specifies `torch==2.10.0`) — model training, inference, tensor ops
  - `torch.distributed` — multi-GPU distributed training (`sam3/train/utils/distributed.py`)
  - `torch.utils.tensorboard` — training metrics logging (`sam3/train/utils/logger.py`)
  - `torch.compile` — optional model compilation for performance (`sam3/perflib/compile.py`)
  - TensorFloat-32 enabled for Ampere GPUs (`sam3/model_builder.py:54–63`)
- **torchvision** — image transforms and utilities (`sam3/train/transforms/basic.py`)
- **timm** `>=1.0.17` — vision transformer backbones (ViT) (`pyproject.toml` dependency)
- **triton** — custom GPU kernels for sigmoid focal loss (`sam3/train/loss/sigmoid_focal_loss.py`)

**Training Infrastructure:**
- **Hydra** (`hydra-core`) — config management, experiment configuration (`sam3/train/train.py`, `sam3/train/trainer.py`)
- **OmegaConf** — config object handling (`sam3/train/train.py:38–41`)
- **submitit** — SLURM cluster job submission (`sam3/train/train.py`)
- **fairscale** — distributed training utilities (optional train dep)
- **fvcore** — training utilities including learning rate schedulers (`sam3/train/optim/optimizer.py`)
- **torchmetrics** — metric computation during training (`sam3/train/loss/loss_fns.py`)
- **tensorboard** — training visualization (`sam3/train/utils/logger.py`)

**Computer Vision:**
- **Pillow (PIL)** — image loading and manipulation throughout `sam3/model/` and `sam3/train/`
- **opencv-python (cv2)** — image/video processing (`sam3/agent/agent_core.py`, `sam3/visualization_utils.py`)
- **decord** — video loading (optional; `sam3/train/data/sam3_image_dataset.py`, `sam3/model/utils/sam2_utils.py`)
- **pycocotools** — COCO-format mask encoding/decoding (`sam3/train/masks_ops.py`, `sam3/agent/helpers/`)
- **scikit-image** — image processing utilities (notebooks + train extras)
- **scikit-learn** — KMeans clustering in visualization (`sam3/visualization_utils.py`)
- **einops** — tensor reshaping (notebooks optional dep)
- **scipy** — linear assignment/Hungarian algorithm (`sam3/train/matcher.py`, `sam3/eval/`)

**NLP / Text:**
- **ftfy** `==6.1.1` — Unicode text fixing for tokenization (`pyproject.toml` required dep)
- **regex** — text processing for tokenizer (`pyproject.toml` required dep)
- Custom CLIP-style tokenizer at `sam3/model/tokenizer_ve.py` using BPE vocab `sam3/assets/bpe_simple_vocab_16e6.txt.gz`

**Utilities:**
- **numpy** `>=1.26,<2` — array operations throughout
- **tqdm** — progress bars (`sam3/model/io_utils.py`, `sam3/train/train.py`)
- **iopath** `>=0.1.10` — filesystem abstraction (local + remote paths) (`sam3/train/utils/`)
- **typing_extensions** — backport typing features
- **zstandard** — compressed data decompression for training datasets (`sam3/train/data/sam3_image_dataset.py`)
- **numba** — JIT compilation for NMS helper (`sam3/train/nms_helper.py`)
- **python-rapidjson** — fast JSON parsing (dev dependency)
- **pandas** — data analysis scripts (`scripts/`)

**Optional Performance:**
- **flash-attn-3** (`flash_attn_interface`) — FlashAttention 3 for FP8 attention (`sam3/perflib/fa3.py`); installed separately from PyTorch wheels
- **cc_torch** (`github.com/ronghanghu/cc_torch`) — custom connected components CUDA op (`sam3/perflib/connected_components.py`)

**Testing:**
- **pytest** — test runner (`pyproject.toml` dev dep)
- **pytest-cov** — coverage reporting
- Test directory: `test/` (root), `sam3/perflib/tests/`

## Dev Tooling

**Formatting:**
- **black** `==24.2.0` — code formatter, line-length 88, targets py38–py312 (`pyproject.toml:95–97`)
- **ufmt** `==2.8.0` — unified formatter combining usort + ruff-api (`pyproject.toml:107–108`)
- **ruff-api** `==0.1.0` — ruff formatter backend for ufmt

**Import Sorting:**
- **usort** `==1.0.2` — import sorter (`pyproject.toml:104–105`)
- **isort** — configured with `profile = "black"` (`pyproject.toml:100–102`)

**Type Checking:**
- **mypy** — static type checker, targets Python 3.12, strict mode enabled (`pyproject.toml:110–128`)
  - Ignores missing imports for: `torch`, `torchvision`, `timm`, `numpy`, `PIL`, `tqdm`, `ftfy`, `regex`, `iopath`
- **pyre** — Meta's type checker (files annotated with `# pyre-unsafe`)

**Linting:**
- **ruff** — via ruff-api (Rust-based linter/formatter)

**VCS:**
- **gitpython** `==3.1.31` — git operations in dev scripts

**Notebooks:**
- **jupyter** / **notebook** / **ipywidgets** / **ipycanvas** / **ipympl** — interactive notebooks in `examples/`
- **yt-dlp** — YouTube video download for dev/testing
- **matplotlib** — visualization in notebooks

## CI/CD

**GitHub Actions:**
- Format check workflow: `.github/workflows/format.yml`
  - Runs on PRs to `main`
  - Checks `sam3/` and `scripts/` with `ufmt` (black `24.2.0` + usort `1.0.2`)
  - Runner: `ubuntu-latest`

## Build Configuration

- `pyproject.toml` — single source for all configuration (build, deps, tools)
- Package name: `sam3`, version from `sam3.__version__` (= `"0.1.0"` in `sam3/__init__.py`)
- Installed as editable (`pip install -e .`)
- Includes `sam3/assets/*.txt.gz` as package data
- Excludes `build*`, `scripts*`, `examples*` from package distribution

---

*Stack analysis: 2025-07-14*
