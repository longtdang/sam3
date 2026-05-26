# External Integrations

**Analysis Date:** 2025-07-14

## Model Hosting & Distribution

**Hugging Face Hub:**
- Integration: `huggingface_hub.hf_hub_download` — downloads model checkpoints and config files on demand
- Repository: `facebook/sam3` and `facebook/sam3.1` on Hugging Face
- Auth: Requires authenticated access (gated repo); users must run `hf auth login` with a Hugging Face access token
- Usage: `sam3/model_builder.py` — `hf_hub_download(repo_id=..., filename=...)` for both config (`.yaml`) and checkpoint (`.pt`) files
- Env var: Uses `HF_TOKEN` / `HUGGING_FACE_HUB_TOKEN` standard hub auth (not explicitly set in code; relies on `huggingface_hub` library defaults)

## LLM / AI Agent Integration

**OpenAI-Compatible API:**
- Client: `openai.OpenAI` — used in agentic mode to call any OpenAI-compatible server
- File: `sam3/agent/client_llm.py`
- Default model: `meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8`
- Config: `server_url` and `api_key` passed at call time (not hardcoded)
- Supports multimodal (image + text) messages with base64 image encoding

**vLLM (direct inference):**
- Integration: `vllm.LLM` and `vllm.SamplingParams` — direct in-process LLM inference without an API server
- File: `sam3/agent/client_llm.py` (function `run_vllm_generate`)
- Usage: LLM instance is passed externally (initialized by caller); used for local GPU inference

## Filesystem & Storage

**iopath (`iopath.common.file_io.g_pathmgr`):**
- Abstraction layer over local and remote (S3/HDFS) file paths
- Used throughout training utilities: `sam3/train/utils/train_utils.py`, `sam3/train/utils/checkpoint_utils.py`, `sam3/train/utils/logger.py`, `sam3/model_builder.py`
- Enables checkpointing and data loading from cloud storage without code changes

## Compute Cluster / Distributed Training

**SLURM + submitit:**
- Integration: `submitit.AutoExecutor` — submits training jobs to SLURM clusters
- File: `sam3/train/train.py`
- Config: `submitit` block in Hydra YAML configs (partition, num GPUs, timeout, etc.)
- Supports both local single-GPU runs and multi-node SLURM cluster runs
- `sam3/train/utils/distributed.py` — PyTorch `torch.distributed` (NCCL backend) for multi-GPU communication

## Video Data / Media

**yt-dlp:**
- Purpose: Download YouTube videos for development/evaluation datasets
- Listed as dev dependency (`pyproject.toml`)

**decord:**
- Purpose: Fast video frame loading
- Files: `sam3/train/data/sam3_image_dataset.py`, `sam3/model/utils/sam2_utils.py`
- Optional import (falls back if not installed)

## Evaluation Datasets (External Benchmarks)

The codebase integrates evaluation toolkits for these external benchmarks:

**COCO / Open-Vocabulary Detection:**
- pycocotools — COCO mask/annotation format (`sam3/eval/coco_eval.py`, `sam3/train/`)
- OdinW-13 benchmark configs: `sam3/train/configs/odinw13/`
- Roboflow VL-100 benchmark configs: `sam3/train/configs/roboflow_v100/`

**YouTube-VIS (Video Instance Segmentation):**
- HOTA eval toolkit (vendored): `sam3/eval/hota_eval_toolkit/`
- TETA eval toolkit (vendored): `sam3/eval/teta_eval_toolkit/`
- YouTube-VIS dataset handling: `sam3/eval/ytvis_eval.py`

**SA-CO (Segment Anything with Concepts — Meta's proprietary dataset):**
- Gold/silver annotation eval: `sam3/eval/cgf1_eval.py`, `sam3/eval/saco_veval_eval.py`
- SAV (Segment Anything Video) evaluation configs: `sam3/train/configs/saco_video_evals/`

**TAO-OW (Open-World Tracking):**
- Dataset handler: `sam3/eval/hota_eval_toolkit/trackeval/datasets/tao_ow.py`

**Other Eval Datasets (silver image evals):**
- BDD100K, DROID, Ego4D, FathomNet, iNaturalist, NGA, SAV, YouTube-1B
- Configs: `sam3/train/configs/silver_image_evals/`

## Optional GPU Performance Libraries

**FlashAttention 3 (`flash_attn_interface`):**
- Purpose: FP8 attention for faster transformer inference on Hopper GPUs
- File: `sam3/perflib/fa3.py`
- Used via `torch.library.custom_op` wrapper (`flash::flash_attn_func`)
- Optional: installed separately with `pip install flash-attn-3 --no-deps`

**cc_torch (ronghanghu/cc_torch):**
- Purpose: Custom CUDA connected-components operation for mask processing
- File: `sam3/perflib/connected_components.py`
- Installed from GitHub: `pip install git+https://github.com/ronghanghu/cc_torch.git`
- Optional

## Monitoring & Experiment Tracking

**TensorBoard:**
- Integration: `torch.utils.tensorboard.SummaryWriter`
- File: `sam3/train/utils/logger.py`
- Logs training metrics (loss, learning rate, etc.) to TensorBoard event files
- Config: `tensorboard` is a `train` optional dependency

## CI/CD

**GitHub Actions:**
- Platform: GitHub (`.github/workflows/format.yml`)
- Trigger: Pull requests to `main` branch
- Action: `omnilib/ufmt@action-v1` — checks code formatting
- No deployment pipeline detected

## Secrets & Authentication

**Required credentials (not hardcoded — passed via environment or CLI):**
- `HF_TOKEN` / Hugging Face auth token — for downloading gated model checkpoints from `facebook/sam3`
- LLM API key — passed as `api_key` parameter to `sam3/agent/client_llm.py:send_generate_request()`
- SLURM credentials — managed by cluster environment via submitit

**No `.env` files or secret management libraries detected.**

## Webhooks & Callbacks

**Incoming:** None detected.
**Outgoing:** None detected (no webhook endpoints).

---

*Integration audit: 2025-07-14*
