<!-- refreshed: 2025-07-14 -->
# Architecture

**Analysis Date:** 2025-07-14

## System Overview

```text
┌────────────────────────────────────────────────────────────────────────────┐
│                        Public API / Entry Points                           │
│  `sam3/__init__.py`  `sam3/model_builder.py`  `sam3/model/sam3_base_predictor.py` │
├────────────────────┬──────────────────────────┬───────────────────────────┤
│    Image Model     │     Video Model (SAM3)    │  Multiplex (SAM3.1)       │
│  `model/sam3_image │  `model/sam3_video_       │  `model/sam3_multiplex_   │
│   .py`             │   inference.py`           │   tracking.py`            │
│  `model/sam3_image │  `model/sam3_video_       │  `model/video_tracking_   │
│   _processor.py`   │   predictor.py`           │   multiplex.py`           │
└────────────────────┴──────────────────┬────────┴────────────┬──────────────┘
                                         │                     │
          ┌──────────────────────────────┼─────────────────────┤
          ▼                              ▼                     ▼
┌─────────────────────┐   ┌─────────────────────┐  ┌─────────────────────┐
│  Detector Module    │   │   Tracker Module     │  │  Multiplex Memory   │
│  `model/sam3_image  │   │  `model/sam3_tracker │  │  `model/video_      │
│   .py`              │   │   _base.py`          │  │   tracking_         │
│  (SAM3VLBackbone +  │   │  `model/sam3_        │  │   multiplex.py`     │
│   Transformer +     │   │   tracking_          │  │                     │
│   Seg Head)         │   │   predictor.py`      │  │                     │
└──────┬──────────────┘   └──────┬───────────────┘  └────────────────────┘
       │                          │
       ▼                          ▼
┌────────────────────────────────────────────┐
│              Backbone Layer                │
│  ViT: `model/vitdet.py`                    │
│  Neck: `model/necks.py`                    │
│  VL Combiner: `model/vl_combiner.py`       │
│  Text Encoder: `model/text_encoder_ve.py`  │
│  Tokenizer: `model/tokenizer_ve.py`        │
└────────────────────────────────────────────┘
       │
       ▼
┌────────────────────────────────────────────┐
│           Transformer / Head Layer         │
│  Encoder: `model/encoder.py`               │
│  Decoder: `model/decoder.py`               │
│  Geometry Encoder: `model/geometry_        │
│    encoders.py`                            │
│  Seg Head: `model/maskformer_              │
│    segmentation.py`                        │
│  Memory: `model/memory.py`                 │
│  SAM Heads: `sam/mask_decoder.py`          │
│             `sam/prompt_encoder.py`        │
│             `sam/transformer.py`           │
└────────────────────────────────────────────┘
       │
       ▼
┌────────────────────────────────────────────┐
│           perflib (CUDA / Triton)          │
│  `perflib/iou.py`  `perflib/nms.py`        │
│  `perflib/masks_ops.py`                    │
│  `perflib/triton/`                         │
└────────────────────────────────────────────┘
```

## Component Responsibilities

| Component | Responsibility | File |
|-----------|----------------|------|
| `build_sam3_image_model` | Factory — assemble image model from sub-modules | `sam3/model_builder.py` |
| `build_sam3_video_model` | Factory — assemble dense video tracking model | `sam3/model_builder.py` |
| `build_sam3_predictor` | Factory — build predictor wrapper (multi-GPU) | `sam3/model_builder.py` |
| `build_tracker` | Factory — build tracker sub-module | `sam3/model_builder.py` |
| `Sam3Image` | Core image detection/segmentation nn.Module | `sam3/model/sam3_image.py` |
| `Sam3VideoInference` | Frame-by-frame video inference loop + state | `sam3/model/sam3_video_inference.py` |
| `Sam3VideoInferenceWithInstanceInteractivity` | Combines Detector + Tracker, det-trk association | `sam3/model/sam3_video_inference.py` |
| `Sam3BasePredictor` | Session management + request dispatch API | `sam3/model/sam3_base_predictor.py` |
| `Sam3VideoPredictor` | SAM3 single-GPU session predictor | `sam3/model/sam3_video_predictor.py` |
| `Sam3VideoPredictorMultiGPU` | Multi-GPU predictor using multiprocessing | `sam3/model/sam3_video_predictor.py` |
| `Sam3MultiplexVideoPredictor` | SAM3.1 predictor with bf16 + warm-up | `sam3/model/sam3_multiplex_video_predictor.py` |
| `Sam3MultiplexTrackerPredictor` | Hydra-config-based multiplex predictor | `sam3/model/sam3_multiplex_base.py` |
| `Sam3TrackerBase` / `Sam3TrackerPredictor` | Memory-based video object tracker | `sam3/model/sam3_tracker_base.py`, `sam3_tracking_predictor.py` |
| `SAM3VLBackbone` | Combines ViT visual neck + text encoder | `sam3/model/vl_combiner.py` |
| `ViT` | Vision Transformer backbone (ViTDet-style) | `sam3/model/vitdet.py` |
| `Sam3DualViTDetNeck` | Feature pyramid neck over ViT output | `sam3/model/necks.py` |
| `VETextEncoder` | CLIP-style text encoder | `sam3/model/text_encoder_ve.py` |
| `TransformerEncoderFusion` | Self+cross attention for image-text fusion | `sam3/model/encoder.py` |
| `TransformerDecoder` | Box-refining query decoder (DINO-style) | `sam3/model/decoder.py` |
| `SequenceGeometryEncoder` | Encodes points/boxes as geometric prompts | `sam3/model/geometry_encoders.py` |
| `UniversalSegmentationHead` | Pixel decoder + mask prediction | `sam3/model/maskformer_segmentation.py` |
| `MaskDecoder` / `PromptEncoder` | SAM-style interactive mask prediction | `sam3/sam/mask_decoder.py`, `sam3/sam/prompt_encoder.py` |
| `SimpleMaskEncoder` | Memory backbone — encodes past mask frames | `sam3/model/memory.py` |
| `MultiplexController` / `MultiplexState` | Batch multiplexing of multi-object tracking | `sam3/model/multiplex_utils.py` |
| `Trainer` | Distributed training loop (Hydra config driven) | `sam3/train/trainer.py` |
| `Sam3Processor` | High-level image inference helper | `sam3/model/sam3_image_processor.py` |
| `agent_core` | LLM-orchestrated segmentation agent | `sam3/agent/agent_core.py` |
| `perflib` | Triton/CUDA kernels for NMS, IoU, masks | `sam3/perflib/` |

## Pattern Overview

**Overall:** Modular deep-learning library (monolith package, no microservices). Architecturally follows the **Encoder–Decoder with cross-modal fusion** pattern (similar to GroundingDINO / Mask2Former / SAM2).

**Key Characteristics:**
- **Dual-path model**: Separate Detector (image grounding) and Tracker (temporal memory) modules that collaborate at inference time
- **Factory pattern**: All model construction is in `sam3/model_builder.py`; callers use `build_sam3_*` functions, never directly instantiate sub-modules
- **Session-based API**: `Sam3BasePredictor` manages inference sessions keyed by `session_id`, with request dispatch via `handle_request()`
- **Multiplex abstraction**: SAM3.1 uses `MultiplexState` to process N objects in parallel within fixed-size GPU buckets (avoiding per-object loop overhead)
- **Hydra configuration**: Training and eval configs are YAML files resolved via Hydra (`sam3/train/configs/`); `instantiate` wires up trainers and datasets
- **Activation checkpointing throughout**: Most heavyweight blocks expose `use_act_checkpoint` to trade compute for memory
- **`pyre-unsafe` annotations**: Facebook-internal static typing marker present in all files

## Layers

**Entry / API Layer:**
- Purpose: Public-facing builders and predictors
- Location: `sam3/__init__.py`, `sam3/model_builder.py`, `sam3/model/sam3_base_predictor.py`
- Contains: Factory functions, session lifecycle, request dispatch
- Depends on: Model layer
- Used by: End users, notebooks, scripts, agent

**Model Layer (Detection):**
- Purpose: Image grounding + segmentation
- Location: `sam3/model/sam3_image.py`, `sam3/model/vl_combiner.py`, `sam3/model/encoder.py`, `sam3/model/decoder.py`, `sam3/model/maskformer_segmentation.py`
- Contains: `nn.Module` classes for forward pass
- Depends on: Backbone layer, SAM head layer, perflib
- Used by: Video inference layer

**Model Layer (Tracking):**
- Purpose: Memory-based video object tracking
- Location: `sam3/model/sam3_tracker_base.py`, `sam3/model/sam3_tracking_predictor.py`, `sam3/model/sam3_video_base.py`, `sam3/model/sam3_video_inference.py`
- Contains: Tracker logic, inference state, det-trk association
- Depends on: Memory layer, SAM heads, backbone
- Used by: Video predictor

**Backbone Layer:**
- Purpose: Visual and language feature extraction
- Location: `sam3/model/vitdet.py`, `sam3/model/necks.py`, `sam3/model/text_encoder_ve.py`, `sam3/model/tokenizer_ve.py`
- Contains: ViT + FPN neck, CLIP-like text encoder
- Depends on: `timm`, `sam3/sam/rope.py`, perflib
- Used by: VL combiner → image model and tracker

**SAM Head Layer:**
- Purpose: Interactive prompt-based mask prediction (backward compatible with SAM1/SAM2)
- Location: `sam3/sam/mask_decoder.py`, `sam3/sam/prompt_encoder.py`, `sam3/sam/transformer.py`, `sam3/sam/rope.py`
- Contains: `MaskDecoder`, `PromptEncoder`, `TwoWayTransformer`
- Depends on: Core PyTorch only
- Used by: Tracker base, multiplex tracking

**Memory Layer:**
- Purpose: Encode and store past-frame masks for temporal conditioning
- Location: `sam3/model/memory.py`, `sam3/model/multiplex_utils.py`, `sam3/model/multiplex_mask_decoder.py`
- Contains: `SimpleMaskEncoder`, `CXBlock`, `SimpleFuser`, `MultiplexState`
- Depends on: Core PyTorch
- Used by: Tracker and multiplex tracker

**Performance Library:**
- Purpose: GPU-accelerated primitives
- Location: `sam3/perflib/`
- Contains: Triton NMS, connected components, IoU, fused ops, Flash Attention 3 wrapper
- Depends on: Triton, CUDA; optional (gated by `USE_PERFLIB` env var)
- Used by: Model layer and eval

**Training Layer:**
- Purpose: Dataset pipelines, losses, optimizer, distributed trainer
- Location: `sam3/train/`
- Contains: `Trainer`, datasets, loss functions, transforms, Hydra configs
- Depends on: Model layer, `hydra-core`, `submitit`, `fairscale`
- Used by: `sam3/train/train.py` entry point

**Evaluation Layer:**
- Purpose: Compute benchmark metrics
- Location: `sam3/eval/`
- Contains: COCO eval, HOTA, TETA, SACo VEval toolkits, postprocessors
- Depends on: `pycocotools`, internal model/inference utilities
- Used by: `scripts/eval/`

**Agent Layer:**
- Purpose: LLM-orchestrated multi-round segmentation
- Location: `sam3/agent/`
- Contains: `agent_core.py` (orchestrator), `client_llm.py` (LLM calls), `client_sam3.py` (SAM3 calls), helpers
- Depends on: Image model API, external LLM API
- Used by: `examples/sam3_agent.ipynb`

## Data Flow

### Image Inference (text-prompted)

1. User calls `Sam3Processor.set_image(image)` (`sam3/model/sam3_image_processor.py:42`)
2. Image is resized to 1008×1008 and normalized
3. `SAM3VLBackbone.forward_image()` runs ViT + neck → feature maps (`sam3/model/vl_combiner.py`)
4. User calls `set_text_prompt(state, prompt)` — tokenizer encodes text (`sam3/model/tokenizer_ve.py`)
5. `Sam3Image.forward()` fuses text+image via `TransformerEncoderFusion`, decodes queries via `TransformerDecoder` (`sam3/model/sam3_image.py`)
6. `SequenceGeometryEncoder` optionally encodes geometric prompts (boxes/points) (`sam3/model/geometry_encoders.py`)
7. `UniversalSegmentationHead` / `PixelDecoder` upsamples and produces mask logits (`sam3/model/maskformer_segmentation.py`)
8. NMS applied via `perflib/nms.py`, output returned as `{boxes, masks, scores}`

### Video Inference (SAM3)

1. `Sam3VideoPredictor.handle_request({type: "start_session", resource_path})` → `init_state()` loads frames (`sam3/model/sam3_video_inference.py:55`)
2. `add_prompt()` — runs Detector on the annotated frame; initializes Tracker's conditioning memory
3. `propagate_in_video()` — per-frame loop:
   a. Detector runs on current frame, produces new detections
   b. Tracker reads memory bank, predicts object masks
   c. `_associate_det_trk_compilable()` associates detections to tracks (`sam3/model/sam3_video_base.py`)
   d. Memory bank updated with current-frame masks
4. Masks streamed back frame-by-frame

### Multiplex (SAM3.1) Video Inference

1. `Sam3MultiplexVideoPredictor.handle_request()` dispatches to model (`sam3/model/sam3_multiplex_video_predictor.py`)
2. `Sam3MultiplexTracking.propagate_in_video()` processes N objects simultaneously using bucketed multiplex tensors
3. `MultiplexState.mux()` / `demux()` converts between flat-batch and bucket representation (`sam3/model/multiplex_utils.py`)
4. Multiplex mask decoder processes all objects in one GPU pass

### Training

1. `sam3/train/train.py` parses args → initializes Hydra config
2. Slurm/local launcher spawns distributed workers via `submitit`
3. Each worker instantiates `Trainer` from Hydra config (`sam3/train/trainer.py`)
4. `Trainer.run()` iterates data loaders → forward → `Sam3Loss.compute_loss()` → backward → optimizer step

## Key Abstractions

**`Sam3BasePredictor`:**
- Purpose: Shared session management and request dispatch for all predictor variants
- Location: `sam3/model/sam3_base_predictor.py`
- Pattern: Template method — subclasses override `remove_object`, `_get_session_stats` etc.

**`MultiplexState`:**
- Purpose: Converts between flat-batch tensors and bucket/slot tensors for multi-object parallel processing
- Location: `sam3/model/multiplex_utils.py`
- Pattern: Data mapper — `mux()` (data→bucket), `demux()` (bucket→data), `add_objects()`, `remove_objects()`

**`Sam3VideoBase` / `Sam3MultiplexBase`:**
- Purpose: Abstract base classes holding inference-state data structures and det-trk logic
- Location: `sam3/model/sam3_video_base.py`, `sam3/model/sam3_multiplex_base.py`
- Pattern: Template method / mixin

**`BatchedDatapoint`:**
- Purpose: Structured container for batched training/inference samples (images, text, boxes, masks)
- Location: `sam3/model/data_misc.py`, `sam3/train/data/collator.py`
- Pattern: Dataclass DTO

## Entry Points

**Package import:**
- Location: `sam3/__init__.py`
- Exports: `build_sam3_image_model`, `build_sam3_predictor`

**Model builders:**
- `build_sam3_image_model()` — `sam3/model_builder.py:573`
- `build_sam3_video_model()` — `sam3/model_builder.py:676`
- `build_sam3_video_predictor()` — `sam3/model_builder.py:817`
- `build_tracker()` — `sam3/model_builder.py:445`

**Training:**
- `sam3/train/train.py` — CLI entry point; uses `ArgumentParser` + Hydra + `submitit`

**Scripts:**
- `scripts/qualitative_test.py` — smoke-test for SAM3/SAM3.1 on a video
- `scripts/measure_speed.py` — benchmarking
- `scripts/eval/gold/eval_sam3.py` — gold evaluation runner

## Architectural Constraints

- **Threading:** Multi-GPU predictor (`Sam3VideoPredictorMultiGPU`) uses Python `multiprocessing` with one process per GPU; each process holds one model replica. Inter-process communication via queues.
- **Global state:** TF32 enabled at module import time in `model_builder.py:63` and `sam3_multiplex_base.py:37`. `perflib.is_enabled` is a module-level flag controlled by `USE_PERFLIB` env var.
- **Circular imports:** `sam3_video_predictor.py` imports `model_builder` inside `__init__` to break potential circular dependency.
- **CUDA requirement:** Most paths assume `cuda` device; `build_sam3_image_model` defaults to `cuda` if available, but `cpu` is supported for image model.
- **Activation checkpointing:** Most transformer blocks accept `use_act_checkpoint` — always use for training; off by default in some eval paths.
- **Checkpoint format:** `.pt` files keyed by `"model"` dict; `detector.*` keys for image weights, `tracker.*` keys for tracker weights. Loaded in `_load_checkpoint()` at `model_builder.py:539`.

## Anti-Patterns

### Direct instantiation of model sub-modules
**What happens:** Code outside `model_builder.py` directly constructs `Sam3Image(...)` with full component graphs.
**Why it's wrong:** Component wiring is complex and version-sensitive; bypassing the builder risks misconfigured models.
**Do this instead:** Always use `build_sam3_image_model()` or `build_sam3_video_model()` from `sam3/model_builder.py`.

### Accessing inference state dict keys directly
**What happens:** Callers index `inference_state["masks"]`, `inference_state["boxes"]` directly as plain dict keys.
**Why it's wrong:** State dict structure is internal and undocumented; keys may change.
**Do this instead:** Use the `Sam3Processor` or predictor API methods which return structured outputs.

## Error Handling

**Strategy:** Exceptions propagate naturally; no global error handler. Distributed training uses `torch.distributed` collective op timeouts (`SAM3_COLLECTIVE_OP_TIMEOUT_SEC`, default 180s).

**Patterns:**
- Missing checkpoint keys are logged with `print(f"Missing keys: {missing_keys}")` — not raised
- Invalid request types raise `RuntimeError(f"invalid request type: {request_type}")` in `Sam3BasePredictor.handle_request`
- `g_pathmgr` (iopath) used for all file I/O to support cloud paths

## Cross-Cutting Concerns

**Logging:** Custom logger via `sam3/logger.py` — `get_logger(__name__)` pattern used throughout. Training uses a richer `Logger` class in `sam3/train/utils/logger.py`.
**Validation:** Input validation minimal at model level; done at Processor/Predictor API boundary (e.g., image type checks in `Sam3Processor.set_image`).
**Authentication:** HuggingFace Hub (`hf_hub_download`) used for checkpoint download; no auth required for public facebook/sam3 and facebook/sam3.1 repos.

---

*Architecture analysis: 2025-07-14*
