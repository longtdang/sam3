# Codebase Structure

**Analysis Date:** 2025-07-14

## Directory Layout

```
sam3/                              # repo root
├── sam3/                          # Main Python package
│   ├── __init__.py                # Public API: build_sam3_image_model, build_sam3_predictor
│   ├── model_builder.py           # All model factory functions
│   ├── logger.py                  # get_logger() helper
│   ├── visualization_utils.py     # Visualization helpers
│   ├── model/                     # Core nn.Module definitions
│   │   ├── sam3_image.py          # Sam3Image — main detection/seg model
│   │   ├── sam3_image_processor.py# Sam3Processor — high-level image API
│   │   ├── sam3_base_predictor.py # Shared session + dispatch base class
│   │   ├── sam3_video_predictor.py# Sam3VideoPredictor / MultiGPU variant
│   │   ├── sam3_video_inference.py# Sam3VideoInference + WithInstanceInteractivity
│   │   ├── sam3_video_base.py     # Sam3VideoBase abstract + det-trk state
│   │   ├── sam3_tracker_base.py   # Sam3TrackerBase (memory attention)
│   │   ├── sam3_tracking_predictor.py # Sam3TrackerPredictor (interactive)
│   │   ├── sam3_tracker_utils.py  # Tracker utility functions
│   │   ├── sam3_multiplex_base.py # Sam3MultiplexTrackerPredictor (Hydra)
│   │   ├── sam3_multiplex_tracking.py # SAM3.1 multiplex tracking logic
│   │   ├── sam3_multiplex_video_predictor.py # User-facing SAM3.1 predictor
│   │   ├── sam3_multiplex_detector.py # Multiplex detector module
│   │   ├── sam3_multiplex_detector_utils.py # Detector helpers
│   │   ├── video_tracking_multiplex.py # VideoTrackingDynamicMultiplex
│   │   ├── video_tracking_multiplex_demo.py # Demo helper variant
│   │   ├── sam1_task_predictor.py # SAM3InteractiveImagePredictor (SAM1 compat)
│   │   ├── vl_combiner.py         # SAM3VLBackbone (vision + language combiner)
│   │   ├── vitdet.py              # ViT backbone (ViTDet-style, 1B-scale)
│   │   ├── necks.py               # Sam3DualViTDetNeck / Sam3TriViTDetNeck
│   │   ├── encoder.py             # TransformerEncoderFusion (image-text)
│   │   ├── decoder.py             # TransformerDecoder (DINO-style query decoder)
│   │   ├── geometry_encoders.py   # SequenceGeometryEncoder (points/boxes)
│   │   ├── maskformer_segmentation.py # PixelDecoder + UniversalSegmentationHead
│   │   ├── memory.py              # SimpleMaskEncoder, CXBlock, SimpleFuser
│   │   ├── multiplex_utils.py     # MultiplexState, MultiplexController
│   │   ├── multiplex_mask_decoder.py # MultiplexMaskDecoder
│   │   ├── model_misc.py          # MLP, DotProductScoring, MultiheadAttentionWrapper
│   │   ├── position_encoding.py   # PositionEmbeddingSine
│   │   ├── text_encoder_ve.py     # VETextEncoder (CLIP-style)
│   │   ├── tokenizer_ve.py        # SimpleTokenizer (BPE)
│   │   ├── box_ops.py             # Box coordinate utilities
│   │   ├── data_misc.py           # BatchedDatapoint, NestedTensor, FindStage
│   │   ├── io_utils.py            # load_resource_as_video_frames, IMAGE_EXTS
│   │   ├── act_ckpt_utils.py      # Activation checkpointing wrappers
│   │   ├── edt.py                 # Euclidean distance transform ops
│   │   └── utils/
│   │       ├── misc.py            # copy_data_to_device, etc.
│   │       ├── sam1_utils.py      # SAM1 compatibility helpers
│   │       └── sam2_utils.py      # load_video_frames, etc.
│   ├── sam/                       # SAM-heritage interactive heads
│   │   ├── mask_decoder.py        # MaskDecoder (predict masks from prompts)
│   │   ├── prompt_encoder.py      # PromptEncoder (points, boxes, masks)
│   │   ├── transformer.py         # TwoWayTransformer, RoPEAttention
│   │   ├── rope.py                # Rotary position embedding utilities
│   │   └── common.py              # LayerNorm2d
│   ├── train/                     # Training infrastructure
│   │   ├── train.py               # CLI entry point (ArgumentParser + Hydra + submitit)
│   │   ├── trainer.py             # Trainer class (full distributed training loop)
│   │   ├── matcher.py             # BinaryHungarianMatcherV2 (for loss)
│   │   ├── masks_ops.py           # RLE encode, mask ops for training
│   │   ├── nms_helper.py          # NMS helpers for training
│   │   ├── data/
│   │   │   ├── sam3_image_dataset.py  # Image dataset
│   │   │   ├── sam3_video_dataset.py  # Video dataset
│   │   │   ├── torch_dataset.py       # Base dataset wrapper
│   │   │   ├── coco_json_loaders.py   # COCO annotation loaders
│   │   │   └── collator.py            # BatchedDatapoint collation
│   │   ├── loss/
│   │   │   ├── sam3_loss.py           # Sam3Loss (top-level loss aggregation)
│   │   │   ├── loss_fns.py            # Individual loss functions
│   │   │   ├── mask_sampling.py       # Mask sampling strategies
│   │   │   └── sigmoid_focal_loss.py  # Focal loss
│   │   ├── optim/
│   │   │   ├── optimizer.py           # construct_optimizer, param groups
│   │   │   └── schedulers.py          # LR schedulers
│   │   ├── transforms/
│   │   │   ├── basic.py               # Standard augmentations
│   │   │   ├── basic_for_api.py       # Inference-time transforms
│   │   │   ├── filter_query_transforms.py
│   │   │   ├── point_sampling.py      # Point prompt sampling
│   │   │   └── segmentation.py        # Segmentation-specific transforms
│   │   ├── utils/
│   │   │   ├── checkpoint_utils.py    # load_state_dict_into_model, etc.
│   │   │   ├── distributed.py         # all_reduce_max, barrier, get_rank
│   │   │   ├── logger.py              # Logger, setup_logging
│   │   │   └── train_utils.py         # AverageMeter, ProgressMeter, makedir
│   │   └── configs/                   # Hydra YAML configs
│   │       ├── gold_image_evals/      # Gold benchmark eval configs
│   │       ├── silver_image_evals/    # Silver benchmark eval configs
│   │       ├── saco_video_evals/      # SACo/VEval video benchmark configs
│   │       ├── odinw13/               # ODinW-13 detection benchmark configs
│   │       └── roboflow_v100/         # Roboflow100 eval configs
│   ├── eval/                      # Evaluation toolkits
│   │   ├── coco_eval.py           # COCO mAP evaluation
│   │   ├── coco_eval_offline.py   # Offline COCO eval
│   │   ├── coco_writer.py         # Write COCO predictions
│   │   ├── cgf1_eval.py           # CGF1 metric
│   │   ├── saco_veval_eval.py     # SACo VEval evaluation runner
│   │   ├── saco_veval_evaluators.py # Per-category evaluators
│   │   ├── ytvis_eval.py          # YouTube-VIS evaluation
│   │   ├── postprocessors.py      # Output post-processing
│   │   ├── conversion_util.py     # Format conversion helpers
│   │   ├── demo_eval.py           # Demo evaluation helper
│   │   ├── hota_eval_toolkit/     # HOTA metric (embedded copy)
│   │   │   └── trackeval/         # TrackEval library subset
│   │   └── teta_eval_toolkit/     # TETA metric (embedded copy)
│   ├── agent/                     # LLM-orchestrated agent
│   │   ├── agent_core.py          # Main agent orchestration loop
│   │   ├── client_llm.py          # LLM API client
│   │   ├── client_sam3.py         # SAM3 inference client
│   │   ├── inference.py           # Inference helper
│   │   ├── viz.py                 # Agent visualization
│   │   └── helpers/               # Agent-specific utilities
│   │       ├── boxes.py, masks.py, keypoints.py
│   │       ├── memory.py, rle.py, roi_align.py
│   │       ├── color_map.py, visualizer.py
│   │       ├── mask_overlap_removal.py
│   │       ├── som_utils.py, zoom_in.py
│   │       └── rotated_boxes.py
│   └── perflib/                   # GPU performance library
│       ├── __init__.py            # USE_PERFLIB gate (env var)
│       ├── iou.py                 # IoU computation
│       ├── nms.py                 # Non-maximum suppression
│       ├── masks_ops.py           # Mask-level operations
│       ├── connected_components.py
│       ├── associate_det_trk.py   # Detection-tracking association
│       ├── compile.py             # torch.compile wrappers
│       ├── fa3.py                 # Flash Attention 3 bindings
│       ├── fused.py               # Fused CUDA ops
│       └── triton/                # Triton kernel implementations
│           ├── nms.py
│           └── connected_components.py
├── examples/                      # Jupyter notebooks
│   ├── sam3_image_predictor_example.ipynb
│   ├── sam3_image_batched_inference.ipynb
│   ├── sam3_image_interactive.ipynb
│   ├── sam3_video_predictor_example.ipynb
│   ├── sam3.1_video_predictor_example.ipynb
│   ├── sam3_for_sam1_task_example.ipynb
│   ├── sam3_for_sam2_video_task_example.ipynb
│   ├── sam3_agent.ipynb
│   ├── saco_gold_silver_eval_example.ipynb
│   ├── saco_gold_silver_vis_example.ipynb
│   ├── saco_veval_eval_example.ipynb
│   └── saco_veval_vis_example.ipynb
├── scripts/                       # Standalone utility scripts
│   ├── qualitative_test.py        # Smoke test (SAM3/SAM3.1)
│   ├── measure_speed.py           # Inference speed benchmark
│   ├── extract_odinw_results.py   # ODinW result extraction
│   ├── extract_roboflow_vl100_results.py
│   └── eval/
│       ├── standalone_cgf1.py     # Standalone CGF1 evaluation
│       ├── gold/eval_sam3.py      # Gold benchmark runner
│       ├── silver/                # Silver data download/prep scripts
│       └── veval/                 # VEval download/annotation scripts
├── test/                          # Tests
│   └── test_io_utils.py
├── assets/                        # Static assets
│   ├── images/                    # Test images (dog.gif, truck.jpg, etc.)
│   ├── videos/0001/               # Test video frames (JPEGs)
│   └── veval/toy_gt_and_pred/     # Toy eval fixtures (JSON)
├── pyproject.toml                 # Build system + dependencies + tool config
├── README.md                      # Project overview
├── README_TRAIN.md                # Training instructions
└── RELEASE_SAM3p1.md              # SAM3.1 release notes
```

## Directory Purposes

**`sam3/model/`:**
- Purpose: All neural network modules (`nn.Module` subclasses) and inference logic
- Contains: Image model, video model, tracker, multiplex tracker, backbones, heads, memory modules
- Key files: `sam3_image.py`, `sam3_video_inference.py`, `sam3_tracker_base.py`, `vl_combiner.py`, `vitdet.py`

**`sam3/sam/`:**
- Purpose: SAM-heritage interactive segmentation heads (forward-compatible with SAM1/SAM2 prompting)
- Contains: `MaskDecoder`, `PromptEncoder`, `TwoWayTransformer`, RoPE utilities
- Key files: `mask_decoder.py`, `prompt_encoder.py`, `transformer.py`

**`sam3/train/`:**
- Purpose: Everything needed to train the model: data, loss, optimizer, trainer, distributed utils
- Contains: Trainer loop, datasets (image + video), loss functions, transforms, Hydra YAML configs
- Key files: `train.py` (entry), `trainer.py` (loop), `loss/sam3_loss.py`, `data/collator.py`

**`sam3/eval/`:**
- Purpose: Benchmark evaluation — COCO, HOTA, TETA, SACo-VEval, YouTube-VIS
- Contains: Metric implementations, prediction writers, embedded copies of eval toolkits
- Key files: `coco_eval.py`, `saco_veval_eval.py`, `hota_eval_toolkit/`, `teta_eval_toolkit/`

**`sam3/agent/`:**
- Purpose: Multi-round LLM-guided segmentation agent that calls SAM3 as a tool
- Contains: Orchestration loop, LLM client, SAM3 client, visualization helpers
- Key files: `agent_core.py`, `client_llm.py`, `client_sam3.py`

**`sam3/perflib/`:**
- Purpose: High-performance GPU primitives replacing generic PyTorch ops where speed matters
- Contains: Triton kernels for NMS and connected components, CUDA IoU, Flash Attention 3 wrapper
- Key files: `nms.py`, `iou.py`, `masks_ops.py`, `triton/nms.py`

**`sam3/train/configs/`:**
- Purpose: Hydra configuration files for training experiments and evaluations
- Contains: YAML configs for ODinW, Roboflow, gold/silver image evals, SACo video evals
- Generated: No — hand-authored
- Committed: Yes

**`examples/`:**
- Purpose: Jupyter notebooks demonstrating usage of every predictor variant
- Contains: Image, video, interactive, agent, eval, and SACo visualization notebooks

**`scripts/`:**
- Purpose: CLI utilities for evaluation, benchmarking, and data preparation
- Contains: Eval runners, data download scripts, speed measurement

**`test/`:**
- Purpose: Unit tests
- Contains: `test_io_utils.py` (io utility tests)
- Note: `pyproject.toml` points pytest at `tests/` (note the trailing `s`); current test file is under `test/` — minor mismatch

**`assets/`:**
- Purpose: Static test media (images, video frames, eval fixtures)
- Contains: Sample images, 200+ test video frames, toy eval JSON files
- Generated: No — included in repo for test/demo purposes

## Key File Locations

**Entry Points:**
- `sam3/__init__.py`: Package public API (`build_sam3_image_model`, `build_sam3_predictor`)
- `sam3/model_builder.py`: All builder/factory functions
- `sam3/train/train.py`: Training CLI entry point

**Configuration:**
- `pyproject.toml`: Package metadata, dependencies, Black/ruff/mypy/pytest config
- `sam3/train/configs/`: Hydra YAML configs for training + eval runs

**Core Logic:**
- `sam3/model/sam3_image.py`: Image detection + segmentation (`Sam3Image`)
- `sam3/model/sam3_video_inference.py`: Video inference (`Sam3VideoInference`, `Sam3VideoInferenceWithInstanceInteractivity`)
- `sam3/model/sam3_multiplex_tracking.py`: SAM3.1 multiplex tracking
- `sam3/model/sam3_tracker_base.py`: Memory-based tracker core
- `sam3/model/vl_combiner.py`: VL backbone fusion
- `sam3/model/vitdet.py`: ViT backbone

**Testing:**
- `test/test_io_utils.py`: IO utility tests
- `sam3/perflib/tests/tests.py`: Perflib kernel tests

## Naming Conventions

**Files:**
- Snake_case throughout: `sam3_video_predictor.py`, `mask_decoder.py`
- Module grouping prefix: files in `model/` use `sam3_` prefix for top-level model files (`sam3_image.py`, `sam3_video_base.py`), no prefix for generic components (`encoder.py`, `decoder.py`, `memory.py`)
- Config YAML files: descriptive names with underscores, prefixed by task area (`sam3_gold_image_`, `saco_veval_`, `odinw_`)

**Directories:**
- Snake_case: `perflib/`, `hota_eval_toolkit/`, `teta_eval_toolkit/`
- Functional grouping: `model/`, `sam/`, `train/`, `eval/`, `agent/`, `perflib/`

**Classes:**
- PascalCase for `nn.Module` subclasses: `Sam3Image`, `Sam3TrackerBase`, `SAM3VLBackbone`, `ViT`
- Prefix convention: `Sam3*` for main model classes; `SAM3*` for legacy/interop classes; `Simple*` for lightweight components
- Predictors suffixed `*Predictor`: `Sam3VideoPredictor`, `Sam3TrackerPredictor`, `Sam3MultiplexVideoPredictor`

**Functions:**
- Snake_case: `build_sam3_image_model`, `_create_vit_backbone`
- Private factory helpers prefixed `_create_*`: `_create_vit_backbone`, `_create_transformer_encoder`
- Public builders prefixed `build_*`: `build_sam3_image_model`, `build_tracker`

## Where to Add New Code

**New model component (nn.Module):**
- Implementation: `sam3/model/<component_name>.py`
- Wire into model: `sam3/model_builder.py` (add `_create_<component>()` helper + integrate into `build_*` function)
- Export from package if public: `sam3/__init__.py`

**New predictor variant:**
- Implementation: `sam3/model/sam3_<variant>_predictor.py`
- Extend `Sam3BasePredictor` (`sam3/model/sam3_base_predictor.py`) for session management
- Builder: add `build_sam3_<variant>_predictor()` to `sam3/model_builder.py`

**New eval metric:**
- Implementation: `sam3/eval/<metric_name>_eval.py`
- Eval config: `sam3/train/configs/<category>/<name>.yaml`

**New training dataset:**
- Dataset class: `sam3/train/data/sam3_<dataset_name>_dataset.py`
- Config: `sam3/train/configs/`

**New data transform:**
- Implementation: `sam3/train/transforms/<transform_name>.py`
- Register in `sam3/train/transforms/__init__.py`

**New perflib kernel:**
- Triton kernel: `sam3/perflib/triton/<kernel_name>.py`
- Python wrapper: `sam3/perflib/<operation>.py`
- Guard with `perflib.is_enabled` check pattern

**New example notebook:**
- Location: `examples/<feature_name>_example.ipynb`

**New utility script:**
- Location: `scripts/<task_name>.py`

## Special Directories

**`sam3/perflib/tests/`:**
- Purpose: Tests for GPU kernels
- Generated: No
- Committed: Yes

**`sam3/eval/hota_eval_toolkit/` and `sam3/eval/teta_eval_toolkit/`:**
- Purpose: Embedded copies of external evaluation frameworks (HOTA, TETA)
- Generated: No — vendored/adapted
- Committed: Yes

**`assets/videos/0001/`:**
- Purpose: 200+ JPEG frames from a sample video for tests and demos
- Generated: No
- Committed: Yes

---

*Structure analysis: 2025-07-14*
