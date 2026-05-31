---
status: fixing
slug: trainer-hf-401-checkpoint
trigger: "GatedRepoError 401 — trainer tries to download checkpoint from HuggingFace instead of using local models/sam3_origin/sam3.pt"
created: 2026-05-30
updated: 2026-05-30
---

## Current Focus
hypothesis: config.yaml trainer.model block missing checkpoint_path and load_from_HF=false, causing build_sam3_image_model to attempt HF download
next_action: patch experiments/config.yaml with checkpoint_path and load_from_HF false

## Evidence
- timestamp: 2026-05-30T03:38:43
  observation: "GatedRepoError: 401 Client Error — Cannot access gated repo facebook/sam3. Access restricted."
  source: trainer traceback
- timestamp: 2026-05-30T03:38:43
  observation: "model_builder.py:646 — load_from_HF=True and checkpoint_path=None triggers download_ckpt_from_hf"
  source: model_builder.py
- timestamp: 2026-05-30T03:38:43
  observation: "models/sam3_origin/sam3.pt exists locally but is not referenced in config.yaml"
  source: filesystem

## Eliminated
- hypothesis: HuggingFace token issue (wrong token)
  reason: The checkpoint is already downloaded locally; the fix is to not contact HF at all

## Resolution
root_cause: experiments/config.yaml trainer.model block does not set checkpoint_path or load_from_HF=false
fix: Add checkpoint_path and load_from_HF=false to trainer.model in config.yaml
verification: Training run starts without HF network call
files_changed:
  - experiments/config.yaml
