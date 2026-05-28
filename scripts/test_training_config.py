#!/usr/bin/env python3
"""
Dry-run validation: confirm that all custom_finetune Hydra configs compose correctly
after Phase 3 changes (augmentation pipeline, val_epoch_freq, TensorBoard, eval metrics).

Run from the sam3 project root:
    python scripts/test_training_config.py

Tests:
  1. base.yaml: val_epoch_freq == 1, TensorBoard writer configured, ColorJitter/GaussianBlur/
     RandomErasingAPI in train_transforms, enable_segmentation == True, iou_type == "segm"
  2. finetune_strategy/decoder_only.yaml: inherits base, val_epoch_freq still 1
  3. finetune_strategy/full_finetune.yaml: inherits base, val_epoch_freq still 1

Uses the Hydra compose API with initialize_config_module("sam3.train"), matching
train.py exactly. Stubs out sam3 top-level __init__ to avoid requiring PyTorch >= 2.3.
Requires: hydra-core, omegaconf, numpy.
"""
import math
import os
import sys
import types

import numpy as np
import hydra.utils
from hydra import compose, initialize_config_module
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf

_SAM3_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _SAM3_ROOT)


def _make_package_stub(name: str, path: str) -> types.ModuleType:
    import importlib.util
    init_file = os.path.join(path, "__init__.py")
    spec = importlib.util.spec_from_file_location(name, init_file, submodule_search_locations=[path])
    m = types.ModuleType(name)
    m.__path__ = [path]  # type: ignore[assignment]
    m.__package__ = name
    m.__file__ = init_file
    m.__spec__ = spec
    return m


sys.modules.setdefault("sam3", _make_package_stub("sam3", os.path.join(_SAM3_ROOT, "sam3")))
sys.modules.setdefault("sam3.train", _make_package_stub("sam3.train", os.path.join(_SAM3_ROOT, "sam3", "train")))


def register_omegaconf_resolvers():
    OmegaConf.register_new_resolver("get_method", hydra.utils.get_method, replace=True)
    OmegaConf.register_new_resolver("get_class", hydra.utils.get_class, replace=True)
    OmegaConf.register_new_resolver("add", lambda x, y: x + y, replace=True)
    OmegaConf.register_new_resolver("times", lambda *a: np.prod(np.array(a)).item(), replace=True)
    OmegaConf.register_new_resolver("divide", lambda x, y: x / y, replace=True)
    OmegaConf.register_new_resolver("pow", lambda x, y: x**y, replace=True)
    OmegaConf.register_new_resolver("subtract", lambda x, y: x - y, replace=True)
    OmegaConf.register_new_resolver("range", lambda x: list(range(x)), replace=True)
    OmegaConf.register_new_resolver("int", lambda x: int(x), replace=True)
    OmegaConf.register_new_resolver("ceil_int", lambda x: int(math.ceil(x)), replace=True)
    OmegaConf.register_new_resolver("merge", lambda *x: OmegaConf.merge(*x), replace=True)
    OmegaConf.register_new_resolver("string", lambda x: str(x), replace=True)


def reset_hydra():
    GlobalHydra.instance().clear()


def compose_config(config_name: str) -> object:
    cfg = compose(config_name=config_name)
    OmegaConf.to_container(cfg, resolve=True, throw_on_missing=False)
    return cfg


def main():
    register_omegaconf_resolvers()
    reset_hydra()
    initialize_config_module("sam3.train", version_base="1.2")

    errors = []

    # --- Test 1: base.yaml ---
    try:
        reset_hydra()
        initialize_config_module("sam3.train", version_base="1.2")
        cfg_base = compose_config("configs/custom_finetune/base")

        # D-03-06 / EVAL-01: validation runs every epoch
        assert cfg_base.trainer.val_epoch_freq == 1, (
            f"Expected val_epoch_freq=1, got {cfg_base.trainer.val_epoch_freq}"
        )

        # D-03-07 / TRAIN-06: TensorBoard writer is configured
        assert cfg_base.trainer.logging.tensorboard_writer is not None, (
            "TensorBoard writer not configured"
        )
        assert cfg_base.trainer.logging.tensorboard_writer._target_ == (
            "sam3.train.utils.logger.make_tensorboard_logger"
        ), (
            f"Unexpected TensorBoard target: "
            f"{cfg_base.trainer.logging.tensorboard_writer._target_}"
        )

        # D-03-04 / TRAIN-04: augmentation entries in train_transforms ComposeAPI
        inner_targets = [
            t.get("_target_", "")
            for t in cfg_base.scratch.train_transforms[0].transforms
        ]
        assert any("ColorJitter" in t for t in inner_targets), (
            f"ColorJitter not in train_transforms. Found: {inner_targets}"
        )
        assert any("GaussianBlur" in t for t in inner_targets), (
            f"GaussianBlur not in train_transforms. Found: {inner_targets}"
        )
        assert any("RandomErasing" in t for t in inner_targets), (
            f"RandomErasing not in train_transforms. Found: {inner_targets}"
        )

        # CFG-05 / TRAIN-01: segmentation must be enabled
        assert cfg_base.scratch.enable_segmentation is True, (
            f"Expected enable_segmentation=True, got {cfg_base.scratch.enable_segmentation}"
        )

        # EVAL-02: iou_type must be segm, not bbox
        assert cfg_base.trainer.meters.val.custom.detection.iou_type == "segm", (
            f"Expected iou_type=segm, got "
            f"{cfg_base.trainer.meters.val.custom.detection.iou_type}"
        )

        print("✓ configs/custom_finetune/base")
    except Exception as e:
        errors.append(f"✗ configs/custom_finetune/base: {e}")
        print(f"✗ configs/custom_finetune/base: {e}")

    # --- Test 2: decoder_only.yaml inherits base changes ---
    try:
        reset_hydra()
        initialize_config_module("sam3.train", version_base="1.2")
        cfg_dec = compose_config(
            "configs/custom_finetune/finetune_strategy/decoder_only"
        )
        assert cfg_dec.trainer.val_epoch_freq == 1, (
            f"decoder_only: expected val_epoch_freq=1, got {cfg_dec.trainer.val_epoch_freq}"
        )
        print("✓ configs/custom_finetune/finetune_strategy/decoder_only")
    except Exception as e:
        errors.append(f"✗ configs/custom_finetune/finetune_strategy/decoder_only: {e}")
        print(f"✗ configs/custom_finetune/finetune_strategy/decoder_only: {e}")

    # --- Test 3: full_finetune.yaml inherits base changes ---
    try:
        reset_hydra()
        initialize_config_module("sam3.train", version_base="1.2")
        cfg_full = compose_config(
            "configs/custom_finetune/finetune_strategy/full_finetune"
        )
        assert cfg_full.trainer.val_epoch_freq == 1, (
            f"full_finetune: expected val_epoch_freq=1, got {cfg_full.trainer.val_epoch_freq}"
        )
        print("✓ configs/custom_finetune/finetune_strategy/full_finetune")
    except Exception as e:
        errors.append(f"✗ configs/custom_finetune/finetune_strategy/full_finetune: {e}")
        print(f"✗ configs/custom_finetune/finetune_strategy/full_finetune: {e}")

    # --- Final result ---
    if errors:
        print(f"\n{len(errors)} config(s) FAILED:")
        for err in errors:
            print(f"  {err}")
        sys.exit(1)
    else:
        print("\nAll configs parsed successfully.")


if __name__ == "__main__":
    main()
