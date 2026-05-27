#!/usr/bin/env python3
"""
Smoke test: validate that all custom_finetune Hydra configs compose without errors.

Run from the sam3 project root:
    python scripts/test_config_parse.py

Tests:
  1. base.yaml composes and resolves all interpolations
  2. finetune_strategy/decoder_only.yaml inherits base, lr_scale == 0.03 (informational),
     lr_vision_backbone == 2.5e-6 (near-frozen), and backbone LR is 10× lower than full_finetune
  3. finetune_strategy/full_finetune.yaml inherits base, lrd_vision_backbone == 0.9,
     lr_vision_backbone ≈ 2.5e-5

Uses the Hydra compose API with initialize_config_module("sam3.train"), matching
train.py exactly. Stubs out sam3 top-level __init__ to avoid requiring PyTorch ≥ 2.3.
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

# Stub out sam3 top-level package to prevent sam3/__init__.py from loading
# model_builder.py → decoder.py → torch.nn.attention (requires PyTorch ≥ 2.3).
# We only need sam3.train for Hydra config discovery, not the model code.
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
    """Register custom OmegaConf resolvers matching sam3/train/utils/train_utils.py.

    Inlined here so the smoke test runs without a complete sam3 model installation
    (train_utils imports torch + iopath which require PyTorch ≥ 2.3 and GPU deps).
    The resolver set is identical to the one registered by train.py at runtime.
    """
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
    """Clear Hydra global state so initialize_config_dir can be called cleanly."""
    GlobalHydra.instance().clear()


def compose_config(config_name: str) -> object:
    """Compose a config by name and trigger full interpolation resolution."""
    cfg = compose(config_name=config_name)
    # resolve=True triggers all ${...} interpolations; throw_on_missing=False
    # allows paths.* = null (REQUIRED at training time, not at parse time)
    OmegaConf.to_container(cfg, resolve=True, throw_on_missing=False)
    return cfg


def main():
    # Register custom OmegaConf resolvers (times, add, divide, etc.) before initializing Hydra
    register_omegaconf_resolvers()

    reset_hydra()
    initialize_config_module("sam3.train", version_base="1.2")

    errors = []

    # -------------------------------------------------------------------------
    # Test 1: base.yaml — full standalone config
    # -------------------------------------------------------------------------
    try:
        cfg_base = compose_config("configs/custom_finetune/base")
        # Verify segmentation is enabled
        assert cfg_base.scratch.enable_segmentation is True, (
            f"Expected enable_segmentation=True, got {cfg_base.scratch.enable_segmentation}"
        )
        # Verify SAM3 norm values (not ImageNet)
        assert list(cfg_base.scratch.train_norm_mean) == [0.5, 0.5, 0.5], (
            f"Expected train_norm_mean=[0.5,0.5,0.5], got {cfg_base.scratch.train_norm_mean}"
        )
        # Verify explicit literal LR (not via ${times:...} resolver — value must be 8e-5)
        assert math.isclose(cfg_base.scratch.lr_transformer, 8e-5, rel_tol=1e-6), (
            f"Expected lr_transformer=8e-5, got {cfg_base.scratch.lr_transformer}"
        )
        assert math.isclose(cfg_base.scratch.lr_vision_backbone, 2.5e-6, rel_tol=1e-6), (
            f"Expected lr_vision_backbone=2.5e-6, got {cfg_base.scratch.lr_vision_backbone}"
        )
        # Verify small-dataset epoch default
        assert cfg_base.scratch.max_data_epochs == 40, (
            f"Expected max_data_epochs=40, got {cfg_base.scratch.max_data_epochs}"
        )
        print("✓ custom_finetune/base")
    except Exception as e:
        errors.append(f"✗ custom_finetune/base: {e}")
        print(f"✗ custom_finetune/base: {e}")

    # -------------------------------------------------------------------------
    # Test 2: decoder_only.yaml — inherits base, lr_scale explicitly 0.03
    # -------------------------------------------------------------------------
    try:
        cfg_decoder = compose_config("configs/custom_finetune/finetune_strategy/decoder_only")
        # Verify lr_scale is still 0.03 (informational field — still present in config)
        assert math.isclose(cfg_decoder.scratch.lr_scale, 0.03, rel_tol=1e-6), (
            f"Expected lr_scale=0.03, got {cfg_decoder.scratch.lr_scale}"
        )
        # Verify inheritance: enable_segmentation is still True from base
        assert cfg_decoder.scratch.enable_segmentation is True, (
            f"decoder_only must inherit enable_segmentation=True from base, "
            f"got {cfg_decoder.scratch.enable_segmentation}"
        )
        # Key strategy assertion: backbone LR must be near-frozen at 2.5e-6 (the literal value).
        # This is ~10× lower than full fine-tune (2.5e-5) — verifies decoder-only truly near-freezes.
        assert math.isclose(cfg_decoder.scratch.lr_vision_backbone, 2.5e-6, rel_tol=1e-6), (
            f"Expected lr_vision_backbone=2.5e-6 (near-frozen decoder-only), "
            f"got {cfg_decoder.scratch.lr_vision_backbone}"
        )
        # Verify decoder-only backbone LR is LOWER than full fine-tune backbone LR (10× lower)
        cfg_full_check = compose_config("configs/custom_finetune/finetune_strategy/full_finetune")
        assert cfg_decoder.scratch.lr_vision_backbone < cfg_full_check.scratch.lr_vision_backbone, (
            f"decoder_only lr_vision_backbone ({cfg_decoder.scratch.lr_vision_backbone}) "
            f"must be lower than full_finetune ({cfg_full_check.scratch.lr_vision_backbone})"
        )
        ratio = cfg_full_check.scratch.lr_vision_backbone / cfg_decoder.scratch.lr_vision_backbone
        assert math.isclose(ratio, 10.0, rel_tol=1e-3), (
            f"Expected full_finetune/decoder_only backbone LR ratio ≈ 10×, got {ratio:.2f}×"
        )
        print("✓ custom_finetune/finetune_strategy/decoder_only")
    except Exception as e:
        errors.append(f"✗ custom_finetune/finetune_strategy/decoder_only: {e}")
        print(f"✗ custom_finetune/finetune_strategy/decoder_only: {e}")

    # -------------------------------------------------------------------------
    # Test 3: full_finetune.yaml — inherits base, overrides backbone LR + LLRD
    # -------------------------------------------------------------------------
    try:
        cfg_full = compose_config("configs/custom_finetune/finetune_strategy/full_finetune")
        # Verify CFG-03: LLRD enabled at 0.9
        assert math.isclose(cfg_full.scratch.lrd_vision_backbone, 0.9, rel_tol=1e-6), (
            f"Expected lrd_vision_backbone=0.9, got {cfg_full.scratch.lrd_vision_backbone}"
        )
        # Verify backbone LR raised 10× vs decoder-only (2.5e-6 → 2.5e-5)
        assert math.isclose(cfg_full.scratch.lr_vision_backbone, 2.5e-5, rel_tol=1e-6), (
            f"Expected lr_vision_backbone=2.5e-5, got {cfg_full.scratch.lr_vision_backbone}"
        )
        # Verify language backbone LR raised 10× (1.5e-6 → 1.5e-5)
        assert math.isclose(cfg_full.scratch.lr_language_backbone, 1.5e-5, rel_tol=1e-6), (
            f"Expected lr_language_backbone=1.5e-5, got {cfg_full.scratch.lr_language_backbone}"
        )
        # Verify inheritance: enable_segmentation still True
        assert cfg_full.scratch.enable_segmentation is True, (
            f"full_finetune must inherit enable_segmentation=True from base, "
            f"got {cfg_full.scratch.enable_segmentation}"
        )
        print("✓ custom_finetune/finetune_strategy/full_finetune")
    except Exception as e:
        errors.append(f"✗ custom_finetune/finetune_strategy/full_finetune: {e}")
        print(f"✗ custom_finetune/finetune_strategy/full_finetune: {e}")

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    if errors:
        print(f"\n{len(errors)} config(s) FAILED:")
        for err in errors:
            print(f"  {err}")
        sys.exit(1)
    else:
        print("\nAll configs parsed successfully.")


if __name__ == "__main__":
    main()
