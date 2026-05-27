# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

# pyre-unsafe

"""Unit tests for scripts/prepare_dataset.py repair logic and split behaviour."""

import collections
import copy
import os
import sys

import pytest

# Script under test lives in scripts/ (not an installable package)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
import prepare_dataset  # noqa: E402


def test_id_reindex(zero_based_coco):
    """Image, annotation, and category IDs are reindexed from 0-based to 1-based."""
    coco = copy.deepcopy(zero_based_coco)
    repaired = prepare_dataset.repair_ids(coco)
    assert all(img["id"] >= 1 for img in repaired["images"]), "image IDs must be >= 1"
    assert all(ann["id"] >= 1 for ann in repaired["annotations"]), "annotation IDs must be >= 1"
    assert all(cat["id"] >= 1 for cat in repaired["categories"]), "category IDs must be >= 1"
    # annotation image_id must reference a valid image id after repair
    valid_img_ids = {img["id"] for img in repaired["images"]}
    assert all(
        ann["image_id"] in valid_img_ids for ann in repaired["annotations"]
    ), "annotation image_id must match repaired image IDs"


def test_filename_prefix_strip(prefixed_fname_coco):
    """file_name 'images/frame_001.jpg' is stripped to 'frame_001.jpg'."""
    coco = copy.deepcopy(prefixed_fname_coco)
    repaired = prepare_dataset.repair_filenames(coco)
    assert repaired["images"][0]["file_name"] == "frame_001.jpg"
    assert "/" not in repaired["images"][0]["file_name"]


def test_category_reindex(noncontiguous_cat_coco):
    """Non-contiguous category IDs [1, 3, 7] are remapped to contiguous [1, 2, 3]."""
    coco = copy.deepcopy(noncontiguous_cat_coco)
    repaired = prepare_dataset.repair_ids(coco)
    cat_ids = sorted(cat["id"] for cat in repaired["categories"])
    assert cat_ids == [1, 2, 3], f"expected [1,2,3], got {cat_ids}"
    valid_cat_ids = set(cat_ids)
    assert all(
        ann["category_id"] in valid_cat_ids for ann in repaired["annotations"]
    ), "all annotation category_ids must reference repaired category IDs"


def test_stratified_split():
    """Stratified split produces non-empty train and val for a 10-image dataset."""
    # Build a minimal dataset: 10 images, 5 with cat_id=1, 5 with cat_id=2
    image_ids = list(range(1, 11))
    anns_by_image = collections.defaultdict(list)
    for i, img_id in enumerate(image_ids):
        cat_id = 1 if i < 5 else 2
        anns_by_image[img_id].append({"image_id": img_id, "category_id": cat_id})
    cat_id_to_name = {1: "cat_a", 2: "cat_b"}

    train_ids, val_ids = prepare_dataset.stratified_split(
        image_ids, anns_by_image, cat_id_to_name, split_ratio=0.8, seed=42
    )
    assert len(train_ids) > 0, "train split must not be empty"
    assert len(val_ids) > 0, "val split must not be empty"
    assert len(train_ids) + len(val_ids) == len(image_ids), "all images must be assigned"
    assert set(train_ids).isdisjoint(set(val_ids)), "no image should appear in both splits"


def test_cli_args(monkeypatch):
    """--split-ratio and --seed CLI flags are accepted and override defaults."""
    monkeypatch.setattr(
        sys, "argv",
        ["prepare_dataset.py",
         "--ann-file", "a.json",
         "--img-folder", "imgs/",
         "--output", "out/",
         "--split-ratio", "0.7",
         "--seed", "99"],
    )
    args = prepare_dataset.parse_args()
    assert args.split_ratio == 0.7
    assert args.seed == 99
    assert args.ann_file == "a.json"
    assert args.img_folder == "imgs/"
    assert args.output == "out/"


def test_malformed_input(capsys):
    """Missing required COCO keys → sys.exit(1) with clear stderr message."""
    bad_data = {"images": [], "annotations": []}  # missing "categories"
    with pytest.raises(SystemExit) as exc_info:
        prepare_dataset.validate_coco(bad_data, "test.json")
    assert exc_info.value.code == 1
    captured = capsys.readouterr()
    assert "categories" in captured.err, f"stderr should mention 'categories', got: {captured.err!r}"


def test_stats_output(minimal_coco, capsys):
    """print_stats outputs total images line and per-category instance counts."""
    coco = copy.deepcopy(minimal_coco)
    # image id=1 is in train, image id=2 has no annotation (excluded before this step)
    prepare_dataset.print_stats(coco, train_ids=[1], val_ids=[])
    captured = capsys.readouterr()
    assert "Total images processed" in captured.out, (
        f"stdout should contain 'Total images processed', got: {captured.out!r}"
    )
    assert "scratch" in captured.out, (
        f"stdout should contain category name 'scratch', got: {captured.out!r}"
    )
