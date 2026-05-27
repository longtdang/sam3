# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

# pyre-unsafe

"""Shared pytest fixtures for prepare_dataset tests."""

import pytest


@pytest.fixture
def minimal_coco():
    """Minimal valid COCO dict with 1-based IDs and one annotated image."""
    return {
        "info": {},
        "licenses": [],
        "categories": [{"id": 1, "name": "scratch", "supercategory": ""}],
        "images": [
            {"id": 1, "file_name": "frame_001.jpg", "width": 640, "height": 480},
            {"id": 2, "file_name": "frame_002.jpg", "width": 640, "height": 480},
        ],
        "annotations": [
            {
                "id": 1, "image_id": 1, "category_id": 1,
                "bbox": [10, 20, 50, 60], "area": 3000,
                "segmentation": [[10, 20, 60, 20, 60, 80, 10, 80]], "iscrowd": 0,
            },
        ],
    }


@pytest.fixture
def zero_based_coco():
    """COCO dict with 0-based image, annotation, and category IDs (CVAT quirk)."""
    return {
        "info": {},
        "licenses": [],
        "categories": [{"id": 0, "name": "scratch", "supercategory": ""}],
        "images": [
            {"id": 0, "file_name": "frame_001.jpg", "width": 640, "height": 480},
            {"id": 1, "file_name": "frame_002.jpg", "width": 640, "height": 480},
        ],
        "annotations": [
            {
                "id": 0, "image_id": 0, "category_id": 0,
                "bbox": [10, 20, 50, 60], "area": 3000,
                "segmentation": [[10, 20, 60, 20, 60, 80, 10, 80]], "iscrowd": 0,
            },
        ],
    }


@pytest.fixture
def prefixed_fname_coco():
    """COCO dict with 'images/' prefix in file_name (standard CVAT export quirk)."""
    return {
        "info": {},
        "licenses": [],
        "categories": [{"id": 1, "name": "scratch", "supercategory": ""}],
        "images": [
            {"id": 1, "file_name": "images/frame_001.jpg", "width": 640, "height": 480},
        ],
        "annotations": [
            {
                "id": 1, "image_id": 1, "category_id": 1,
                "bbox": [10, 20, 50, 60], "area": 3000,
                "segmentation": [], "iscrowd": 0,
            },
        ],
    }


@pytest.fixture
def noncontiguous_cat_coco():
    """COCO dict with non-contiguous category IDs [1, 3, 7] (CVAT deleted-category quirk)."""
    return {
        "info": {},
        "licenses": [],
        "categories": [
            {"id": 1, "name": "cat_a", "supercategory": ""},
            {"id": 3, "name": "cat_b", "supercategory": ""},
            {"id": 7, "name": "cat_c", "supercategory": ""},
        ],
        "images": [
            {"id": 1, "file_name": "frame_001.jpg", "width": 640, "height": 480},
        ],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 1, "bbox": [0, 0, 10, 10], "area": 100, "segmentation": [], "iscrowd": 0},
            {"id": 2, "image_id": 1, "category_id": 3, "bbox": [0, 0, 10, 10], "area": 100, "segmentation": [], "iscrowd": 0},
            {"id": 3, "image_id": 1, "category_id": 7, "bbox": [0, 0, 10, 10], "area": 100, "segmentation": [], "iscrowd": 0},
        ],
    }
