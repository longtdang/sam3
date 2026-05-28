#!/usr/bin/env python3
"""
Generate a minimal synthetic COCO dataset for CI smoke-testing the end-to-end
training pipeline (VAL-01). No real data required.

Run from the sam3 project root:
    python scripts/generate_fake_dataset.py --out /tmp/fake_defect_data

Output:
    <out>/images/fake_0001.png ... fake_0005.png  (5 synthetic 64x64 RGB PNGs)
    <out>/train.json   (COCO format, all 5 images, 1 annotation each)
    <out>/val.json     (same as train.json — acceptable for smoke test)

Prints the absolute output path to stdout for shell capture:
    OUT=$(python scripts/generate_fake_dataset.py --out /tmp/fake_defect_data)
"""
import argparse
import json
import os

import numpy as np
from PIL import Image


def generate_fake_dataset(out_dir, n_images=5, img_size=64):
    """
    Create a minimal synthetic COCO dataset under out_dir.

    Args:
        out_dir:   Directory to create (created if not exists).
        n_images:  Number of fake images to generate (default 5).
        img_size:  Width and height of each image in pixels (default 64).

    Returns:
        Absolute path of out_dir.
    """
    out_dir = os.path.abspath(out_dir)
    img_dir = os.path.join(out_dir, "images")
    os.makedirs(img_dir, exist_ok=True)

    # All IDs are 1-based to match SAM3 COCO loader expectations.
    images = []
    annotations = []
    ann_id = 1
    margin = img_size // 4        # 16px for 64px images
    bbox_side = img_size // 2     # 32px wide/tall bounding box

    for i in range(1, n_images + 1):
        fname = f"fake_{i:04d}.png"
        # Random RGB pixels — content does not matter, only shape/format
        arr = np.random.randint(0, 256, (img_size, img_size, 3), dtype=np.uint8)
        Image.fromarray(arr).save(os.path.join(img_dir, fname))

        images.append({
            "id": i,
            "file_name": fname,
            "width": img_size,
            "height": img_size,
        })

        x0, y0 = margin, margin
        x1, y1 = x0 + bbox_side, y0 + bbox_side
        annotations.append({
            "id": ann_id,
            "image_id": i,
            "category_id": 1,
            "bbox": [x0, y0, bbox_side, bbox_side],       # [x, y, w, h]
            "area": bbox_side * bbox_side,
            "segmentation": [[x0, y0, x1, y0, x1, y1, x0, y1]],  # rectangle polygon
            "iscrowd": 0,
        })
        ann_id += 1

    coco = {
        "info": {"description": "Fake defect dataset for CI smoke testing"},
        "licenses": [],
        "categories": [{"id": 1, "name": "defect", "supercategory": "defect"}],
        "images": images,
        "annotations": annotations,
    }

    # Both splits contain all images — acceptable for smoke test (D-04-05)
    for split in ("train", "val"):
        split_path = os.path.join(out_dir, f"{split}.json")
        with open(split_path, "w") as f:
            json.dump(coco, f, indent=2)

    return out_dir


def main():
    parser = argparse.ArgumentParser(
        description="Generate a minimal synthetic COCO defect dataset for CI smoke testing."
    )
    parser.add_argument(
        "--out",
        required=True,
        help="Output directory (created if not exists). E.g. /tmp/fake_defect_data",
    )
    parser.add_argument(
        "--n-images",
        type=int,
        default=5,
        help="Number of fake images to generate (default: 5)",
    )
    parser.add_argument(
        "--img-size",
        type=int,
        default=64,
        help="Image size in pixels, width=height (default: 64)",
    )
    args = parser.parse_args()

    out = generate_fake_dataset(args.out, n_images=args.n_images, img_size=args.img_size)
    print(out)  # print path for shell capture: OUT=$(python ... --out ...)


if __name__ == "__main__":
    main()
