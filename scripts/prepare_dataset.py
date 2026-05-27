# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

# pyre-unsafe

"""
Prepares a CVAT COCO export for SAM3 fine-tuning.

Repairs known CVAT quirks, performs a stratified train/val split, and writes
two SAM3-compatible JSON files.

Usage:
    python3 scripts/prepare_dataset.py \\
        --ann-file /path/to/instances_default.json \\
        --img-folder /path/to/images \\
        --output /path/to/output \\
        [--split-ratio 0.8] [--seed 42]
"""

import argparse
import collections
import copy
import json
import os
import random
import sys

REQUIRED_KEYS = {"images", "annotations", "categories"}


def parse_args():
    parser = argparse.ArgumentParser("Dataset preparation script for SAM3 fine-tuning")
    parser.add_argument("--ann-file", required=True, type=str,
                        help="Path to CVAT COCO annotation JSON")
    parser.add_argument("--img-folder", required=True, type=str,
                        help="Path to directory containing the images")
    parser.add_argument("--output", required=True, type=str,
                        help="Output directory for train.json and val.json")
    parser.add_argument("--split-ratio", type=float, default=0.8,
                        help="Fraction of images for training (default: 0.8)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility (default: 42)")
    return parser.parse_args()


def validate_coco(data: dict, path: str) -> None:
    """Exit with code 1 if required COCO keys are missing."""
    missing = REQUIRED_KEYS - set(data.keys())
    if missing:
        print(
            f"ERROR: {path} is missing required COCO keys: {sorted(missing)}",
            file=sys.stderr,
        )
        sys.exit(1)


def repair_filenames(coco: dict) -> dict:
    """Strip any leading directory prefix from file_name (e.g. 'images/frame.jpg' -> 'frame.jpg')."""
    for img in coco["images"]:
        img["file_name"] = os.path.basename(img["file_name"])
    return coco


def repair_ids(coco: dict) -> dict:
    """
    Reindex 0-based IDs to 1-based and make category IDs contiguous.
    Each ID type is checked independently (handles mixed 0/1-based exports).
    """
    images = coco["images"]
    anns = coco["annotations"]
    cats = coco["categories"]

    # Step 1: Image IDs — reindex if min == 0
    img_id_map: dict = {}
    if images and min(img["id"] for img in images) == 0:
        for img in images:
            img_id_map[img["id"]] = img["id"] + 1
            img["id"] += 1

    # Step 2: Annotation IDs — reindex if min == 0; patch image_id refs
    if anns and min(ann["id"] for ann in anns) == 0:
        for ann in anns:
            ann["id"] += 1
    if img_id_map:
        for ann in anns:
            if ann["image_id"] in img_id_map:
                ann["image_id"] = img_id_map[ann["image_id"]]

    # Step 3: Category IDs — always sort + remap to 1..N (handles gaps AND 0-based)
    sorted_cats = sorted(cats, key=lambda c: c["id"])
    cat_id_map = {cat["id"]: i + 1 for i, cat in enumerate(sorted_cats)}
    for cat in cats:
        cat["id"] = cat_id_map[cat["id"]]
    for ann in anns:
        ann["category_id"] = cat_id_map[ann["category_id"]]

    return coco


def warn_orphan_annotations(coco: dict) -> None:
    """Warn about and remove annotations referencing unknown image IDs."""
    valid_ids = {img["id"] for img in coco["images"]}
    orphans = [ann for ann in coco["annotations"] if ann["image_id"] not in valid_ids]
    if orphans:
        print(
            f"Warning: {len(orphans)} annotation(s) reference unknown image IDs "
            f"and will be skipped: {sorted({a['image_id'] for a in orphans})}"
        )
    coco["annotations"] = [a for a in coco["annotations"] if a["image_id"] in valid_ids]


def exclude_zero_annotation_images(coco: dict) -> set:
    """
    Return the set of image IDs that have at least one annotation.
    Prints a warning listing excluded filenames.
    """
    annotated_ids = {ann["image_id"] for ann in coco["annotations"]}
    excluded = [img["file_name"] for img in coco["images"] if img["id"] not in annotated_ids]
    if excluded:
        print(
            f"Warning: {len(excluded)} image(s) with zero annotations excluded from splits: {excluded}"
        )
    return annotated_ids


def stratified_split(
    image_ids: list,
    anns_by_image: dict,
    cat_id_to_name: dict,
    split_ratio: float,
    seed: int,
) -> tuple:
    """
    Multi-label greedy stratified split.
    Assigns most-constrained images first; falls back to random for rare classes (D-02).
    """
    rng = random.Random(seed)  # noqa: F841 — available for future tie-breaking use

    # Build per-image category sets
    img_categories: dict = collections.defaultdict(set)
    for img_id in image_ids:
        for ann in anns_by_image.get(img_id, []):
            img_categories[img_id].add(ann["category_id"])

    # Count images per category for scoring
    cat_image_count: dict = collections.Counter()
    for cats in img_categories.values():
        for cat_id in cats:
            cat_image_count[cat_id] += 1

    # D-02: warn on rare classes (only 1 image)
    rare_cats = {cat_id for cat_id, count in cat_image_count.items() if count == 1}
    if rare_cats:
        rare_names = [cat_id_to_name.get(c, str(c)) for c in sorted(rare_cats)]
        print(
            f"Warning: {len(rare_cats)} rare class(es) with only 1 annotated image "
            f"(will be randomly assigned to a split): {rare_names}"
        )

    train_ids: list = []
    val_ids: list = []
    train_cat_count: dict = collections.Counter()
    val_cat_count: dict = collections.Counter()

    # Most-constrained images first (most categories → process first)
    sorted_images = sorted(
        image_ids, key=lambda img_id: len(img_categories[img_id]), reverse=True
    )

    for img_id in sorted_images:
        cats = img_categories[img_id]
        if not cats:
            continue  # zero-annotation images excluded before this step

        # Assign to whichever split is most behind its target for this image's rarest category
        rarest = min(cats, key=lambda c: cat_image_count.get(c, 0))
        total = max(cat_image_count.get(rarest, 1), 1)
        train_fill = train_cat_count.get(rarest, 0) / (total * split_ratio + 1e-9)
        val_fill = val_cat_count.get(rarest, 0) / (total * (1 - split_ratio) + 1e-9)

        if train_fill <= val_fill:
            train_ids.append(img_id)
            for c in cats:
                train_cat_count[c] += 1
        else:
            val_ids.append(img_id)
            for c in cats:
                val_cat_count[c] += 1

    # D-04: warn if val is empty
    if not val_ids:
        print(
            "Warning: val split is empty. Consider increasing dataset size or adjusting --split-ratio."
        )

    return train_ids, val_ids


def filter_split(coco: dict, keep_ids: set) -> dict:
    """Filter a repaired COCO dict to only include images/annotations for keep_ids.
    Output file_name values are already bare basenames (repair_filenames ran before this).
    """
    keep = set(keep_ids)
    return {
        "info": coco.get("info", {}),
        "licenses": coco.get("licenses", []),
        "categories": copy.deepcopy(coco["categories"]),
        "images": [img for img in coco["images"] if img["id"] in keep],
        "annotations": [ann for ann in coco["annotations"] if ann["image_id"] in keep],
    }


def print_stats(coco: dict, train_ids: list, val_ids: list) -> None:
    """Print dataset summary: total images, per-split count, per-category instance counts."""
    cat_id_to_name = {cat["id"]: cat["name"] for cat in coco["categories"]}
    train_set = set(train_ids)
    val_set = set(val_ids)

    train_counts: dict = collections.Counter()
    val_counts: dict = collections.Counter()
    for ann in coco["annotations"]:
        if ann["image_id"] in train_set:
            train_counts[ann["category_id"]] += 1
        elif ann["image_id"] in val_set:
            val_counts[ann["category_id"]] += 1

    total = len(train_ids) + len(val_ids)
    print(f"\nDataset summary:")
    print(f"  Total images processed : {total}  (train: {len(train_ids)}, val: {len(val_ids)})")
    print(f"  Category instance counts:")
    for cat_id, name in sorted(cat_id_to_name.items()):
        print(
            f"    {name:30s}  train: {train_counts.get(cat_id, 0):5d}  val: {val_counts.get(cat_id, 0):5d}"
        )
    print(
        f"\nNOTE: Set img_folder in your Hydra config to the directory directly "
        f"containing the image files (not the parent)."
    )


def main(args):
    # 1. Load
    try:
        with open(args.ann_file) as f:
            coco = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        print(f"ERROR: Could not read {args.ann_file}: {e}", file=sys.stderr)
        sys.exit(1)

    # 2. Validate — D-13: exit(1) on missing required keys
    validate_coco(coco, args.ann_file)

    # 3. Repair pass (all silent — D-11)
    repair_filenames(coco)   # D-06: strip file_name prefix
    repair_ids(coco)         # D-10: 0-based → 1-based + contiguous categories
    warn_orphan_annotations(coco)  # D-12: warn + drop orphan annotations

    # 4. Filter zero-annotation images — D-03
    annotated_ids = exclude_zero_annotation_images(coco)
    image_ids = [img["id"] for img in coco["images"] if img["id"] in annotated_ids]

    # 5. Stratified split — D-01
    anns_by_image = collections.defaultdict(list)
    for ann in coco["annotations"]:
        anns_by_image[ann["image_id"]].append(ann)
    cat_id_to_name = {cat["id"]: cat["name"] for cat in coco["categories"]}

    train_ids, val_ids = stratified_split(
        image_ids, anns_by_image, cat_id_to_name, args.split_ratio, args.seed
    )

    # 6. Write output — D-09
    os.makedirs(args.output, exist_ok=True)
    train_coco = filter_split(coco, set(train_ids))
    val_coco = filter_split(coco, set(val_ids))
    with open(os.path.join(args.output, "train.json"), "w") as f:
        json.dump(train_coco, f)
    with open(os.path.join(args.output, "val.json"), "w") as f:
        json.dump(val_coco, f)

    # 7. Print stats — D-14
    print_stats(coco, train_ids, val_ids)


if __name__ == "__main__":
    main(parse_args())
