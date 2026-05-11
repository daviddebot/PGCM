"""
SAM-based segmentation pipeline for CLEVR-Hans3 images.

For each image in the dataset, uses bounding boxes from scene JSON files
as prompts for SAM to produce per-object masked images. Output filenames
encode object attributes (size, color, shape, material).
"""

import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np
import torch
from segment_anything import SamPredictor, sam_model_registry
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description="Segment CLEVR-Hans3 objects using SAM with bbox prompts."
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="../",
        help="Path to CLEVR-Hans3 root directory (default: ../)",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default="./output",
        help="Path to write segmentation results (default: ./output)",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val", "test"],
        help="Splits to process (default: train val test)",
    )
    parser.add_argument(
        "--sam-checkpoint",
        type=str,
        default="./sam_vit_h_4b8939.pth",
        help="Path to SAM checkpoint (default: ./sam_vit_h_4b8939.pth)",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="vit_h",
        help="SAM model type (default: vit_h)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (default: auto-detect CUDA/CPU)",
    )
    return parser.parse_args()


def load_sam(checkpoint: str, model_type: str, device: str) -> SamPredictor:
    """Load SAM model and return a predictor."""
    print(f"Loading SAM model ({model_type}) from {checkpoint} on {device}...")
    # Build model architecture without loading checkpoint internally
    sam = sam_model_registry[model_type]()
    # Load state dict manually (handles both legacy and zip .pth formats)
    state_dict = torch.load(checkpoint, map_location=device, weights_only=False)
    sam.load_state_dict(state_dict)
    sam.to(device=device)
    return SamPredictor(sam)


def bbox_xywh_to_xyxy(bbox: list) -> np.ndarray:
    """Convert [x, y, w, h] bbox to [x1, y1, x2, y2] format."""
    x, y, w, h = bbox
    return np.array([x, y, x + w, y + h])


def build_object_filename(obj: dict) -> str:
    """Build output filename from object attributes.

    Format: obj{idx}_{size}_{color}_{shape}_{material}.png
    """
    idx = obj["idx"]
    size = obj["size"]
    color = obj["color"]
    shape = obj["shape"]
    material = obj["material"]
    return f"obj{idx}_{size}_{color}_{shape}_{material}.png"


def process_image(
    predictor: SamPredictor,
    image_path: Path,
    scene_data: dict,
    output_dir: Path,
) -> None:
    """Process a single image: segment each object and save masked results."""
    # Load image (BGR -> RGB for SAM)
    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        print(f"  WARNING: Could not read image {image_path}, skipping.")
        return
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    # Set image embedding once
    predictor.set_image(image_rgb)

    # Create output directory for this image
    output_dir.mkdir(parents=True, exist_ok=True)

    for obj in scene_data["objects"]:
        # Convert bbox to xyxy
        box = bbox_xywh_to_xyxy(obj["bbox"])

        # Run SAM prediction with box prompt
        masks, scores, _ = predictor.predict(
            box=box,
            multimask_output=True,
        )

        # Take the mask with the highest score
        best_idx = np.argmax(scores)
        mask = masks[best_idx]  # (H, W) boolean

        # Apply mask to original image (black background)
        masked_image = np.zeros_like(image_rgb)
        masked_image[mask] = image_rgb[mask]

        # Build filename and save (RGB -> BGR for cv2)
        filename = build_object_filename(obj)
        out_path = output_dir / filename
        cv2.imwrite(str(out_path), cv2.cvtColor(masked_image, cv2.COLOR_RGB2BGR))


def process_split(
    predictor: SamPredictor,
    data_root: Path,
    output_root: Path,
    split: str,
) -> None:
    """Process all images in a single split."""
    scenes_dir = data_root / split / "scenes"
    images_dir = data_root / split / "images"
    split_output_dir = output_root / split

    if not scenes_dir.exists():
        print(f"  Scenes directory not found: {scenes_dir}, skipping split '{split}'.")
        return
    if not images_dir.exists():
        print(f"  Images directory not found: {images_dir}, skipping split '{split}'.")
        return

    # Collect and sort scene files
    scene_files = sorted(scenes_dir.glob("*.json"))
    print(f"  Found {len(scene_files)} scene files in '{split}'.")

    for scene_file in tqdm(scene_files, desc=f"  {split}"):
        # Load scene data
        with open(scene_file, "r") as f:
            scene_data = json.load(f)

        # Resolve image path
        image_filename = scene_data["image_filename"]
        image_path = images_dir / image_filename

        # Output subdirectory named after the image (without extension)
        image_stem = Path(image_filename).stem
        output_dir = split_output_dir / image_stem

        # Process
        process_image(predictor, image_path, scene_data, output_dir)


def main():
    args = parse_args()

    # Resolve device
    device = args.device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    data_root = Path(args.data_root).resolve()
    output_root = Path(args.output_root).resolve()
    sam_checkpoint = Path(args.sam_checkpoint).resolve()

    print(f"Data root:      {data_root}")
    print(f"Output root:    {output_root}")
    print(f"SAM checkpoint: {sam_checkpoint}")
    print(f"Device:         {device}")
    print(f"Splits:         {args.splits}")
    print()

    # Load SAM once
    predictor = load_sam(str(sam_checkpoint), args.model_type, device)

    # Process each split
    for split in args.splits:
        print(f"Processing split: {split}")
        process_split(predictor, data_root, output_root, split)
        print()

    print("Done.")


if __name__ == "__main__":
    main()
