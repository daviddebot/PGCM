"""
Build presegmented .pt dataset files for CLEVR-Hans3 (centered + padded variant).

Each segmented object is cropped to its
bounding box and placed at its **original size** in the centre of a black
canvas of size (resolution, resolution).  No enlarging or stretching is
applied — the object keeps its original proportions.  If the crop happens
to be larger than the target resolution in either dimension it is scaled
*down* (preserving aspect ratio) so that it fits.

    {
        'segmented_objects': float32  (N, MAX_OBJECTS, 3, res, res)
        'predicted_masks':   int8     (N, MAX_OBJECTS, res, res)
        'concepts':          int8     (N, MAX_OBJECTS, NUM_CONCEPTS)
        'tasks':             int8     (N, NUM_CLASSES)
        'original_images':   float32  (N, 3, res, res)
    }

Usage:
    python build_presegmented_centered_padded.py \\
        --data-root ../ \\
        --segmentation-root ./output \\
        --output-dir ./presegmented_centered_padded \\
        --resolution 128
"""

import argparse
import json
import os
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

# ── CLEVR-Hans concept encoding ────────────────────────────────────────────────

SIZES = ['large', 'small']
COLORS = ['gray', 'red', 'blue', 'green', 'brown', 'purple', 'cyan', 'yellow']
SHAPES = ['cube', 'sphere', 'cylinder']
MATERIALS = ['rubber', 'metal']

NUM_CONCEPTS = len(SIZES) + len(COLORS) + len(SHAPES) + len(MATERIALS)  # 15
NUM_CLASSES = 3
MAX_OBJECTS = 10


def obj_to_concept_vec(obj: dict) -> torch.Tensor:
    """One-hot encode an object's attributes into a vector of length 15."""
    vec = torch.zeros(NUM_CONCEPTS)
    vec[SIZES.index(obj['size'])] = 1.0
    vec[2 + COLORS.index(obj['color'])] = 1.0
    vec[10 + SHAPES.index(obj['shape'])] = 1.0
    vec[13 + MATERIALS.index(obj['material'])] = 1.0
    return vec


def obj_filename(obj: dict) -> str:
    """Reproduce the filename used by main.py's SAM segmentation."""
    return f"obj{obj['idx']}_{obj['size']}_{obj['color']}_{obj['shape']}_{obj['material']}.png"


def crop_and_center_on_canvas(img: Image.Image, bbox, resolution: int) -> Image.Image:
    """
    Crop *img* to the bounding box and paste it centred on a black canvas.

    The crop is kept at its original pixel size.  If the crop is larger
    than *resolution* in either dimension it is scaled down (preserving
    aspect ratio) so that it fits.  No up-scaling is ever performed.

    Parameters
    ----------
    img : PIL.Image
        Full-resolution image (segmented object or original).
    bbox : list[int]
        [x, y, w, h] in the original image coordinate system.
    resolution : int
        Target square canvas size.

    Returns
    -------
    PIL.Image
        A (resolution x resolution) image with the crop centred.
    """
    bx, by, bw, bh = bbox
    cropped = img.crop((bx, by, bx + bw, by + bh))

    cw, ch = cropped.size  # width, height of the crop

    # Scale down (preserving aspect ratio) only if necessary
    if cw > resolution or ch > resolution:
        scale = min(resolution / cw, resolution / ch)
        new_w = int(cw * scale)
        new_h = int(ch * scale)
        cropped = cropped.resize((new_w, new_h), Image.BILINEAR)
        cw, ch = new_w, new_h

    # Create black canvas and paste centred
    canvas = Image.new("RGB", (resolution, resolution), (0, 0, 0))
    paste_x = (resolution - cw) // 2
    paste_y = (resolution - ch) // 2
    canvas.paste(cropped, (paste_x, paste_y))
    return canvas


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build presegmented .pt files for CLEVR-Hans3 (centered + padded)."
    )
    parser.add_argument(
        "--data-root", type=str, default="../",
        help="Path to CLEVR-Hans3 root (default: ../)",
    )
    parser.add_argument(
        "--segmentation-root", type=str, default="./output",
        help="Path to SAM segmentation output (default: ./output)",
    )
    parser.add_argument(
        "--output-dir", type=str, default="./presegmented_centered_padded",
        help="Where to write train.pt / val.pt / test.pt (default: ./presegmented_centered_padded)",
    )
    parser.add_argument(
        "--resolution", type=int, default=128,
        help="Target resolution for images (default: 128)",
    )
    parser.add_argument(
        "--splits", nargs="+", default=["train", "val", "test"],
        help="Splits to process (default: train val test)",
    )
    return parser.parse_args()


def build_split(data_root: Path, seg_root: Path, split: str, resolution: int):
    """
    Build the presegmented tensors for one split.

    Returns a dict matching the _serialize_presegmented_dataset format.
    """
    scenes_dir = data_root / split / "scenes"
    images_dir = data_root / split / "images"
    seg_split_dir = seg_root / split

    scene_files = sorted([f for f in os.listdir(scenes_dir) if f.endswith(".json")])
    print(f"[{split}] Found {len(scene_files)} scenes.")

    to_tensor = transforms.ToTensor()  # → [0,1] float32, (C,H,W)

    resize_original = transforms.Compose([
        transforms.Resize((resolution, resolution)),
        transforms.ToTensor(),
    ])

    seg_objects_list = []
    masks_list = []
    concepts_list = []
    tasks_list = []
    images_list = []

    for scene_file in tqdm(scene_files, desc=f"  {split}"):
        with open(scenes_dir / scene_file, "r") as f:
            scene = json.load(f)

        # ── Original image ───────────────────────────────────────────────
        img = Image.open(images_dir / scene["image_filename"]).convert("RGB")
        img_tensor = resize_original(img)  # (3, res, res)

        # ── Per-object segmented images, masks, concepts ─────────────────
        seg_objs = torch.zeros(MAX_OBJECTS, 3, resolution, resolution)
        masks = torch.zeros(MAX_OBJECTS, resolution, resolution)
        concepts = torch.zeros(MAX_OBJECTS, NUM_CONCEPTS)

        image_stem = Path(scene["image_filename"]).stem
        seg_image_dir = seg_split_dir / image_stem

        for obj in scene["objects"]:
            idx = obj["idx"] - 1  # 1-based → 0-based
            if idx >= MAX_OBJECTS:
                continue

            bbox = obj["bbox"]  # [x, y, w, h] in original coords

            # Load SAM segmented object image (if it exists)
            seg_path = seg_image_dir / obj_filename(obj)
            if seg_path.exists():
                seg_img = Image.open(seg_path).convert("RGB")
                # Crop to bounding box, centre on black canvas (no enlarging)
                seg_centered = crop_and_center_on_canvas(seg_img, bbox, resolution)
                seg_tensor = to_tensor(seg_centered)  # (3, res, res)
                seg_objs[idx] = seg_tensor

                # Derive mask: any pixel that is not black in the result
                mask = (seg_tensor.sum(dim=0) > 0.01).float()  # (res, res)
                masks[idx] = mask
            else:
                # Fallback: crop original image at bbox, centre on canvas
                obj_centered = crop_and_center_on_canvas(img, bbox, resolution)
                seg_tensor = to_tensor(obj_centered)
                seg_objs[idx] = seg_tensor
                # Mask from non-black pixels of the centred crop
                mask = (seg_tensor.sum(dim=0) > 0.01).float()
                masks[idx] = mask

            concepts[idx] = obj_to_concept_vec(obj)

        # ── Task label ───────────────────────────────────────────────────
        task = torch.zeros(NUM_CLASSES)
        task[scene["class_id"]] = 1.0

        seg_objects_list.append(seg_objs)
        masks_list.append(masks)
        concepts_list.append(concepts)
        tasks_list.append(task)
        images_list.append(img_tensor)

    # Stack and serialize to the expected format / dtypes
    result = {
        "segmented_objects": torch.stack(seg_objects_list).to(torch.float32),
        "predicted_masks": (torch.stack(masks_list) > 0.5).to(torch.int8),
        "concepts": torch.stack(concepts_list).to(torch.int8),
        "tasks": torch.stack(tasks_list).to(torch.int8),
        "original_images": torch.stack(images_list).to(torch.float32),
    }
    for k, v in result.items():
        print(f"    {k}: {v.shape} {v.dtype}")
    return result


def main():
    args = parse_args()
    data_root = Path(args.data_root).resolve()
    seg_root = Path(args.segmentation_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Data root:          {data_root}")
    print(f"Segmentation root:  {seg_root}")
    print(f"Output dir:         {output_dir}")
    print(f"Resolution:         {args.resolution}")
    print()

    for split in args.splits:
        print(f"Building {split}...")
        payload = build_split(data_root, seg_root, split, args.resolution)
        out_path = output_dir / f"{split}.pt"
        torch.save(payload, out_path)
        print(f"  Saved → {out_path}\n")

    print("Done.")


if __name__ == "__main__":
    main()
