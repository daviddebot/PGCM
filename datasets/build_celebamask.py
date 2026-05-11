import os
import glob
import math
from PIL import Image
from tqdm import tqdm
import torch
import torchvision.transforms as T

def create_celebahq_shards_optimized(
    img_dir: str,
    mask_root: str,
    attr_path: str,
    out_dir: str,
    shard_size: int = 1000,
    resize: int | None = None,
    max_images: int | None = None,
    extensions: tuple = ('.jpg', '.jpeg', '.png'),
    verbose: bool = True,
    all_possible_masks = ['cloth', 'ear_r', 'eye_g', 'hair', 'hat', 'l_brow', 'l_ear', 'l_eye', 'l_lip', 'mouth', 'neck', 'neck_l', 'nose', 'r_brow', 'r_ear', 'r_eye', 'skin', 'u_lip']
):
    os.makedirs(out_dir, exist_ok=True)

    # --- gather image paths ---
    img_paths = []
    for ext in extensions:
        img_paths.extend(glob.glob(os.path.join(img_dir, f'*{ext}')))
        img_paths.extend(glob.glob(os.path.join(img_dir, f'*{ext.upper()}')))
    img_paths = sorted(img_paths)
    if max_images:
        img_paths = img_paths[:max_images]
    n = len(img_paths)

    # --- load attributes ---
    with open(attr_path, 'r') as f:
        lines = f.readlines()
    header = lines[1].strip().split()[1:]
    concept_names = header
    attr_map = {}
    for line in lines[2:]:
        parts = line.strip().split()
        img_id = parts[0].split('.')[0]
        # Attributes are usually -1 or 1. We can keep float or use int8.
        attrs = [int(x) for x in parts[1:]] 
        attr_map[img_id.zfill(5) + '.jpg'] = torch.tensor(attrs, dtype=torch.int8) 

    # --- prepare transform ---
    # We define separate resizing for PIL images to ensure we stay in PIL land 
    # before converting to tensor to avoid implicit float conversion
    resizer = T.Resize(resize, antialias=True) if resize else None

    num_shards = math.ceil(n / shard_size)
    for shard_idx in range(num_shards):
        start = shard_idx * shard_size
        end = min(start + shard_size, n)
        shard_paths = img_paths[start:end]

        img_tensors, mask_tensors, concept_tensors, masks_names, filenames = [], [], [], [], []

        for img_path in tqdm(shard_paths, disable=not verbose, desc=f"Shard {shard_idx+1}/{num_shards}"):
            fname = os.path.basename(img_path)
            img_id = os.path.splitext(fname)[0]
            filenames.append(fname)

            # --- image ---
            with Image.open(img_path) as img:
                img = img.convert('RGB')
                if resizer:
                    img = resizer(img)
                # PILToTensor keeps data as uint8 (0-255)
                # Shape: [3, H, W]
                t_img = T.PILToTensor()(img)
                img_tensors.append(t_img)

            # --- masks ---
            img_id = img_id.zfill(5)
            masks = []
            masks_names_found = []
            
            # Determine size for zero masks
            h, w = t_img.shape[1], t_img.shape[2]

            for possible_mask in all_possible_masks:
                subfolder_idx = int(img_id) // 2000
                mask_path = os.path.join(mask_root, str(subfolder_idx), f"{img_id}_{possible_mask}.png")
                
                if os.path.exists(mask_path):
                    masks_names_found.append(possible_mask)
                    with Image.open(mask_path) as m:
                        m = m.convert('L')
                        if resizer:
                            m = resizer(m)
                        # PILToTensor keeps data as uint8 (0-255)
                        t_m = T.PILToTensor()(m) 
                        
                        # Convert 255 to 1 for boolean/mask logic
                        t_m = (t_m > 128).to(torch.uint8) # Shape [1, H, W]
                        masks.append(t_m)
                else:
                    masks_names_found.append('not_found')
                    # Create uint8 zeros
                    t_m = torch.zeros((1, h, w), dtype=torch.uint8)
                    masks.append(t_m)

            # Concatenate along channel dim. 
            # Original code had unsqueeze(1) which made it [18, 1, H, W]. 
            # Better to be [18, H, W] for storage.
            t_masks = torch.cat(masks, dim=0) 
            
            # --- concepts ---
            fkey = img_id + '.jpg'
            t_concepts = attr_map.get(fkey, torch.zeros(len(concept_names), dtype=torch.int8))

            mask_tensors.append(t_masks)
            masks_names.append(masks_names_found)
            concept_tensors.append(t_concepts)

        # --- stack everything ---
        imgs = torch.stack(img_tensors)       # [B, 3, H, W]  dtype=uint8
        masks = torch.stack(mask_tensors)     # [B, 18, H, W] dtype=uint8 (0 or 1)
        concepts = torch.stack(concept_tensors)

        shard_dict = {
            'images': imgs,
            'masks': masks,
            'concepts': concepts,
            'concept_names': concept_names,
            'mask_names': masks_names,
            'filenames': filenames,
        }

        # Verify size reduction
        if verbose and shard_idx == 0:
            print(f"Data types: Img={imgs.dtype}, Mask={masks.dtype}")
            
        out_path = os.path.join(out_dir, f'shard_{shard_idx:04d}.pt')
        torch.save(shard_dict, out_path)
        if verbose:
            print(f"Saved shard {shard_idx+1}/{num_shards}: {out_path}")
# create_image_tensor_shards(
#         src_dir="./datasets/celebamask/CelebAMask-HQ/CelebA-HQ-img",
#         out_dir="./datasets/celebamask/CelebAMask-HQ/tensors",
#         shard_size=500,
#         resize=256,   # set to None to keep original sizes (will center-crop per shard)
#         max_images=None
#     )

create_celebahq_shards_optimized(
        img_dir="./datasets/celebamask/CelebAMask-HQ/CelebA-HQ-img",
        mask_root="./datasets/celebamask/CelebAMask-HQ/CelebAMask-HQ-mask-anno",
        attr_path="./datasets/celebamask/CelebAMask-HQ/CelebAMask-HQ-attribute-anno.txt",
        out_dir="./datasets/celebamask/CelebAMask-HQ/tensors_fixed_128",
        shard_size=2000,
        resize=128,   # set to None to keep original sizes (will center-crop per shard)
        max_images=None
    )