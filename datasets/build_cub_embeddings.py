"""
Standalone script to generate CUB pretrained embeddings using BlackBox model.

Uses hardcoded ResNet18 backbone (resnet18-5c106cde.pth).

USAGE:
    # Option 1: Pass parent directory (CUB_200_2011/)
    python generate_cub_embeddings.py \
        --cub_data_root /path/to/CUB_200_2011 \
        --output_dir ./cub_embeddings \
        --batch_size 64
    
    # Option 2: Pass data directory directly (class_attr_data_10/)
    python generate_cub_embeddings.py \
        --cub_data_root /path/to/CUB_200_2011/class_attr_data_10 \
        --output_dir ./cub_embeddings \
        --batch_size 64

OUTPUT:
    output_dir/train_x.pt, train_c.pt, train_y.pt
    output_dir/val_x.pt, val_c.pt, val_y.pt
    output_dir/test_x.pt, test_c.pt, test_y.pt
    
Where:
    - x = embeddings from ResNet18 backbone
    - c = concept labels
    - y = task labels
"""

import argparse
import os
import pickle
from pathlib import Path

import torch
import torch.nn as nn
import torchvision
import pytorch_lightning as pl
import numpy as np
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms


class MinimalBlackBox(pl.LightningModule):
    """Minimal BlackBox implementation for embedding extraction only."""
    
    def __init__(self):
        super().__init__()
        # Load ResNet18 backbone
        self.backbone = torchvision.models.resnet18(weights=None)
        self.backbone.load_state_dict(torch.load("./resnet18_checkpoint.pth", weights_only=False))
        n_features = self.backbone.fc.in_features
        self.backbone.fc = torch.nn.Identity()
    
    def backbone_forward(self, x):
        """Extract embeddings from input triplet (image, concepts, labels)."""
        image, c_true, y_true = x
        emb = self.backbone(image)
        return emb, c_true, y_true


class CUBDataset(Dataset):
    """Standalone CUB dataset loader from pickle files."""
    
    def __init__(self, pickle_path, root_dir, transform=None):
        """
        Args:
            pickle_path: Path to .pkl file containing list of dicts with:
                         {'img_path': str, 'attribute_label': list, 'class_label': int}
            root_dir: Root directory for image paths
            transform: Optional image transforms
        """
        with open(pickle_path, 'rb') as f:
            self.data = pickle.load(f)
        
        self.root_dir = root_dir
        
        # Default transform: ResNet18 expects 224x224 with ImageNet normalization
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
        else:
            self.transform = transform
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_data = self.data[idx]
        
        # Get image path from pickle (may be absolute like /juice/scr/.../CUB_200_2011/images/...)
        img_path = img_data['img_path']
        
        # Extract the relevant part after 'CUB_200_2011'
        if 'CUB_200_2011' in img_path:
            parts = img_path.split('CUB_200_2011')
            if len(parts) > 1:
                # Get the part after CUB_200_2011 and remove leading /
                img_path = parts[-1].lstrip('/')
        else:
            # Already relative - remove leading ./ or /
            img_path = img_path.lstrip('./')
        
        # Construct full path
        full_path = os.path.join(self.root_dir, img_path)
        
        # Load image
        if not os.path.exists(full_path):
            raise FileNotFoundError(f"Image not found: {full_path}\n  Original: {img_data['img_path']}\n  Root: {self.root_dir}")
        
        image = Image.open(full_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        
        # Get attributes and class label
        attributes = torch.tensor(img_data['attribute_label'], dtype=torch.float32)
        label = torch.tensor(img_data['class_label'], dtype=torch.long)
        
        return image, attributes, label


def get_cub_dataloaders(batch_size, cub_data_root):
    """Load CUB train/val/test dataloaders from pickle files.
    
    Args:
        batch_size: Batch size for loaders
        cub_data_root: Path to class_attr_data_10 or parent CUB_200_2011 directory
    
    Returns:
        train_loader, val_loader, test_loader
    """
    # Normalize the path
    cub_data_root = os.path.abspath(cub_data_root)
    
    # Handle both cases: /path/to/CUB_200_2011 or /path/to/class_attr_data_10
    if os.path.basename(cub_data_root) == 'class_attr_data_10':
        data_dir = cub_data_root
        # Look for CUB_200_2011 as a sibling (same parent)
        parent = os.path.dirname(cub_data_root)
        cub_dir = os.path.join(parent, 'CUB_200_2011')
        if os.path.exists(cub_dir):
            image_root = cub_dir
        else:
            # Fallback to parent if CUB_200_2011 doesn't exist
            image_root = parent
    else:
        # Assume it's the CUB_200_2011 directory
        cub_dir = cub_data_root
        data_dir = os.path.join(cub_dir, 'class_attr_data_10')
        image_root = cub_dir
    
    print(f"  Data dir: {data_dir}")
    print(f"  Image root: {image_root}")
    
    train_dataset = CUBDataset(
        os.path.join(data_dir, 'train.pkl'),
        image_root
    )
    val_dataset = CUBDataset(
        os.path.join(data_dir, 'val.pkl'),
        image_root
    )
    test_dataset = CUBDataset(
        os.path.join(data_dir, 'test.pkl'),
        image_root
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    return train_loader, val_loader, test_loader


def extract_cub_embeddings(
    model,
    train_loader,
    val_loader,
    test_loader,
    output_dir,
    device="cuda" if torch.cuda.is_available() else "cpu",
):
    """Extract embeddings from model backbone and save to .pt files.
    
    Args:
        model: Model with backbone_forward(inputs) method
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        test_loader: DataLoader for test data
        output_dir: Directory to save embeddings
        device: Device to run on (cuda or cpu)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    model = model.to(device)
    model.eval()
    
    xs, cs, ys = dict(), dict(), dict()
    
    # Extract embeddings without keeping autograd graph and save efficiently
    with torch.no_grad():
        for loader, loader_type in [
            (train_loader, "train"),
            (val_loader, "val"),
            (test_loader, "test"),
        ]:
            print(f"Processing {loader_type} split...")
            xs[loader_type] = []
            cs[loader_type] = []
            ys[loader_type] = []
            
            for batch_idx, batch in enumerate(loader):
                if batch_idx % 10 == 0:
                    print(f"  Batch {batch_idx}/{len(loader)}")
                
                # move inputs to device and run backbone forward under no_grad
                inputs = (
                    batch[0].to(device),
                    batch[1].to(device),
                    batch[2].to(device),
                )
                emb, c_true, y_true = model.backbone_forward(inputs)
                
                # detach and move to CPU immediately to avoid retaining graph / GPU memory
                xs[loader_type].append(emb.detach().cpu())
                cs[loader_type].append(c_true.detach().cpu())
                ys[loader_type].append(y_true.detach().cpu())
            
            # concatenate and save (more memory-efficient than torch.tensor(list_of_tensors))
            try:
                xs_cat = torch.cat(xs[loader_type], dim=0)
            except Exception:
                xs_cat = torch.stack(xs[loader_type], dim=0)
            
            try:
                cs_cat = torch.cat(cs[loader_type], dim=0)
            except Exception:
                cs_cat = torch.stack(cs[loader_type], dim=0)
            
            try:
                ys_cat = torch.cat(ys[loader_type], dim=0)
            except Exception:
                ys_cat = torch.stack(ys[loader_type], dim=0)
            
            # Save tensors
            xs_path = output_dir / f"{loader_type}_x.pt"
            cs_path = output_dir / f"{loader_type}_c.pt"
            ys_path = output_dir / f"{loader_type}_y.pt"
            
            torch.save(xs_cat, xs_path)
            torch.save(cs_cat, cs_path)
            torch.save(ys_cat, ys_path)
            
            print(f"  Saved {loader_type} embeddings:")
            print(f"    x: {xs_cat.shape} -> {xs_path}")
            print(f"    c: {cs_cat.shape} -> {cs_path}")
            print(f"    y: {ys_cat.shape} -> {ys_path}")
    
    print("\nEmbedding extraction complete!")


def main():
    parser = argparse.ArgumentParser(
        description="Generate CUB embeddings using BlackBox model"
    )
    parser.add_argument(
        "--cub_data_root",
        type=str,
        required=True,
        help="Path to CUB data root (CUB_200_2011/ or CUB_200_2011/class_attr_data_10/)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="cub_embeddings",
        help="Output directory for embeddings (default: cub_embeddings)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size for data loading (default: 64)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use (cuda or cpu)",
    )
    
    args = parser.parse_args()
    
    # Load the checkpoint
    checkpoint_path = "./resnet18-5c106cde.pth"
    print(f"Loading BlackBox model with ResNet18 from {checkpoint_path}...")
    model = MinimalBlackBox()
    
    # Load CUB data
    print(f"Loading CUB data from {args.cub_data_root}...")
    train_loader, val_loader, test_loader = get_cub_dataloaders(args.batch_size, args.cub_data_root)
    
    # Extract and save embeddings
    print(f"\nExtracting embeddings to {args.output_dir}...")
    extract_cub_embeddings(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        output_dir=args.output_dir,
        device=args.device,
    )


if __name__ == "__main__":
    main()
