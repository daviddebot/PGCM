"""Dataset loading and presegmentation utilities.

Provides data loaders for every supported benchmark: MNIST (digit addition),
CelebA-Mask (face attribute prediction), CLEVR-Hans3 (compositional visual
reasoning), CUB-200-2011 (pre-segmented bird parts), and CUB-EMB (pretrained
feature embeddings).

For image-based datasets the module also supports a two-stage workflow:
1. **Segmenter pretraining** — ``pretrain_segmenter.py`` trains a U-Net
   segmenter on ground-truth masks and saves presegmented tensors via
   ``save_presegmented_datasets``.
2. **Presegmented loading** — ``load_presegmented_dataloaders`` loads those
   tensors at training time so the segmenter need not run online.

MNIST loading logic originally adapted from
https://github.com/pietrobarbiero/interpretable-relational-reasoning
"""


import json
import os
import random
from pathlib import Path
from typing import List, Tuple
import torch.nn.functional as F
import torch
import torchvision
from torchvision import transforms as transforms
from torchvision.datasets import MNIST
from torch.utils.data import TensorDataset, DataLoader
from itertools import product
from collections import defaultdict
from torch.utils.data.dataloader import default_collate
from tqdm import tqdm

import neural_networks
from typing import Dict
from torch.utils.data import Dataset

DEFAULT_CUB_ROOT = "/cw/dtaijupiter/NoCsBack/dtai/david/residual/cub_new_embeddings"
VALID_SPLITS = ("train", "val", "test")


class CubDataset(Dataset):
    """Dataset wrapper that keeps the mask slot empty."""

    def __init__(self, x: torch.Tensor, c: torch.Tensor, y: torch.Tensor):
        self.x = x
        self.c = c
        self.y = y

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, index: int):
        return self.x[index], None, self.c[index], self.y[index]

    @property
    def tensors(self):
        return self.x, self.c, self.y


def _collate_cub_batch(batch):
    x, _m, c, y = zip(*batch)
    return default_collate(x), None, default_collate(c), default_collate(y)


def _get_cub_collate_fn(device, num_workers):
    if num_workers == 0:
        def _collate_and_move(batch):
            x, _m, c, y = _collate_cub_batch(batch)
            return x.to(device), None, c.to(device), y.to(device)

        return _collate_and_move

    return _collate_cub_batch


def load_cub_split(
    dataset_root: os.PathLike[str] | str,
    split: str,
    num_classes: int | None = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Load one CUB split from `*_x.pt`, `*_c.pt`, and `*_y.pt` files."""

    if split not in VALID_SPLITS:
        raise ValueError(f"Unknown split: {split!r}. Expected one of {VALID_SPLITS!r}.")

    root = Path(dataset_root)
    x = torch.load(root / f"{split}_x.pt", map_location="cpu")
    c = torch.load(root / f"{split}_c.pt", map_location="cpu").float()
    y = torch.load(root / f"{split}_y.pt", map_location="cpu")

    if num_classes is not None and (y.ndim == 1 or (y.ndim == 2 and y.shape[1] == 1)):
        y = F.one_hot(y.reshape(-1).long(), num_classes=num_classes)
    y = y.float()

    if c.ndim == 3:
        c = c.max(dim=1).values

    return x, c, y


def load_cub_splits(dataset_root: os.PathLike[str] | str = DEFAULT_CUB_ROOT) -> Dict[str, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Load all standard CUB splits into memory."""

    root = Path(dataset_root)
    raw_splits = {split: load_cub_split(root, split, num_classes=None) for split in VALID_SPLITS}

    num_classes = 0
    for _x, _c, y in raw_splits.values():
        if y.ndim == 1:
            num_classes = max(num_classes, int(y.max().item()) + 1)
        elif y.ndim == 2 and y.shape[1] == 1:
            num_classes = max(num_classes, int(y.max().item()) + 1)
        else:
            num_classes = max(num_classes, y.shape[1])

    splits = {}
    for split, (x, c, y) in raw_splits.items():
        if y.ndim == 1 or (y.ndim == 2 and y.shape[1] == 1):
            y = F.one_hot(y.reshape(-1).long(), num_classes=num_classes).float()
        else:
            y = y.float()
        splits[split] = (x, c, y)

    return splits


def build_cub_datasets(
    dataset_root: os.PathLike[str] | str = DEFAULT_CUB_ROOT,
) -> Dict[str, CubDataset]:
    """Build datasets for the train, validation, and test splits."""

    splits = load_cub_splits(dataset_root)
    return {split: CubDataset(*tensors) for split, tensors in splits.items()}


def get_cub_dataloaders(
    batch_size: int,
    dataset_root: os.PathLike[str] | str = DEFAULT_CUB_ROOT,
    shuffle_train: bool = False,
    num_workers: int = 0,
    device: os.PathLike[str] | str = "cpu",
) -> Tuple[DataLoader, DataLoader, DataLoader, int, int]:
    """Return train/val/test loaders plus the concept/task dimensions.

    The function keeps the API simple on purpose: it only loads the tensors
    that already exist on disk and infers the tensor dimensions from them.
    """

    datasets = build_cub_datasets(dataset_root)
    _, train_c, train_y = datasets["train"].tensors
    n_concepts = train_c.shape[1]
    n_tasks = train_y.shape[1]

    train_loader = DataLoader(
        datasets["train"],
        batch_size=batch_size,
        shuffle=shuffle_train,
        num_workers=num_workers,
        collate_fn=_get_cub_collate_fn(device, num_workers),
    )
    val_loader = DataLoader(
        datasets["val"],
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=_get_cub_collate_fn(device, num_workers),
    )
    test_loader = DataLoader(
        datasets["test"],
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=_get_cub_collate_fn(device, num_workers),
    )

    return train_loader, val_loader, test_loader, n_concepts, n_tasks



transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
)
data_root = Path(__file__).parent / ".." / "data"

CELEBA_DIR = '/cw/dtaiexp/2025-Stefano/tensors_fixed_128'

TRAIN_SHARDS = [0,1,2,3,4,5,6,7,8,9,10,11] 
TEST_SHARDS = [12,13,14]

CLEVR_HANS_DIR = '/cw/dtaiexp/2023-DavidDebot/my_clevr/CLEVR-Hans3'


def _collate_to_device(device):
    return lambda batch: tuple(item.to(device) for item in default_collate(batch))


def _get_collate_fn(device, num_workers):
    # Moving tensors to GPU inside DataLoader workers is unsafe.
    # Keep the existing direct-to-device path only for single-worker loading.
    if num_workers == 0:
        return _collate_to_device(device)
    return default_collate


def _resolve_segmenter_state_dict_path(path):
    if path is None:
        raise ValueError("pretrained_segmenter_path must be provided when use_pretrained_segmenter is enabled.")

    if os.path.isdir(path):
        candidates = [
            os.path.join(path, 'outputs', 'segmenter_state_dict.pt'),
            os.path.join(path, 'segmenter_state_dict.pt'),
        ]
        for candidate in candidates:
            if os.path.exists(candidate):
                return candidate
        raise FileNotFoundError(f"Could not find segmenter_state_dict.pt under directory: {path}")

    if not os.path.exists(path):
        raise FileNotFoundError(f"Pretrained segmenter weights not found: {path}")

    return path


def _build_presegmented_dataset(loader, segmenter, desc):
    # Precompute segmentation outputs once so training and evaluation can reuse
    # fixed tensors instead of running the segmenter inside every batch.
    segmented_objects = []
    predicted_masks = []
    concepts = []
    tasks = []
    original_images = []

    segmenter.eval()
    with torch.no_grad():
        for x, _, c, y in tqdm(loader, desc=desc):
            predicted_mask_logits = segmenter(x)
            predicted_mask_probs = torch.sigmoid(predicted_mask_logits)
            object_images = predicted_mask_probs.unsqueeze(2) * x.unsqueeze(1)

            segmented_objects.append(object_images.cpu())
            predicted_masks.append(predicted_mask_probs.cpu())
            concepts.append(c.cpu())
            tasks.append(y.cpu())
            original_images.append(x.cpu())

    return TensorDataset(
        torch.cat(segmented_objects, dim=0),
        torch.cat(predicted_masks, dim=0),
        torch.cat(concepts, dim=0),
        torch.cat(tasks, dim=0),
        torch.cat(original_images, dim=0),
    )


def _serialize_presegmented_dataset(dataset):
    segmented_objects, predicted_masks, concepts, tasks, original_images = dataset.tensors
    return {
        # Keep image-like continuous tensors in float16 to reduce disk usage.
        'segmented_objects': segmented_objects.to(torch.float16).cpu(),
        'predicted_masks': (predicted_masks > 0.5).to(torch.int8).cpu(),
        'concepts': concepts.to(torch.int8).cpu(),
        'tasks': tasks.to(torch.int8).cpu(),
        'original_images': original_images.to(torch.float16).cpu(),
    }


def _deserialize_presegmented_dataset(payload):
    return TensorDataset(
        payload['segmented_objects'].to(torch.float32),
        payload['predicted_masks'].to(torch.float32),
        payload['concepts'].to(torch.float32),
        payload['tasks'].to(torch.float32),
        payload['original_images'].to(torch.float32),
    )


def presegment_dataloaders(train_loader, val_loader, test_loader, dataset_name, segmentation_method, pretrained_segmenter_path, device='cpu', num_workers=0):
    state_dict_path = _resolve_segmenter_state_dict_path(pretrained_segmenter_path)

    sample_batch = next(iter(train_loader))
    _, sample_masks, _, _ = sample_batch
    n_class = sample_masks.shape[1]

    segmenter = neural_networks.build_segmenter(
        dataset_name=dataset_name,
        segmentation_method=segmentation_method,
        n_class=n_class,
    )
    state_dict = torch.load(state_dict_path, map_location=device)
    segmenter.load_state_dict(state_dict)
    segmenter.to(device)

    transformed_train = _build_presegmented_dataset(
        train_loader,
        segmenter,
        desc='Precomputing train segmentations',
    )
    transformed_val = _build_presegmented_dataset(
        val_loader,
        segmenter,
        desc='Precomputing val segmentations',
    )
    transformed_test = _build_presegmented_dataset(
        test_loader,
        segmenter,
        desc='Precomputing test segmentations',
    )

    train_loader_preseg = DataLoader(
        transformed_train,
        batch_size=train_loader.batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=_get_collate_fn(device, num_workers),
    )
    val_loader_preseg = DataLoader(
        transformed_val,
        batch_size=val_loader.batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=_get_collate_fn(device, num_workers),
    )
    test_loader_preseg = DataLoader(
        transformed_test,
        batch_size=test_loader.batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=_get_collate_fn(device, num_workers),
    )

    return train_loader_preseg, val_loader_preseg, test_loader_preseg


def save_presegmented_datasets(train_loader, val_loader, test_loader, segmenter, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    train_dataset = _build_presegmented_dataset(train_loader, segmenter, desc='Precomputing train segmentations')
    val_dataset = _build_presegmented_dataset(val_loader, segmenter, desc='Precomputing val segmentations')
    test_dataset = _build_presegmented_dataset(test_loader, segmenter, desc='Precomputing test segmentations')

    torch.save(_serialize_presegmented_dataset(train_dataset), os.path.join(output_dir, 'train.pt'))
    torch.save(_serialize_presegmented_dataset(val_dataset), os.path.join(output_dir, 'val.pt'))
    torch.save(_serialize_presegmented_dataset(test_dataset), os.path.join(output_dir, 'test.pt'))


def _inject_concept_noise_tensor(concepts, noisy_concept_indices, target_noisy_concept_indices, noisy_prob):
    if noisy_prob <= 0.0 or noisy_concept_indices is None or target_noisy_concept_indices is None:
        return concepts

    noisy_concepts = concepts.clone()
    for noisy_idx, target_idx in zip(noisy_concept_indices, target_noisy_concept_indices):
        present_mask = noisy_concepts[..., noisy_idx] == 1.0
        flip_mask = torch.rand(present_mask.shape, device=noisy_concepts.device) < noisy_prob
        to_flip = present_mask & flip_mask
        noisy_concepts[..., noisy_idx][to_flip] = 0.0
        noisy_concepts[..., target_idx][to_flip] = 1.0
    return noisy_concepts


def _inject_concept_noise_presegmented_dataset(dataset, noisy_concept_names, target_noisy_concept_names, noisy_prob, concept_names):
    if noisy_concept_names is None or target_noisy_concept_names is None:
        return dataset

    if len(noisy_concept_names) != len(target_noisy_concept_names):
        raise ValueError("noisy_concept_names and target_noisy_concept_names must have the same length.")

    if concept_names is None:
        raise ValueError("concept_names must be provided to inject concept noise on presegmented datasets.")

    noisy_concept_indices = [concept_names.index(name) for name in noisy_concept_names]
    target_noisy_concept_indices = [concept_names.index(name) for name in target_noisy_concept_names]

    segmented_objects, predicted_masks, concepts, tasks, original_images = dataset.tensors
    concepts_with_noise = _inject_concept_noise_tensor(
        concepts=concepts,
        noisy_concept_indices=noisy_concept_indices,
        target_noisy_concept_indices=target_noisy_concept_indices,
        noisy_prob=noisy_prob,
    )

    print(
        f"[Presegmented] Injecting concept noise: {noisy_concept_names} -> "
        f"{target_noisy_concept_names} with probability {noisy_prob}"
    )

    return TensorDataset(segmented_objects, predicted_masks, concepts_with_noise, tasks, original_images)


def load_presegmented_dataloaders(
    presegmented_datasets_path,
    batch_size,
    device='cpu',
    num_workers=0,
    noisy_concept_names=None,
    target_noisy_concept_names=None,
    noisy_prob=0.0,
    concept_names=None,
):
    train_path = os.path.join(presegmented_datasets_path, 'train.pt')
    val_path = os.path.join(presegmented_datasets_path, 'val.pt')
    test_path = os.path.join(presegmented_datasets_path, 'test.pt')

    split_paths = [
        ('train', train_path),
        ('val', val_path),
        ('test', test_path),
    ]

    for _, required_path in tqdm(split_paths, desc='Checking presegmented splits', unit='split'):
        if not os.path.exists(required_path):
            raise FileNotFoundError(f"Missing presegmented split file: {required_path}")

    datasets = {}
    for split_name, split_path in tqdm(split_paths, desc='Loading presegmented splits', unit='split'):
        datasets[split_name] = _deserialize_presegmented_dataset(torch.load(split_path, map_location='cpu'))

    train_dataset = datasets['train']
    val_dataset = datasets['val']
    test_dataset = datasets['test']

    # Match raw CLEVR-Hans behavior: apply concept noise only to training data.
    train_dataset = _inject_concept_noise_presegmented_dataset(
        train_dataset,
        noisy_concept_names=noisy_concept_names,
        target_noisy_concept_names=target_noisy_concept_names,
        noisy_prob=noisy_prob,
        concept_names=concept_names,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=_get_collate_fn(device, num_workers),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=_get_collate_fn(device, num_workers),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=_get_collate_fn(device, num_workers),
    )

    return train_loader, val_loader, test_loader


# ── CUB-200-2011 Pre-Segmented ──────────────────────────────────────────────────

CUB_PRESEGM_DIR = '/cw/dtaiexp/2026-Stefano/cub/CUB_200_2011/segmentation/presegmented'
CUB_PRESEGM_ATTRIBUTES_FILE = '/cw/dtaiexp/2026-Stefano/cub/CUB_200_2011/attributes/attributes.txt'

CUB_PRESEGM_PARTS = ['beak', 'head', 'body', 'wings', 'tail', 'legs']
CUB_PRESEGM_PART_TO_SLOT = {p: i for i, p in enumerate(CUB_PRESEGM_PARTS)}
CUB_PRESEGM_NUM_PARTS = len(CUB_PRESEGM_PARTS)
CUB_PRESEGM_NUM_CLASSES = 200

CUB_PRESEGM_ATTRIBUTE_PREFIX_TO_PART = {
    'has_bill_shape': 'beak',
    'has_head_pattern': 'head',
    'has_upperparts_color': 'body',
    'has_wing_color': 'wings',
    'has_tail_shape': 'tail',
    'has_leg_color': 'legs',
}


def _parse_cub_attribute_names(attributes_file=CUB_PRESEGM_ATTRIBUTES_FILE):
    """Parse attribute names from the CUB attributes.txt file."""
    attrs = []
    with open(attributes_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            attrs.append(parts[1])  # e.g. "has_bill_shape::curved_(up_or_down)"
    return attrs


def _build_cub_part_to_attribute_indices(attribute_names, prefix_to_part=None, used_parts=None):
    """Build a dict mapping each part name to the list of attribute indices it owns.

    Args:
        attribute_names: list of 312 attribute name strings.
        prefix_to_part: dict mapping attribute prefix -> part name.
            Defaults to CUB_PRESEGM_ATTRIBUTE_PREFIX_TO_PART.
        used_parts: list of part names to include. Defaults to CUB_PRESEGM_PARTS.

    Returns:
        dict[str, list[int]]: part_name -> list of attribute indices.
    """
    if prefix_to_part is None:
        prefix_to_part = CUB_PRESEGM_ATTRIBUTE_PREFIX_TO_PART
    if used_parts is None:
        used_parts = CUB_PRESEGM_PARTS

    part_to_indices = {p: [] for p in used_parts}
    for i, attr_name in enumerate(attribute_names):
        prefix = attr_name.split('::')[0]
        part = prefix_to_part.get(prefix)
        if part is not None and part in part_to_indices:
            part_to_indices[part].append(i)
    return part_to_indices


def _apply_cub_concept_to_part_mask(concepts, part_to_indices, used_parts):
    """Zero-out attribute indices that don't belong to each part.

    Args:
        concepts: tensor of shape (N, num_used_parts, num_attributes).
        part_to_indices: dict mapping part name -> list of attribute indices.
        used_parts: ordered list of part names corresponding to dim-1.

    Returns:
        Masked concepts tensor of the same shape.
    """
    masked = torch.zeros_like(concepts)
    for part_idx, part_name in enumerate(used_parts):
        indices = part_to_indices.get(part_name, [])
        if indices:
            idx_tensor = torch.tensor(indices, dtype=torch.long)
            masked[:, part_idx, idx_tensor] = concepts[:, part_idx, idx_tensor]
    return masked


def load_cub_presegmented_dataloaders(
    presegmented_dir,
    batch_size,
    part_to_attribute_indices,
    used_parts=None,
    val_split=0.1,
    device='cpu',
    num_workers=0,
):
    """Load CUB pre-segmented .pt files and return train/val/test DataLoaders.

    Handles:
    - Splitting train into train+val (cached as val.pt on first run)
    - Applying concept-to-part masking
    - Returning 5-element tuples: (segmented_objects, masks, concepts, tasks, original_images)

    Args:
        presegmented_dir: path to directory containing train.pt (and test.pt).
        batch_size: batch size for DataLoaders.
        part_to_attribute_indices: dict mapping part name -> list of attribute indices.
        used_parts: ordered list of part names. Defaults to CUB_PRESEGM_PARTS.
        val_split: fraction of training data to use as validation.
        device: target device.
        num_workers: DataLoader workers.
    """
    if used_parts is None:
        used_parts = CUB_PRESEGM_PARTS

    train_path = os.path.join(presegmented_dir, 'train.pt')
    test_path = os.path.join(presegmented_dir, 'test.pt')
    val_path = os.path.join(presegmented_dir, 'val.pt')

    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Missing CUB presegmented train file: {train_path}")
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Missing CUB presegmented test file: {test_path}")

    # Load raw data
    print("[CUB-presegm] Loading presegmented data...")
    train_raw = torch.load(train_path, map_location='cpu')
    test_raw = torch.load(test_path, map_location='cpu')

    # Create val split if it doesn't exist
    if os.path.exists(val_path):
        print("[CUB-presegm] Loading cached val.pt...")
        val_raw = torch.load(val_path, map_location='cpu')
        # Reload train_path in case it was already split
    else:
        print(f"[CUB-presegm] Creating val split ({val_split*100:.0f}% of train)...")
        n_train = train_raw['segmented_objects'].shape[0]
        n_val = int(n_train * val_split)
        n_train_new = n_train - n_val

        # Shuffle indices deterministically
        gen = torch.Generator()
        gen.manual_seed(42)
        perm = torch.randperm(n_train, generator=gen)
        train_indices = perm[:n_train_new]
        val_indices = perm[n_train_new:]

        val_raw = {k: v[val_indices] for k, v in train_raw.items()}
        train_raw = {k: v[train_indices] for k, v in train_raw.items()}

        # Save val and updated train for future runs
        torch.save(val_raw, val_path)
        print(f"[CUB-presegm] Saved val.pt ({n_val} samples) to {val_path}")

    def _make_dataset(raw_dict):
        """Deserialize, slice to used parts, and apply concept-to-part mask."""
        # Get slot indices for requested parts
        slot_indices = [CUB_PRESEGM_PART_TO_SLOT[p] for p in used_parts]
        
        # Slice tensors to only include requested parts
        seg_obj = raw_dict['segmented_objects'][:, slot_indices].to(torch.float32)
        masks = raw_dict['predicted_masks'][:, slot_indices].to(torch.float32)
        concepts = raw_dict['concepts'][:, slot_indices].to(torch.float32)
        
        # Keep task and original images as they are
        tasks = raw_dict['tasks'].to(torch.float32)
        orig_imgs = raw_dict['original_images'].to(torch.float32)

        # Apply concept-to-part masking
        concepts = _apply_cub_concept_to_part_mask(concepts, part_to_attribute_indices, used_parts)

        return TensorDataset(seg_obj, masks, concepts, tasks, orig_imgs)

    train_dataset = _make_dataset(train_raw)
    val_dataset = _make_dataset(val_raw)
    test_dataset = _make_dataset(test_raw)

    n_train = len(train_dataset)
    n_val = len(val_dataset)
    n_test = len(test_dataset)
    print(f"[CUB-presegm] Splits: train={n_train}, val={n_val}, test={n_test}")

    collate_fn = _get_collate_fn(device, num_workers)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, collate_fn=collate_fn)

    return train_loader, val_loader, test_loader


def load_cub_presegmented_dataloaders_nonpgcm(
    presegmented_dir,
    batch_size,
    part_to_attribute_indices,
    used_parts=None,
    val_split=0.1,
    device='cpu',
    num_workers=0,
):
    """Load CUB pre-segmented data for non-PGCM models as 4-tuples.

    Reuses `load_cub_presegmented_dataloaders` and converts each split from
    (segmented_objects, masks, concepts, tasks, original_images)
    to
    (original_images, masks, concepts, tasks).
    """
    train_loader_5, val_loader_5, test_loader_5 = load_cub_presegmented_dataloaders(
        presegmented_dir=presegmented_dir,
        batch_size=batch_size,
        part_to_attribute_indices=part_to_attribute_indices,
        used_parts=used_parts,
        val_split=val_split,
        device=device,
        num_workers=num_workers,
    )

    def _to_4tuple_dataset(dataset_5):
        seg_obj, masks, concepts, tasks, orig_imgs = dataset_5.tensors
        return TensorDataset(
            orig_imgs.to(torch.float32),
            masks.to(torch.float32),
            concepts.to(torch.float32),
            tasks.to(torch.float32),
        )

    collate_fn = _get_collate_fn(device, num_workers)

    train_loader = DataLoader(
        _to_4tuple_dataset(train_loader_5.dataset),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        _to_4tuple_dataset(val_loader_5.dataset),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        _to_4tuple_dataset(test_loader_5.dataset),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )

    return train_loader, val_loader, test_loader


def get_mnist_data(train: bool) -> MNIST:
    return torchvision.datasets.MNIST(
        root=str(data_root / "raw/"), train=train, download=True, transform=transform
    )

colors_dict = {'red': (255,0,0), 'blue': (0,0,255), 'green': (0,255,0)}

def get_colors_concepts(X, c, i, j, p, allowed_colors, color_as_concepts=True):
    # it should depend on the number of allowed colors and works independently
    # allowed colors is a list of the allowed colors
    # for example if allowed colors contains only elements then is half the probability for the first and half for the second
    chunks = 1 / len(allowed_colors)
    for k, color in enumerate(allowed_colors):
        if p < (k+1)*chunks:
            X[i][j, 0, :, :] = X[i][j, 0, :, :] * colors_dict[color][0] / 255
            X[i][j, 1, :, :] = X[i][j, 1, :, :] * colors_dict[color][1] / 255
            X[i][j, 2, :, :] = X[i][j, 2, :, :] * colors_dict[color][2] / 255
            if color_as_concepts:
                c[i][j, -len(allowed_colors)+k] = 1.0
            break
    X[i][j] = X[i][j] / 255
    return X, c

def check_eyes_aggregation(masks_names, eyes_aggregation):
    if not eyes_aggregation:
        return True
    left_eye_present = 'l_eye' in masks_names
    right_eye_present = 'r_eye' in masks_names
    return left_eye_present and right_eye_present

def check_lips_aggregation(masks_names, lips_aggregation):
    if not lips_aggregation:
        return True
    left_lip_present = 'l_lip' in masks_names
    upper_lip_present = 'u_lip' in masks_names
    return left_lip_present and upper_lip_present

def celeba_masks_dataset(train: bool, used_masks=['skin', 'nose', 'l_lip', 'u_lip', 'hair', 'l_eye', 'r_eye', 'neck', 'l_brow', 'r_brow'], concepts_for_masks={'skin': [0, 1]}, task_indices=[], noisy_part=None, noisy_target_part=None, noisy_prob=0.2) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    '''
        Loads the CelebA dataset with the specified masks. The data is in the folder in the form of shards. Each shard contain:
        - images = shard['images']          # (N, 3, H, W)
        - masks = shard['masks']            # (N, nb_masks, 1, H, W)
        - concepts = shard['concepts']      # (N, nb_concepts)
        - masks_names = shard['mask_names'] # (N, nb_masks) list of strings where not_found is used for empty masks
    '''
    split = 'train' if train else 'test'
    if train:
        shards = [torch.load(os.path.join(CELEBA_DIR, f'shard_{i:04d}.pt'), mmap=False) for i in TRAIN_SHARDS]
        print(f"Loaded {len(shards)} shards for train set")
    else:
        shards = [torch.load(os.path.join(CELEBA_DIR, f'shard_{i:04d}.pt'), mmap=False) for i in TEST_SHARDS]
        print(f"Loaded {len(shards)} shards for test set")

    if noisy_part is not None and noisy_target_part is not None and len(used_masks) >  1:
        raise ValueError("When injecting noise, used_masks must be a single mask corresponding to the noisy_part.")

    fullmask = 'full' in used_masks
    if fullmask:
        used_masks.remove('full')

    eyes_aggregation = 'eyes' in used_masks
    if eyes_aggregation:
        used_masks.remove('eyes')
    
    lips_aggregation = 'lips' in used_masks
    if lips_aggregation:
        used_masks.remove('lips')
    
    X_list = []
    M_list = []
    c_list = []
    y_list = []
    for shard in tqdm(shards, desc=f"Processing {split} shards"):
        images = shard['images'].float()/255          # (N, 3, H, W)
        masks = shard['masks'].float()            # (N, nb_masks, H, W)
        concepts = shard['concepts']      # (N, nb_concepts)
        masks_names = shard['mask_names'] # (N, nb_masks) list of strings where not_found is used for empty masks

        for i in range(images.shape[0]):
            # append only if it has all the masks
            # mask_indices = [masks_names.index(name) for name in masks_names[i] if name in used_masks]
            mask_indices = [masks_names[i].index(name) for name in used_masks if name in masks_names[i]]
            if len(mask_indices) == len(used_masks) and check_eyes_aggregation(masks_names[i], eyes_aggregation) and check_lips_aggregation(masks_names[i], lips_aggregation):
                X_list.append(images[i])

                # for each used mask, get the associated concepts if provided
                # the final concept list should be (len(used_masks), nb_concepts) masked through concepts_for_masks
                zero_one_concepts = (concepts[i]+1)/2
                if concepts_for_masks is not None:
                    #pad with zero the non selected ones
                    selected_concepts = []
                    for mask_name in used_masks:
                        associated_concept_indices = concepts_for_masks.get(mask_name, [])
                        if len(associated_concept_indices) == 0:
                            raise ValueError(f"No associated concepts provided for mask {mask_name} in concepts_for_masks.")
                        mask_concepts = torch.zeros_like(zero_one_concepts)
                        for idx in associated_concept_indices:
                            mask_concepts[idx] = zero_one_concepts[idx]
                        selected_concepts.append(mask_concepts)

                    if eyes_aggregation:
                        # aggregate left and right eye
                        associated_concept_inddex_left_eye = concepts_for_masks.get('l_eye', [])
                        associated_concept_inddex_right_eye = concepts_for_masks.get('r_eye', [])
                        eyes_concepts = torch.zeros_like(zero_one_concepts)
                        # apply the index to zero one concepts, then sum them and clamp to 1
                        for idx in associated_concept_inddex_left_eye:
                            eyes_concepts[idx] += zero_one_concepts[idx]
                        for idx in associated_concept_inddex_right_eye:
                            eyes_concepts[idx] += zero_one_concepts[idx]
                        eyes_concepts = eyes_concepts.clamp(max=1.0)
                        selected_concepts.append(eyes_concepts)
                    
                    if lips_aggregation:
                        # aggregate lower and upper lip
                        associated_concept_inddex_lower_lip = concepts_for_masks.get('l_lip', [])
                        associated_concept_inddex_upper_lip = concepts_for_masks.get('u_lip', [])
                        lips_concepts = torch.zeros_like(zero_one_concepts)
                        # apply the index to zero one concepts, then sum them and clamp to 1
                        for idx in associated_concept_inddex_lower_lip:
                            lips_concepts[idx] += zero_one_concepts[idx]
                        for idx in associated_concept_inddex_upper_lip:
                            lips_concepts[idx] += zero_one_concepts[idx]
                        lips_concepts = lips_concepts.clamp(max=1.0)
                        selected_concepts.append(lips_concepts)
                    
                    if not selected_concepts == []:
                        c_list.append(torch.stack(selected_concepts, dim=0))  # (nb_used_masks, nb_concepts)
                else:
                    c_list.append(zero_one_concepts)  # (nb_concepts,)

                # return also the interesting masks as output
                interesting_masks = [masks[i, j] for j in mask_indices]

                if eyes_aggregation:
                    # aggregate left and right eye
                    left_eye_index = masks_names[i].index('l_eye')
                    right_eye_index = masks_names[i].index('r_eye')
                    left_eye_mask = masks[i, left_eye_index]
                    right_eye_mask = masks[i, right_eye_index]
                    eyes_mask = torch.maximum(left_eye_mask, right_eye_mask)
                    interesting_masks.append(eyes_mask)

                if lips_aggregation:
                    # aggregate lower and upper lip
                    lower_lip_index = masks_names[i].index('l_lip')
                    upper_lip_index = masks_names[i].index('u_lip')
                    lower_lip_mask = masks[i, lower_lip_index]
                    upper_lip_mask = masks[i, upper_lip_index]
                    lips_mask = torch.maximum(lower_lip_mask, upper_lip_mask)
                    interesting_masks.append(lips_mask)

                if not interesting_masks == []:
                    M_list.append(torch.stack(interesting_masks, dim=0))  # (nb_used_masks, 1, H, W)

                if fullmask:
                    M_list.append(torch.ones((1, masks.shape[2], masks.shape[3])))  # full mask
                    # c_list.append(torch.zeros_like(zero_one_concepts))  # no concepts for full mask
                    c_list.append(zero_one_concepts)  # no concepts for full mask

                # now the task labels
                if len(task_indices) > 0:
                    task_labels = zero_one_concepts[task_indices]
                    y_list.append(task_labels)
            else:
                continue

    X = torch.stack(X_list, 0)
    M = torch.stack(M_list, 0)
    c = torch.stack(c_list, 0)

    if len(c.shape) == 2:
        c = c.unsqueeze(1)

    if len(task_indices) > 0:
        y = torch.stack(y_list, 0)
    else:
        y = torch.ones((X.shape[0], 1))
    
    if noisy_part is not None and noisy_target_part is not None:
        # images with label noisy_part will be changed to noisy_target_part with probability noisy_prob
        for i in range(c.shape[0]):
            if c.shape[1] != 1:
                raise ValueError("When injecting noise, concepts tensor c must have shape (N, 1, nb_concepts).")
            if c[i,0,noisy_part] == 1.0:
                if random.random() < noisy_prob:
                    c[i,0,noisy_part] = 0.0
                    c[i,0,noisy_target_part] = 1.0

    # loop over all Xs, loop over all masks for it, apply each mask to the image, save to disk (fixed name, overrides)
    # for idx in range(100):
    #     img = X[idx].numpy().transpose(1, 2, 0)
    #     import matplotlib.pyplot as plt
    #     plt.imsave(f'celeba_image.png', img)
    #     print('FBWEJBFWHBFJrh')
    #     for m_idx in range(M.shape[1]):
    #         mask = M[idx, m_idx].numpy().squeeze()
    #         masked_img = img * mask[:, :, None]
    #         plt.imsave(f'celeba_image_masked.png', masked_img)
    #         input('Life is short...')

    return X, M, c, y


def addition_dataset(train, num_digits, digit_limit=10, allowed_digits=[0,1],  allowed_colors=['red', 'blue', 'green'], color_as_concepts=True, noisy_digit=None, noisy_prob=0.2, noisy_target_digit=None):
    dataset = get_mnist_data(train)
    X, y = dataset.data, dataset.targets

    X = X[torch.isin(y, torch.tensor(allowed_digits))]
    y = y[torch.isin(y, torch.tensor(allowed_digits))]

    # if noisy_digit is not None and noisy_target_digit is not None:
    #     noise_mask = torch.rand(len(y)) < noisy_prob
    #     noisy_digit_tensor = torch.tensor(noisy_digit)
    #     noisy_target_digit_tensor = torch.tensor(noisy_target_digit)
    #     y[noise_mask] = torch.where(y[noise_mask] == noisy_digit_tensor, noisy_target_digit_tensor, y[noise_mask])

    # i want a version that works if noisy_digit is a list
    if noisy_digit is not None and noisy_target_digit is not None:
        if isinstance(noisy_digit, int) and isinstance(noisy_target_digit, int):
            print("Injecting noise into MNIST dataset with single noisy digit...")
            noise_mask = torch.rand(len(y)) < noisy_prob
            noisy_digit_tensor = torch.tensor(noisy_digit)
            noisy_target_digit_tensor = torch.tensor(noisy_target_digit)
            y[noise_mask] = torch.where(y[noise_mask] == noisy_digit_tensor, noisy_target_digit_tensor, y[noise_mask])
        # if is a list of noisy digits
        elif isinstance(noisy_digit, List) and isinstance(noisy_target_digit, List):
            print("Injecting noise into MNIST dataset with multiple noisy digits...")
            for nd, ntd in zip(noisy_digit, noisy_target_digit):
                noise_mask = torch.rand(len(y)) < noisy_prob
                noisy_digit_tensor = torch.tensor(nd)
                noisy_target_digit_tensor = torch.tensor(ntd)
                y[noise_mask] = torch.where(y[noise_mask] == noisy_digit_tensor, noisy_target_digit_tensor, y[noise_mask])
        else:
            raise ValueError("noisy_digit and noisy_target_digit must be both int or both List[int].") 



    X = torch.unsqueeze(X, 1).float()
    size = len(X) // num_digits
    X, y = torch.split(X, size), torch.split(y, size)

    # # i want to apply noise nly at the digits appearing as first in the tuples in X
    # if noisy_digit is not None and noisy_target_digit is not None:
    #     print("Injecting noise into MNIST addition dataset...")
    #     if isinstance(noisy_digit, int) and isinstance(noisy_target_digit, int):
    #         noise_mask = torch.rand(len(y[0])) < noisy_prob
    #         noisy_digit_tensor = torch.tensor(noisy_digit)
    #         noisy_target_digit_tensor = torch.tensor(noisy_target_digit)
    #         y[0][noise_mask] = torch.where(y[0][noise_mask] == noisy_digit_tensor, noisy_target_digit_tensor, y[0][noise_mask])
    #     # if is a list of noisy digits
    #     elif isinstance(noisy_digit, List) and isinstance(noisy_target_digit, List):
    #         for nd, ntd in zip(noisy_digit, noisy_target_digit):
    #             noise_mask = torch.rand(len(y[0])) < noisy_prob
    #             noisy_digit_tensor = torch.tensor(nd)
    #             noisy_target_digit_tensor = torch.tensor(ntd)
    #             y[0][noise_mask] = torch.where(y[0][noise_mask] == noisy_digit_tensor, noisy_target_digit_tensor, y[0][noise_mask])
    #     else:
    #         raise ValueError("noisy_digit and noisy_target_digit must be both int or both List[int].")

    if len(X) % num_digits != 0:
        X = X[:-1]
        y = y[:-1]

    if color_as_concepts:
        c = [torch.zeros((len(X[0]), len(allowed_digits)+len(allowed_colors))).float() for _ in range(len(X))]
    else:
        c = [torch.zeros((len(X[0]), len(allowed_digits))).float() for _ in range(len(X))]
    
    for i, ys in enumerate(y):
        for j, yi in enumerate(ys):
            c[i][j, yi] = 1.0

    y = torch.sum(torch.stack(y, 0), 0)

    ## Color the digits
    # X is tuple, 2 images, each has shape (nb_data, 1, 28, 28)
    X = [xi.repeat(1, 3, 1, 1) for xi in X]

    # new: pad to each has double width
    # X = [F.pad(xi, (0, xi.shape[3], 0, 0), "constant", 0) for xi in X]  # pad right side with zeros
    
    for i in range(len(X)):
        for j in range(X[i].shape[0]):
            p = random.random()
        #     X, c = get_colors_concepts(X, c, i, j, p, allowed_colors)
        #     if p < 1/3:
        #         X[i][j, 1:3, :, :] = 0  # keep only red channel
        #         if color_as_concepts:
        #             c[i][j, -3] = 1.0  # red  TODO
        #     elif p < 2/3:
        #         X[i][j, 0:2, :, :] = 0  # keep only blue channel
        #         if color_as_concepts:
        #             c[i][j, -2] = 1.0  # TODO
        #     else:
        #         X[i][j, 0, :, :] = 0  # keep only green channel
        #         X[i][j, 2, :, :] = 0
        #         if color_as_concepts:
        #             c[i][j, -1] = 1.0  # TODO
        # X[i] = X[i] / 255
            X, c = get_colors_concepts(X, c, i, j, p, allowed_colors, color_as_concepts=color_as_concepts)

    return X, c, y


def create_single_digit_addition(num_digits, digit_limit=10, allowed_digits=[0, 1]):
    concept_names = ["x%d%d" % (i, j) for i, j in product(range(num_digits), range(digit_limit))]


    sums = defaultdict(list)
    for d in product(*[range(len(allowed_digits)) for _ in range(num_digits)]):
        conj = []
        z = 0
        for i, n in enumerate(d):
            conj.append("x%d%d" % (i,n))
            z += n
        sums[z].append("(" + " & ".join(conj) + ")")


    explanations = {}
    class_names = ["z%d" % z for z in range(len(allowed_digits)*num_digits - num_digits + 1)]
    for z in range(len(allowed_digits)*num_digits - num_digits + 1):

        explanations["z%d" % z] = {"name": "%d" % z,
                                   "explanation": "(" + " | ".join(sums[z]) + ")"}

    return concept_names, class_names, explanations


def get_mnist(batch_size, digit_limit=10, allowed_digits=[0, 1], allowed_colors=['red', 'blue', 'green'], color_as_concepts=True, device='cpu', noisy_digit=None, noisy_prob=0.2, noisy_target_digit=None, num_workers=0):
    def get_mask(x):
        shape = x.shape[0], x.shape[2], x.shape[3]  # batch, H, W
        m0, m1 = torch.ones(shape), torch.ones(shape)
        m0[:, (x.shape[2] // 2):, :] = 0
        m1[:, :(x.shape[2] // 2), :] = 0
        m = torch.stack([m0, m1], dim=1)
        return m
    x_train, c_train, y_train = addition_dataset(True, 2, digit_limit, allowed_digits, allowed_colors=allowed_colors, color_as_concepts=color_as_concepts, noisy_digit=noisy_digit, noisy_prob=noisy_prob, noisy_target_digit=noisy_target_digit)
    x_test, c_test, y_test = addition_dataset(False, 2, digit_limit, allowed_digits, allowed_colors=allowed_colors, color_as_concepts=color_as_concepts)
    x_train = torch.cat(x_train, dim=2)
    c_train = torch.stack(c_train, dim=1)
    y_train = F.one_hot(y_train.unsqueeze(-1).long().ravel()).float()
    m_train = get_mask(x_train)

    val_split = 0.1
    train_set_size = int(len(x_train) * (1 - val_split))
    x_val, c_val, y_val = x_train[train_set_size:], c_train[train_set_size:], y_train[train_set_size:]
    x_train, c_train, y_train = x_train[:train_set_size], c_train[:train_set_size], y_train[:train_set_size]
    m_val = m_train[train_set_size:]
    m_train = m_train[:train_set_size]

    x_test = torch.cat(x_test, dim=2)
    c_test = torch.stack(c_test, dim=1)
    y_test = F.one_hot(y_test.unsqueeze(-1).long().ravel()).float()
    m_test = get_mask(x_test)

    # take first 1000 images from each set for faster testing
    # x_train, m_train, c_train, y_train = x_train[:1000], m_train[:1000], c_train[:1000], y_train[:1000]
    # x_test, m_test, c_test, y_test = x_test[:1000], m_test[:1000], c_test[:1000], y_test[:1000]
    # x_val, m_val, c_val, y_val = x_val[:1000], m_val[:1000], c_val[:1000], y_val[:1000]
    
    train_loader = DataLoader(TensorDataset(x_train, m_train, c_train, y_train), batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=_get_collate_fn(device, num_workers))
    test_loader = DataLoader(TensorDataset(x_test, m_test, c_test, y_test), batch_size=batch_size, num_workers=num_workers, collate_fn=_get_collate_fn(device, num_workers))
    val_loader = DataLoader(TensorDataset(x_val, m_val, c_val, y_val), batch_size=batch_size, num_workers=num_workers, collate_fn=_get_collate_fn(device, num_workers))

    # import matplotlib.pyplot as plt
    # for x, c, y in train_loader:
    #     for i in range(5):
    #         img = x[i].cpu().numpy().transpose(1, 2, 0)
    #         plt.imshow(img)
    #         plt.show()
    #     break


    return train_loader, val_loader, test_loader

def get_celeba_dataset(batch_size, used_masks=[], concepts_for_masks=None, task_indices=[], noisy_part=None, noisy_target_part=None, noisy_prob=0.2, device='cpu', num_workers=0):
    
    if len(used_masks) == 0:
        raise ValueError("used_masks must be a non-empty list of mask names.")
    
    used_mask_copy = used_masks.copy()
    xtrain, mtrain, ctrain, ytrain = celeba_masks_dataset(train=True, used_masks=used_mask_copy, concepts_for_masks=concepts_for_masks, task_indices=task_indices, noisy_part=noisy_part, noisy_target_part=noisy_target_part, noisy_prob=noisy_prob)

    used_mask_copy = used_masks.copy()
    xtest, mtest, ctest, ytest = celeba_masks_dataset(train=False, used_masks=used_mask_copy, concepts_for_masks=concepts_for_masks, task_indices=task_indices)

    val_split = 0.1
    train_set_size = int(len(xtrain) * (1 - val_split))
    xval, mval, cval, yval = xtrain[train_set_size:], mtrain[train_set_size:], ctrain[train_set_size:], ytrain[train_set_size:]
    xtrain, mtrain, ctrain, ytrain = xtrain[:train_set_size], mtrain[:train_set_size], ctrain[:train_set_size], ytrain[:train_set_size]
   
    train_loader = DataLoader(TensorDataset(xtrain, mtrain, ctrain, ytrain), batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=_get_collate_fn(device, num_workers))
    val_loader = DataLoader(TensorDataset(xval, mval, cval, yval), batch_size=batch_size, num_workers=num_workers, collate_fn=_get_collate_fn(device, num_workers))
    test_loader = DataLoader(TensorDataset(xtest, mtest, ctest, ytest), batch_size=batch_size, num_workers=num_workers, collate_fn=_get_collate_fn(device, num_workers))

    return train_loader, val_loader, test_loader


# ── CLEVR-Hans3 ──────────────────────────────────────────────────────────────────

CLEVR_HANS_SIZES = ['large', 'small']
CLEVR_HANS_COLORS = ['gray', 'red', 'blue', 'green', 'brown', 'purple', 'cyan', 'yellow']
CLEVR_HANS_SHAPES = ['cube', 'sphere', 'cylinder']
CLEVR_HANS_MATERIALS = ['rubber', 'metal']

CLEVR_HANS_ALL_CONCEPTS = (
    [f'size_{s}' for s in CLEVR_HANS_SIZES] +
    [f'color_{c}' for c in CLEVR_HANS_COLORS] +
    [f'shape_{s}' for s in CLEVR_HANS_SHAPES] +
    [f'material_{m}' for m in CLEVR_HANS_MATERIALS]
)  # 15 concepts total

CLEVR_HANS_NUM_CONCEPTS = len(CLEVR_HANS_ALL_CONCEPTS)  # 15
CLEVR_HANS_NUM_CLASSES = 3
CLEVR_HANS_MAX_OBJECTS = 10


def _clevrhans_obj_to_concept_vec(obj: dict) -> torch.Tensor:
    """Convert a CLEVR-Hans object dict to a one-hot concept vector of length 15."""
    vec = torch.zeros(CLEVR_HANS_NUM_CONCEPTS)
    # Size (2)
    vec[CLEVR_HANS_SIZES.index(obj['size'])] = 1.0
    # Color (8) — offset by 2
    vec[2 + CLEVR_HANS_COLORS.index(obj['color'])] = 1.0
    # Shape (3) — offset by 10
    vec[10 + CLEVR_HANS_SHAPES.index(obj['shape'])] = 1.0
    # Material (2) — offset by 13
    vec[13 + CLEVR_HANS_MATERIALS.index(obj['material'])] = 1.0
    return vec


def clevrhans_dataset(split: str, resolution: int = 128, data_dir: str = None, noisy_concept_names=None, target_noisy_concept_names=None, noisy_prob=0.0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Load CLEVR-Hans3 split and return (X, M, C, Y) tensors.

    X: (N, 3, res, res) — images resized to resolution×resolution, RGB, [0,1]
    M: (N, MAX_OBJECTS, res, res) — binary bbox masks per object slot
    C: (N, MAX_OBJECTS, NUM_CONCEPTS) — one-hot concept vectors per object slot
    Y: (N, NUM_CLASSES) — one-hot class labels
    
    Args:
        noisy_concept_names: List of concept names to corrupt (e.g., ['color_red', 'size_large'])
        target_noisy_concept_names: List of concept names to corrupt them into
        noisy_prob: Probability of flipping each occurrence
    """
    from PIL import Image

    if data_dir is None:
        data_dir = CLEVR_HANS_DIR

    # Validate and map concept names to indices
    noisy_concept_indices = None
    target_noisy_concept_indices = None
    if noisy_concept_names is not None and target_noisy_concept_names is not None:
        try:
            noisy_concept_indices = [CLEVR_HANS_ALL_CONCEPTS.index(name) for name in noisy_concept_names]
            target_noisy_concept_indices = [CLEVR_HANS_ALL_CONCEPTS.index(name) for name in target_noisy_concept_names]
            print(f"[CLEVR-Hans3] Injecting noise: concepts {noisy_concept_names} → {target_noisy_concept_names} with probability {noisy_prob}")
        except ValueError as e:
            raise ValueError(f"Invalid concept name. Valid concepts are: {CLEVR_HANS_ALL_CONCEPTS}") from e

    scenes_dir = os.path.join(data_dir, split, 'scenes')
    images_dir = os.path.join(data_dir, split, 'images')

    scene_files = sorted([f for f in os.listdir(scenes_dir) if f.endswith('.json')])
    print(f"[CLEVR-Hans3] Loading {len(scene_files)} scenes from '{split}' split (resolution={resolution})...")

    resize_transform = transforms.Compose([
        transforms.Resize((resolution, resolution)),
        transforms.ToTensor(),  # converts to [0,1] float and (C,H,W)
    ])

    X_list, M_list, C_list, Y_list = [], [], [], []

    for scene_file in tqdm(scene_files, desc=f"  {split}"):
        with open(os.path.join(scenes_dir, scene_file), 'r') as f:
            scene = json.load(f)

        # Load and resize image (RGBA → RGB)
        img_path = os.path.join(images_dir, scene['image_filename'])
        img = Image.open(img_path).convert('RGB')
        orig_w, orig_h = img.size  # PIL gives (width, height)
        img_tensor = resize_transform(img)  # (3, res, res)

        # Build masks and concepts for each object (up to MAX_OBJECTS)
        masks = torch.zeros(CLEVR_HANS_MAX_OBJECTS, resolution, resolution)
        concepts = torch.zeros(CLEVR_HANS_MAX_OBJECTS, CLEVR_HANS_NUM_CONCEPTS)

        for obj in scene['objects']:
            idx = obj['idx'] - 1  # scene JSONs use 1-based indexing
            if idx >= CLEVR_HANS_MAX_OBJECTS:
                continue

            # Build bbox mask — scale bbox from original resolution to target resolution
            bx, by, bw, bh = obj['bbox']
            # Scale to target resolution
            sx = resolution / orig_w
            sy = resolution / orig_h
            x1 = max(0, int(bx * sx))
            y1 = max(0, int(by * sy))
            x2 = min(resolution, int((bx + bw) * sx))
            y2 = min(resolution, int((by + bh) * sy))
            masks[idx, y1:y2, x1:x2] = 1.0

            # Build concept vector
            concepts[idx] = _clevrhans_obj_to_concept_vec(obj)

        # Build task label (one-hot class_id)
        task = torch.zeros(CLEVR_HANS_NUM_CLASSES)
        task[scene['class_id']] = 1.0

        X_list.append(img_tensor)
        M_list.append(masks)
        C_list.append(concepts)
        Y_list.append(task)

    X = torch.stack(X_list, 0)  # (N, 3, res, res)
    M = torch.stack(M_list, 0)  # (N, MAX_OBJECTS, res, res)
    C = torch.stack(C_list, 0)  # (N, MAX_OBJECTS, NUM_CONCEPTS)
    Y = torch.stack(Y_list, 0)  # (N, NUM_CLASSES)

    print(f"  Loaded: X={X.shape}, M={M.shape}, C={C.shape}, Y={Y.shape}")
    
    # Apply noise to concepts if requested (vectorized for efficiency)
    if noisy_concept_indices is not None and target_noisy_concept_indices is not None:
        print(f"[CLEVR-Hans3] Applying noise to {len(X)} samples...")
        for noisy_idx, target_idx in zip(noisy_concept_indices, target_noisy_concept_indices):
            # Find where the noisy concept is present
            mask = C[:, :, noisy_idx] == 1.0
            # Generate random values for all locations at once
            noise_flip = torch.rand(C.shape[0], C.shape[1]) < noisy_prob
            # Combine masks - only flip where concept exists AND random flip succeeds
            to_flip = mask & noise_flip
            # Apply the noise
            C[to_flip, noisy_idx] = 0.0
            C[to_flip, target_idx] = 1.0
    
    return X, M, C, Y


def get_clevrhans_dataset(batch_size, resolution=128, data_dir=None, device='cpu', num_workers=0, noisy_concept_names=None, target_noisy_concept_names=None, noisy_prob=0.0):
    """Load CLEVR-Hans3 train/val/test and return DataLoaders.
    
    Args:
        noisy_concept_names: List of concept names to corrupt in train/val sets
        target_noisy_concept_names: List of target concept names for corruption
        noisy_prob: Probability of flipping each occurrence
    """
    # Apply noise to training and validation sets; keep test clean.
    xtrain, mtrain, ctrain, ytrain = clevrhans_dataset('train', resolution=resolution, data_dir=data_dir,
                                                        noisy_concept_names=noisy_concept_names,
                                                        target_noisy_concept_names=target_noisy_concept_names,
                                                        noisy_prob=noisy_prob)
    xval, mval, cval, yval = clevrhans_dataset('val', resolution=resolution, data_dir=data_dir,
                                               noisy_concept_names=noisy_concept_names,
                                               target_noisy_concept_names=target_noisy_concept_names,
                                               noisy_prob=noisy_prob)
    xtest, mtest, ctest, ytest = clevrhans_dataset('test', resolution=resolution, data_dir=data_dir)

    train_loader = DataLoader(TensorDataset(xtrain, mtrain, ctrain, ytrain), batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=_get_collate_fn(device, num_workers))
    val_loader = DataLoader(TensorDataset(xval, mval, cval, yval), batch_size=batch_size, num_workers=num_workers, collate_fn=_get_collate_fn(device, num_workers))
    test_loader = DataLoader(TensorDataset(xtest, mtest, ctest, ytest), batch_size=batch_size, num_workers=num_workers, collate_fn=_get_collate_fn(device, num_workers))

    return train_loader, val_loader, test_loader


def get_cub_dataset(batch_size, dataset_root=None, device='cpu', num_workers=0, shuffle_train=False):
    """Load precomputed CUB tensors and return DataLoaders plus basic metadata."""
    dataset_root = DEFAULT_CUB_ROOT if dataset_root is None else dataset_root
    train_loader, val_loader, test_loader, n_concepts, n_tasks = get_cub_dataloaders(
        batch_size=batch_size,
        dataset_root=dataset_root,
        shuffle_train=shuffle_train,
        num_workers=num_workers,
        device=device,
    )
    sample_x = train_loader.dataset.x
    resolution = tuple(sample_x.shape[-2:]) if sample_x.ndim >= 4 else None
    return train_loader, val_loader, test_loader, n_concepts, n_tasks, resolution


if __name__ == "__main__":
    X, M, c, y = celeba_masks_dataset(train=True)
    print(X.shape, M.shape, c.shape, y.shape)
    X, M, c, y = celeba_masks_dataset(train=False)
    print(X.shape, M.shape, c.shape, y.shape)


