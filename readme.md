# Prototype-Grounded Concept Models for Verifiable Concept Alignment (ICML 2026)

PGCMs learn a set of prototype embeddings that are jointly decoded into concept probability vectors and visual representations.
Each concept prediction can be traced back to a specific prototype, enabling *verifiable* concept alignment: users can inspect whether the model's learned concepts truly match the intended semantics, rather than relying solely on aggregate accuracy metrics.

---

## Overview

| Module | Description |
|---|---|
| `main.py` | Training & evaluation entry point for PGCM and all baselines |
| `model.py` | PGCM `LightningModule` — prototype learning, swapping, reconstruction |
| `competitors.py` | Baseline models: CBM, CRM, CMR, DNN |
| `neural_networks.py` | Shared building blocks: embedders, decoders, segmenters, concept/task heads |
| `dataset.py` | Data loaders and presegmentation utilities for all benchmarks |
| `pretrain_segmenter.py` | Standalone segmenter pretraining and presegmented dataset generation |
| `intervene.py` | Post-hoc prototype inspection and concept intervention tool |
| `plot_intervenability.py` | Intervenability curve generation across concept intervention budgets |
| `utils.py` | Metrics (balanced accuracy), visualization, entropy schedules |
| `datasets/` | Scripts to create the datasets |

---

## Installation

```bash
git clone https://github.com/StefanoColamonaco/HigherOrderCBMs.git
cd HigherOrderCBMs
pip install -r requirements.txt
```

Requires Python ≥ 3.12 and a CUDA-capable GPU for reasonable training times.

---

## Datasets

The project uses MNIST Addition, CelebA-Mask, CLEVR-Hans3, and CUB-200.

For detailed information about each dataset, including characteristics and instructions on how to download or generate them, please see the [Datasets README](datasets/README.md).

---

## Training

All experiments are launched through `main.py` with a YAML config file.

### PGCM

```bash
# MNIST (with presegmented masks)
python main.py --config configs/config_mnist_pre_segm.yml --device cuda:0

# CelebA-Mask (with presegmented masks)
python main.py --config configs/config_celebamask_pre_segm.yml --device cuda:0

# CLEVR-Hans3 (with presegmented masks)
python main.py --config configs/config_clevrhans_pre_segm.yml --device cuda:0

# CUB-200 feature embeddings
python main.py --config configs/config_cubEMB.yml --device cuda:0
```

### Competitor baselines

You can manually change the competitor (CBM / CRM / CMR / DNN) in the correspond YAML files by changing `model:` in the config.

```bash
# Competitors on MNIST
python main.py --config configs/config_competitors_mnist.yml --device cuda:0

# Competitors on CUB embeddings
python main.py --config configs/config_cubEMB_competitors.yml --device cuda:0
```

### Key config parameters

| Parameter | Effect |
|---|---|
| `nb_proto` | Number of learned prototypes |
| `lam_entropy` | Entropy regularization on prototype assignment (sharpness) |
| `lam_reconstruction` | Weight of the reconstruction loss |
| `lam_segmentation` | Weight of the segmentation loss |
| `intv_prob` | Probability of random concept interventions during training |
| `use_pretrained_segmenter` | Skip online segmenter; load presegmented tensors instead |
| `presegmented_datasets_path` | Path to saved presegmented `.pt` files |
| `concepts_to_task` | `"thresholding"` (use predicted concepts) or `"ground truth"` |
| `use_balanced_accuracy` | Use balanced accuracy for concept metrics (recommended for CUB) |

---

## Segmenter Pretraining

For image-based datasets (MNIST, CelebA-Mask, CLEVR-Hans3), PGCM benefits from a pretrained object segmenter.  The two-stage workflow is:

### 1. Train the segmenter

```bash
# MNIST
python pretrain_segmenter.py --config configs/config_segmenter_mnist.yml --device cuda:0

# CelebA-Mask
python pretrain_segmenter.py --config configs/config_segmenter_celebamask.yml --device cuda:0
```

This trains a ResNet-UNet on ground-truth masks, saves the segmenter weights, and writes presegmented tensors to disk.

### 2. Point the PGCM config to the presegmented output

In your PGCM config, set:

```yaml
use_pretrained_segmenter: True
presegmented_datasets_path: /path/to/outputs/presegmented_datasets
```

---

## Post-hoc Prototype Editing

The following file allows manual inspection and editing of learned prototypes after training:

```bash
python intervene.py
```

This tool supports:
- Visualizing each prototype's decoded image and concept probabilities
- Masking (deleting) prototypes that do not represent meaningful concepts
- Forcing specific concept values for individual prototypes
- Re-evaluating task and concept accuracy after edits

This can be used to replicate the model interventions experiment of the paper, editing models trained on noisy concept labels (see below).

The framework supports injecting controlled concept noise to study robustness. See `configs/config_intervention.yml` for a complete example.

---

## Outputs

Each run creates a timestamped directory under `new_outputs/` containing:

```
new_outputs/<timestamp>_<dataset>_<extra>/
├── outputs/           # Prototype visualizations, args dump
│   └── prototypes/    # Decoded prototype images per epoch
├── checkpoints/       # Lightning checkpoints (best before/after swap)
└── wandb/             # Weights & Biases local logs
```


---

## Citation

If you use this code in your research, please cite:

```bibtex
@article{colamonaco2026prototype,
  title={Prototype-Grounded Concept Models for Verifiable Concept Alignment},
  author={Colamonaco, Stefano and Debot, David and Barbiero, Pietro and Marra, Giuseppe},
  journal={arXiv preprint arXiv:2604.16076},
  year={2026}
}
```
