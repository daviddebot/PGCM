# Datasets

| Dataset | Key | Input | Objects | Concepts | Task |
|---|---|---|---|---|---|
| MNIST Addition | `mnist` | 56×28 RGB | 2 digits | 10 digit classes | Sum (19 classes) |
| CelebA-Mask | `celebamask` | 128×128 RGB | 4 face parts | 40 attributes | Attractive / Male / Young |
| CLEVR-Hans3 | `clevrhans` | 128×128 RGB | 10 object slots | 15 (size, color, shape, material) | 3 compositional rules |
| CUB-200 Embeddings | `cubEMB` | Pretrained features | 1 | 112 binary | 200 species |

**Data paths** are configured as constants at the top of `dataset.py` (in the root folder) and in YAML config files under `configs/`.  Update them to point to your local copies of each dataset before running.

### Getting the datasets

- **MNIST** is downloaded automatically the first time you run a loader, because the code uses `download=True` in the dataset setup.
- **CelebA** is not downloaded by the code. Place the preprocessed shard files in the directory referenced by `CELEBA_DIR` in `dataset.py`. The shards can be acquired by downloading CelebA from [here](https://www.kaggle.com/datasets/ipythonx/celebamaskhq) and running `build_celebamask.py`.
- **CLEVR-Hans3**: download CLEVR-Hans3 from [here](https://github.com/ml-research/CLEVR-Hans), run `sam_segment.py` and then `build_presegmented_clevr.py`.
- **CUB-200 embeddings**: download CUB and then run `build_cub_embeddings.py`.

If you move the data, update the constants in `dataset.py` and the relevant values in `configs/` before launching a run.
