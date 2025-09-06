# Project: PLE Census‑Income (KDD) Preparation

> IMPORTANT: Use Python 3.10+ and activate a virtual environment before running. Example (macOS/Linux): `python -m venv .venv && source .venv/bin/activate`.

## Goal
Prepare the Census‑Income (KDD) dataset for multi‑task learning as in PLE:
- Task 1: predict income >50K (binary)
- Task 2: predict marital‑status == "Never married" (binary)

Outputs are train/val/test NumPy arrays and lightweight PyTorch dataset/dataloader utilities.

---

## Folder Structure (self‑contained)
- `prepare_census_income.py` — main CLI to fetch, preprocess, split, and save artifacts
- `dataset.py` — `CensusKDDDataset` and `make_dataloaders` utilities
- `requirements.txt` — minimal dependencies for this subproject
- `README.md` — quickstart, examples, and usage
- `__init__.py` — enables package imports

Artifacts written to `--output_dir`:
- `X_{train,val,test}.npy` (float32)
- `y_income_{train,val,test}.npy` (uint8)
- `y_nevermarried_{train,val,test}.npy` (uint8)
- `feature_meta.json` (column lists, one‑hot categories, scaler stats, sklearn version, pipeline hash)

---

## Data Source
- Primary: OpenML "Census‑Income (KDD)" via `sklearn.datasets.fetch_openml(as_frame=True)`
- Fallback: UCI repository with a known schema; rows concatenated from `data` and `test` if both are available

Targets are detected robustly:
- Income column: inferred among names like `income`, `class`, `>50K`, `<=50K`; mapped to `{0: <=50K, 1: >50K}`
- Marital column: any column containing `marital`; `"Never married" → 1`, otherwise `0`

---

## Feature Processing
- Auto infer types: `object` → categorical, others → numeric (excluding targets)
- Numeric: median impute → StandardScaler
- Categorical: mode impute → OneHotEncoder(`handle_unknown="ignore"`, `min_frequency=<arg>`, dense output)
- ColumnTransformer outputs a single dense `np.float32` matrix

---

## Splits
Two‑stage split with stratification on income>50K:
1) Train+Val vs Test using `--test_size`
2) Train vs Val using `val_size / (1 - test_size)`

Printed report includes split sizes, final feature dimensionality, and class distributions for both tasks.

---

## CLI Usage
Run from the repository root or this folder:

```
python ple_experiment/prepare_census_income.py \
  --output_dir ./data/census_kdd \
  --test_size 0.15 \
  --val_size 0.10 \
  --batch_size 4096 \
  --num_workers 4 \
  --onehot_min_freq 10 \
  --seed 42
```

---

## Dev Setup
- Create/activate venv (example):
  - `python -m venv .venv && source .venv/bin/activate`
- Install deps for this subproject:
  - `python -m pip install -r ple_experiment/requirements.txt`

### Colab Quickstart (copy/paste)
```
repo_url = 'https://github.com/allyoushawn/recsys_playground.git'
repo_dir = 'recsys_playground'

import os, sys
if not os.path.exists(repo_dir):
    !git clone $repo_url
%cd $repo_dir
!pip -q install -r ple_experiment/requirements.txt

# Run prep
!python ple_experiment/prepare_census_income.py \
  --output_dir ./data/census_kdd \
  --test_size 0.15 \
  --val_size 0.10 \
  --batch_size 4096 \
  --num_workers 4 \
  --onehot_min_freq 10 \
  --seed 42

# Minimal load
from ple_experiment.dataset import CensusKDDDataset, make_dataloaders
ds = CensusKDDDataset('./data/census_kdd', split='train')
print('Train shapes:', ds.X.shape, ds.y_income.shape, ds.y_never.shape)
```

---

## Validations & Reproducibility
- Global seeds set for `random`, `numpy`, and `torch`
- Basic checks on saved arrays: no NaNs in `X_*`, `X_*` is `float32`, labels in `{0,1}`, shapes consistent
- `feature_meta.json` records numeric scaler stats, one‑hot categories, sklearn version, and a pipeline parameter hash

---

## Current Status
- [x] Data fetching (OpenML with UCI fallback)
- [x] Preprocessing pipeline & dense float32 features
- [x] Stratified train/val/test splits
- [x] Saved artifacts and feature metadata
- [x] PyTorch dataset/dataloaders + sanity check
- [x] README and self‑contained requirements

## Next Tasks
- Optional unit tests for IO and splits
- Add caching for OpenML data and UCI downloads
- Example training script for a simple MTL/PLE model
- CI smoke test to run the prep on a small subset

