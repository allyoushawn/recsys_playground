PLE Census-Income (KDD) Prep
============================

This self-contained project downloads and prepares the Census-Income (KDD) dataset for multi-task learning (as used in PLE):
- Task 1: predict income >50K (binary)
- Task 2: predict marital-status == "Never married" (binary)

Requirements
- Python 3.10+
- pandas, numpy, pyarrow, scikit-learn, torch

Setup
- Install deps:
  - `python -m pip install -r ple_experiment/requirements.txt`

Usage
- Prepare data and artifacts (example):
  - `python ple_experiment/prepare_census_income.py \
     --output_dir ./data/census_kdd \
     --test_size 0.15 \
     --val_size 0.10 \
     --batch_size 4096 \
     --num_workers 4 \
     --onehot_min_freq 10 \
     --seed 42`

What it does
- Fetches Census-Income (KDD) from OpenML (preferred), with UCI fallback.
- Builds preprocessing pipeline:
  - Numeric: median impute + StandardScaler
  - Categorical: mode impute + OneHotEncoder(handle_unknown="ignore", min_frequency)
- Splits dataset into train/val/test with stratification on income>50K.
- Saves artifacts into `--output_dir`:
  - `X_{train,val,test}.npy` (float32)
  - `y_income_{train,val,test}.npy` (uint8)
  - `y_nevermarried_{train,val,test}.npy` (uint8)
  - `feature_meta.json` (column lists, categories, scaler stats, sklearn version, pipeline hash)
- Performs basic validations and a quick PyTorch DataLoader sanity check.

Using the Dataset utilities
```
from ple_experiment.dataset import CensusKDDDataset, make_dataloaders

loaders = make_dataloaders(
    data_dir="./data/census_kdd",
    batch_size=4096,
    num_workers=4,
)
batch = next(iter(loaders["train"]))
print(batch["x"].shape, batch["y_income"].shape, batch["y_never_married"].shape)
```

Notes
- The OpenML dataset includes column names and is header-safe; if OpenML is unavailable, the script downloads from UCI and applies a known schema to avoid manual parsing issues.
- The script normalizes label strings (strip/lower/punctuation) to robustly find the `income` and `marital-status` targets.

