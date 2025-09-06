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

PLE (Progressive Layered Extraction) on Census‑Income (KDD)
===========================================================

This folder implements a 2‑level PLE multi‑task model in PyTorch for two binary tasks:
- Task 1 (income): predict income >50K
- Task 2 (never_married): predict marital‑status == "Never married"

Prerequisites
- Python 3.10+
- Install deps for this subproject:
  - `python -m pip install -r ple_experiment/requirements.txt`
- Prepare data artifacts first (see `ple_experiment/README.md`) or run the prep script:
  - `python ple_experiment/prepare_census_income.py --output_dir ./data/census_kdd`

Train
```
python ple_experiment/train_ple.py \
  --data_dir ./data/census_kdd \
  --epochs 15 \
  --batch_size 4096 \
  --num_workers 4 \
  --lr 2e-3 \
  --weight_decay 1e-4 \
  --d_model 128 \
  --expert_hidden 256 \
  --num_levels 2 \
  --num_shared_experts 2 \
  --num_task_experts 2 \
  --dropout 0.1 \
  --w_income 1.0 \
  --w_never_married 1.0 \
  --use_pos_weight true \
  --grad_clip 1.0 \
  --mixed_precision true \
  --seed 42 \
  --out_dir ./runs/ple_census
```

What gets saved under `--out_dir`
- `config.json`: exact hyperparameters used (includes inferred `d_in`)
- `metrics.jsonl`: one JSON line per epoch with train/val metrics
- `best.pt`: best model checkpoint by validation combined AUC
- `last.pt`: last model checkpoint
- `test_report.json`: metrics on the test set using the best checkpoint

Metrics
- Per task: BCE loss, Accuracy, AUROC
- Combined score: mean of the two AUCs (used for model selection)

Tasks mapping
- Income head: uses label `y_income` (0/1)
- Never‑married head: uses label `y_never_married` (0/1)

Resume
- To resume training manually, load `best.pt` or `last.pt` with:
```
ckpt = torch.load('./runs/ple_census/best.pt', map_location='cpu')
from ple_experiment.model import PLEModel
model = PLEModel(**{k: v for k, v in ckpt['cfg'].items() if k in ['d_in','d_model','expert_hidden','num_levels','num_shared_experts','num_task_experts','dropout']})
model.load_state_dict(ckpt['model'])
```

Notes
- Mixed precision (AMP) is enabled by default when CUDA is available.
- Pos‑class weighting can be enabled via `--use_pos_weight` (computes `neg/pos` from train labels per task).
- The implementation follows the PLE routing scheme with two levels, shared + task‑specific experts per level, and per‑task/shared gates mixing all experts at that level.

