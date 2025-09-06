# Project: MovieLens-100K Recommender System

> IMPORTANT: Before running any commands or notebooks, activate the virtual environment `vevn` (e.g., `source venv/bin/activate`).

## Goal
Build a research prototype for recommendation models using the MovieLens-100K dataset.  
Objectives:
1. Data ingestion & preprocessing
2. Exploratory analysis
3. Baseline recommenders (popular, user/movie means, ALS/Matrix Factorization)
4. Evaluation metrics (RMSE, MAE, HR@K, NDCG@K)
5. Prepare for scaling to larger MovieLens datasets (1M, 20M)

---

## Repo Structure
- `notebooks/`
  - Jupyter/Colab notebooks for experiments
- `src/data/`
  - Data loading & preprocessing utils
- `src/models/`
  - Baseline and ML models
- `tests/`
  - Unit tests
- `requirements.txt`
  - Dependencies
- `README.md`
  - Project overview + Colab badge

---

## Current Status
- [x] Workspace initialized
- [x] Loader for MovieLens-100K dataset (`src/data/movielens.py`)
 - [x] Baseline popularity recommender
- [ ] ALS model (PySpark / Surprise / Implicit)
- [ ] Evaluation metrics implementation
- [ ] First results notebook

---

## Next Tasks for Codex
- Implement ALS/MF baseline (Surprise/Implicit/PySpark)
- Add evaluation metrics (RMSE, MAE, HR@K, NDCG@K)
- Expand results notebook with metrics + comparisons
- Prepare code paths for scaling to ML-1M/20M

---

## Conventions
- Use Python 3.10+
- Follow PEP8 style
- Write unit tests in `tests/`
- Document functions with short docstrings
- Ensure new code runs in Google Colab (keep dependencies minimal)

---

## Dev Setup
- Activate virtual environment (use `vevn` before anything else):
  - macOS/Linux (zsh/bash): `source vevn/bin/activate`
  - Alternatives (if applicable): `source .venv/bin/activate` or `source venv/bin/activate`
  - Verify: `python -V && which python`
  - Deactivate: `deactivate`
- Install dependencies:
  - Base: `python -m pip install -r requirements.txt`
  - Tests (optional): `python -m pip install pytest`

## Running
- Tests: `pytest -q`
  - Note: `tests/test_data_loading.py` downloads MovieLens-100K on first run.
    Ensure network access is available or run in an environment that allows it.
- Notebook: open `notebooks/movielens_baseline.ipynb` and run cells.

## Recent Changes
- Added popularity baseline: `src/models/popularity.py` with `get_top_n(train_ratings, n=10)`
- Added unit tests: `tests/test_popularity.py`
- Updated notebook to demo popularity baseline: `notebooks/movielens_baseline.ipynb`
