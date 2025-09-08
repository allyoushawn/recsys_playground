# MovieLens-100K Recommender System

This folder contains the MovieLens-100K prototype: data loading utilities, baseline models, tests, and a demo notebook under `../notebooks/`.

## Colab

Open `../notebooks/movie_lens_100k/movielens_baseline.ipynb` in Colab. The first cell clones the repo and installs dependencies from this folder:

```
!pip -q install -r movie_lens_100k/requirements.txt
```

It also adds `movie_lens_100k/src` to `sys.path` so imports like `from data.movielens import load_movielens_100k` work.

## Local Setup

- Python 3.10+ recommended.
- From the repository root, create and activate a virtual environment, then install deps for this module:

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
python -m pip install -r movie_lens_100k/requirements.txt
```

## Structure (this folder)

- `src/data/` → dataset loading & preprocessing utilities
- `src/models/` → baseline models (e.g., popularity)
- `tests/` → unit tests for this module
- `requirements.txt` → minimal dependencies for MovieLens-100K

Related notebooks live at `../notebooks/`.

## Dataset

`src/data/movielens.py:load_movielens_100k()` downloads and caches MovieLens-100K on first use, returning a pandas DataFrame with columns: `user_id`, `movie_id`, `rating`, `timestamp`.

## Notes

- The baseline notebook runs EDA and a popularity baseline. After the refactor, imports are resolved by adding `movie_lens_100k/src` to `sys.path` in the notebook.
- To run tests, execute from this folder or add it to `PYTHONPATH`:
  - `cd movie_lens_100k && pytest -q`
