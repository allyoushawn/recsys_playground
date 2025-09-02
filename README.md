# MovieLens-100K Exploration

This repository is a clean workspace for exploring the MovieLens-100K dataset, including simple data loading utilities, a baseline exploration notebook, and minimal tests.

## Colab

Run the notebook in Google Colab using the badge below:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](COLAB_LINK_PLACEHOLDER)

In Colab, you may need to install dependencies at the top of the notebook:

```
!pip install pandas numpy scikit-learn jupyter
```

## Local Setup

- Python 3.9+ recommended.
- Create and activate a virtual environment (optional but recommended).

```bash
python -m venv .venv
source .venv/bin/activate  # on Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
```

## Structure

- `notebooks/` → Jupyter/Colab notebooks
- `src/data/` → dataset loading & preprocessing utilities
- `src/models/` → baseline models
- `tests/` → unit tests

## Dataset

The function `src/data/movielens.py:load_movielens_100k()` downloads and caches the MovieLens-100K dataset from GroupLens on first use, then loads `u.data` into a pandas DataFrame with columns: `user_id`, `movie_id`, `rating`, `timestamp`.

## Notes

- The included notebook performs a quick EDA (head, shape, ratings histogram), then splits the data into an 80/20 train/test split and prints summary stats.
- If running locally and you don’t have matplotlib installed, Colab already provides it. Locally, install `matplotlib` if you want to view plots interactively.

