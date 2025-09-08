# Recsys Playground

A multi-project workspace for recommender systems research prototypes. Each project is self-contained with its own `src/`, `tests/`, and Colab-friendly notebooks under `notebooks/`.

## Projects

- MovieLens‑100K (`movie_lens_100k/`)
  - Baselines and utilities for MovieLens‑100K (data loader, popularity model; ALS/MF planned).
  - Notebook: `notebooks/movie_lens_100k/movielens_baseline.ipynb`
  - Install deps: `python -m pip install -r movie_lens_100k/requirements.txt`
  - Run tests: `cd movie_lens_100k && pytest -q`

- PLE/MMoE Census (KDD) (`ple_experiment/`)
  - Progressive Layered Extraction (PLE) and MMoE experiments on Census‑Income (KDD).
  - Notebook: `notebooks/ple_experiment/run_experiment.ipynb`
  - Extra scripts under `ple_experiment/` (data prep and training).

- TIGER SemanticID on Amazon Beauty (`tiger_semantic_id_amazon_beauty/`)
  - Planned: Semantic ID pipeline with RQ‑VAE + seq2seq generative retrieval on Amazon Beauty 5‑core.
  - Plan: `tiger_semantic_id_amazon_beauty/AGENTS.md`
  - Notebook (to be added): `notebooks/tiger_semantic_id_amazon_beauty/TIGER_SemanticID_AmazonBeauty.ipynb`

## Quickstart

1) Python & venv

```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
python -V && which python
```

2) Install per‑project dependencies (from repo root)

```bash
python -m pip install -r movie_lens_100k/requirements.txt
# Optional: PLE extras
python -m pip install -r ple_experiment/requirements.txt  # if present
```

3) Notebooks (Colab‑friendly)
- Open the desired notebook under `notebooks/...`.
- The first cell typically installs the matching project requirements, e.g.:

```python
!pip -q install -r movie_lens_100k/requirements.txt
```

- Notebooks add the project `src` to `sys.path` (e.g., `movie_lens_100k/src`) for imports like `from data.movielens import load_movielens_100k`.

## Notes
- Python 3.10+ recommended.
- Some tests and notebooks download datasets on first run (network required).
- Keep dependencies minimal to run on Google Colab GPUs.

## Repository Layout

```
notebooks/
  movie_lens_100k/movielens_baseline.ipynb
  ple_experiment/run_experiment.ipynb
  tiger_semantic_id_amazon_beauty/  # (planned notebooks)
movie_lens_100k/
  src/  tests/  requirements.txt  README.md  AGENTS.md
ple_experiment/
  *.py  README.md  AGENTS.md  requirements.txt (if present)
tiger_semantic_id_amazon_beauty/
  AGENTS.md  # src/tests/README/requirements to be added
```

---
For details and next steps on each project, see the respective `README.md` or `AGENTS.md` in its folder.
