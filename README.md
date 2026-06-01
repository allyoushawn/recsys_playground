# Recsys Playground

A multi-project workspace for recommender systems research prototypes. Each project is self-contained with its own `src/`, `tests/`, and Colab-friendly notebooks under `notebooks/`.

## Datasets (read first)

**[`DATASETS.md`](./DATASETS.md) is the single source of truth** for which datasets are experiment-ready, for which models, where the data/cache lives on Drive, and what to build next. Check it before starting any modeling work.

| Dataset | Solves | Ready for | Status |
|---|---|---|---|
| **Ali-CCP** | CTR / CVR / CTCVR | ESMM, SharedBottom, MMoE, PLE, Wide&Deep, DeepFM, DCNv2 | ✅ ready now |
| **Amazon Video Games** | rating regression (1–5) | regression MLP/MF/NeuMF, DCNv2, PLE heads | ⚠️ partial (not CTR/ranking/DIN) |
| **Amazon Beauty (TIGER)** | generative semantic-ID retrieval | RQ-VAE + seq2seq SemanticID, LLM-SID | ⚠️ partial (not wired to DCN/DIN/DeepFM) |

**Routing:** Wide&Deep / DeepFM / DCN and ESMM / MMoE / PLE → Ali-CCP (ready). DIN → not ready (needs an Amazon sequence datamodule with negative sampling). SASRec / BERT4Rec / GRU4Rec / TIGER → Amazon Beauty / Video_Games. See [`DATASETS.md`](./DATASETS.md) for the standard cache contract and the sequenced build plan.

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

- TIGER SemanticID (`tiger_semantic_id/`)
  - Semantic ID pipeline with RQ‑VAE + seq2seq generative retrieval on Amazon 5‑core datasets (Beauty, Video_Games).
  - Plan: `tiger_semantic_id/AGENTS.md`
  - Notebook: `notebooks/tiger_semantic_id/TIGER_SemanticID.ipynb`

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
  tiger_semantic_id/  # (planned notebooks)
movie_lens_100k/
  src/  tests/  requirements.txt  README.md  AGENTS.md
ple_experiment/
  *.py  README.md  AGENTS.md  requirements.txt (if present)
tiger_semantic_id/
  AGENTS.md  # src/tests/README/requirements to be added
```

---
For details and next steps on each project, see the respective `README.md` or `AGENTS.md` in its folder.
