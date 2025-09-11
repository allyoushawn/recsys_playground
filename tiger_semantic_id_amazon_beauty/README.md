# TIGER SemanticID on Amazon Beauty

This module implements a Colab-friendly pipeline for TIGER (Semantic IDs via RQ‑VAE + generative retrieval) on the Amazon Beauty 5‑core dataset.

- Notebook: `notebooks/tiger_semantic_id_amazon_beauty/TIGER_SemanticID_AmazonBeauty.ipynb`
- Source: `tiger_semantic_id_amazon_beauty/src`
- Artifacts: saved to `/content/artifacts` in Colab

Quickstart (Colab)
- Open the notebook above in Google Colab.
- First cell installs deps from this folder and adds `tiger_semantic_id_amazon_beauty/src` to `sys.path`.

Local Dev
- Activate the repo venv first: `source venv/bin/activate` (or `.venv`).
- Install deps: `python -m pip install -r tiger_semantic_id_amazon_beauty/requirements.txt`.
- Minimal tests (optional): `pytest -q` from repo root or this folder after setting `PYTHONPATH`.

See `tiger_semantic_id_amazon_beauty/AGENTS.md` for the full plan and acceptance criteria.
