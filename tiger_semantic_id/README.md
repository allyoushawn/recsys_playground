# TIGER SemanticID on Amazon Datasets

This module implements a Colab-friendly pipeline for TIGER (Semantic IDs via RQ‑VAE + generative retrieval) on Amazon 5‑core datasets.

- Notebook: `notebooks/tiger_semantic_id/TIGER_SemanticID.ipynb`
- Source: `tiger_semantic_id/src`
- Artifacts: saved to `/content/artifacts` in Colab

## Supported Datasets

The pipeline supports multiple Amazon 5-core datasets:
- **Beauty** (default)
- **Video_Games**

To switch datasets, modify the `dataset_name` parameter in the notebook's Config cell:

```python
@dataclass
class Config:
    dataset_name: str = 'Video_Games'  # Change from 'Beauty' to 'Video_Games'
    # ... other parameters
```

## Quickstart (Colab)
- Open the notebook above in Google Colab.
- (Optional) Change `dataset_name` in the Config cell to your desired dataset.
- Run all cells - the pipeline will automatically download and process the selected dataset.

## Local Dev
- Activate the repo venv first: `source venv/bin/activate` (or `.venv`).
- Install deps: `python -m pip install -r tiger_semantic_id/requirements.txt`.
- Minimal tests (optional): `pytest -q` from repo root or this folder after setting `PYTHONPATH`.

## Adding New Datasets

To add support for additional Amazon datasets:

1. Update `DATASET_URLS` in `src/data.py`:
```python
DATASET_URLS = {
    "Beauty": {...},
    "Video_Games": {...},
    "YourDataset": {
        "reviews": "http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_YourDataset_5.json.gz",
        "meta": "http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/meta_YourDataset.json.gz",
    },
}
```

2. Add filename mapping in `DatasetConfig.get_filenames()` method (optional, generic fallback exists).

See `tiger_semantic_id/AGENTS.md` for the full plan and acceptance criteria.
