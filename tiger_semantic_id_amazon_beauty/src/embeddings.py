from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd
import torch

# Optional import: allow tests to run without installing sentence-transformers.
try:  # pragma: no cover - behavior verified via tests with monkeypatch
    from sentence_transformers import SentenceTransformer  # type: ignore
except Exception:  # pragma: no cover
    SentenceTransformer = None  # type: ignore


def build_item_text(meta_df: pd.DataFrame) -> Dict[int, str]:
    """Create a compact text string per item_idx from metadata DataFrame.

    Expected columns: item_idx, title, brand, category_leaf, price (optional)
    """
    texts: Dict[int, str] = {}
    for row in meta_df.itertuples(index=False):
        title = getattr(row, "title", "") or ""
        brand = getattr(row, "brand", "") or ""
        cat = getattr(row, "category_leaf", "") or ""
        price = getattr(row, "price", None)
        parts = []
        if title:
            parts.append(f"{title}.")
        if brand:
            parts.append(f"Brand: {brand}.")
        if cat:
            parts.append(f"Category: {cat}.")
        if price and not (isinstance(price, float) and np.isnan(price)):
            try:
                parts.append(f"Price: ${float(price):.2f}.")
            except Exception:
                pass
        text = " ".join(parts).strip()
        texts[int(getattr(row, "item_idx"))] = text if text else "(unknown item)"
    return texts


def encode_items(
    item_texts: Dict[int, str], model_name: str = "sentence-t5-base", batch_size: int = 256
) -> torch.Tensor:
    """Encode item texts with SentenceTransformer -> embeddings [num_items, hidden].

    In test environments without the dependency, monkeypatch `SentenceTransformer`
    in this module to a fake encoder that provides `.encode(...)`.
    """
    if SentenceTransformer is None:
        raise ImportError(
            "sentence-transformers is not installed. Install it or monkeypatch SentenceTransformer for tests."
        )
    model = SentenceTransformer(model_name)
    # Keep input order stable by sorting by index
    idxs = sorted(item_texts.keys())
    texts = [item_texts[i] for i in idxs]
    emb = model.encode(texts, batch_size=batch_size, show_progress_bar=True, convert_to_tensor=True)
    # Reorder to original idx positions
    emb = emb.detach().cpu()
    if max(idxs, default=-1) + 1 == len(idxs):
        return emb
    # If item_idx not contiguous, expand to full array and scatter
    dim = emb.shape[1]
    out = torch.zeros(max(idxs) + 1, dim, dtype=emb.dtype)
    for j, i in enumerate(idxs):
        out[i] = emb[j]
    return out
