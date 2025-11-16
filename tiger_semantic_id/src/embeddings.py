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
    """Create a text string per item_idx from metadata DataFrame.

    Supports both legacy (2014 SNAP) and rich (2023 Amazon) metadata formats.
    - Legacy format (Beauty): Simple compact text with periods
    - Rich format (Video_Games): Includes description/features with rich formatting

    Expected columns: item_idx, title, [description, features, brand, category_leaf, price, ...]
    """
    texts: Dict[int, str] = {}

    # Check if we have rich metadata (Amazon 2023 format)
    has_description = "description" in meta_df.columns
    has_features = "features" in meta_df.columns
    use_rich_format = has_description or has_features

    for row in meta_df.itertuples(index=False):
        title = getattr(row, "title", "") or ""
        description = getattr(row, "description", "") or ""
        features = getattr(row, "features", "") or ""
        brand = getattr(row, "brand", "") or ""
        cat = getattr(row, "category_leaf", "") or ""
        price = getattr(row, "price", None)

        parts = []

        if use_rich_format:
            # Rich format for Amazon 2023 datasets (Video_Games)
            # Build context in reference implementation format
            if title:
                parts.append(f"Product: {title}")

            if has_description and description:
                # Truncate very long descriptions
                desc_text = description[:500] if len(description) > 500 else description
                parts.append(f"Description: {desc_text}")

            if has_features and features:
                # Truncate very long feature lists
                feat_text = features[:300] if len(features) > 300 else features
                parts.append(f"Features: {feat_text}")

            if cat:
                parts.append(f"Category: {cat}")

            if brand:
                parts.append(f"Brand: {brand}")

            if price and not (isinstance(price, float) and np.isnan(price)):
                try:
                    parts.append(f"Price: ${float(price):.2f}")
                except Exception:
                    pass

            # Join with newlines for rich format
            text = "\n\n".join(parts).strip()
        else:
            # Legacy format for 2014 SNAP datasets (Beauty)
            # Preserves exact format for backward compatibility
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

            # Join with spaces for legacy format
            text = " ".join(parts).strip()

        texts[int(getattr(row, "item_idx"))] = text if text else "(unknown item)"

    return texts


def encode_items(
    item_texts: Dict[int, str], model_name: str = "sentence-t5-base", batch_size: int = 256,
    device: str | None = None
) -> torch.Tensor:
    """Encode item texts with SentenceTransformer -> embeddings [num_items, hidden].

    Args:
        item_texts: Dictionary mapping item indices to text descriptions
        model_name: Name of the SentenceTransformer model
        batch_size: Batch size for encoding
        device: Device to use ('cuda', 'cpu', or None for auto-detect)

    In test environments without the dependency, monkeypatch `SentenceTransformer`
    in this module to a fake encoder that provides `.encode(...)`.
    """
    if SentenceTransformer is None:
        raise ImportError(
            "sentence-transformers is not installed. Install it or monkeypatch SentenceTransformer for tests."
        )
    
    # Auto-detect device if not specified
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Using device: {device} for SentenceTransformer encoding")
    
    # Create model and move to specified device
    model = SentenceTransformer(model_name, device=device)
    
    # Keep input order stable by sorting by index
    idxs = sorted(item_texts.keys())
    texts = [item_texts[i] for i in idxs]
    
    # Encode with GPU acceleration if available
    emb = model.encode(
        texts, 
        batch_size=batch_size, 
        show_progress_bar=True, 
        convert_to_tensor=True,
        device=device  # Ensure encoding happens on specified device
    )
    
    # Keep on GPU for now, only move to CPU when needed
    if max(idxs, default=-1) + 1 == len(idxs):
        # Contiguous indices - return directly (keep on device)
        return emb.detach()
    
    # If item_idx not contiguous, expand to full array and scatter
    dim = emb.shape[1]
    out = torch.zeros(max(idxs) + 1, dim, dtype=emb.dtype, device=emb.device)
    for j, i in enumerate(idxs):
        out[i] = emb[j]
    
    return out.detach()
