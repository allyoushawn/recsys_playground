import types
import torch
import pandas as pd

import tiger_semantic_id_amazon_beauty.src.embeddings as emb_mod


def test_build_item_text_basic():
    df = pd.DataFrame(
        {
            "item_idx": [0, 1],
            "title": ["Face Cream", "Lipstick"],
            "brand": ["BrandA", ""],
            "category_leaf": ["Skin Care", "Makeup"],
            "price": [19.99, None],
        }
    )
    texts = emb_mod.build_item_text(df)
    assert 0 in texts and 1 in texts
    assert "Face Cream" in texts[0]
    assert "Brand: BrandA" in texts[0]
    assert "Category: Skin Care" in texts[0]
    assert "Price:" in texts[0]
    assert "Lipstick" in texts[1]


def test_encode_items_with_mock(monkeypatch):
    # Monkeypatch SentenceTransformer inside module to avoid network/model download
    class FakeModel:
        def __init__(self, *args, **kwargs):
            pass
        def encode(self, texts, batch_size=32, show_progress_bar=False, convert_to_tensor=False):
            import torch

            n = len(texts)
            d = 8
            out = torch.arange(n * d, dtype=torch.float32).reshape(n, d)
            return out if convert_to_tensor else out.numpy()

    monkeypatch.setattr(emb_mod, "SentenceTransformer", FakeModel, raising=True)
    item_texts = {0: "a", 1: "b", 2: "c"}
    emb = emb_mod.encode_items(item_texts, model_name="ignored", batch_size=2)
    assert isinstance(emb, torch.Tensor)
    assert emb.shape == (3, 8)
