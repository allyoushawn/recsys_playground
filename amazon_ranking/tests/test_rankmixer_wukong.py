"""Tests for RankMixer and Wukong feature-interaction ranking models.

Mirrors the style of test_baselines.py: fixtures, parametrize over both models,
short synthetic train loop, guard tests.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from amazon_ranking.src.datamodule import DataModuleConfig, SequenceRankingDataModule
from amazon_ranking.src.din import DINTrainConfig, train_din, pad_histories
from amazon_ranking.src.onetrans.rankmixer import RankMixer
from amazon_ranking.src.onetrans.wukong import Wukong
from amazon_ranking.src.onetrans.registry import build_model


# ---------------------------------------------------------------------------
# Shared fixtures (mirrors test_baselines.py)
# ---------------------------------------------------------------------------

def _make_reviews(n_users: int = 40, items_per_user: int = 8) -> pd.DataFrame:
    rows = []
    ts = 1
    for u in range(n_users):
        for i in range(items_per_user):
            rows.append({"user_id": f"u{u}", "item_id": f"i{(u * 3 + i) % 60}", "ts": ts})
            ts += 1
    return pd.DataFrame(rows)


def _module(seed: int = 0, max_hist_len: int = 8) -> SequenceRankingDataModule:
    cfg = DataModuleConfig(
        max_hist_len=max_hist_len,
        min_user_interactions=5,
        n_eval_negatives=20,
        n_train_negatives=4,
        seed=seed,
    )
    dm = SequenceRankingDataModule.from_reviews(_make_reviews(), cfg)
    dm.build()
    return dm


ALL_MODELS = ["rankmixer", "wukong"]
ALL_CLASSES = {"rankmixer": RankMixer, "wukong": Wukong}


# ---------------------------------------------------------------------------
# 1. Output shape [B] and finite values
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", ALL_MODELS)
def test_forward_shape_and_finite(name):
    dm = _module()
    model = build_model(name, num_items=dm.num_items, embed_dim=16, n_field_tokens=4, token_dim=16)
    assert model.pad_idx == dm.num_items
    ids, mask = pad_histories([[0, 1], [2]], max_hist_len=8, pad_idx=model.pad_idx)
    logits = model(ids, mask, torch.tensor([3, 4], dtype=torch.long))
    assert logits.shape == (2,), f"expected shape (2,), got {logits.shape}"
    assert torch.isfinite(logits).all(), f"non-finite logits: {logits}"


# ---------------------------------------------------------------------------
# 2. Empty history rows produce finite (no NaN)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", ALL_MODELS)
def test_empty_history_no_nan(name):
    dm = _module()
    model = build_model(name, num_items=dm.num_items, embed_dim=16, n_field_tokens=4, token_dim=16)
    ids, mask = pad_histories([[]], max_hist_len=8, pad_idx=model.pad_idx)
    logits = model(ids, mask, torch.tensor([0], dtype=torch.long))
    assert torch.isfinite(logits).all(), f"NaN/Inf on empty history: {logits}"


# ---------------------------------------------------------------------------
# 3. Short train loop lowers BCE loss
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", ALL_MODELS)
def test_train_loop_lowers_bce(name):
    dm = _module(seed=0)
    torch.manual_seed(42)
    model = build_model(name, num_items=dm.num_items, embed_dim=16, n_field_tokens=4, token_dim=16)
    out = train_din(
        model,
        dm.train_examples(),
        max_hist_len=dm.cfg.max_hist_len,
        cfg=DINTrainConfig(embed_dim=16, epochs=15, batch_size=64, lr=1e-2, seed=0),
    )
    assert np.isfinite(out["first_epoch_loss"]), "first_epoch_loss is not finite"
    assert np.isfinite(out["last_epoch_loss"]), "last_epoch_loss is not finite"
    assert out["last_epoch_loss"] < out["first_epoch_loss"], (
        f"loss did not decrease: first={out['first_epoch_loss']:.4f}, last={out['last_epoch_loss']:.4f}"
    )


# ---------------------------------------------------------------------------
# 4. embed_dim % n_field_tokens != 0 raises ValueError
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", ALL_MODELS)
def test_embed_dim_divisibility_guard(name):
    cls = ALL_CLASSES[name]
    with pytest.raises(ValueError, match="n_field_tokens"):
        cls(num_items=50, embed_dim=16, n_field_tokens=3)  # 16 % 3 != 0


# ---------------------------------------------------------------------------
# 5. Registry build_model works for both names
# ---------------------------------------------------------------------------

def test_registry_rankmixer():
    model = build_model("rankmixer", num_items=50, embed_dim=8, n_field_tokens=2)
    assert isinstance(model, RankMixer)
    assert model.pad_idx == 50


def test_registry_wukong():
    model = build_model("wukong", num_items=50, embed_dim=8, n_field_tokens=2)
    assert isinstance(model, Wukong)
    assert model.pad_idx == 50


# ---------------------------------------------------------------------------
# 6. Output is invariant to extra right-padding (mask-aware pooling)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", ALL_MODELS)
def test_output_invariant_to_padding_amount(name):
    torch.manual_seed(0)
    model = build_model(name, num_items=50, embed_dim=8, n_field_tokens=2, token_dim=8)
    history = [3, 7]
    target = torch.tensor([5], dtype=torch.long)
    ids4, mask4 = pad_histories([history], max_hist_len=4, pad_idx=model.pad_idx)
    ids8, mask8 = pad_histories([history], max_hist_len=8, pad_idx=model.pad_idx)
    assert torch.allclose(
        model(ids4, mask4, target), model(ids8, mask8, target), atol=1e-6
    ), "output changed with different padding lengths"
