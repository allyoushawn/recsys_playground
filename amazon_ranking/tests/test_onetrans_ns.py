"""Tests for the category-enriched NS-token ablation (onetrans_ns).

Covers:
- OneTrans with item_category + num_categories builds and forwards to [B] finite
  output; works with use_mixed_param True/False.
- Category NS-token actually changes logits vs the same model without category.
- build_item_category(..., synthetic=True) returns a [num_items] LongTensor in
  range and the correct num_categories.
- build_model("onetrans_ns", ..., item_category=..., num_categories=K) works via
  the registry.
- Existing onetrans (no category) path is UNCHANGED: no category_emb attribute.
"""

from __future__ import annotations

import sys
import os

import pytest

torch = pytest.importorskip("torch")

# Ensure the experiment dir is on sys.path so category_features can be imported.
_EXPERIMENT_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "experiments", "20260621_onetrans_reproduction"
)
if _EXPERIMENT_DIR not in sys.path:
    sys.path.insert(0, os.path.abspath(_EXPERIMENT_DIR))

from amazon_ranking.src.onetrans.onetrans import OneTrans
from amazon_ranking.src.onetrans.registry import build_model
from category_features import build_item_category

NUM_ITEMS = 50
EMBED_DIM = 8
HIST_LEN = 6
BATCH = 4
NUM_CATEGORIES = 5


def _cat_tensor(num_items: int = NUM_ITEMS, num_cats: int = NUM_CATEGORIES, seed: int = 0):
    """Deterministic category assignment for tests."""
    import numpy as np
    rng = np.random.default_rng(seed)
    return torch.tensor(rng.integers(0, num_cats, size=num_items), dtype=torch.long)


def _random_batch(B: int = BATCH, L: int = HIST_LEN, seed: int = 0):
    rng = torch.Generator()
    rng.manual_seed(seed)
    valid_lens = torch.randint(1, L + 1, (B,), generator=rng)
    hist_ids = torch.full((B, L), NUM_ITEMS, dtype=torch.long)
    hist_mask = torch.zeros(B, L)
    for i, vl in enumerate(valid_lens):
        items = torch.randint(0, NUM_ITEMS, (int(vl),), generator=rng)
        hist_ids[i, :vl] = items
        hist_mask[i, :vl] = 1.0
    target_ids = torch.randint(0, NUM_ITEMS, (B,), generator=rng)
    return hist_ids, hist_mask, target_ids


# ---------------------------------------------------------------------------
# Basic build + forward: use_mixed_param=True
# ---------------------------------------------------------------------------

def test_onetrans_ns_forward_mixed_param_true():
    """OneTrans with category NS-token + use_mixed_param=True: shape [B] and finite."""
    torch.manual_seed(0)
    cat = _cat_tensor()
    model = OneTrans(
        num_items=NUM_ITEMS,
        embed_dim=EMBED_DIM,
        n_layers=1,
        n_heads=2,
        n_ns_tokens=2,
        max_len=10,
        use_mixed_param=True,
        item_category=cat,
        num_categories=NUM_CATEGORIES,
    )
    hist_ids, hist_mask, target_ids = _random_batch()
    with torch.no_grad():
        logits = model(hist_ids, hist_mask, target_ids)
    assert logits.shape == (BATCH,), f"expected ({BATCH},), got {logits.shape}"
    assert torch.isfinite(logits).all(), "onetrans_ns output contains non-finite values"


# ---------------------------------------------------------------------------
# Basic build + forward: use_mixed_param=False
# ---------------------------------------------------------------------------

def test_onetrans_ns_forward_mixed_param_false():
    """OneTrans with category NS-token + use_mixed_param=False: shape [B] and finite."""
    torch.manual_seed(1)
    cat = _cat_tensor()
    model = OneTrans(
        num_items=NUM_ITEMS,
        embed_dim=EMBED_DIM,
        n_layers=1,
        n_heads=2,
        n_ns_tokens=2,
        max_len=10,
        use_mixed_param=False,
        item_category=cat,
        num_categories=NUM_CATEGORIES,
    )
    hist_ids, hist_mask, target_ids = _random_batch()
    with torch.no_grad():
        logits = model(hist_ids, hist_mask, target_ids)
    assert logits.shape == (BATCH,), f"expected ({BATCH},), got {logits.shape}"
    assert torch.isfinite(logits).all(), "onetrans_ns (no mixed_param) output not finite"


# ---------------------------------------------------------------------------
# Category token actually changes logits
# ---------------------------------------------------------------------------

def test_category_token_changes_logits():
    """The extra category NS-token must produce different logits from no-category model.

    Both models start from the same random seed so the only structural difference
    is the presence of the category embedding and the +1 NS-token.
    """
    torch.manual_seed(42)
    base_model = OneTrans(
        num_items=NUM_ITEMS,
        embed_dim=EMBED_DIM,
        n_layers=1,
        n_heads=2,
        n_ns_tokens=2,
        max_len=10,
    )
    torch.manual_seed(42)
    cat = _cat_tensor()
    ns_model = OneTrans(
        num_items=NUM_ITEMS,
        embed_dim=EMBED_DIM,
        n_layers=1,
        n_heads=2,
        n_ns_tokens=2,
        max_len=10,
        item_category=cat,
        num_categories=NUM_CATEGORIES,
    )

    hist_ids, hist_mask, target_ids = _random_batch()
    base_model.eval()
    ns_model.eval()
    with torch.no_grad():
        logits_base = base_model(hist_ids, hist_mask, target_ids)
        logits_ns = ns_model(hist_ids, hist_mask, target_ids)

    assert torch.isfinite(logits_base).all()
    assert torch.isfinite(logits_ns).all()
    assert not torch.allclose(logits_base, logits_ns, atol=1e-6), (
        "Category NS-token had no effect on logits — it may not be wired in correctly."
    )


# ---------------------------------------------------------------------------
# build_item_category synthetic mode
# ---------------------------------------------------------------------------

def test_build_item_category_synthetic():
    """build_item_category(synthetic=True) returns correct shape, dtype, and range."""
    item2id = {f"item_{i}": i for i in range(30)}
    cat_tensor, num_cats = build_item_category(
        dataset="Beauty",
        data_dir="/tmp",
        item2id=item2id,
        synthetic=True,
        num_synth_categories=7,
        seed=0,
    )
    assert isinstance(cat_tensor, torch.Tensor), "should return a torch.Tensor"
    assert cat_tensor.dtype == torch.long, f"expected long, got {cat_tensor.dtype}"
    assert cat_tensor.shape == (30,), f"expected (30,), got {cat_tensor.shape}"
    assert num_cats == 7
    assert cat_tensor.min().item() >= 0
    assert cat_tensor.max().item() < 7


def test_build_item_category_synthetic_deterministic():
    """Same seed must produce identical results."""
    item2id = {f"i{i}": i for i in range(20)}
    t1, n1 = build_item_category("Beauty", "/tmp", item2id, synthetic=True, seed=3)
    t2, n2 = build_item_category("Beauty", "/tmp", item2id, synthetic=True, seed=3)
    assert torch.equal(t1, t2)
    assert n1 == n2


# ---------------------------------------------------------------------------
# Registry: build_model("onetrans_ns", ...) works
# ---------------------------------------------------------------------------

def test_registry_builds_onetrans_ns():
    """build_model('onetrans_ns', ...) must return an OneTrans with category_emb."""
    torch.manual_seed(0)
    cat = _cat_tensor()
    model = build_model(
        "onetrans_ns",
        num_items=NUM_ITEMS,
        embed_dim=EMBED_DIM,
        n_heads=2,
        item_category=cat,
        num_categories=NUM_CATEGORIES,
    )
    assert isinstance(model, OneTrans)
    assert hasattr(model, "category_emb"), "onetrans_ns must have category_emb"
    hist_ids, hist_mask, target_ids = _random_batch(B=2)
    with torch.no_grad():
        logits = model(hist_ids, hist_mask, target_ids)
    assert logits.shape == (2,)
    assert torch.isfinite(logits).all()


# ---------------------------------------------------------------------------
# Existing onetrans path is unchanged
# ---------------------------------------------------------------------------

def test_onetrans_baseline_unchanged_no_category_emb():
    """OneTrans without item_category has NO category_emb and produces [B] finite logits."""
    torch.manual_seed(0)
    model = OneTrans(num_items=NUM_ITEMS, embed_dim=EMBED_DIM, n_heads=2)
    assert not hasattr(model, "category_emb"), (
        "Baseline OneTrans (no item_category) must NOT have category_emb"
    )
    hist_ids, hist_mask, target_ids = _random_batch()
    with torch.no_grad():
        logits = model(hist_ids, hist_mask, target_ids)
    assert logits.shape == (BATCH,)
    assert torch.isfinite(logits).all()
