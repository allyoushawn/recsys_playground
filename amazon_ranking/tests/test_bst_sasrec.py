"""Tests for BST and SASRec sequence-transformer ranking models.

Covers:
- Output shape [B] and finite values
- Masks respected: padded vs. unpadded rows differ
- Empty-history rows produce finite output (no NaNs)
- Short training loop on synthetic data reduces BCE loss
- Registry integration via build_model
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn

from amazon_ranking.src.din import DINTrainConfig, pad_histories, train_din
from amazon_ranking.src.onetrans.bst import BST
from amazon_ranking.src.onetrans.sasrec import SASRec
from amazon_ranking.src.onetrans.registry import build_model


# ---------------------------------------------------------------------------
# Shared fixtures / helpers
# ---------------------------------------------------------------------------

NUM_ITEMS = 50
EMBED_DIM = 8
HIST_LEN = 6
BATCH = 4


def _make_bst(**kwargs) -> BST:
    defaults = dict(n_layers=1, n_heads=2)
    defaults.update(kwargs)
    return BST(num_items=NUM_ITEMS, embed_dim=EMBED_DIM, **defaults)


def _make_sasrec(**kwargs) -> SASRec:
    defaults = dict(n_layers=1, n_heads=1)
    defaults.update(kwargs)
    return SASRec(num_items=NUM_ITEMS, embed_dim=EMBED_DIM, **defaults)


def _random_batch(B: int = BATCH, L: int = HIST_LEN, seed: int = 0):
    """Return (hist_ids, hist_mask, target_ids) with mixed padding."""
    rng = torch.Generator()
    rng.manual_seed(seed)
    # Each row has a random valid history length in [1, L].
    valid_lens = torch.randint(1, L + 1, (B,), generator=rng)
    hist_ids = torch.full((B, L), NUM_ITEMS, dtype=torch.long)  # pad_idx = NUM_ITEMS
    hist_mask = torch.zeros(B, L)
    for i, vl in enumerate(valid_lens):
        items = torch.randint(0, NUM_ITEMS, (int(vl),), generator=rng)
        hist_ids[i, :vl] = items
        hist_mask[i, :vl] = 1.0
    target_ids = torch.randint(0, NUM_ITEMS, (B,), generator=rng)
    return hist_ids, hist_mask, target_ids


def _synthetic_examples(n: int = 80, L: int = HIST_LEN, seed: int = 42):
    """Create synthetic training examples in the DIN harness format."""
    rng = np.random.default_rng(seed)
    examples = []
    for _ in range(n):
        vl = rng.integers(1, L + 1)
        history = rng.integers(0, NUM_ITEMS, size=int(vl)).tolist()
        target_idx = int(rng.integers(0, NUM_ITEMS))
        label = float(rng.integers(0, 2))
        examples.append({"history": history, "target_idx": target_idx, "label": label})
    return examples


# ---------------------------------------------------------------------------
# BST tests
# ---------------------------------------------------------------------------

def test_bst_output_shape_and_finite():
    """BST forward produces shape [B] with all finite values."""
    torch.manual_seed(0)
    model = _make_bst()
    hist_ids, hist_mask, target_ids = _random_batch()
    logits = model(hist_ids, hist_mask, target_ids)
    assert logits.shape == (BATCH,), f"expected ({BATCH},), got {logits.shape}"
    assert torch.isfinite(logits).all(), "BST output contains non-finite values"


def test_bst_empty_history_no_nan():
    """BST handles all-padding (empty history) without producing NaNs."""
    torch.manual_seed(0)
    model = _make_bst()
    ids, mask = pad_histories([[]], max_hist_len=HIST_LEN, pad_idx=model.pad_idx)
    target = torch.tensor([5], dtype=torch.long)
    logits = model(ids, mask, target)
    assert torch.isfinite(logits).all(), "BST output is NaN for empty history"


def test_bst_masks_respected():
    """Padded positions must not affect the logit: padding the same history to
    different lengths should yield the same result."""
    torch.manual_seed(1)
    model = _make_bst()
    model.eval()
    history = [3, 7, 12]
    target = torch.tensor([5], dtype=torch.long)
    ids4, mask4 = pad_histories([history], max_hist_len=4, pad_idx=model.pad_idx)
    ids6, mask6 = pad_histories([history], max_hist_len=6, pad_idx=model.pad_idx)
    with torch.no_grad():
        out4 = model(ids4, mask4, target)
        out6 = model(ids6, mask6, target)
    assert torch.allclose(out4, out6, atol=1e-5), (
        f"BST logit changed with extra padding: {out4.item():.6f} vs {out6.item():.6f}"
    )


def test_bst_padded_vs_unpadded_differ():
    """A row with history must produce a different logit from a fully-masked
    (empty) row for the same target — confirming masking is not clamping everything
    to the same value."""
    torch.manual_seed(2)
    model = _make_bst()
    model.eval()
    target = torch.tensor([5, 5], dtype=torch.long)
    ids, mask = pad_histories([[], [3, 7]], max_hist_len=HIST_LEN, pad_idx=model.pad_idx)
    with torch.no_grad():
        logits = model(ids, mask, target)
    assert torch.isfinite(logits).all()
    assert not torch.allclose(logits[0], logits[1]), (
        "Empty-history and non-empty-history rows produced identical BST logits"
    )


def test_bst_training_reduces_bce():
    """A short training loop on synthetic binary data should reduce BCE loss."""
    torch.manual_seed(0)
    model = _make_bst(n_layers=1)
    examples = _synthetic_examples(n=100)
    out = train_din(
        model,
        examples,
        max_hist_len=HIST_LEN,
        cfg=DINTrainConfig(embed_dim=EMBED_DIM, epochs=20, batch_size=32, lr=5e-3, seed=0),
    )
    assert np.isfinite(out["first_epoch_loss"]) and np.isfinite(out["last_epoch_loss"]), (
        "BST training produced non-finite loss"
    )
    assert out["last_epoch_loss"] < out["first_epoch_loss"], (
        f"BST loss did not decrease: {out['first_epoch_loss']:.4f} -> {out['last_epoch_loss']:.4f}"
    )


# ---------------------------------------------------------------------------
# SASRec tests
# ---------------------------------------------------------------------------

def test_sasrec_output_shape_and_finite():
    """SASRec forward produces shape [B] with all finite values."""
    torch.manual_seed(0)
    model = _make_sasrec()
    hist_ids, hist_mask, target_ids = _random_batch()
    logits = model(hist_ids, hist_mask, target_ids)
    assert logits.shape == (BATCH,), f"expected ({BATCH},), got {logits.shape}"
    assert torch.isfinite(logits).all(), "SASRec output contains non-finite values"


def test_sasrec_empty_history_no_nan():
    """SASRec handles all-padding (empty history) without producing NaNs."""
    torch.manual_seed(0)
    model = _make_sasrec()
    ids, mask = pad_histories([[]], max_hist_len=HIST_LEN, pad_idx=model.pad_idx)
    target = torch.tensor([5], dtype=torch.long)
    logits = model(ids, mask, target)
    assert torch.isfinite(logits).all(), "SASRec output is NaN for empty history"


def test_sasrec_masks_respected():
    """Extra right-padding must not change the SASRec logit for a fixed history."""
    torch.manual_seed(1)
    model = _make_sasrec()
    model.eval()
    history = [3, 7, 12]
    target = torch.tensor([5], dtype=torch.long)
    ids4, mask4 = pad_histories([history], max_hist_len=4, pad_idx=model.pad_idx)
    ids6, mask6 = pad_histories([history], max_hist_len=6, pad_idx=model.pad_idx)
    with torch.no_grad():
        out4 = model(ids4, mask4, target)
        out6 = model(ids6, mask6, target)
    assert torch.allclose(out4, out6, atol=1e-5), (
        f"SASRec logit changed with extra padding: {out4.item():.6f} vs {out6.item():.6f}"
    )


def test_sasrec_padded_vs_unpadded_differ():
    """Empty-history vs. non-empty-history rows must yield different SASRec logits."""
    torch.manual_seed(2)
    model = _make_sasrec()
    model.eval()
    target = torch.tensor([5, 5], dtype=torch.long)
    ids, mask = pad_histories([[], [3, 7]], max_hist_len=HIST_LEN, pad_idx=model.pad_idx)
    with torch.no_grad():
        logits = model(ids, mask, target)
    assert torch.isfinite(logits).all()
    assert not torch.allclose(logits[0], logits[1]), (
        "Empty-history and non-empty-history rows produced identical SASRec logits"
    )


def test_sasrec_training_reduces_bce():
    """A short training loop on synthetic binary data should reduce BCE loss."""
    torch.manual_seed(0)
    model = _make_sasrec(n_layers=1)
    examples = _synthetic_examples(n=100)
    out = train_din(
        model,
        examples,
        max_hist_len=HIST_LEN,
        cfg=DINTrainConfig(embed_dim=EMBED_DIM, epochs=20, batch_size=32, lr=5e-3, seed=0),
    )
    assert np.isfinite(out["first_epoch_loss"]) and np.isfinite(out["last_epoch_loss"]), (
        "SASRec training produced non-finite loss"
    )
    assert out["last_epoch_loss"] < out["first_epoch_loss"], (
        f"SASRec loss did not decrease: {out['first_epoch_loss']:.4f} -> {out['last_epoch_loss']:.4f}"
    )


# ---------------------------------------------------------------------------
# Empty-history NaN safety (explicit mixed batch with one all-zero hist_mask row)
# ---------------------------------------------------------------------------

def test_bst_empty_history_mixed_batch_nan_safe():
    """BST: batch with one all-padding row + real rows must all be finite."""
    torch.manual_seed(7)
    model = _make_bst()
    model.eval()
    B, L = 4, HIST_LEN
    # Row 0: fully padded (empty history); rows 1-3: random real history.
    hist_ids = torch.full((B, L), NUM_ITEMS, dtype=torch.long)
    hist_mask = torch.zeros(B, L)
    for i in range(1, B):
        vl = torch.randint(1, L + 1, (1,)).item()
        hist_ids[i, :vl] = torch.randint(0, NUM_ITEMS, (vl,))
        hist_mask[i, :vl] = 1.0
    target_ids = torch.randint(0, NUM_ITEMS, (B,))
    with torch.no_grad():
        logits = model(hist_ids, hist_mask, target_ids)
    assert torch.isfinite(logits).all(), (
        f"BST mixed-batch logits not finite: {logits}"
    )


def test_sasrec_empty_history_mixed_batch_nan_safe():
    """SASRec: batch with one all-padding row + real rows must all be finite."""
    torch.manual_seed(7)
    model = _make_sasrec()
    model.eval()
    B, L = 4, HIST_LEN
    hist_ids = torch.full((B, L), NUM_ITEMS, dtype=torch.long)
    hist_mask = torch.zeros(B, L)
    for i in range(1, B):
        vl = torch.randint(1, L + 1, (1,)).item()
        hist_ids[i, :vl] = torch.randint(0, NUM_ITEMS, (vl,))
        hist_mask[i, :vl] = 1.0
    target_ids = torch.randint(0, NUM_ITEMS, (B,))
    with torch.no_grad():
        logits = model(hist_ids, hist_mask, target_ids)
    assert torch.isfinite(logits).all(), (
        f"SASRec mixed-batch logits not finite: {logits}"
    )


# ---------------------------------------------------------------------------
# Registry integration
# ---------------------------------------------------------------------------

def test_registry_builds_bst():
    """build_model('bst', ...) must return a BST instance."""
    model = build_model("bst", num_items=NUM_ITEMS, embed_dim=EMBED_DIM)
    assert isinstance(model, BST)
    assert model.pad_idx == NUM_ITEMS


def test_registry_builds_sasrec():
    """build_model('sasrec', ...) must return a SASRec instance."""
    model = build_model("sasrec", num_items=NUM_ITEMS, embed_dim=EMBED_DIM)
    assert isinstance(model, SASRec)
    assert model.pad_idx == NUM_ITEMS
