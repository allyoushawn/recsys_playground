"""Tests for OneTrans sequence-transformer ranking model.

Covers:
- Output shape [B] and finite values
- All 4 toggle combinations (use_mixed_param x use_pyramid)
- Masked vs unmasked history rows produce different logits
- Short training loop on synthetic data reduces BCE loss
- embed_dim % n_heads guard raises ValueError
- Empty-history row produces finite output (no NaNs)
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn

from amazon_ranking.src.din import DINTrainConfig, pad_histories, train_din
from amazon_ranking.src.onetrans.onetrans import OneTrans
from amazon_ranking.src.onetrans.registry import build_model


# ---------------------------------------------------------------------------
# Shared fixtures / helpers
# ---------------------------------------------------------------------------

NUM_ITEMS = 50
EMBED_DIM = 8
HIST_LEN = 6
BATCH = 4


def _make_onetrans(**kwargs) -> OneTrans:
    defaults = dict(n_layers=1, n_heads=2, n_ns_tokens=2, max_len=10)
    defaults.update(kwargs)
    return OneTrans(num_items=NUM_ITEMS, embed_dim=EMBED_DIM, **defaults)


def _random_batch(B: int = BATCH, L: int = HIST_LEN, seed: int = 0):
    """Return (hist_ids, hist_mask, target_ids) with mixed padding."""
    rng = torch.Generator()
    rng.manual_seed(seed)
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
# Basic forward tests
# ---------------------------------------------------------------------------

def test_onetrans_output_shape_and_finite():
    """OneTrans forward produces shape [B] with all finite values."""
    torch.manual_seed(0)
    model = _make_onetrans()
    hist_ids, hist_mask, target_ids = _random_batch()
    logits = model(hist_ids, hist_mask, target_ids)
    assert logits.shape == (BATCH,), f"expected ({BATCH},), got {logits.shape}"
    assert torch.isfinite(logits).all(), "OneTrans output contains non-finite values"


def test_onetrans_empty_history_no_nan():
    """OneTrans handles all-padding (empty history) without producing NaNs."""
    torch.manual_seed(0)
    model = _make_onetrans()
    ids, mask = pad_histories([[]], max_hist_len=HIST_LEN, pad_idx=model.pad_idx)
    target = torch.tensor([5], dtype=torch.long)
    logits = model(ids, mask, target)
    assert torch.isfinite(logits).all(), "OneTrans output is NaN for empty history"


# ---------------------------------------------------------------------------
# All 4 toggle combinations (use_mixed_param x use_pyramid)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("use_mixed_param,use_pyramid", [
    (True, False),
    (True, True),
    (False, False),
    (False, True),
])
def test_onetrans_toggle_combinations(use_mixed_param, use_pyramid):
    """OneTrans runs without errors for all 4 toggle combinations."""
    torch.manual_seed(0)
    model = _make_onetrans(
        n_layers=2,
        use_mixed_param=use_mixed_param,
        use_pyramid=use_pyramid,
    )
    hist_ids, hist_mask, target_ids = _random_batch()
    logits = model(hist_ids, hist_mask, target_ids)
    assert logits.shape == (BATCH,), (
        f"toggle ({use_mixed_param}, {use_pyramid}): expected ({BATCH},), got {logits.shape}"
    )
    assert torch.isfinite(logits).all(), (
        f"toggle ({use_mixed_param}, {use_pyramid}): non-finite logits"
    )


# ---------------------------------------------------------------------------
# Masking behavior
# ---------------------------------------------------------------------------

def test_onetrans_masked_vs_unmasked_differ():
    """A row with history must produce a different logit from a fully-masked
    (empty-history) row for the same target, confirming masking is active."""
    torch.manual_seed(2)
    model = _make_onetrans()
    model.eval()
    target = torch.tensor([5, 5], dtype=torch.long)
    ids, mask = pad_histories([[], [3, 7]], max_hist_len=HIST_LEN, pad_idx=model.pad_idx)
    with torch.no_grad():
        logits = model(ids, mask, target)
    assert torch.isfinite(logits).all()
    assert not torch.allclose(logits[0], logits[1]), (
        "Empty-history and non-empty-history rows produced identical OneTrans logits"
    )


# ---------------------------------------------------------------------------
# Training convergence
# ---------------------------------------------------------------------------

def test_onetrans_training_reduces_bce():
    """A short training loop on synthetic binary data should reduce BCE loss."""
    torch.manual_seed(0)
    model = _make_onetrans(n_layers=1)
    examples = _synthetic_examples(n=100)
    out = train_din(
        model,
        examples,
        max_hist_len=HIST_LEN,
        cfg=DINTrainConfig(embed_dim=EMBED_DIM, epochs=20, batch_size=32, lr=5e-3, seed=0),
    )
    assert np.isfinite(out["first_epoch_loss"]) and np.isfinite(out["last_epoch_loss"]), (
        "OneTrans training produced non-finite loss"
    )
    assert out["last_epoch_loss"] < out["first_epoch_loss"], (
        f"OneTrans loss did not decrease: "
        f"{out['first_epoch_loss']:.4f} -> {out['last_epoch_loss']:.4f}"
    )


# ---------------------------------------------------------------------------
# Empty-history NaN safety (explicit mixed batch with one all-zero hist_mask row)
# ---------------------------------------------------------------------------

def test_onetrans_empty_history_mixed_batch_nan_safe():
    """OneTrans: batch with one all-padding row + real rows must all be finite."""
    torch.manual_seed(7)
    model = _make_onetrans()
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
        f"OneTrans mixed-batch logits not finite: {logits}"
    )


# ---------------------------------------------------------------------------
# Pyramid freeze produces different logits from non-pyramid
# ---------------------------------------------------------------------------

def test_onetrans_pyramid_changes_logits():
    """use_pyramid=True must produce DIFFERENT logits from use_pyramid=False.

    Requires n_layers >= 3 so that at least one layer (block 2) attends to S keys
    that were updated differently: in the flat model the inactive S rows are updated
    by block 1 attention + FFN, while in the pyramid model they carry frozen block-0
    residuals.  Block 2 NS queries then attend to these divergent S keys, producing
    different NS hidden states and therefore different final logits.

    With only 2 layers the divergence is invisible because the final head only uses
    NS tokens, and NS output at block 1 depends on S keys from block 0 — which are
    identical for both models at that point.
    """
    torch.manual_seed(42)
    # n_layers=3: blocks 0, 1 (pyramid kicks in), 2 (sees divergent S keys).
    # L=6 > 2, pyramid_keep defaults to max(1, 6//2)=3, leaving 3 rows frozen.
    model_flat = _make_onetrans(n_layers=3, use_pyramid=False)
    model_flat.eval()

    # Build model_pyr with the same weights so the only difference is the flag.
    model_pyr = _make_onetrans(n_layers=3, use_pyramid=True)
    model_pyr.load_state_dict(model_flat.state_dict())
    model_pyr.eval()

    hist_ids, hist_mask, target_ids = _random_batch(B=4, L=HIST_LEN)
    with torch.no_grad():
        logits_flat = model_flat(hist_ids, hist_mask, target_ids)
        logits_pyr = model_pyr(hist_ids, hist_mask, target_ids)

    assert torch.isfinite(logits_flat).all()
    assert torch.isfinite(logits_pyr).all()
    assert not torch.allclose(logits_flat, logits_pyr, atol=1e-6), (
        "use_pyramid=True and use_pyramid=False produced identical logits — "
        "pyramid freeze has no effect.  With n_layers=3 the divergent S key "
        "states should propagate to NS tokens in block 2."
    )


# ---------------------------------------------------------------------------
# Guard / validation tests
# ---------------------------------------------------------------------------

def test_onetrans_embed_dim_n_heads_guard():
    """OneTrans must raise ValueError when embed_dim is not divisible by n_heads."""
    with pytest.raises(ValueError, match="embed_dim"):
        OneTrans(num_items=NUM_ITEMS, embed_dim=9, n_heads=4)


# ---------------------------------------------------------------------------
# Registry integration
# ---------------------------------------------------------------------------

def test_registry_builds_onetrans():
    """build_model('onetrans', ...) must return an OneTrans instance."""
    torch.manual_seed(0)
    model = build_model("onetrans", num_items=NUM_ITEMS, embed_dim=EMBED_DIM, n_heads=2)
    assert isinstance(model, OneTrans)
    assert model.pad_idx == NUM_ITEMS
    # Quick forward check.
    hist_ids, hist_mask, target_ids = _random_batch(B=2)
    with torch.no_grad():
        logits = model(hist_ids, hist_mask, target_ids)
    assert logits.shape == (2,)
    assert torch.isfinite(logits).all()
