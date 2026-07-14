"""Tests for ESMM_DCMT (Zhu et al., ICDE 2023, arXiv:2302.06141).

Spec: dcmt-formula-verification.md (6 resolved ambiguities A1-A6, 16 divergences).
Mirrors tests/test_egean.py style.
"""

import sys
import os

_IMPL_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "experiments", "20260404_ali_cpp_esmm",
)
if _IMPL_DIR not in sys.path:
    sys.path.insert(0, _IMPL_DIR)

import numpy as np
import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn
import torch.nn.functional as F

from esmm_ali_ccp_impl import ESMM_DCMT, ESMMModel_Wide, count_parameters

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

CARDS = [10, 10, 10]
NUM_DENSE = 3
EMBED_DIM = 8
BATCH = 32


def _mk_batch(seed=0):
    torch.manual_seed(seed)
    sparse_x = torch.randint(0, 10, (BATCH, len(CARDS)))
    dense_x = torch.randn(BATCH, NUM_DENSE)
    return sparse_x, dense_x


def _mk_labels(seed=0, click_rate=0.5, cvr_rate=0.3):
    torch.manual_seed(seed + 100)
    y_click = (torch.rand(BATCH) < click_rate).float()
    y_purchase = torch.where(
        y_click > 0.5,
        (torch.rand(BATCH) < cvr_rate).float(),
        torch.zeros(BATCH),
    )
    return y_click, y_purchase


def _make_model(**kw):
    return ESMM_DCMT(CARDS, NUM_DENSE, embed_dim=EMBED_DIM, **kw)


# ---------------------------------------------------------------------------
# 1. Shape and range tests
# ---------------------------------------------------------------------------

def test_forward_shapes():
    model = _make_model()
    model.eval()
    sp, dn = _mk_batch()
    with torch.no_grad():
        p_ctr, p_cvr, p_ctcvr = model(sp, dn)
    for name, p in [('p_ctr', p_ctr), ('p_cvr', p_cvr), ('p_ctcvr', p_ctcvr)]:
        assert p.shape == (BATCH,), f"{name}: expected ({BATCH},), got {p.shape}"
        assert torch.all(p >= 0.0) and torch.all(p <= 1.0), f"{name} out of [0,1]"
        assert torch.all(torch.isfinite(p)), f"{name} contains non-finite values"


def test_forward_ctcvr_product():
    """p_ctcvr = p_ctr * p_cvr (clamped)."""
    model = _make_model()
    model.eval()
    sp, dn = _mk_batch()
    with torch.no_grad():
        p_ctr, p_cvr, p_ctcvr = model(sp, dn)
    eps = 1e-7
    expected = (p_ctr * p_cvr).clamp(eps, 1 - eps)
    torch.testing.assert_close(p_ctcvr, expected, atol=1e-6, rtol=1e-5)


def test_forward_side_effect_p_cvr_cf():
    """forward() must set _last_p_cvr_cf with shape (B,) and values in (0, 1)."""
    model = _make_model()
    model.eval()
    sp, dn = _mk_batch()
    with torch.no_grad():
        model(sp, dn)
    assert model._last_p_cvr_cf.shape == (BATCH,), (
        f"_last_p_cvr_cf shape: expected ({BATCH},), got {model._last_p_cvr_cf.shape}"
    )
    assert torch.all(model._last_p_cvr_cf > 0) and torch.all(model._last_p_cvr_cf < 1), (
        "_last_p_cvr_cf not in open interval (0, 1)"
    )
    assert torch.all(torch.isfinite(model._last_p_cvr_cf)), "_last_p_cvr_cf non-finite"


# ---------------------------------------------------------------------------
# 2. Loss finite + grads to both twin towers
# ---------------------------------------------------------------------------

def test_loss_finite():
    torch.manual_seed(42)
    model = _make_model()
    model.train()
    sp, dn = _mk_batch()
    y_click, y_purchase = _mk_labels()
    model(sp, dn)
    loss = model.compute_dcmt_loss(y_click, y_purchase)
    assert torch.isfinite(loss), f"loss is not finite: {loss.item()}"


def test_grads_flow_to_both_twin_towers():
    """After backward, both factual CVR tower and counterfactual CVR head must have
    non-zero gradients — both towers are trained via their respective loss terms."""
    torch.manual_seed(7)
    model = _make_model()
    model.train()
    sp, dn = _mk_batch()
    y_click, y_purchase = _mk_labels()
    model(sp, dn)
    loss = model.compute_dcmt_loss(y_click, y_purchase)
    loss.backward()

    def _has_nonzero_grad(param_list):
        for p in param_list:
            if p.grad is not None and p.grad.abs().sum().item() > 0:
                return True
        return False

    # Factual CVR tower (shared body cvr_tower + wide_cvr)
    factual_params = list(model.cvr_tower.parameters()) + list(model.wide_cvr.parameters())
    # Counterfactual CVR head + wide
    cf_params = list(model.cvr_cf_head.parameters()) + list(model.wide_cvr_cf.parameters())
    # CTR tower
    ctr_params = list(model.ctr_tower.parameters()) + list(model.wide_ctr.parameters())

    assert _has_nonzero_grad(factual_params), "Factual CVR tower received no gradients"
    assert _has_nonzero_grad(cf_params), "Counterfactual CVR head received no gradients"
    assert _has_nonzero_grad(ctr_params), "CTR tower received no gradients"


def test_all_param_grads_finite():
    torch.manual_seed(13)
    model = _make_model()
    model.train()
    sp, dn = _mk_batch()
    y_click, y_purchase = _mk_labels()
    model(sp, dn)
    loss = model.compute_dcmt_loss(y_click, y_purchase)
    loss.backward()
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            assert torch.all(torch.isfinite(param.grad)), f"Non-finite grad in {name}"


# ---------------------------------------------------------------------------
# 3. SNIPS normalization hand-check on tiny batch
# ---------------------------------------------------------------------------

def test_snips_normalization_hand_check():
    """Eq. 13: SNIPS_factual = sum(e_f * o/ô) / sum(o/ô) — hand-computed tiny batch.

    Uses lambda1=0, lambda2=0, wctcvr=0, wcvr=1 to isolate E_DCMT.
    lambda1=0 removes L_cf; lambda2=0 removes L2; wctcvr=0 removes L_CTCVR.
    wcvr=1 keeps E_DCMT.

    We also zero out the CTR loss by setting wctcvr=0 and using lambda1=0, lambda2=0,
    and note that l_ctr is always present (can't zero it). Instead we check the
    DCMT portion by computing by hand and comparing to the full loss minus l_ctr.
    """
    torch.manual_seed(77)
    model = _make_model(lambda1=0.0, lambda2=0.0, wcvr=1.0, wctcvr=0.0)
    model.eval()

    # Tiny batch of 6 rows: 4 clicked, 2 non-clicked
    B = 6
    yc = torch.tensor([1., 1., 1., 1., 0., 0.])
    yp = torch.tensor([1., 0., 1., 0., 0., 0.])

    # Manual p_ctr and p_cvr values; override stash directly
    p_ctr_v = torch.tensor([0.6, 0.7, 0.4, 0.8, 0.3, 0.5])
    p_cvr_v = torch.tensor([0.7, 0.3, 0.6, 0.2, 0.5, 0.4])
    # CF preds
    p_cvr_cf_v = torch.tensor([0.4, 0.6, 0.35, 0.75, 0.55, 0.5])

    model._last_p_ctr     = p_ctr_v
    model._last_p_cvr     = p_cvr_v
    model._last_p_ctcvr   = (p_ctr_v * p_cvr_v).clamp(1e-7, 1 - 1e-7)
    model._last_p_cvr_cf  = p_cvr_cf_v
    model._last_h_cvr     = torch.zeros(B, 80)   # dummy; not used in loss

    loss = model.compute_dcmt_loss(yc, yp)

    # Hand-compute SNIPS factual
    eps = 1e-7
    ô = p_ctr_v.clamp(1e-6, 1 - 1e-6).numpy()
    yc_np = yc.numpy()
    yp_np = yp.numpy()
    yp_cf_np = 1.0 - yp_np

    p_cvr_np    = p_cvr_v.clamp(eps, 1 - eps).numpy()
    p_cvr_cf_np = p_cvr_cf_v.clamp(eps, 1 - eps).numpy()

    e_f  = -(yp_np    * np.log(p_cvr_np)    + (1 - yp_np)    * np.log(1 - p_cvr_np))
    e_cf = -(yp_cf_np * np.log(p_cvr_cf_np) + (1 - yp_cf_np) * np.log(1 - p_cvr_cf_np))

    click_idx    = np.where(yc_np > 0.5)[0]
    nonclick_idx = np.where(yc_np < 0.5)[0]

    inv_o    = 1.0 / ô
    inv_1mo  = 1.0 / (1.0 - ô)

    snips_num_f  = float(np.sum(e_f  * yc_np * inv_o))
    snips_den_f  = float(np.sum(inv_o[click_idx]))
    snips_f      = snips_num_f / max(snips_den_f, eps)

    snips_num_cf = float(np.sum(e_cf * (1 - yc_np) * inv_1mo))
    snips_den_cf = float(np.sum(inv_1mo[nonclick_idx]))
    snips_cf     = snips_num_cf / max(snips_den_cf, eps)

    e_dcmt_main = snips_f + snips_cf   # SNIPS already self-normalised; no /B

    # l_cf = 0 (lambda1=0), l2 = 0 (lambda2=0), wctcvr=0 → l_ctcvr dropped
    # l_ctr is still present
    p_ctr_c = p_ctr_v.clamp(eps, 1 - eps).numpy()
    l_ctr = float(np.mean(-(yc_np * np.log(p_ctr_c) + (1 - yc_np) * np.log(1 - p_ctr_c))))

    expected = l_ctr + 1.0 * e_dcmt_main   # wcvr=1.0

    torch.testing.assert_close(
        loss, torch.tensor(expected, dtype=torch.float32),
        atol=1e-4, rtol=1e-4,
        msg=f"SNIPS normalization mismatch: got {loss.item():.6f}, expected {expected:.6f}"
    )


# ---------------------------------------------------------------------------
# 4. Counterfactual label flip semantics
# ---------------------------------------------------------------------------

def test_counterfactual_label_flip():
    """r* = 1 − r (Trick 1): counterfactual tower must be trained on FLIPPED labels.

    Verification: set lambda1=0, lambda2=0, wctcvr=0, wcvr=1, and use a batch
    where all rows are non-click (yc=0) so only the counterfactual term contributes
    to E_DCMT. The factual SNIPS is 0 (no clicks). We check the loss equals
    l_ctr + snips_cf where snips_cf uses yp_cf = 1 - yp = 1 - 0 = 1 (all-1 cf labels).
    """
    torch.manual_seed(55)
    model = _make_model(lambda1=0.0, lambda2=0.0, wcvr=1.0, wctcvr=0.0)
    model.eval()

    B = 4
    yc = torch.zeros(B)                         # all non-click
    yp = torch.zeros(B)                         # r = 0 for all
    # r* = 1 - r = 1 for all non-click rows

    p_ctr_v    = torch.tensor([0.4, 0.5, 0.3, 0.6])
    p_cvr_v    = torch.tensor([0.5, 0.4, 0.6, 0.7])   # not used in SNIPS_f (no clicks)
    p_cvr_cf_v = torch.tensor([0.7, 0.8, 0.6, 0.9])   # CF tower preds
    eps = 1e-7

    model._last_p_ctr    = p_ctr_v
    model._last_p_cvr    = p_cvr_v
    model._last_p_ctcvr  = (p_ctr_v * p_cvr_v).clamp(eps, 1 - eps)
    model._last_p_cvr_cf = p_cvr_cf_v
    model._last_h_cvr    = torch.zeros(B, 80)

    loss = model.compute_dcmt_loss(yc, yp)

    # Hand-compute
    ô_np = p_ctr_v.clamp(1e-6, 1 - 1e-6).numpy()
    yc_np = yc.numpy()
    yp_cf_np = (1.0 - yp).numpy()    # all 1.0
    p_cvr_cf_np = p_cvr_cf_v.clamp(eps, 1 - eps).numpy()
    # e_cf = BCE(p_cvr_cf, 1) = -log(p_cvr_cf) for all rows
    e_cf = -np.log(p_cvr_cf_np)

    inv_1mo = 1.0 / (1.0 - ô_np)
    snips_num_cf = float(np.sum(e_cf * (1 - yc_np) * inv_1mo))
    snips_den_cf = float(np.sum(inv_1mo))  # all non-click rows → all rows
    snips_cf = snips_num_cf / max(snips_den_cf, eps)
    e_dcmt_main = snips_cf   # snips_f = 0 (no clicks); SNIPS already self-normalised

    p_ctr_c = p_ctr_v.clamp(eps, 1 - eps).numpy()
    l_ctr = float(np.mean(-(yc_np * np.log(p_ctr_c) + (1 - yc_np) * np.log(1 - p_ctr_c))))

    expected = l_ctr + e_dcmt_main

    torch.testing.assert_close(
        loss, torch.tensor(expected, dtype=torch.float32),
        atol=1e-4, rtol=1e-4,
        msg=f"CF label flip mismatch: got {loss.item():.6f}, expected {expected:.6f}"
    )


# ---------------------------------------------------------------------------
# 5. Propensity detach: CTR grads unaffected by DCMT CVR loss
# ---------------------------------------------------------------------------

def test_propensity_detach_ctr_grads_identical():
    """CTR tower gradients must be identical whether wcvr=0 or wcvr=1.

    The propensity ô in the SNIPS denominators is detached from the CTR tower
    (Ambiguity A2), so the CVR/CF loss terms do NOT backpropagate into the CTR tower.
    We verify this by comparing CTR gradients with and without the DCMT CVR term.
    """
    seed = 42
    sp, dn = _mk_batch(seed)
    y_click, y_purchase = _mk_labels(seed)

    # wcvr=0: no DCMT CVR contribution
    torch.manual_seed(seed)
    model0 = _make_model(lambda1=0.0, lambda2=0.0, wcvr=0.0, wctcvr=0.0)
    model0.train()
    model0(sp.clone(), dn.clone())
    loss0 = model0.compute_dcmt_loss(y_click.clone(), y_purchase.clone())
    loss0.backward()
    ctr_grads_0 = {
        n: p.grad.detach().clone()
        for n, p in model0.named_parameters()
        if p.requires_grad and p.grad is not None
        and ('ctr_tower' in n or 'wide_ctr' in n)
    }

    # wcvr=1.0: DCMT CVR term active
    torch.manual_seed(seed)
    model1 = _make_model(lambda1=0.0, lambda2=0.0, wcvr=1.0, wctcvr=0.0)
    model1.train()
    model1.load_state_dict(model0.state_dict())
    model1(sp.clone(), dn.clone())
    loss1 = model1.compute_dcmt_loss(y_click.clone(), y_purchase.clone())
    loss1.backward()
    ctr_grads_1 = {
        n: p.grad.detach().clone()
        for n, p in model1.named_parameters()
        if p.requires_grad and p.grad is not None
        and ('ctr_tower' in n or 'wide_ctr' in n)
    }

    assert set(ctr_grads_0.keys()) == set(ctr_grads_1.keys()), (
        "CTR parameter names differ between wcvr=0 and wcvr=1 models"
    )
    for name in ctr_grads_0:
        torch.testing.assert_close(
            ctr_grads_0[name], ctr_grads_1[name], atol=1e-6, rtol=1e-5,
            msg=f"CTR grad for {name} changed with wcvr=1 (propensity detach violated)"
        )


# ---------------------------------------------------------------------------
# 6. Constraint term: L_cf = lambda1 * |1 - (r̂ + r̂*)| over 𝒟
# ---------------------------------------------------------------------------

def test_constraint_term_semantics():
    """L_cf = lambda1 * mean(|1 - (r̂ + r̂*)|) over 𝒟.

    With lambda1=1.0, wcvr=0, wctcvr=0, lambda2=0: the loss should equal
    l_ctr + mean(|1 - (p_cvr + p_cvr_cf)|).
    """
    torch.manual_seed(66)
    model = _make_model(lambda1=1.0, lambda2=0.0, wcvr=1.0, wctcvr=0.0)
    model.eval()

    B = 8
    sp, dn = _mk_batch(66)
    y_click, y_purchase = _mk_labels(66)

    # After forward(), SNIPS and constraint will be computed together.
    with torch.no_grad():
        model(sp[:B], dn[:B])

    yc = y_click[:B]
    yp = y_purchase[:B]
    loss = model.compute_dcmt_loss(yc, yp)

    # Verify: constraint term must be >= 0 and the loss should change if we zero it.
    model2 = _make_model(lambda1=0.0, lambda2=0.0, wcvr=1.0, wctcvr=0.0)
    model2.load_state_dict(model.state_dict())
    model2.eval()
    with torch.no_grad():
        model2(sp[:B], dn[:B])
    loss_no_constraint = model2.compute_dcmt_loss(yc, yp)

    # The constraint term = lambda1 * |1 - (r̂ + r̂*)|.mean()
    # Since lambda1=1 vs lambda1=0, losses should differ
    p_cvr    = model._last_p_cvr.detach()
    p_cvr_cf = model._last_p_cvr_cf.detach()
    constraint = (1.0 - (p_cvr + p_cvr_cf)).abs().mean()

    # Losses should differ by the constraint value (same SNIPS because same model state)
    diff = abs(loss.item() - loss_no_constraint.item())
    expected_diff = constraint.item()
    assert abs(diff - expected_diff) < 1e-4, (
        f"Constraint contribution mismatch: diff={diff:.6f}, expected={expected_diff:.6f}"
    )
    assert torch.isfinite(loss), "Loss with constraint term is non-finite"


# ---------------------------------------------------------------------------
# 7. Counterfactual tower is separate from factual at inference
# ---------------------------------------------------------------------------

def test_counterfactual_tower_does_not_affect_factual_inference():
    """r̂* (counterfactual) must differ from r̂ (factual) for a non-trivial batch.

    The twin towers share deep params θ_d but have separate heads,
    so their outputs should generally differ (no full parameter sharing of heads).
    """
    torch.manual_seed(17)
    model = _make_model()
    model.eval()
    sp, dn = _mk_batch()
    with torch.no_grad():
        p_ctr, p_cvr, p_ctcvr = model(sp, dn)
        p_cvr_cf = model._last_p_cvr_cf

    # p_cvr and p_cvr_cf should differ (separate CF head params)
    diff = (p_cvr - p_cvr_cf).abs().mean().item()
    assert diff > 1e-4, (
        f"p_cvr and p_cvr_cf are identical (mean diff={diff:.2e}); "
        "CF head may not be separate from factual head"
    )

    # Forward return value must only contain factual predictions
    assert p_cvr.shape == (BATCH,)
    assert p_ctcvr.shape == (BATCH,)
    torch.testing.assert_close(
        p_ctcvr, (p_ctr * p_cvr).clamp(1e-7, 1 - 1e-7),
        atol=1e-6, rtol=1e-5,
        msg="p_ctcvr must be p_ctr * p_cvr (factual only; CF discarded at inference)"
    )
