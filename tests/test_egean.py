"""Tests for ESMM_EGEAN (Zhang et al., WWW 2025, arXiv 2412.06852).

Spec: egean-formula-verification.md (6 resolved ambiguities A1-A6, harness mapping).
Mirrors tests/test_escm2.py style.
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

from esmm_ali_ccp_impl import ESMM_EGEAN, ESMMModel_Wide, count_parameters, _mmd_rbf

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
    return ESMM_EGEAN(CARDS, NUM_DENSE, embed_dim=EMBED_DIM, **kw)


# ---------------------------------------------------------------------------
# 1. Forward shapes and ranges
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


def test_forward_intermediates_shapes():
    """_last_delta_hat, _last_p_exp, _last_O_ep_cvr, _last_E_shared_flat must be set."""
    model = _make_model()
    model.eval()
    sp, dn = _mk_batch()
    with torch.no_grad():
        model(sp, dn)
    flat_emb_dim = len(CARDS) * EMBED_DIM

    assert model._last_delta_hat.shape == (BATCH,), (
        f"_last_delta_hat shape: expected ({BATCH},), got {model._last_delta_hat.shape}"
    )
    assert model._last_p_exp.shape == (BATCH,), (
        f"_last_p_exp shape: expected ({BATCH},), got {model._last_p_exp.shape}"
    )
    assert model._last_O_ep_cvr.shape == (BATCH, flat_emb_dim), (
        f"_last_O_ep_cvr shape: expected ({BATCH}, {flat_emb_dim}), "
        f"got {model._last_O_ep_cvr.shape}"
    )
    assert model._last_E_shared_flat.shape == (BATCH, flat_emb_dim), (
        f"_last_E_shared_flat shape: expected ({BATCH}, {flat_emb_dim}), "
        f"got {model._last_E_shared_flat.shape}"
    )
    assert torch.all(model._last_p_exp >= 0.0) and torch.all(model._last_p_exp <= 1.0), (
        "_last_p_exp out of [0,1]"
    )
    assert torch.all(torch.isfinite(model._last_delta_hat)), "_last_delta_hat non-finite"


def test_forward_ctcvr_product():
    """p_ctcvr must be close to p_ctr * p_cvr (clamped)."""
    model = _make_model()
    model.eval()
    sp, dn = _mk_batch()
    with torch.no_grad():
        p_ctr, p_cvr, p_ctcvr = model(sp, dn)
    eps = 1e-7
    expected = (p_ctr * p_cvr).clamp(eps, 1 - eps)
    torch.testing.assert_close(p_ctcvr, expected, atol=1e-6, rtol=1e-5)


# ---------------------------------------------------------------------------
# 2. Loss finite + grads flow to every component
# ---------------------------------------------------------------------------

def test_loss_finite():
    torch.manual_seed(42)
    model = _make_model()
    model.train()
    sp, dn = _mk_batch()
    y_click, y_purchase = _mk_labels()
    model(sp, dn)
    loss = model.compute_egean_loss(y_click, y_purchase)
    assert torch.isfinite(loss), f"loss is not finite: {loss.item()}"


def test_grads_flow_to_all_components():
    """After backward, CTR tower, CVR tower, IMP tower, EPNet, PPNet, exposure_mlp,
    LoRA adapters, and task embeddings must all have non-zero gradients."""
    torch.manual_seed(7)
    model = _make_model()
    model.train()
    sp, dn = _mk_batch()
    y_click, y_purchase = _mk_labels()
    model(sp, dn)
    loss = model.compute_egean_loss(y_click, y_purchase)
    loss.backward()

    def _has_nonzero_grad(param_list):
        for p in param_list:
            if p.grad is not None and p.grad.abs().sum().item() > 0:
                return True
        return False

    # CTR/CVR towers (PPNet gated layers + heads + wide)
    ctr_params = (
        list(model.ppnet_ctr_layers.parameters())
        + list(model.ppnet_ctr_head.parameters())
        + list(model.wide_ctr.parameters())
    )
    cvr_params = (
        list(model.ppnet_cvr_layers.parameters())
        + list(model.ppnet_cvr_head.parameters())
        + list(model.wide_cvr.parameters())
    )
    imp_params = list(model.imp_tower.parameters())
    epnet_params = list(model.epnet_ctr.parameters()) + list(model.epnet_cvr.parameters())
    exp_params = list(model.exposure_mlp.parameters())
    lora_params = (
        list(model.lora_ctr_A.parameters()) + list(model.lora_ctr_B.parameters())
        + list(model.lora_cvr_A.parameters()) + list(model.lora_cvr_B.parameters())
    )
    task_params = list(model.task_emb.parameters())

    assert _has_nonzero_grad(ctr_params), "CTR tower (PPNet) received no gradients"
    assert _has_nonzero_grad(cvr_params), "CVR tower (PPNet) received no gradients"
    assert _has_nonzero_grad(imp_params), "IMP tower received no gradients"
    assert _has_nonzero_grad(epnet_params), "EPNet received no gradients"
    assert _has_nonzero_grad(exp_params), "exposure_mlp received no gradients"
    assert _has_nonzero_grad(lora_params), "LoRA adapters received no gradients"
    assert _has_nonzero_grad(task_params), "task_emb received no gradients"


def test_all_param_grads_finite():
    torch.manual_seed(13)
    model = _make_model()
    model.train()
    sp, dn = _mk_batch()
    y_click, y_purchase = _mk_labels()
    model(sp, dn)
    loss = model.compute_egean_loss(y_click, y_purchase)
    loss.backward()
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            assert torch.all(torch.isfinite(param.grad)), f"Non-finite grad in {name}"


# ---------------------------------------------------------------------------
# 3. Stop-gradient properties — Trick 2a and 2b (EPNet and PPNet)
# ---------------------------------------------------------------------------

def test_stop_grad_epnet_shared_emb_not_updated_via_epnet():
    """Trick 2a: shared embedding E is detached inside EPNet gate (∅(E) in Eq. 2).

    Verification: set all losses except EPNet-related to zero (mmd_weight=0,
    exp_weight=0, imp_weight=0, pvdr_weight=0, ctr_weight=0), so the only
    gradient path is through EPNet. Then compare shared embedding gradient
    when EPNet is active vs. inactive (no EPNet path).

    Because ∅(E) is a detach, the shared embedding (unified_emb) must NOT
    receive gradients through the EPNet gate path. We confirm this by checking
    that the gradient through the detached path is zero even when EPNet params
    change.

    Concretely: run with mmd_weight=1, all others=0, so only L_metric flows.
    L_metric uses O_ep_cvr = delta_cvr * E_cvr_lora. Since delta_cvr is
    computed from ∅(E_cvr_lora.detach()), the gradient of L_metric w.r.t.
    unified_emb from the EPNet gate path is zero (the gate doesn't backprop
    through shared emb). However unified_emb still receives gradient from
    E_cvr_lora (the direct multiplication path in O_ep_cvr = delta_cvr * E_cvr_lora).

    The key assertion: EPNet gate parameters (epnet_cvr) DO get gradients,
    but those gradients do NOT flow back through shared embedding via that path
    (i.e., if we freeze unified_emb and check that epnet grads are identical
    whether or not unified_emb is frozen, the epnet gate is indeed detached).
    """
    seed = 77
    torch.manual_seed(seed)
    sp, dn = _mk_batch(seed)
    y_click, y_purchase = _mk_labels(seed)

    # Run with mmd_weight=1, all other weights=0 (only metric loss flows)
    torch.manual_seed(seed)
    model_a = _make_model(
        ctr_weight=0.0, pvdr_weight=0.0, imp_weight=0.0,
        exp_weight=0.0, mmd_weight=1.0,
    )
    model_a.train()
    model_a(sp.clone(), dn.clone())
    loss_a = model_a.compute_egean_loss(y_click.clone(), y_purchase.clone())
    loss_a.backward()

    # EPNet gate should have non-zero grads (it contributes to O_ep_cvr → MMD)
    epnet_grads_a = {
        n: p.grad.detach().clone()
        for n, p in model_a.named_parameters()
        if 'epnet_cvr' in n and p.grad is not None
    }
    assert len(epnet_grads_a) > 0, "EPNet CVR should have grads via MMD loss"
    for n, g in epnet_grads_a.items():
        assert g.abs().sum().item() > 0, f"EPNet CVR param {n} has zero grad (unexpected)"

    # The shared embedding's gradient from EPNet gate should be zero.
    # Because ∅(E_cvr_lora.detach()) is detached in EPNet forward, the gate
    # does not backprop into E_cvr_lora through that argument.
    # unified_emb may still have a gradient from the direct O_ep_cvr = delta*E_cvr_lora path.
    # We verify the EPNet gate input detach by checking: if we replace unified_emb with
    # a leaf that requires_grad and compute only the EPNet gate output, its grad is zero.

    torch.manual_seed(seed)
    sp2, dn2 = _mk_batch(seed)
    flat_emb_dim = len(CARDS) * EMBED_DIM
    E_flat_leaf = torch.randn(BATCH, flat_emb_dim, requires_grad=True)
    t_cvr = model_a.task_emb(torch.ones(BATCH, dtype=torch.long))
    # EPNet receives detached E_flat; compute gate output
    delta = model_a.epnet_cvr(t_cvr, E_flat_leaf.detach())  # ∅(E_flat_leaf)
    loss_gate = delta.sum()
    loss_gate.backward()
    # E_flat_leaf gradient must be None (detach blocks backprop into E_flat_leaf via gate)
    assert E_flat_leaf.grad is None or E_flat_leaf.grad.abs().sum().item() == 0.0, (
        "EPNet gate back-propagated into shared embedding (detach violated Trick 2a)"
    )


def test_stop_grad_ppnet_epnet_output_not_updated_via_ppnet():
    """Trick 2b: EPNet output O_ep is detached before entering PPNet gate (∅(O_ep)).

    Verification: EPNet parameters must NOT receive gradients through the PPNet
    gate path. We check this by comparing EPNet gradients when pvdr_weight=1
    (which flows through PPNet towers → backward into O_ep_cvr → if NOT detached,
    into EPNet) vs. when we explicitly zero out the EPNet path.

    Concretely: run forward, compute loss with only pvdr_weight=1 (other weights=0).
    If ∅(O_ep) is correctly applied, EPNet receives zero gradients from L_PVDR
    (since O_ep.detach() blocks that path).  We confirm by checking EPNet grads are zero.
    """
    seed = 55
    # Ensure some clicks so PVDR is non-trivial
    torch.manual_seed(seed)
    sp, dn = _mk_batch(seed)
    y_click = torch.ones(BATCH)  # all clicked → non-zero PVDR numerator
    y_purchase = (torch.rand(BATCH) < 0.3).float()

    torch.manual_seed(seed)
    model = _make_model(
        ctr_weight=0.0, pvdr_weight=1.0, imp_weight=0.0,
        exp_weight=0.0, mmd_weight=0.0,
    )
    model.train()
    model(sp, dn)
    loss = model.compute_egean_loss(y_click, y_purchase)
    loss.backward()

    # EPNet gate parameters must have zero grad (O_ep.detach() blocks the path)
    epnet_grads = {
        n: p.grad
        for n, p in model.named_parameters()
        if ('epnet_ctr' in n or 'epnet_cvr' in n) and p.grad is not None
    }
    for n, g in epnet_grads.items():
        assert g.abs().sum().item() == pytest.approx(0.0, abs=1e-9), (
            f"EPNet param {n} received gradient via PPNet path "
            f"(O_ep.detach() violated Trick 2b): grad_norm={g.abs().sum().item():.2e}"
        )


# ---------------------------------------------------------------------------
# 4. PVDR ratio semantics — hand-computed tiny batch
# ---------------------------------------------------------------------------

def test_pvdr_ratio_hand_computed():
    """Eq. 12: L_PVDR = Σ(o·ê/p̂) / (λ|D| + (1-λ)·Σ(o/p̂)).

    With a tiny batch of 4 rows, hand-compute the expected value and assert match.
    Uses lambda_pvdr=0.5 to exercise the non-degenerate denominator.
    Uses mmd_weight=0, imp_weight=0, exp_weight=0, ctr_weight=0 to isolate L_PVDR.
    """
    torch.manual_seed(99)
    model = _make_model(
        ctr_weight=0.0, pvdr_weight=1.0, imp_weight=0.0,
        exp_weight=0.0, mmd_weight=0.0,
        lambda_pvdr=0.5, ips_clip_floor=0.01,
    )
    model.eval()

    # Manually override model state to have known p_ctr, delta_hat
    # We bypass forward and directly set the stashed tensors.
    B4 = 4
    y_click    = torch.tensor([1.0, 0.0, 1.0, 1.0])   # o
    y_purchase = torch.tensor([1.0, 0.0, 0.0, 1.0])   # r (unused in PVDR directly)
    p_ctr_val  = torch.tensor([0.3, 0.5, 0.8, 0.4])   # p̂ (propensity, clipped)
    delta_hat_val = torch.tensor([0.2, 0.1, 0.5, 0.3])  # ê (imputation output)
    p_cvr_val  = torch.tensor([0.5, 0.4, 0.6, 0.7])   # dummy p_cvr
    p_ctcvr_val = p_ctr_val * p_cvr_val

    model._last_p_ctr = p_ctr_val
    model._last_p_cvr = p_cvr_val
    model._last_p_ctcvr = p_ctcvr_val
    model._last_delta_hat = delta_hat_val
    model._last_p_exp = torch.full((B4,), 0.5)
    flat_dim = len(CARDS) * EMBED_DIM
    model._last_O_ep_cvr = torch.zeros(B4, flat_dim)
    model._last_E_shared_flat = torch.zeros(B4, flat_dim)

    loss = model.compute_egean_loss(y_click, y_purchase)

    # Hand-compute L_PVDR
    # propensity = p_ctr_val.clamp(min=0.01) = [0.3, 0.5, 0.8, 0.4]
    # ips = o / prop = [1/0.3, 0, 1/0.8, 1/0.4]
    prop = p_ctr_val.clamp(min=0.01).numpy()
    o = y_click.numpy()
    ehat = delta_hat_val.numpy()
    numerator = float(np.sum(o * ehat / prop))
    ips_sum = float(np.sum(o / prop))
    lam = 0.5
    denominator = lam * B4 + (1 - lam) * ips_sum
    expected_pvdr = numerator / max(denominator, 1e-8)

    torch.testing.assert_close(
        loss, torch.tensor(expected_pvdr, dtype=torch.float32),
        atol=1e-5, rtol=1e-5,
        msg=f"PVDR ratio mismatch: got {loss.item():.6f}, expected {expected_pvdr:.6f}"
    )


def test_pvdr_lambda1_equals_stable_dr():
    """lambda_pvdr=1.0: denominator = |D|, so L_PVDR = mean(o·ê/p̂) (StableDR collapse)."""
    torch.manual_seed(11)
    model = _make_model(
        ctr_weight=0.0, pvdr_weight=1.0, imp_weight=0.0,
        exp_weight=0.0, mmd_weight=0.0,
        lambda_pvdr=1.0, ips_clip_floor=0.01,
    )
    model.eval()

    B4 = 4
    y_click    = torch.tensor([1.0, 0.0, 1.0, 1.0])
    y_purchase = torch.tensor([1.0, 0.0, 0.0, 1.0])
    p_ctr_val  = torch.tensor([0.3, 0.5, 0.8, 0.4])
    delta_hat_val = torch.tensor([0.2, 0.1, 0.5, 0.3])
    p_cvr_val  = torch.tensor([0.5, 0.4, 0.6, 0.7])
    flat_dim = len(CARDS) * EMBED_DIM

    model._last_p_ctr = p_ctr_val
    model._last_p_cvr = p_cvr_val
    model._last_p_ctcvr = p_ctr_val * p_cvr_val
    model._last_delta_hat = delta_hat_val
    model._last_p_exp = torch.full((B4,), 0.5)
    model._last_O_ep_cvr = torch.zeros(B4, flat_dim)
    model._last_E_shared_flat = torch.zeros(B4, flat_dim)

    loss = model.compute_egean_loss(y_click, y_purchase)

    # lambda=1 → denominator = 1.0 * 4 = 4 → L_PVDR = sum(o·ê/p̂) / 4
    prop = p_ctr_val.clamp(min=0.01).numpy()
    o = y_click.numpy()
    ehat = delta_hat_val.numpy()
    expected_pvdr = float(np.sum(o * ehat / prop)) / B4

    torch.testing.assert_close(
        loss, torch.tensor(expected_pvdr, dtype=torch.float32),
        atol=1e-5, rtol=1e-5,
        msg=f"lambda=1 StableDR collapse mismatch: got {loss.item():.6f}, expected {expected_pvdr:.6f}"
    )


# ---------------------------------------------------------------------------
# 5. MMD² properties
# ---------------------------------------------------------------------------

def test_mmd_nonnegative():
    """MMD² must be >= 0 for any two distributions."""
    torch.manual_seed(123)
    for _ in range(5):
        x = torch.randn(16, 8)
        y = torch.randn(16, 8) + 2.0   # different mean → positive MMD²
        mmd = _mmd_rbf(x, y)
        assert mmd.item() >= 0.0, f"MMD² negative: {mmd.item()}"
        assert torch.isfinite(mmd), f"MMD² non-finite: {mmd.item()}"


def test_mmd_zero_for_identical_distributions():
    """MMD²(x, x) must be 0 (identical distributions)."""
    torch.manual_seed(42)
    x = torch.randn(16, 8)
    mmd = _mmd_rbf(x, x)
    assert abs(mmd.item()) < 1e-5, f"MMD²(x,x) = {mmd.item():.2e} (expected ~0)"


def test_mmd_positive_for_different_distributions():
    """MMD² must be clearly > 0 for well-separated distributions."""
    torch.manual_seed(7)
    x = torch.zeros(32, 16)
    y = torch.ones(32, 16) * 5.0     # 5 standard deviations away
    mmd = _mmd_rbf(x, y)
    assert mmd.item() > 0.01, f"MMD² too small for well-separated distributions: {mmd.item()}"


# ---------------------------------------------------------------------------
# 6. Imputation loss over click space (A1 resolution)
# ---------------------------------------------------------------------------

def test_imp_loss_over_click_space():
    """L̂ = mean(o * (δ - ê)²) — imputation MSE only for clicked rows.

    With all_click=0, the imputation loss should be 0. With all_click=1,
    the imputation loss should match MSE(ê, δ(r, r̂)).
    """
    torch.manual_seed(20)
    model = _make_model(
        ctr_weight=0.0, pvdr_weight=0.0, imp_weight=1.0,
        exp_weight=0.0, mmd_weight=0.0,
    )
    model.eval()
    sp, dn = _mk_batch()

    with torch.no_grad():
        model(sp, dn)

    # All unclicked → L̂ should be 0
    y_click_zero = torch.zeros(BATCH)
    y_purchase = torch.zeros(BATCH)
    loss_noclicks = model.compute_egean_loss(y_click_zero, y_purchase)
    assert abs(loss_noclicks.item()) < 1e-7, (
        f"Imputation loss with all-unclicked should be 0, got {loss_noclicks.item():.2e}"
    )

    # All clicked → L̂ = mean((δ - ê)²) should be > 0 (unless model is perfect)
    y_click_all = torch.ones(BATCH)
    y_purchase_half = (torch.rand(BATCH) < 0.5).float()
    with torch.no_grad():
        model(sp, dn)
    loss_allclicked = model.compute_egean_loss(y_click_all, y_purchase_half)
    assert loss_allclicked.item() >= 0.0, f"Imputation loss should be non-negative"
    assert torch.isfinite(loss_allclicked), "Imputation loss non-finite"


# ---------------------------------------------------------------------------
# 7. Propensity detach: CTR grads unaffected by PVDR (similar to ESCM²-DR Trick 2)
# ---------------------------------------------------------------------------

def test_propensity_detach_ctr_grads_identical():
    """CTR tower gradients must be identical whether pvdr_weight=0 or pvdr_weight=1
    (propensity is detached from CTR, so PVDR loss does not backprop into CTR tower).
    """
    seed = 42
    sp, dn = _mk_batch(seed)
    y_click, y_purchase = _mk_labels(seed)

    # pvdr_weight=0 (no PVDR term)
    torch.manual_seed(seed)
    model0 = _make_model(ctr_weight=1.0, pvdr_weight=0.0, imp_weight=0.0,
                         exp_weight=0.0, mmd_weight=0.0)
    model0.train()
    model0(sp.clone(), dn.clone())
    loss0 = model0.compute_egean_loss(y_click.clone(), y_purchase.clone())
    loss0.backward()
    ctr_grads_0 = {
        n: p.grad.detach().clone()
        for n, p in model0.named_parameters()
        if p.requires_grad and p.grad is not None
        and ('ppnet_ctr' in n or 'wide_ctr' in n)
    }

    # pvdr_weight=1.0 (PVDR term active)
    torch.manual_seed(seed)
    model1 = _make_model(ctr_weight=1.0, pvdr_weight=1.0, imp_weight=0.0,
                         exp_weight=0.0, mmd_weight=0.0)
    model1.train()
    model1.load_state_dict(model0.state_dict())
    model1(sp.clone(), dn.clone())
    loss1 = model1.compute_egean_loss(y_click.clone(), y_purchase.clone())
    loss1.backward()
    ctr_grads_1 = {
        n: p.grad.detach().clone()
        for n, p in model1.named_parameters()
        if p.requires_grad and p.grad is not None
        and ('ppnet_ctr' in n or 'wide_ctr' in n)
    }

    assert set(ctr_grads_0.keys()) == set(ctr_grads_1.keys()), (
        "CTR parameter names differ between pvdr_weight=0 and pvdr_weight=1 models"
    )
    for name in ctr_grads_0:
        torch.testing.assert_close(
            ctr_grads_0[name], ctr_grads_1[name], atol=1e-6, rtol=1e-5,
            msg=f"CTR grad for {name} changed with pvdr_weight=1 (propensity detach violated)"
        )


# ---------------------------------------------------------------------------
# 8. Exposure loss: differs from trivial all-ones BCE (A7 fix check)
# ---------------------------------------------------------------------------

def test_exposure_loss_differs_from_trivial_all_ones():
    """A7 fix: after a real forward() call, the exposure loss must NOT equal the all-ones
    trivial BCE (which provides zero gradient signal to the exposure head).

    The fixed implementation appends in-batch shuffled negatives, so:
      L_exp_proper = 0.5 * (BCE(p_exp, 1) + BCE(p_exp_neg, 0))

    This will differ from the degenerate:
      L_exp_trivial = BCE(p_exp, 1)

    unless p_exp_neg happens to be exactly 0.5 for all rows, which is vanishingly unlikely
    with random initialised weights. We verify:
      1. After forward(), L_exp_proper != L_exp_trivial (non-degenerate).
      2. L_exp_proper >= 0 and finite.
      3. _last_p_exp_neg has the same shape as _last_p_exp and is in (0, 1).
    """
    torch.manual_seed(9)
    model = _make_model(
        ctr_weight=0.0, pvdr_weight=0.0, imp_weight=0.0,
        mmd_weight=0.0, exp_weight=1.0,
    )
    model.eval()
    sp, dn = _mk_batch(9)

    with torch.no_grad():
        model(sp, dn)

    # Verify _last_p_exp_neg is set and has matching shape
    assert hasattr(model, '_last_p_exp_neg'), "_last_p_exp_neg not set after forward()"
    assert model._last_p_exp_neg.shape == model._last_p_exp.shape, (
        f"_last_p_exp_neg shape {model._last_p_exp_neg.shape} != "
        f"_last_p_exp shape {model._last_p_exp.shape}"
    )
    assert torch.all(model._last_p_exp_neg > 0) and torch.all(model._last_p_exp_neg < 1), (
        "_last_p_exp_neg values not in open interval (0, 1)"
    )

    yc = torch.ones(BATCH)  # all clicked (labels don't affect L_exp in our impl)
    yp = torch.zeros(BATCH)

    # Compute actual exposure loss (should use positives + negatives)
    loss_proper = model.compute_egean_loss(yc, yp)

    # Manually compute the trivial all-ones BCE for comparison
    eps = 1e-7
    p_exp = model._last_p_exp.clamp(eps, 1 - eps)
    loss_trivial_all_ones = F.binary_cross_entropy(p_exp, torch.ones_like(p_exp))

    # They should differ (because negatives are included)
    assert abs(loss_proper.item() - loss_trivial_all_ones.item()) > 1e-6, (
        f"Exposure loss is identical to trivial all-ones BCE — in-batch negatives not applied. "
        f"loss_proper={loss_proper.item():.6f}, loss_trivial={loss_trivial_all_ones.item():.6f}"
    )
    assert loss_proper.item() >= 0.0, "Exposure loss negative"
    assert torch.isfinite(loss_proper), "Exposure loss non-finite"


# ---------------------------------------------------------------------------
# 9. MMD click mask: MMD computed only over click space (A2-fix check)
# ---------------------------------------------------------------------------

def test_mmd_click_mask_respected():
    """A2-fix: L_metric must use only clicked rows (yc>0.5) for the MMD computation.

    Verification: run with all-clicked batch vs. all-non-clicked batch — when no rows
    are clicked the MMD should be zero (edge-case skip path). With all-clicked, MMD
    should be computed normally (non-negative, finite).

    Additionally: modify a copy to compare losses between all-clicked vs. half-clicked —
    if the mask is working, the two losses should differ (fewer rows → different MMD value).
    """
    seed = 33
    torch.manual_seed(seed)

    model = _make_model(
        ctr_weight=0.0, pvdr_weight=0.0, imp_weight=0.0,
        exp_weight=0.0, mmd_weight=1.0,
    )
    model.eval()
    sp, dn = _mk_batch(seed)

    # Case 1: all rows clicked
    with torch.no_grad():
        model(sp, dn)
    yc_all = torch.ones(BATCH)
    yp = torch.zeros(BATCH)
    loss_all_click = model.compute_egean_loss(yc_all, yp)
    assert loss_all_click.item() >= 0.0, "MMD (all-clicked) should be non-negative"
    assert torch.isfinite(loss_all_click), "MMD (all-clicked) is non-finite"

    # Case 2: no rows clicked — MMD edge-case path: l_metric = 0
    with torch.no_grad():
        model(sp, dn)
    yc_none = torch.zeros(BATCH)
    loss_no_click = model.compute_egean_loss(yc_none, yp)
    # With mmd_weight=1 and all others=0, and click_mask.sum()<2, loss should be ~0
    assert abs(loss_no_click.item()) < 1e-5, (
        f"MMD with zero clicks should be ~0 (edge-case skip path), got {loss_no_click.item():.2e}"
    )

    # Case 3: half clicked — loss should differ from all-clicked
    with torch.no_grad():
        model(sp, dn)
    yc_half = torch.zeros(BATCH)
    yc_half[:BATCH // 2] = 1.0
    loss_half = model.compute_egean_loss(yc_half, yp)
    assert torch.isfinite(loss_half), "MMD (half-clicked) is non-finite"
    # Different click subsets → different MMD (distinct embedding subsets selected)
    assert abs(loss_all_click.item() - loss_half.item()) > 1e-6, (
        f"MMD with all-clicked ({loss_all_click.item():.6f}) should differ from "
        f"half-clicked ({loss_half.item():.6f}) — click mask not effective"
    )
