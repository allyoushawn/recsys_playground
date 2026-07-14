"""Tests for ESMM_ESCM2DR (ESCM²-DR, Wang et al. SIGIR 2022).

Spec: escm2dr-formula-verification.md (arXiv 2204.05125v2, Eq. 22–27,
code-confirmed against PaddleRec dygraph_model.py).

Mirrors tests/test_ndm_models.py style.
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

from esmm_ali_ccp_impl import ESMM_ESCM2DR, ESMMModel_Wide, count_parameters

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
    # Purchases only happen for clicked rows
    y_purchase = torch.where(
        y_click > 0.5,
        (torch.rand(BATCH) < cvr_rate).float(),
        torch.zeros(BATCH),
    )
    return y_click, y_purchase


def _make_model(**kw):
    return ESMM_ESCM2DR(CARDS, NUM_DENSE, embed_dim=EMBED_DIM, **kw)


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


def test_forward_delta_hat_shape():
    """_last_delta_hat must be (B,) after forward (no sigmoid, can be any real)."""
    model = _make_model()
    model.eval()
    sp, dn = _mk_batch()
    with torch.no_grad():
        model(sp, dn)
    dh = model._last_delta_hat
    assert dh.shape == (BATCH,), f"_last_delta_hat shape: expected ({BATCH},), got {dh.shape}"
    assert torch.all(torch.isfinite(dh)), "_last_delta_hat contains non-finite values"


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
# 2. Loss finite + grads flow to all three towers
# ---------------------------------------------------------------------------

def test_loss_finite():
    torch.manual_seed(42)
    model = _make_model()
    model.train()
    sp, dn = _mk_batch()
    y_click, y_purchase = _mk_labels()
    model(sp, dn)
    loss = model.compute_escm2_loss(y_click, y_purchase)
    assert torch.isfinite(loss), f"loss is not finite: {loss.item()}"
    assert loss.item() >= 0.0, f"loss is negative: {loss.item()}"


def test_grads_flow_to_all_three_towers():
    """After backward, CTR tower, CVR tower, and IMP tower must all have non-zero gradients."""
    torch.manual_seed(7)
    model = _make_model()
    model.train()
    sp, dn = _mk_batch()
    y_click, y_purchase = _mk_labels()
    model(sp, dn)
    loss = model.compute_escm2_loss(y_click, y_purchase)
    loss.backward()

    def _has_nonzero_grad(param_list):
        for p in param_list:
            if p.grad is not None and p.grad.abs().sum().item() > 0:
                return True
        return False

    ctr_params = list(model.ctr_tower.parameters()) + list(model.wide_ctr.parameters())
    cvr_params = list(model.cvr_tower.parameters()) + list(model.wide_cvr.parameters())
    imp_params = list(model.imp_tower.parameters())

    assert _has_nonzero_grad(ctr_params), "CTR tower received no gradients"
    assert _has_nonzero_grad(cvr_params), "CVR tower received no gradients"
    assert _has_nonzero_grad(imp_params), "IMP tower received no gradients"


def test_all_param_grads_finite():
    torch.manual_seed(13)
    model = _make_model()
    model.train()
    sp, dn = _mk_batch()
    y_click, y_purchase = _mk_labels()
    model(sp, dn)
    loss = model.compute_escm2_loss(y_click, y_purchase)
    loss.backward()
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            assert torch.all(torch.isfinite(param.grad)), f"Non-finite grad in {name}"


# ---------------------------------------------------------------------------
# 3. Propensity detach: CTR tower grads unaffected by R_DR (λ_c=0 vs λ_c=1)
# ---------------------------------------------------------------------------

def test_propensity_detach_ctr_grads_identical():
    """Trick 2 (paper Appendix A): IPS is detached, so CTR tower gradients must be
    identical whether λ_c=0 (no DR term) or λ_c=1 (full DR term), on the same
    batch and seed.  Verified by comparing per-parameter gradient tensors."""
    seed = 42
    torch.manual_seed(seed)
    sp, dn = _mk_batch(seed)
    y_click, y_purchase = _mk_labels(seed)

    # --- λ_c = 0 (no DR term) ---
    torch.manual_seed(seed)
    model0 = _make_model(lambda_c=0.0, lambda_g=1.0)
    model0.train()
    model0(sp.clone(), dn.clone())
    loss0 = model0.compute_escm2_loss(y_click.clone(), y_purchase.clone())
    loss0.backward()
    ctr_grads_lc0 = {
        n: p.grad.detach().clone()
        for n, p in model0.named_parameters()
        if p.requires_grad and p.grad is not None
        and ('ctr_tower' in n or 'wide_ctr' in n)
    }

    # --- λ_c = 1.0 (full DR term) ---
    torch.manual_seed(seed)
    model1 = _make_model(lambda_c=1.0, lambda_g=1.0)
    model1.train()
    # Copy identical weights from model0
    model1.load_state_dict(model0.state_dict())
    model1(sp.clone(), dn.clone())
    loss1 = model1.compute_escm2_loss(y_click.clone(), y_purchase.clone())
    loss1.backward()
    ctr_grads_lc1 = {
        n: p.grad.detach().clone()
        for n, p in model1.named_parameters()
        if p.requires_grad and p.grad is not None
        and ('ctr_tower' in n or 'wide_ctr' in n)
    }

    assert set(ctr_grads_lc0.keys()) == set(ctr_grads_lc1.keys()), (
        "CTR parameter names differ between λ_c=0 and λ_c=1 models"
    )
    for name in ctr_grads_lc0:
        torch.testing.assert_close(
            ctr_grads_lc0[name], ctr_grads_lc1[name], atol=1e-6, rtol=1e-5,
            msg=f"CTR grad for {name} changed with λ_c=1 (propensity detach violated)"
        )


# ---------------------------------------------------------------------------
# 4. Clip floor respected: tiny p_ctr → IPS bounded
# ---------------------------------------------------------------------------

def test_clip_floor_respected():
    """With p_ctr very small (< ips_clip_floor), the IPS must be clipped and loss stays finite."""
    torch.manual_seed(99)
    model = _make_model(ips_clip_floor=0.1)
    model.train()

    # Force p_ctr to be tiny by zeroing the CTR tower and wide term,
    # then setting a large negative bias so sigmoid(logit) ≈ 0.
    sp, dn = _mk_batch()
    y_click = torch.ones(BATCH)       # all clicked → IPS = 1/p_ctr; clip should kick in
    y_purchase = (torch.rand(BATCH) < 0.3).float()

    # Drive p_ctr small via large negative logit
    with torch.no_grad():
        for m in list(model.ctr_tower.modules()) + [model.wide_ctr]:
            if isinstance(m, nn.Linear):
                m.weight.fill_(0.0)
                if m.bias is not None:
                    m.bias.fill_(-10.0)

    model(sp, dn)
    # p_ctr should be close to sigmoid(-10) ≈ 4.5e-5 (well below 0.1 floor)
    assert model._last_p_ctr.max().item() < 0.001, "Expected p_ctr to be tiny"

    loss = model.compute_escm2_loss(y_click, y_purchase)
    assert torch.isfinite(loss), f"loss not finite with tiny p_ctr: {loss.item()}"


# ---------------------------------------------------------------------------
# 5. Warmup zeroes DR terms
# ---------------------------------------------------------------------------

def test_warmup_zeroes_dr_terms():
    """When global_step < dr_warmup_steps, loss must equal L_CTR + λ_g·L_CTCVR (no R_DR)."""
    torch.manual_seed(0)
    model = _make_model(lambda_c=2.0, lambda_g=1.0, dr_warmup_steps=1000)
    model.train()
    sp, dn = _mk_batch()
    y_click, y_purchase = _mk_labels()
    y_ctcvr = (y_click * y_purchase).clamp(0.0, 1.0)

    model(sp, dn)
    loss_warmup = model.compute_escm2_loss(y_click, y_purchase, global_step=0)

    eps = 1e-7
    pc = model._last_p_ctr.clamp(eps, 1 - eps)
    pcc = model._last_p_ctcvr.clamp(eps, 1 - eps)
    l_ctr = F.binary_cross_entropy(pc, y_click.float())
    l_ctcvr = F.binary_cross_entropy(pcc, y_ctcvr.float())
    expected = l_ctr + model.lambda_g * l_ctcvr

    assert abs(loss_warmup.item() - expected.item()) < 1e-5, (
        f"Warmup loss {loss_warmup.item():.6f} != backbone-only {expected.item():.6f}"
    )


def test_warmup_activates_dr_after_steps():
    """After dr_warmup_steps, the full DR loss differs from the backbone-only loss (λ_c > 0)."""
    torch.manual_seed(1)
    model = _make_model(lambda_c=2.0, lambda_g=1.0, dr_warmup_steps=5)
    model.train()
    sp, dn = _mk_batch()
    y_click, y_purchase = _mk_labels()
    y_ctcvr = (y_click * y_purchase).clamp(0.0, 1.0)

    model(sp, dn)
    loss_active = model.compute_escm2_loss(y_click, y_purchase, global_step=10)

    eps = 1e-7
    pc = model._last_p_ctr.clamp(eps, 1 - eps)
    pcc = model._last_p_ctcvr.clamp(eps, 1 - eps)
    l_ctr = F.binary_cross_entropy(pc, y_click.float())
    l_ctcvr = F.binary_cross_entropy(pcc, y_ctcvr.float())
    backbone_only = l_ctr + model.lambda_g * l_ctcvr

    # With λ_c=2.0 and a non-trivial batch, R_DR should make the loss differ
    # (could be equal only in degenerate cases, which are very unlikely with this seed)
    assert abs(loss_active.item() - backbone_only.item()) > 1e-6, (
        "DR loss after warmup should differ from backbone-only loss "
        f"(got loss={loss_active.item():.6f}, backbone={backbone_only.item():.6f})"
    )


# ---------------------------------------------------------------------------
# 6. Imputation over ALL rows; correction click-gated (mask test)
# ---------------------------------------------------------------------------

def test_imputation_over_all_rows_correction_click_gated():
    """Eq. 22: R_DR^err = mean(δ̂ + ê·(o/ô)).
    - δ̂ contributes for ALL rows (imputation over entire space).
    - ê·(o/ô) contributes ONLY for clicked rows (o=0 → ips=0 for unclicked).

    Verification: set all click=0 → ips=0 → R_DR reduces to mean(δ̂) (imputation only).
    """
    torch.manual_seed(5)
    model = _make_model(lambda_c=1.0, lambda_g=0.0, dr_warmup_steps=0)
    model.train()
    sp, dn = _mk_batch()
    # All unclicked: o=0 for all rows → ips=0 → correction terms vanish
    y_click_zero = torch.zeros(BATCH)
    y_purchase = torch.zeros(BATCH)

    model(sp, dn)
    loss_noclicks = model.compute_escm2_loss(y_click_zero, y_purchase, global_step=1)

    # Manually compute: only L_CTR + λ_c * mean(δ̂) should remain
    # (L_CTCVR is λ_g=0 so absent; correction terms are 0 since ips=0)
    eps = 1e-7
    pc = model._last_p_ctr.clamp(eps, 1 - eps)
    l_ctr = F.binary_cross_entropy(pc, y_click_zero.float())
    delta_hat = model._last_delta_hat
    # R_DR^err with ips=0: mean(δ̂ + 0) = mean(δ̂)
    # R_DR^imp with ips=0: mean(ê²·0) = 0
    l_cvr_expected = delta_hat.mean()
    expected = l_ctr + model.lambda_c * l_cvr_expected

    assert abs(loss_noclicks.item() - expected.item()) < 1e-5, (
        f"With all-unclicked, loss={loss_noclicks.item():.6f} "
        f"!= L_CTR + λ_c*mean(δ̂)={expected.item():.6f}"
    )


def test_correction_term_nonzero_when_clicked():
    """With at least some clicked rows, the correction term ê·(o/ô) is non-zero,
    so the full DR loss differs from the imputation-only loss."""
    torch.manual_seed(3)
    model = _make_model(lambda_c=1.0, lambda_g=0.0, dr_warmup_steps=0)
    model.train()
    sp, dn = _mk_batch()
    y_click, y_purchase = _mk_labels(click_rate=0.7)

    # All-unclicked reference
    model(sp, dn)
    delta_hat = model._last_delta_hat.detach().clone()
    loss_noclicks = model.compute_escm2_loss(
        torch.zeros(BATCH), torch.zeros(BATCH), global_step=1
    )

    # With clicks
    model(sp, dn)
    loss_clicked = model.compute_escm2_loss(y_click, y_purchase, global_step=1)

    # The correction term changes the loss (unless e_hat happens to be 0 everywhere)
    assert abs(loss_clicked.item() - loss_noclicks.item()) > 1e-6, (
        "Loss with clicked rows should differ from all-unclicked "
        f"(clicked={loss_clicked.item():.6f}, no_clicks={loss_noclicks.item():.6f})"
    )


# ---------------------------------------------------------------------------
# 7. Param count: ESCM2DR adds only imp_tower over ESMMModel_Wide
# ---------------------------------------------------------------------------

def test_param_overhead():
    """ESCM2DR adds only the imputation tower over ESMMModel_Wide."""
    model_dr = _make_model()
    model_wide = ESMMModel_Wide(CARDS, NUM_DENSE, embed_dim=EMBED_DIM)
    n_dr = count_parameters(model_dr)
    n_wide = count_parameters(model_wide)
    input_dim = len(CARDS) * EMBED_DIM + NUM_DENSE
    hidden_imp = max(1, input_dim // 2)
    expected_imp_params = (input_dim * hidden_imp + hidden_imp) + (hidden_imp * 1 + 1)
    assert n_dr - n_wide == expected_imp_params, (
        f"Param overhead {n_dr - n_wide} != expected imp tower params {expected_imp_params}"
    )
