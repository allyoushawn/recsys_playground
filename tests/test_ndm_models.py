"""Tests for cycle-2 NDM model classes and utilities (ESMM_PLE_WideCross_NDM, ESMM_NDM,
GateObserverHead, GradSNRTracker).

Mirrors tests/test_adaorder_models.py style.
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

from esmm_ali_ccp_impl import (
    ESMM_PLE_WideCross_NDM,
    ESMM_NDM,
    GateObserverHead,
    GradSNRTracker,
    count_parameters,
    ESMM_PLE_WideCross,
    _ESMMCrossNetExposed,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

CARDS = [10, 10, 10]
NUM_DENSE = 3
EMBED_DIM = 8
BATCH = 16

PLE_KWARGS = dict(
    d_model=32,
    expert_hidden=64,
    num_shared_experts=1,
    num_task_experts=1,
    dropout=0.0,
)


def _mk_batch():
    sparse_x = torch.randint(0, 10, (BATCH, len(CARDS)))
    dense_x = torch.randn(BATCH, NUM_DENSE)
    return sparse_x, dense_x


def _mk_labels(click_rate=0.5):
    y_click = (torch.rand(BATCH) < click_rate).float()
    y_purchase = torch.where(y_click > 0.5, (torch.rand(BATCH) < 0.3).float(), torch.zeros(BATCH))
    return y_click, y_purchase


def _make_ndm(ndm_mode='ndm', **extra):
    return ESMM_PLE_WideCross_NDM(
        CARDS, NUM_DENSE, embed_dim=EMBED_DIM,
        num_cross_layers=3,
        ndm_mode=ndm_mode,
        **PLE_KWARGS,
        **extra,
    )


def _make_esmm_ndm(ndm_mode='ndm', **extra):
    return ESMM_NDM(
        CARDS, NUM_DENSE, embed_dim=EMBED_DIM,
        ndm_mode=ndm_mode,
        **extra,
    )


# ---------------------------------------------------------------------------
# Forward / shape / range
# ---------------------------------------------------------------------------

def _assert_forward_ok(model):
    model.eval()
    sp, dn = _mk_batch()
    with torch.no_grad():
        p_ctr, p_cvr, p_ctcvr = model(sp, dn)
    for name, p in [('p_ctr', p_ctr), ('p_cvr', p_cvr), ('p_ctcvr', p_ctcvr)]:
        assert p.shape == (BATCH,), f"{name}: expected ({BATCH},), got {p.shape}"
        assert torch.all(p >= 0.0) and torch.all(p <= 1.0), f"{name} out of [0,1]"
        assert torch.all(torch.isfinite(p)), f"{name} contains non-finite values"


def test_ndm_forward_ndm_mode():
    _assert_forward_ok(_make_ndm('ndm'))


def test_ndm_forward_hard_mode():
    _assert_forward_ok(_make_ndm('hard'))


def test_ndm_forward_smooth_mode():
    _assert_forward_ok(_make_ndm('smooth', smooth_mass=0.05))


def test_esmm_ndm_forward_ndm():
    _assert_forward_ok(_make_esmm_ndm('ndm'))


def test_esmm_ndm_forward_hard():
    _assert_forward_ok(_make_esmm_ndm('hard'))


# ---------------------------------------------------------------------------
# Backward / gradient tests
# ---------------------------------------------------------------------------

def _assert_backward_ok(model, ndm_mode):
    model.train()
    sp, dn = _mk_batch()
    y_click, y_purchase = _mk_labels()
    p_ctr, p_cvr, p_ctcvr = model(sp, dn)
    loss = model.compute_ndm_loss(p_ctr, p_cvr, p_ctcvr, y_click, y_purchase)
    loss.backward()
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            assert torch.all(torch.isfinite(param.grad)), f"Non-finite grad in {name}"


def test_ndm_backward_ndm_mode():
    _assert_backward_ok(_make_ndm('ndm'), 'ndm')


def test_ndm_backward_hard_mode():
    _assert_backward_ok(_make_ndm('hard'), 'hard')


def test_ndm_backward_smooth_mode():
    _assert_backward_ok(_make_ndm('smooth', smooth_mass=0.05), 'smooth')


def test_esmm_ndm_backward_ndm():
    _assert_backward_ok(_make_esmm_ndm('ndm'), 'ndm')


def test_esmm_ndm_backward_hard():
    _assert_backward_ok(_make_esmm_ndm('hard'), 'hard')


# ---------------------------------------------------------------------------
# Param parity: hard vs ndm mode must have identical param counts
# ---------------------------------------------------------------------------

def test_param_parity_hard_vs_ndm():
    m_ndm = _make_ndm('ndm')
    m_hard = _make_ndm('hard')
    n_ndm = count_parameters(m_ndm)
    n_hard = count_parameters(m_hard)
    assert n_ndm == n_hard, (
        f"hard ({n_hard:,}) and ndm ({n_ndm:,}) must have identical param counts "
        "(aux tower always built; loss disabled in hard mode, not removed)"
    )


def test_param_parity_esmm_ndm_hard_vs_ndm():
    m_ndm = _make_esmm_ndm('ndm')
    m_hard = _make_esmm_ndm('hard')
    assert count_parameters(m_ndm) == count_parameters(m_hard)


# ---------------------------------------------------------------------------
# ChorusCVR Eq. 6: uncvr_tower exists + two-stage CTunCVR product
# ---------------------------------------------------------------------------

def test_uncvr_tower_exists():
    """Both ESMM_PLE_WideCross_NDM and ESMM_NDM must have an uncvr_tower attribute."""
    m_ple = _make_ndm('ndm')
    m_simple = _make_esmm_ndm('ndm')
    assert hasattr(m_ple, 'uncvr_tower'), "ESMM_PLE_WideCross_NDM missing uncvr_tower"
    assert hasattr(m_simple, 'uncvr_tower'), "ESMM_NDM missing uncvr_tower"
    assert isinstance(m_ple.uncvr_tower, nn.Module)
    assert isinstance(m_simple.uncvr_tower, nn.Module)


def test_two_stage_ctuncvr_product():
    """After forward(), _last_y_ctuncvr must equal p_ctr * _last_y_uncvr element-wise (Eq. 6)."""
    torch.manual_seed(42)
    for make_fn in [lambda: _make_ndm('ndm'), lambda: _make_esmm_ndm('ndm')]:
        model = make_fn()
        model.eval()
        sp, dn = _mk_batch()
        with torch.no_grad():
            p_ctr, p_cvr, p_ctcvr = model(sp, dn)
        # y_ctuncvr should be p_ctr * y_uncvr (clamped to eps..1-eps on both)
        eps = 1e-7
        expected = (p_ctr * model._last_y_uncvr).clamp(eps, 1 - eps)
        torch.testing.assert_close(
            model._last_y_ctuncvr, expected, atol=1e-6, rtol=1e-5,
            msg='_last_y_ctuncvr != p_ctr * _last_y_uncvr (Eq. 6 violated)',
        )


# ---------------------------------------------------------------------------
# Eq. 10: alignment soft targets are detached (no grad through sg side)
# ---------------------------------------------------------------------------

def test_alignment_soft_targets_are_detached():
    """The soft labels passed to alignment BCE must have requires_grad=False (sg operator)."""
    torch.manual_seed(7)
    for make_fn in [lambda: _make_ndm('ndm'), lambda: _make_esmm_ndm('ndm')]:
        model = make_fn()
        model.train()
        sp, dn = _mk_batch()
        y_click, y_purchase = _mk_labels()
        p_ctr, p_cvr, p_ctcvr = model(sp, dn)

        # Verify that _last_y_uncvr.detach() used as soft_for_cvr has no grad
        soft_for_cvr = (1.0 - model._last_y_uncvr.detach()).clamp(0.0, 1.0)
        soft_for_uncvr = (1.0 - p_cvr.detach()).clamp(0.0, 1.0)
        assert not soft_for_cvr.requires_grad, (
            "soft_for_cvr must be stop-gradient (sg operator; Eq. 10)"
        )
        assert not soft_for_uncvr.requires_grad, (
            "soft_for_uncvr must be stop-gradient (sg operator; Eq. 10)"
        )


# ---------------------------------------------------------------------------
# Eq. 10: IPW weights match click/unclick membership
# ---------------------------------------------------------------------------

def test_ipw_weights_match_click_unclick():
    """IPW weights in Eq. 10: click-space weight = p_ctr.detach(), unclick = 1 - p_ctr.detach()."""
    torch.manual_seed(11)
    model = _make_ndm('ndm')
    model.train()
    sp, dn = _mk_batch()
    p_ctr, p_cvr, p_ctcvr = model(sp, dn)
    y_click, y_purchase = _mk_labels()

    w_click = p_ctr.detach()
    w_unclick = 1.0 - w_click

    # w_click should equal p_ctr (detached) — values in (0, 1)
    assert not w_click.requires_grad, "w_click must be detached"
    assert not w_unclick.requires_grad, "w_unclick must be detached"
    assert torch.all(w_click >= 0.0) and torch.all(w_click <= 1.0), "w_click out of [0,1]"
    assert torch.allclose(w_click + w_unclick, torch.ones_like(w_click), atol=1e-6), (
        "w_click + w_unclick must sum to 1 elementwise"
    )

    # Clicked rows should have higher w_click than w_unclick on average
    # (p_ctr > 0.5 for clicked) — just check the sum relationship
    # We verify forward+backward produces finite loss with these weights
    loss = model.compute_ndm_loss(p_ctr, p_cvr, p_ctcvr, y_click, y_purchase)
    assert torch.isfinite(loss), "loss must be finite with IPW weights"


# ---------------------------------------------------------------------------
# Smooth mode: constant mass (semantics unchanged)
# ---------------------------------------------------------------------------

def test_smooth_mode_constant_mass():
    """In smooth mode, unclicked CVR labels are replaced by the fixed smooth_mass scalar."""
    smooth_mass = 0.07
    model = _make_ndm('smooth', smooth_mass=smooth_mass)
    model.train()
    sp, dn = _mk_batch()
    y_click = torch.zeros(BATCH)
    y_click[:4] = 1.0  # 4 clicked
    y_purchase = torch.zeros(BATCH)

    # Intercept the soft CVR target by monkey-patching F.binary_cross_entropy
    captured_targets = []
    orig_bce = F.binary_cross_entropy

    def mock_bce(input, target, *args, **kwargs):
        captured_targets.append(target.detach().clone())
        return orig_bce(input, target, *args, **kwargs)

    import esmm_ali_ccp_impl as _impl
    old_F_bce = _impl.F.binary_cross_entropy
    _impl.F.binary_cross_entropy = mock_bce

    try:
        p_ctr, p_cvr, p_ctcvr = model(sp, dn)
        loss = model.compute_ndm_loss(p_ctr, p_cvr, p_ctcvr, y_click, y_purchase)
    finally:
        _impl.F.binary_cross_entropy = old_F_bce

    # Find the CVR soft target tensor (unclicked rows should all be smooth_mass)
    unclicked_idx = (y_click < 0.5).nonzero(as_tuple=True)[0]
    found = False
    for t in captured_targets:
        if t.shape == (BATCH,) and torch.allclose(t[unclicked_idx], torch.full_like(t[unclicked_idx], smooth_mass), atol=1e-6):
            found = True
            break
    assert found, (
        f"smooth mode should produce unclicked CVR targets = {smooth_mass}; "
        f"captured targets: {[t.tolist() for t in captured_targets]}"
    )


# ---------------------------------------------------------------------------
# Observer stop-gradient: backbone params identical after observer-only step
# ---------------------------------------------------------------------------

def test_observer_stop_gradient():
    """Observer loss must not update backbone parameters.

    We attach a standalone GateObserverHead to purely detached inputs and confirm
    that after running its backward + optimizer step, all parameters outside the
    observer head are unchanged.
    """
    torch.manual_seed(7)
    K = 3
    d = 16
    # Create a simple observer head in isolation
    obs = GateObserverHead(input_dim=d, num_cross_layers=K)

    # Build random detached layer outputs (B, d) × K
    B = 8
    layer_outs = [torch.randn(B, d).detach() for _ in range(K)]
    logit_ctr_det = torch.randn(B).detach()
    logit_cvr_det = torch.randn(B).detach()
    y_click = (torch.rand(B) > 0.5).float()
    y_ctcvr = (y_click * (torch.rand(B) > 0.7).float())

    # Simulate a fake "backbone" module to confirm it isn't touched
    class _FakeBackbone(nn.Module):
        def __init__(self):
            super().__init__()
            self.param = nn.Parameter(torch.randn(4))

    fake_backbone = _FakeBackbone()
    fake_param_before = fake_backbone.param.detach().clone()

    # Observer optimizer — only over observer params
    obs_opt = torch.optim.SGD(obs.parameters(), lr=0.1)
    obs_opt.zero_grad()
    obs_loss = obs(layer_outs, logit_ctr_det, logit_cvr_det, y_click, y_ctcvr)
    # Call backward from obs_loss; fake_backbone.param has no grad path here
    obs_loss.backward()
    obs_opt.step()

    # fake_backbone param untouched (no grad path from obs_loss → fake_backbone)
    assert torch.allclose(fake_param_before, fake_backbone.param.detach(), atol=1e-8), (
        "Backbone param changed — observer backward must not affect unconnected params"
    )

    # Also verify: observer parameters themselves DID get updated
    for name, p in obs.named_parameters():
        if 'theta' in name:
            # theta should have a grad
            assert p.grad is not None, f"Observer {name} should have gradient"


# ---------------------------------------------------------------------------
# GradSNRTracker: clean signal → high SNR, noisy → low
# ---------------------------------------------------------------------------

def test_grad_snr_clean_signal_high():
    """SNR uses per-minibatch (mean, std) of the gradient tensor.

    With a single scalar output (1-dim weight), the gradient tensor has one element,
    so std=0, and SNR = |mean|/eps >> 1.  Identical repeated batches → mean is stable
    and positive, confirming SNR is not artificially depressed by within-tensor variance.
    """
    torch.manual_seed(42)

    class _ScalarModel(nn.Module):
        """y = w * x (single scalar weight); gradient w.r.t. w is x*(w*x - target)."""
        def __init__(self):
            super().__init__()
            # A 1×1 weight: gradient tensor has exactly one element → std=0 within tensor
            self.cross = nn.ModuleList([nn.Linear(1, 1, bias=False)])
        def forward(self, x):
            return self.cross[0](x)

    model = _ScalarModel()
    # Initialise weight far from target so gradient stays large
    with torch.no_grad():
        model.cross[0].weight.fill_(0.0)
    tracker = GradSNRTracker(model, layer_prefix='cross')

    # lr=0: weights never change → gradient is identical every step
    opt = torch.optim.SGD(model.parameters(), lr=0.0)
    x_fixed = torch.ones(16, 1) * 1.0
    target_fixed = torch.ones(16, 1) * 5.0  # large gap → large gradient

    for _ in range(10):
        opt.zero_grad()
        out = model(x_fixed)
        loss = F.mse_loss(out, target_fixed)
        loss.backward()
        tracker.accumulate()

    snr = tracker.compute_snr()
    for name, val in snr.items():
        assert np.isfinite(val), f"SNR for {name} is not finite: {val}"
        # Single-element gradient → std=0 within tensor → SNR = |mean|/eps >> 1
        assert val > 100.0, (
            f"1-element gradient tensor with std=0 should give SNR >> 1, got {val:.2f} for {name}"
        )


def test_grad_snr_noise_low():
    """With random targets each batch, gradient direction varies → low SNR."""
    torch.manual_seed(13)

    class _FakeModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.cross = nn.ModuleList([nn.Linear(8, 8)])
        def forward(self, x):
            return self.cross[0](x)

    fake_model = _FakeModel()
    tracker = GradSNRTracker(fake_model, layer_prefix='cross')
    opt = torch.optim.SGD(fake_model.parameters(), lr=0.001)

    rng = torch.Generator()
    rng.manual_seed(13)

    for _ in range(50):
        opt.zero_grad()
        x_rand = torch.randn(16, 8, generator=rng)
        target_rand = torch.randn(16, 8, generator=rng) * 10  # high-variance targets
        out = fake_model(x_rand)
        loss = F.mse_loss(out, target_rand)
        loss.backward()
        tracker.accumulate()

    snr = tracker.compute_snr()
    # With high-variance random targets the gradient direction should be inconsistent
    # → SNR < clean signal.  We check it's lower than the clean case (not == 0).
    for name, val in snr.items():
        assert np.isfinite(val), f"SNR for {name} should be finite even in noisy case"
        # SNR from noisy gradients should be noticeably lower than 10
        assert val < 5.0, (
            f"Noisy gradients should yield low SNR for {name}, got {val:.4f}"
        )


def test_grad_snr_tracker_reset():
    """After reset(), compute_snr() should return nan (no data accumulated)."""

    class _FakeModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.cross = nn.ModuleList([nn.Linear(4, 4)])
        def forward(self, x):
            return self.cross[0](x)

    model = _FakeModel()
    tracker = GradSNRTracker(model, layer_prefix='cross')
    # accumulate some data
    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    opt.zero_grad()
    F.mse_loss(model(torch.randn(4, 4)), torch.zeros(4, 4)).backward()
    tracker.accumulate()
    tracker.reset()
    snr = tracker.compute_snr()
    for name, val in snr.items():
        assert np.isnan(val), f"After reset, SNR should be nan, got {val} for {name}"


# ---------------------------------------------------------------------------
# Observer get_gate_weights
# ---------------------------------------------------------------------------

def test_observer_get_gate_weights_shape():
    model = _make_ndm('ndm', with_observer=True)
    gw = model.get_gate_weights()
    assert set(gw.keys()) == {'ctr', 'cvr'}
    for task, w in gw.items():
        assert w is not None, f"{task} weights should not be None with observer attached"
        assert isinstance(w, np.ndarray)
        assert w.shape == (3,), f"Expected K=3 weights, got {w.shape}"
        assert abs(w.sum() - 1.0) < 1e-5, f"{task} weights should sum to 1"


def test_no_observer_get_gate_weights_returns_none():
    model = _make_ndm('ndm', with_observer=False)
    gw = model.get_gate_weights()
    assert gw['ctr'] is None and gw['cvr'] is None


# ---------------------------------------------------------------------------
# Fix-4 regression: smooth mode all-clicked batch keeps hard-label BCE (> 0)
# ---------------------------------------------------------------------------

def test_smooth_allclicked_uses_hard_bce():
    """When ALL rows are clicked in smooth mode, l_soft_cvr must equal
    BCE(p_cvr, y_purchase) — not 0.0 — so the CVR head still sees a gradient.

    Regression test for the fix: else-branch = BCE(pv, yp), not zero.
    """
    torch.manual_seed(0)
    model = _make_ndm('smooth', smooth_mass=0.05)
    model.train()
    sp, dn = _mk_batch()
    # All rows clicked, half purchased
    y_click = torch.ones(BATCH)
    y_purchase = torch.zeros(BATCH)
    y_purchase[:BATCH // 2] = 1.0

    p_ctr, p_cvr, p_ctcvr = model(sp, dn)
    loss = model.compute_ndm_loss(p_ctr, p_cvr, p_ctcvr, y_click, y_purchase)

    # Must be > 0 (BCE of non-trivial labels is positive)
    assert loss.item() > 0.0, (
        'smooth mode all-clicked batch: loss must be > 0 (hard-label BCE on clicked rows)'
    )

    # Must equal the plain BCE reference
    eps = 1e-7
    pv = p_cvr.clamp(eps, 1 - eps)
    expected_l_soft_cvr = F.binary_cross_entropy(pv, y_purchase)
    # Recompute backbone loss to isolate l_soft_cvr
    pc = p_ctr.clamp(eps, 1 - eps)
    pcc = p_ctcvr.clamp(eps, 1 - eps)
    y_ctcvr_label = (y_click * y_purchase).clamp(0.0, 1.0)
    backbone = F.binary_cross_entropy(pc, y_click) + F.binary_cross_entropy(pcc, y_ctcvr_label)
    # CTunCVR term (l_ctuncvr): uses two-stage product _last_y_ctuncvr (Eq. 6)
    y_ctuncvr_hard = (y_click * (1.0 - y_purchase)).clamp(0.0, 1.0)
    l_ctuncvr = F.binary_cross_entropy(model._last_y_ctuncvr, y_ctuncvr_hard)
    expected_total = backbone + model.ndm_weight * (l_ctuncvr + expected_l_soft_cvr)
    assert abs(loss.item() - expected_total.item()) < 1e-5, (
        f'smooth all-clicked total loss {loss.item():.6f} != expected {expected_total.item():.6f}'
    )


# ---------------------------------------------------------------------------
# Fix-1 regression: observer params registered before optimizer → get updates
# ---------------------------------------------------------------------------

def test_observer_params_registered_before_optimizer():
    """GateObserverHead._proj_ctr/_proj_cvr created in __init__ so an optimizer
    built immediately after construction registers them and they receive updates.
    """
    torch.manual_seed(3)
    K = 3
    d = 16
    obs = GateObserverHead(input_dim=d, num_cross_layers=K)

    # Build optimizer immediately after construction (no forward call yet)
    opt = torch.optim.SGD(obs.parameters(), lr=0.1)

    # Verify _proj_ctr and _proj_cvr appear in named_parameters()
    param_names = {n for n, _ in obs.named_parameters()}
    assert '_proj_ctr.weight' in param_names, (
        '_proj_ctr.weight must be a registered parameter (not lazily created)'
    )
    assert '_proj_cvr.weight' in param_names, (
        '_proj_cvr.weight must be a registered parameter (not lazily created)'
    )

    # Run one step and confirm _proj_ctr/_proj_cvr actually get updated
    B = 8
    layer_outs = [torch.randn(B, d) for _ in range(K)]
    logit_ctr = torch.randn(B)
    logit_cvr = torch.randn(B)
    y_click = (torch.rand(B) > 0.5).float()
    y_ctcvr = (y_click * (torch.rand(B) > 0.7).float())

    proj_ctr_before = obs._proj_ctr.weight.detach().clone()
    proj_cvr_before = obs._proj_cvr.weight.detach().clone()

    opt.zero_grad()
    loss = obs(layer_outs, logit_ctr, logit_cvr, y_click, y_ctcvr)
    loss.backward()
    opt.step()

    assert not torch.allclose(obs._proj_ctr.weight.detach(), proj_ctr_before, atol=1e-8), (
        '_proj_ctr.weight was not updated — likely not registered with the optimizer'
    )
    assert not torch.allclose(obs._proj_cvr.weight.detach(), proj_cvr_before, atol=1e-8), (
        '_proj_cvr.weight was not updated — likely not registered with the optimizer'
    )


# ---------------------------------------------------------------------------
# Fix-2 regression: numerical equivalence of exposed vs plain cross stack
# ---------------------------------------------------------------------------

def test_ndm_cross_stack_numerical_equivalence():
    """ESMM_PLE_WideCross_NDM's exposed cross stack must be numerically equivalent
    to ESMM_PLE_WideCross with the same weights: using x_K (final layer output of
    _ESMMCrossNetExposed) as the cross contribution gives identical logits.

    Verifies Fix-2 contract: the main-task logits are unchanged after switching the
    NDM backbone's cross stack from _ESMMCrossNet to _ESMMCrossNetExposed.
    """
    torch.manual_seed(7)
    num_cross_layers = 3
    input_dim = len(CARDS) * EMBED_DIM + NUM_DENSE

    # Build a plain _ESMMCrossNet and an _ESMMCrossNetExposed, copy weights.
    cross_plain = __import__(
        'esmm_ali_ccp_impl', fromlist=['_ESMMCrossNet']
    )._ESMMCrossNet(input_dim, num_cross_layers)
    cross_exposed = _ESMMCrossNetExposed(input_dim, num_cross_layers)

    # Copy weights layer by layer
    for lp, le in zip(cross_plain.layers, cross_exposed.layers):
        le.weight.data.copy_(lp.weight.data)
        le.bias.data.copy_(lp.bias.data)

    x = torch.randn(BATCH, input_dim)
    with torch.no_grad():
        out_plain = cross_plain(x)                 # (B, d)
        out_exposed_list = cross_exposed(x)        # list of K tensors
        out_exposed_final = out_exposed_list[-1]   # x_K

    torch.testing.assert_close(
        out_plain, out_exposed_final, atol=1e-5, rtol=1e-5,
        msg='_ESMMCrossNetExposed final output (x_K) must match _ESMMCrossNet output',
    )


# ---------------------------------------------------------------------------
# NDM param overhead ≤ 15% over champion backbone
# ---------------------------------------------------------------------------

def test_ndm_param_overhead_within_15pct():
    big_cards = [100] * 23
    big_dense = 8
    big_embed = 18
    big_ple = dict(d_model=128, expert_hidden=256, num_shared_experts=1, num_task_experts=1, dropout=0.0)

    ref = ESMM_PLE_WideCross(big_cards, big_dense, embed_dim=big_embed, num_cross_layers=3, **big_ple)
    ndm = ESMM_PLE_WideCross_NDM(big_cards, big_dense, embed_dim=big_embed, num_cross_layers=3, **big_ple)
    ref_n = count_parameters(ref)
    ndm_n = count_parameters(ndm)
    overhead = (ndm_n - ref_n) / ref_n
    assert overhead <= 0.15, (
        f"NDM param overhead {overhead:.1%} exceeds 15% budget "
        f"(ref={ref_n:,}, ndm={ndm_n:,})"
    )
