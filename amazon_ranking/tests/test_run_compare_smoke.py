"""Smoke tests for experiments/20260621_onetrans_reproduction/run_compare.py.

Tests that:
- The ``din`` model runs end-to-end on synthetic data via ``run()`` and
  produces a result dict with valid metrics (sampled_auc > 0.5).
- Milestone-pending models raise ``NotImplementedError`` via the registry.
"""

import argparse
import sys
import types

import pandas as pd
import pytest

torch = pytest.importorskip("torch")

# Make the experiment module importable regardless of cwd / install state.
import importlib
import importlib.util
import pathlib

_COMPARE_PATH = (
    pathlib.Path(__file__).parent.parent.parent
    / "experiments"
    / "20260621_onetrans_reproduction"
    / "run_compare.py"
)


def _import_run_compare():
    spec = importlib.util.spec_from_file_location("run_compare", _COMPARE_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


run_compare = _import_run_compare()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_args(**overrides) -> argparse.Namespace:
    """Return a minimal Namespace that run() accepts for synthetic smoke."""
    defaults = dict(
        model="din",
        dataset="Beauty",
        synthetic=True,
        data_dir="/tmp",
        cache_dir=None,
        out=None,
        epochs=2,
        embed_dim=8,
        batch_size=64,
        lr=1e-2,
        max_hist_len=5,
        n_eval_negatives=10,
        n_train_negatives=1,
        seed=0,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_run_compare_din_smoke():
    """End-to-end smoke: run() on synthetic DIN returns a valid result dict."""
    args = _make_args(model="din", epochs=3, embed_dim=8)
    result = run_compare.run(args)

    # Required top-level keys
    assert "model" in result
    assert "metrics" in result

    # sampled_auc must be present and plausible
    assert "sampled_auc" in result["metrics"], (
        f"sampled_auc missing from metrics: {result['metrics']}"
    )
    auc = result["metrics"]["sampled_auc"]
    assert auc > 0.5, (
        f"Expected sampled_auc > 0.5 on synthetic data after training, got {auc:.4f}"
    )


def test_run_compare_result_schema():
    """Result dict contains all expected top-level keys and config fields."""
    args = _make_args(model="din", epochs=2)
    result = run_compare.run(args)

    for key in ("model", "dataset", "config", "git_sha", "metrics", "wall_seconds", "timestamp_note"):
        assert key in result, f"missing key {key!r}"

    assert result["timestamp_note"] == "set-by-caller"
    assert result["model"] == "din"

    config = result["config"]
    for field in ("epochs", "embed_dim", "batch_size", "lr", "max_hist_len",
                  "n_eval_negatives", "n_train_negatives", "seed"):
        assert field in config, f"config missing field {field!r}"


def test_all_models_build_and_forward():
    """Every model in ALL_MODELS builds via the registry and forwards to [B] logits.

    All 9 are now implemented (M1-M4 complete), so the registry has no remaining
    milestone-pending entries. RankMixer/Wukong need ``embed_dim % n_field_tokens == 0``.
    """
    import torch

    from amazon_ranking.src.onetrans.registry import ALL_MODELS, build_model

    torch.manual_seed(0)
    hist = torch.randint(0, 50, (4, 6))
    mask = torch.ones(4, 6)
    target = torch.randint(0, 50, (4,))
    for name in ALL_MODELS:
        kw = {"n_field_tokens": 2} if name in ("rankmixer", "wukong") else {}
        model = build_model(name, num_items=50, embed_dim=8, **kw)
        out = model(hist, mask, target)
        assert out.shape == (4,), (name, out.shape)
        assert torch.isfinite(out).all(), name
