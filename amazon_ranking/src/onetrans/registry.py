"""Unified model registry for sequence-ranking experiments.

Provides a single ``build_model`` entry point that maps model names to their
constructors, delegating to existing modules for current models and raising a
clear ``NotImplementedError`` for milestone-pending future models.
"""

from __future__ import annotations

import torch.nn as nn

from amazon_ranking.src.baselines import build_baseline
from amazon_ranking.src.din import DIN

ALL_MODELS = [
    "meanpool",
    "deepfm",
    "dcn",
    "din",
    "bst",
    "sasrec",
    "rankmixer",
    "wukong",
    "onetrans",
    "onetrans_ns",
]

# Lazy import thunks for milestone-pending models.  Each entry maps a model
# name to a callable ``() -> type[nn.Module]`` that imports the class only
# when the model is actually requested.  Wiring a new model later is a
# one-line addition here.
_LAZY_REGISTRY: dict[str, object] = {
    "bst": lambda: __import__(
        "amazon_ranking.src.onetrans.bst", fromlist=["BST"]
    ).BST,
    "sasrec": lambda: __import__(
        "amazon_ranking.src.onetrans.sasrec", fromlist=["SASRec"]
    ).SASRec,
    "rankmixer": lambda: __import__(
        "amazon_ranking.src.onetrans.rankmixer", fromlist=["RankMixer"]
    ).RankMixer,
    "wukong": lambda: __import__(
        "amazon_ranking.src.onetrans.wukong", fromlist=["Wukong"]
    ).Wukong,
    "onetrans": lambda: __import__(
        "amazon_ranking.src.onetrans.onetrans", fromlist=["OneTrans"]
    ).OneTrans,
    "onetrans_ns": lambda: __import__(
        "amazon_ranking.src.onetrans.onetrans", fromlist=["OneTrans"]
    ).OneTrans,
}

_BASELINES = {"meanpool", "deepfm", "dcn"}


def build_model(name: str, num_items: int, embed_dim: int = 32, **kwargs) -> nn.Module:
    """Construct a ranking model by name.

    Parameters
    ----------
    name:
        One of :data:`ALL_MODELS`.
    num_items:
        Vocabulary size (number of distinct items).  The pad index is set to
        ``num_items`` following the convention in ``DIN`` / ``_SeqRankModel``.
    embed_dim:
        Item embedding dimension.
    **kwargs:
        Extra keyword arguments forwarded to the model constructor (ignored
        for baselines that do not accept them).

    Returns
    -------
    nn.Module
        A module with the shared forward signature
        ``forward(hist_ids, hist_mask, target_ids) -> logits [B]``,
        a ``pad_idx`` attribute, and an ``item_emb`` embedding table.

    Raises
    ------
    ValueError
        When ``name`` is not in :data:`ALL_MODELS`.
    NotImplementedError
        When the module for a milestone-pending model has not yet been
        implemented (the import fails with ``ImportError``).
    """
    if name not in ALL_MODELS:
        raise ValueError(
            f"unknown model {name!r}; options: {ALL_MODELS}"
        )

    if name in _BASELINES:
        return build_baseline(name, num_items=num_items, embed_dim=embed_dim)

    if name == "din":
        return DIN(num_items=num_items, embed_dim=embed_dim, **kwargs)

    # Milestone-pending: attempt lazy import; raise descriptively on failure.
    thunk = _LAZY_REGISTRY[name]
    try:
        cls = thunk()  # type: ignore[operator]
    except ImportError:
        raise NotImplementedError(
            f"{name} not yet implemented (milestone pending)"
        ) from None

    return cls(num_items=num_items, embed_dim=embed_dim, **kwargs)
