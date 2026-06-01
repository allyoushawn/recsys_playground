import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from amazon_ranking.src.datamodule import DataModuleConfig, SequenceRankingDataModule
from amazon_ranking.src.din import (
    DIN,
    DINTrainConfig,
    evaluate_ranking,
    pad_histories,
    rank_candidates,
    score_candidates,
    train_din,
)


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


def test_pad_histories_shapes_and_mask():
    ids, mask = pad_histories([[1, 2, 3], [], [9]], max_hist_len=4, pad_idx=99)
    assert ids.shape == (3, 4) and mask.shape == (3, 4)
    assert ids[0].tolist() == [1, 2, 3, 99]
    assert mask[0].tolist() == [1.0, 1.0, 1.0, 0.0]
    assert mask[1].tolist() == [0.0, 0.0, 0.0, 0.0]  # empty history fully masked
    # keeps the most recent max_hist_len items
    ids2, _ = pad_histories([[1, 2, 3, 4, 5]], max_hist_len=3, pad_idx=99)
    assert ids2[0].tolist() == [3, 4, 5]


def test_din_forward_shapes_and_finite():
    dm = _module()
    model = DIN(num_items=dm.num_items, embed_dim=16)
    ids, mask = pad_histories([[0, 1], [2]], max_hist_len=8, pad_idx=model.pad_idx)
    target = torch.tensor([3, 4], dtype=torch.long)
    logits = model(ids, mask, target)
    assert logits.shape == (2,)
    assert torch.isfinite(logits).all()


def test_din_handles_empty_history_without_nan():
    dm = _module()
    model = DIN(num_items=dm.num_items, embed_dim=16)
    ids, mask = pad_histories([[]], max_hist_len=8, pad_idx=model.pad_idx)
    logits = model(ids, mask, torch.tensor([0], dtype=torch.long))
    assert torch.isfinite(logits).all()


def test_din_trains_and_reduces_loss():
    dm = _module(seed=0)
    model = DIN(num_items=dm.num_items, embed_dim=16)
    out = train_din(
        model,
        dm.train_examples(),
        max_hist_len=dm.cfg.max_hist_len,
        cfg=DINTrainConfig(embed_dim=16, epochs=15, batch_size=64, lr=1e-2, seed=0),
    )
    assert np.isfinite(out["first_epoch_loss"]) and np.isfinite(out["last_epoch_loss"])
    # Training should reduce BCE loss on this small learnable set.
    assert out["last_epoch_loss"] < out["first_epoch_loss"]


def test_din_end_to_end_ranking_metrics():
    dm = _module(seed=0)
    model = DIN(num_items=dm.num_items, embed_dim=16)
    train_din(
        model,
        dm.train_examples(),
        max_hist_len=dm.cfg.max_hist_len,
        cfg=DINTrainConfig(embed_dim=16, epochs=5, batch_size=64, lr=1e-2, seed=0),
    )
    # rank_candidates returns a permutation of the candidates
    entry = next(iter(dm.eval_examples("test").values()))
    ranked = rank_candidates(
        model, [int(i) for i in entry["history"]], [int(c) for c in entry["candidates"]],
        max_hist_len=dm.cfg.max_hist_len,
    )
    assert sorted(ranked) == sorted(int(c) for c in entry["candidates"])

    res = evaluate_ranking(model, dm.eval_examples("test"), max_hist_len=dm.cfg.max_hist_len, ks=(5, 10))
    for key in ("recall@10", "ndcg@10", "mrr@10", "sampled_auc"):
        assert key in res
        assert 0.0 <= res[key] <= 1.0


def test_din_output_invariant_to_padding_amount():
    """Masking semantics lock: padded positions get zero attention weight, so the
    logit for a fixed (history, target) must not depend on how much right-padding
    is added. Guards the masked-softmax pooling against padding leakage."""
    torch.manual_seed(0)
    model = DIN(num_items=50, embed_dim=8)
    history = [3, 7]
    target = torch.tensor([5], dtype=torch.long)
    ids4, mask4 = pad_histories([history], max_hist_len=4, pad_idx=model.pad_idx)
    ids8, mask8 = pad_histories([history], max_hist_len=8, pad_idx=model.pad_idx)
    out4 = model(ids4, mask4, target)
    out8 = model(ids8, mask8, target)
    assert torch.allclose(out4, out8, atol=1e-6)


def test_din_empty_history_collapses_but_nonempty_does_not():
    """Only genuinely empty histories should pool to zero interest. A non-empty
    history must produce a different logit from the empty one for the same
    target (i.e. its attention is not also being zeroed)."""
    torch.manual_seed(0)
    model = DIN(num_items=50, embed_dim=8)
    target = torch.tensor([5, 5], dtype=torch.long)
    ids, mask = pad_histories([[], [3, 7]], max_hist_len=4, pad_idx=model.pad_idx)
    out = model(ids, mask, target)
    assert torch.isfinite(out).all()
    assert not torch.allclose(out[0], out[1])  # empty vs non-empty differ


def test_din_reproducible_when_seeded_before_construction():
    """Per the train_din docstring, seeding torch BEFORE constructing the model
    yields end-to-end reproducible training (init + batch order + optimizer)."""

    def run():
        torch.manual_seed(123)
        dm = _module(seed=0)
        model = DIN(num_items=dm.num_items, embed_dim=16)
        return train_din(
            model,
            dm.train_examples(),
            max_hist_len=dm.cfg.max_hist_len,
            cfg=DINTrainConfig(embed_dim=16, epochs=5, batch_size=64, lr=1e-2, seed=0),
        )

    a, b = run(), run()
    assert a["first_epoch_loss"] == pytest.approx(b["first_epoch_loss"])
    assert a["last_epoch_loss"] == pytest.approx(b["last_epoch_loss"])


def test_din_score_candidates_is_single_ranking_primitive():
    """rank_candidates must derive its order from score_candidates (one forward
    pass), not an independent scoring path."""
    dm = _module(seed=0)
    model = DIN(num_items=dm.num_items, embed_dim=16)
    entry = next(iter(dm.eval_examples("test").values()))
    history = [int(i) for i in entry["history"]]
    cands = [int(c) for c in entry["candidates"]]
    scores = score_candidates(model, history, cands, max_hist_len=dm.cfg.max_hist_len)
    assert scores.shape == (len(cands),)
    ranked = rank_candidates(model, history, cands, max_hist_len=dm.cfg.max_hist_len)
    expected = [cands[i] for i in np.argsort(-scores, kind="stable")]
    assert ranked == expected
