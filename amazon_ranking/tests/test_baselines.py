import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from amazon_ranking.src.datamodule import DataModuleConfig, SequenceRankingDataModule
from amazon_ranking.src.din import DINTrainConfig, evaluate_ranking, pad_histories, train_din
from amazon_ranking.src.baselines import DCN, DeepFM, MeanPoolMLP, build_baseline


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


ALL = ["meanpool", "deepfm", "dcn"]


@pytest.mark.parametrize("name", ALL)
def test_baseline_forward_shapes_and_finite(name):
    dm = _module()
    model = build_baseline(name, num_items=dm.num_items, embed_dim=16)
    assert model.pad_idx == dm.num_items  # same padding convention as DIN
    ids, mask = pad_histories([[0, 1], [2]], max_hist_len=8, pad_idx=model.pad_idx)
    logits = model(ids, mask, torch.tensor([3, 4], dtype=torch.long))
    assert logits.shape == (2,)
    assert torch.isfinite(logits).all()


@pytest.mark.parametrize("name", ALL)
def test_baseline_handles_empty_history_without_nan(name):
    dm = _module()
    model = build_baseline(name, num_items=dm.num_items, embed_dim=16)
    ids, mask = pad_histories([[]], max_hist_len=8, pad_idx=model.pad_idx)
    logits = model(ids, mask, torch.tensor([0], dtype=torch.long))
    assert torch.isfinite(logits).all()


@pytest.mark.parametrize("name", ALL)
def test_baseline_trains_and_reduces_loss(name):
    dm = _module(seed=0)
    model = build_baseline(name, num_items=dm.num_items, embed_dim=16)
    out = train_din(  # the harness is model-agnostic — works for any compatible model
        model,
        dm.train_examples(),
        max_hist_len=dm.cfg.max_hist_len,
        cfg=DINTrainConfig(embed_dim=16, epochs=15, batch_size=64, lr=1e-2, seed=0),
    )
    assert np.isfinite(out["first_epoch_loss"]) and np.isfinite(out["last_epoch_loss"])
    assert out["last_epoch_loss"] < out["first_epoch_loss"]


@pytest.mark.parametrize("name", ALL)
def test_baseline_evaluate_ranking_in_range(name):
    dm = _module(seed=0)
    model = build_baseline(name, num_items=dm.num_items, embed_dim=16)
    train_din(
        model,
        dm.train_examples(),
        max_hist_len=dm.cfg.max_hist_len,
        cfg=DINTrainConfig(embed_dim=16, epochs=5, batch_size=64, lr=1e-2, seed=0),
    )
    res = evaluate_ranking(model, dm.eval_examples("test"), max_hist_len=dm.cfg.max_hist_len, ks=(5, 10))
    for key in ("recall@10", "ndcg@10", "mrr@10", "sampled_auc"):
        assert key in res and 0.0 <= res[key] <= 1.0


def test_deepfm_fm_term_equals_dot_product_for_two_fields():
    """With exactly two fields the FM identity must equal <interest, target>."""
    torch.manual_seed(0)
    model = DeepFM(num_items=50, embed_dim=8)
    ids, mask = pad_histories([[3, 7]], max_hist_len=8, pad_idx=model.pad_idx)
    target_ids = torch.tensor([5], dtype=torch.long)
    interest = model._pool(ids, mask)
    target = model.item_emb(target_ids)
    expected_fm = (interest * target).sum(dim=1)  # <interest, target>
    fields = torch.stack([interest, target], dim=1)
    fm = 0.5 * ((fields.sum(dim=1) ** 2) - (fields ** 2).sum(dim=1)).sum(dim=1)
    assert torch.allclose(fm, expected_fm, atol=1e-5)


def test_build_baseline_unknown_raises():
    with pytest.raises(ValueError):
        build_baseline("nope", num_items=10)


def test_baseline_output_invariant_to_padding_amount():
    """Mean pooling is mask-aware → extra right-padding must not change the logit."""
    for name in ALL:
        torch.manual_seed(0)
        model = build_baseline(name, num_items=50, embed_dim=8)
        history = [3, 7]
        target = torch.tensor([5], dtype=torch.long)
        ids4, mask4 = pad_histories([history], max_hist_len=4, pad_idx=model.pad_idx)
        ids8, mask8 = pad_histories([history], max_hist_len=8, pad_idx=model.pad_idx)
        assert torch.allclose(model(ids4, mask4, target), model(ids8, mask8, target), atol=1e-6)
