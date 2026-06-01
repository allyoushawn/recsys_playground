import pandas as pd

from amazon_ranking.src.datamodule import DataModuleConfig, SequenceRankingDataModule


def _make_reviews(n_users: int = 6, items_per_user: int = 6) -> pd.DataFrame:
    # Overlapping windows over a larger item pool so each user sees only a
    # subset, leaving genuine unseen items to sample negatives from.
    rows = []
    ts = 1
    for u in range(n_users):
        for i in range(items_per_user):
            rows.append({"user_id": f"u{u}", "item_id": f"i{u * 4 + i}", "ts": ts})
            ts += 1
    return pd.DataFrame(rows)


def _build_module(seed: int = 0) -> SequenceRankingDataModule:
    cfg = DataModuleConfig(n_eval_negatives=3, seed=seed, min_user_interactions=5)
    dm = SequenceRankingDataModule.from_reviews(_make_reviews(), cfg)
    dm.build()
    return dm


def test_build_train_examples_and_histories():
    dm = _build_module()
    assert dm.num_users > 0
    assert dm.num_items > 0
    assert len(dm.train_examples()) > 0
    for ex in dm.train_examples():
        assert ex["label"] == 1
        assert isinstance(ex["history"], list)
        assert len(ex["history"]) > 0
        assert all(isinstance(i, int) for i in ex["history"])

    val = dm.eval_examples("val")
    test = dm.eval_examples("test")
    for user_idx in test:
        assert len(test[user_idx]["history"]) == len(val[user_idx]["history"]) + 1


def test_eval_candidates_are_leakage_free():
    dm = _build_module()
    test = dm.eval_examples("test")
    n_neg = dm.cfg.n_eval_negatives
    for user_idx, entry in test.items():
        cands = entry["candidates"]
        assert cands[0] == entry["positive"]
        assert len(cands) == n_neg + 1
        seen = dm._user_seen[user_idx]
        negatives = set(cands[1:].tolist())
        assert not (negatives & seen)


def test_cache_round_trip(tmp_path):
    dm = _build_module(seed=0)
    cache_dir = str(tmp_path / "cache")
    dm.save_cache(cache_dir)

    fresh = SequenceRankingDataModule.from_reviews(
        _make_reviews(), DataModuleConfig(n_eval_negatives=3, seed=0, min_user_interactions=5)
    )
    assert fresh.load_cache(cache_dir) is True
    for split in ("val", "test"):
        original = dm.eval_examples(split)
        restored = fresh.eval_examples(split)
        assert set(original.keys()) == set(restored.keys())
        for user_idx in original:
            assert original[user_idx]["positive"] == restored[user_idx]["positive"]
            assert (
                original[user_idx]["candidates"].tolist()
                == restored[user_idx]["candidates"].tolist()
            )

    changed = SequenceRankingDataModule.from_reviews(
        _make_reviews(), DataModuleConfig(n_eval_negatives=3, seed=7, min_user_interactions=5)
    )
    assert changed.load_cache(cache_dir) is False


def test_train_negatives_are_emitted_and_leakage_free():
    cfg = DataModuleConfig(
        n_eval_negatives=3, n_train_negatives=2, seed=0, min_user_interactions=5
    )
    dm = SequenceRankingDataModule.from_reviews(_make_reviews(), cfg)
    dm.build()
    examples = dm.train_examples()
    labels = {ex["label"] for ex in examples}
    assert labels == {0, 1}
    n_pos = sum(1 for ex in examples if ex["label"] == 1)
    n_neg = sum(1 for ex in examples if ex["label"] == 0)
    assert n_neg == n_pos * cfg.n_train_negatives
    for ex in examples:
        if ex["label"] == 0:
            assert ex["target_idx"] not in dm._user_seen[ex["user_idx"]]


def test_build_with_cache_dir_reuses_negatives(tmp_path):
    cache_dir = str(tmp_path / "cache")
    cfg = lambda: DataModuleConfig(n_eval_negatives=3, seed=0, min_user_interactions=5)
    a = SequenceRankingDataModule.from_reviews(_make_reviews(), cfg())
    a.build(cache_dir=cache_dir)
    b = SequenceRankingDataModule.from_reviews(_make_reviews(), cfg())
    b.build(cache_dir=cache_dir)  # should reuse the cached eval candidates
    assert len(b.train_examples()) > 0  # and still have training data
    for split in ("val", "test"):
        for u in a.eval_examples(split):
            assert (
                a.eval_examples(split)[u]["candidates"].tolist()
                == b.eval_examples(split)[u]["candidates"].tolist()
            )


def test_truly_fresh_instance_can_load_cache(tmp_path):
    cache_dir = str(tmp_path / "cache")
    built = _build_module(seed=0)
    built.save_cache(cache_dir)
    # No from_reviews(): a bare instance must still restore eval + id maps.
    fresh = SequenceRankingDataModule(DataModuleConfig(n_eval_negatives=3, seed=0, min_user_interactions=5))
    assert fresh.load_cache(cache_dir) is True
    assert fresh.num_items == built.num_items
    assert fresh.num_users == built.num_users
    for u in built.eval_examples("test"):
        assert (
            fresh.eval_examples("test")[u]["candidates"].tolist()
            == built.eval_examples("test")[u]["candidates"].tolist()
        )


def test_test_history_capped_at_max_hist_len():
    cfg = DataModuleConfig(n_eval_negatives=3, seed=0, min_user_interactions=5, max_hist_len=3)
    dm = SequenceRankingDataModule.from_reviews(_make_reviews(), cfg)
    dm.build()
    for entry in dm.eval_examples("test").values():
        assert len(entry["history"]) <= cfg.max_hist_len
