import os

import numpy as np
import pytest

from amazon_ranking.src.negatives import (
    NegativeSampler,
    build_eval_candidates,
    cache_version,
    load_candidates,
    save_candidates,
)


def test_sample_distinct_and_excluded():
    sampler = NegativeSampler(num_items=10, seed=0)
    out = sampler.sample(5, exclude={0, 1})
    assert len(out) == 5
    assert len(set(out.tolist())) == 5
    assert all(0 <= x < 10 for x in out.tolist())
    assert not (set(out.tolist()) & {0, 1})


def test_sample_is_deterministic():
    a = NegativeSampler(num_items=10, seed=0).sample(5, exclude={0, 1})
    b = NegativeSampler(num_items=10, seed=0).sample(5, exclude={0, 1})
    assert a.tolist() == b.tolist()


def test_sample_too_many_raises():
    sampler = NegativeSampler(num_items=10, seed=0)
    with pytest.raises(ValueError):
        sampler.sample(9, exclude={0, 1})


def test_popularity_strategy_requires_popularity():
    with pytest.raises(ValueError):
        NegativeSampler(num_items=5, strategy="popularity")


def test_build_eval_candidates_excludes_seen():
    sampler = NegativeSampler(num_items=20, seed=1)
    user_seen = {0: {0, 1, 2}, 1: {5, 6}}
    pairs = [(0, 0), (1, 5)]
    out = build_eval_candidates(pairs, user_seen, sampler, n_negatives=4)
    for user_idx, positive in pairs:
        entry = out[user_idx]
        assert entry["candidates"][0] == positive
        assert len(entry["candidates"]) == 5
        negatives = set(entry["candidates"][1:].tolist())
        assert not (negatives & user_seen[user_idx])


def test_cache_round_trip_and_version_mismatch(tmp_path):
    candidates = {
        0: {"positive": 3, "candidates": np.array([3, 7, 9], dtype=np.int64)},
        2: {"positive": 1, "candidates": np.array([1, 4, 8], dtype=np.int64)},
    }
    version = cache_version("uniform", seed=0, n_negatives=2, num_items=10, num_users=3)
    path = os.path.join(tmp_path, "cand.npz")
    save_candidates(path, candidates, version)

    loaded = load_candidates(path, version)
    assert loaded is not None
    assert set(loaded.keys()) == {0, 2}
    assert loaded[0]["positive"] == 3
    assert loaded[0]["candidates"].tolist() == [3, 7, 9]

    other = cache_version("uniform", seed=99, n_negatives=2, num_items=10, num_users=3)
    assert load_candidates(path, other) is None
    assert load_candidates(os.path.join(tmp_path, "missing.npz"), version) is None
