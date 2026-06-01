import math

import pytest

from amazon_ranking.src.metrics import (
    hit_at_k,
    mean_metrics,
    mrr_at_k,
    ndcg_at_k,
    recall_at_k,
    sampled_auc,
)


def test_single_relevant_outside_top_k():
    ranked = [5, 2, 9, 1]
    assert hit_at_k(ranked, 9, 2) == 0.0
    assert recall_at_k(ranked, 9, 2) == 0.0
    assert ndcg_at_k(ranked, 9, 2) == 0.0
    assert mrr_at_k(ranked, 9, 2) == 0.0


def test_single_relevant_inside_top_k():
    ranked = [5, 2, 9, 1]
    assert hit_at_k(ranked, 9, 3) == 1.0
    assert recall_at_k(ranked, 9, 3) == 1.0
    assert ndcg_at_k(ranked, 9, 3) == 0.5
    assert mrr_at_k(ranked, 9, 3) == pytest.approx(1 / 3)


def test_multiple_relevant():
    ranked = [5, 2, 9, 1]
    relevant = {5, 9}
    assert recall_at_k(ranked, relevant, 3) == pytest.approx(1.0)
    expected = (1 / math.log2(2) + 1 / math.log2(4)) / (
        1 / math.log2(2) + 1 / math.log2(3)
    )
    assert ndcg_at_k(ranked, relevant, 3) == pytest.approx(expected)


def test_sampled_auc():
    assert sampled_auc(0.8, [0.1, 0.5, 0.9, 0.8]) == pytest.approx(0.625)
    assert sampled_auc(1.0, [0.0, 0.0]) == 1.0
    assert math.isnan(sampled_auc(0.5, []))


def test_mean_metrics_ignores_nan():
    rows = [
        {"hit": 1.0, "auc": float("nan")},
        {"hit": 0.0, "auc": 0.5},
    ]
    out = mean_metrics(rows)
    assert out["hit"] == pytest.approx(0.5)
    assert out["auc"] == pytest.approx(0.5)
