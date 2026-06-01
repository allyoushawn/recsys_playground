"""Ranking metrics over a single ranked candidate list.

Every function operates on one query: an ordered ``ranked_items`` sequence
(best first) and a ``relevant`` target that is either a single item id or a
set of item ids. Metrics are pure and free of side effects.
"""

from __future__ import annotations

import math
from typing import Dict, Iterable, List, Sequence, Set, Union

Relevant = Union[int, Set[int]]


def _as_set(relevant: Relevant) -> Set[int]:
    """Normalize ``relevant`` into a set of item ids."""
    if isinstance(relevant, (set, frozenset)):
        return set(relevant)
    return {relevant}


def hit_at_k(ranked_items: Sequence[int], relevant: Relevant, k: int) -> float:
    """Return 1.0 if any relevant item appears in the top-k, else 0.0."""
    rel = _as_set(relevant)
    top_k = ranked_items[:k]
    return 1.0 if any(item in rel for item in top_k) else 0.0


def recall_at_k(ranked_items: Sequence[int], relevant: Relevant, k: int) -> float:
    """Return |relevant ∩ top-k| / |relevant|."""
    rel = _as_set(relevant)
    if not rel:
        return 0.0
    top_k = set(ranked_items[:k])
    return len(top_k & rel) / len(rel)


def ndcg_at_k(ranked_items: Sequence[int], relevant: Relevant, k: int) -> float:
    """Return normalized DCG@k with binary gains.

    DCG sums ``rel(rank) / log2(rank + 1)`` over the top-k with 1-based rank;
    the ideal DCG packs all relevant items into the first positions.
    """
    rel = _as_set(relevant)
    if not rel:
        return 0.0
    dcg = 0.0
    for rank, item in enumerate(ranked_items[:k], start=1):
        if item in rel:
            dcg += 1.0 / math.log2(rank + 1)
    ideal_hits = min(len(rel), k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    if idcg == 0.0:
        return 0.0
    return dcg / idcg


def mrr_at_k(ranked_items: Sequence[int], relevant: Relevant, k: int) -> float:
    """Return 1/rank of the first relevant item within top-k, else 0.0."""
    rel = _as_set(relevant)
    for rank, item in enumerate(ranked_items[:k], start=1):
        if item in rel:
            return 1.0 / rank
    return 0.0


def sampled_auc(pos_score: float, neg_scores: Iterable[float]) -> float:
    """Return the sampled AUC of one positive against a set of negatives.

    Equals ``(#neg < pos + 0.5 * #neg == pos) / len(neg)``. Returns NaN when
    ``neg_scores`` is empty.
    """
    neg = list(neg_scores)
    n = len(neg)
    if n == 0:
        return float("nan")
    wins = sum(1 for s in neg if s < pos_score)
    ties = sum(1 for s in neg if s == pos_score)
    return (wins + 0.5 * ties) / n


def mean_metrics(per_query: List[Dict[str, float]]) -> Dict[str, float]:
    """Average each metric key across per-query dicts, ignoring NaNs."""
    sums: Dict[str, float] = {}
    counts: Dict[str, int] = {}
    for row in per_query:
        for key, value in row.items():
            if value is None or (isinstance(value, float) and math.isnan(value)):
                continue
            sums[key] = sums.get(key, 0.0) + float(value)
            counts[key] = counts.get(key, 0) + 1
    return {key: sums[key] / counts[key] for key in sums if counts[key] > 0}
