"""Leakage-free negative sampling plus a reproducible eval-candidate cache.

The cache contract lets evaluation negatives be sampled once and reused across
runs. A stored cache is only honored when its ``version`` dict matches the
current configuration, otherwise callers must rebuild.
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np


class NegativeSampler:
    """Sample item ids uniformly or proportional to popularity.

    The sampler owns a seeded ``np.random.default_rng`` so that draws are
    deterministic for a given seed and call sequence.
    """

    def __init__(
        self,
        num_items: int,
        strategy: str = "uniform",
        item_popularity: Optional[np.ndarray] = None,
        seed: int = 0,
    ) -> None:
        if strategy not in {"uniform", "popularity"}:
            raise ValueError(f"Unknown strategy: {strategy!r}")
        if strategy == "popularity":
            if item_popularity is None:
                raise ValueError("item_popularity is required for the 'popularity' strategy")
            pop = np.asarray(item_popularity, dtype=np.float64)
            if pop.shape != (num_items,):
                raise ValueError(
                    f"item_popularity must have shape ({num_items},), got {pop.shape}"
                )
            if np.any(pop < 0):
                raise ValueError("item_popularity must be non-negative")
            total = pop.sum()
            if total <= 0:
                raise ValueError("item_popularity must have positive total mass")
            self._probs: Optional[np.ndarray] = pop / total
        else:
            self._probs = None

        self.num_items = int(num_items)
        self.strategy = strategy
        self.seed = int(seed)
        self.rng = np.random.default_rng(seed)

    def sample(self, n: int, exclude: Set[int]) -> np.ndarray:
        """Return ``n`` distinct item ids in [0, num_items), none in ``exclude``.

        Deterministic given the seed and the sequence of prior calls.
        """
        exclude = set(exclude)
        available = self.num_items - len(exclude)
        if n > available:
            raise ValueError(
                f"Cannot sample {n} distinct items: only {available} available "
                f"(num_items={self.num_items}, excluded={len(exclude)})"
            )
        if n <= 0:
            return np.empty(0, dtype=np.int64)

        if self.strategy == "uniform":
            return self._sample_uniform(n, exclude)
        return self._sample_popularity(n, exclude)

    def _sample_uniform(self, n: int, exclude: Set[int]) -> np.ndarray:
        chosen: List[int] = []
        seen: Set[int] = set()
        # Rejection sampling keeps the draw order stable across exclude sizes.
        while len(chosen) < n:
            remaining = n - len(chosen)
            # Oversample to reduce the number of RNG rounds.
            batch = self.rng.integers(0, self.num_items, size=max(remaining * 2, 1))
            for item in batch:
                item = int(item)
                if item in exclude or item in seen:
                    continue
                seen.add(item)
                chosen.append(item)
                if len(chosen) == n:
                    break
        return np.asarray(chosen, dtype=np.int64)

    def _sample_popularity(self, n: int, exclude: Set[int]) -> np.ndarray:
        assert self._probs is not None
        chosen: List[int] = []
        seen: Set[int] = set()
        while len(chosen) < n:
            remaining = n - len(chosen)
            batch = self.rng.choice(
                self.num_items, size=max(remaining * 2, 1), p=self._probs
            )
            for item in batch:
                item = int(item)
                if item in exclude or item in seen:
                    continue
                seen.add(item)
                chosen.append(item)
                if len(chosen) == n:
                    break
        return np.asarray(chosen, dtype=np.int64)


def build_eval_candidates(
    eval_pairs: Sequence[Tuple[int, int]],
    user_seen: Dict[int, Set[int]],
    sampler: NegativeSampler,
    n_negatives: int,
) -> Dict[int, Dict[str, object]]:
    """Build a ``{user_idx: {"positive", "candidates"}}`` map for evaluation.

    ``candidates`` is ``[positive] + n_negatives`` sampled negatives (positive
    first). Negatives exclude every item in ``user_seen[user_idx]`` so that they
    are genuinely unseen by the user.
    """
    out: Dict[int, Dict[str, object]] = {}
    for user_idx, positive in eval_pairs:
        seen = set(user_seen.get(user_idx, set()))
        seen.add(positive)
        negatives = sampler.sample(n_negatives, exclude=seen)
        candidates = np.concatenate(([positive], negatives)).astype(np.int64)
        out[user_idx] = {"positive": int(positive), "candidates": candidates}
    return out


def cache_version(
    strategy: str,
    seed: int,
    n_negatives: int,
    num_items: int,
    num_users: int,
) -> Dict[str, object]:
    """Build the version dict that identifies a candidate cache."""
    return {
        "strategy": str(strategy),
        "seed": int(seed),
        "n_negatives": int(n_negatives),
        "num_items": int(num_items),
        "num_users": int(num_users),
    }


def save_candidates(path: str, candidates: Dict[int, Dict[str, object]], version: Dict[str, object]) -> None:
    """Persist ``candidates`` plus its ``version`` to an ``.npz`` file."""
    directory = os.path.dirname(os.path.abspath(path))
    if directory:
        os.makedirs(directory, exist_ok=True)

    arrays: Dict[str, np.ndarray] = {}
    user_order: List[int] = []
    for user_idx in sorted(candidates.keys()):
        entry = candidates[user_idx]
        user_order.append(int(user_idx))
        arrays[f"cand__{user_idx}"] = np.asarray(entry["candidates"], dtype=np.int64)
        arrays[f"pos__{user_idx}"] = np.asarray([entry["positive"]], dtype=np.int64)

    arrays["__users__"] = np.asarray(user_order, dtype=np.int64)
    arrays["__version__"] = np.frombuffer(
        json.dumps(version, sort_keys=True).encode("utf-8"), dtype=np.uint8
    )
    np.savez(path, **arrays)


def load_candidates(path: str, version: Dict[str, object]) -> Optional[Dict[int, Dict[str, object]]]:
    """Load candidates if the stored version matches; else return None.

    A missing file or a version mismatch returns ``None`` to signal a rebuild.
    """
    if not os.path.exists(path):
        return None

    with np.load(path, allow_pickle=False) as data:
        stored_raw = bytes(data["__version__"].tobytes()).decode("utf-8")
        stored_version = json.loads(stored_raw)
        if stored_version != json.loads(json.dumps(version, sort_keys=True)):
            return None
        users = [int(u) for u in data["__users__"].tolist()]
        out: Dict[int, Dict[str, object]] = {}
        for user_idx in users:
            candidates = data[f"cand__{user_idx}"].astype(np.int64)
            positive = int(data[f"pos__{user_idx}"][0])
            out[user_idx] = {"positive": positive, "candidates": candidates}
    return out
