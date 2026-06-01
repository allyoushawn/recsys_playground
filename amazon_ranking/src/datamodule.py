"""Sequence ranking data module.

Builds ``(history, target, label)`` training examples and cached, leakage-free
evaluation candidates on top of the shared Amazon dataset preprocessing in
``tiger_semantic_id.src.data``.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

from tiger_semantic_id.src.data import (
    DatasetConfig,
    apply_id_maps,
    build_id_maps,
    filter_and_split,
)

from .negatives import (
    NegativeSampler,
    build_eval_candidates,
    cache_version,
    load_candidates,
    save_candidates,
)


@dataclass
class DataModuleConfig:
    """Configuration for :class:`SequenceRankingDataModule`."""

    max_hist_len: int = 20
    min_user_interactions: int = 5
    n_eval_negatives: int = 100
    n_train_negatives: int = 0  # label=0 examples emitted per positive train pair
    neg_strategy: str = "uniform"
    seed: int = 0


class SequenceRankingDataModule:
    """Turn raw reviews into train pairs and cached eval candidates."""

    def __init__(self, dm_cfg: DataModuleConfig) -> None:
        self.cfg = dm_cfg
        self.user2id: Dict[str, int] = {}
        self.item2id: Dict[str, int] = {}
        self._num_users = 0
        self._num_items = 0
        self._train_df: Optional[pd.DataFrame] = None
        self._val_df: Optional[pd.DataFrame] = None
        self._test_df: Optional[pd.DataFrame] = None

        self._train_examples: List[Dict[str, object]] = []
        self._user_seen: Dict[int, Set[int]] = {}
        self._eval: Dict[str, Dict[int, Dict[str, object]]] = {"val": {}, "test": {}}

    @classmethod
    def from_reviews(
        cls, reviews_df: pd.DataFrame, dm_cfg: DataModuleConfig
    ) -> "SequenceRankingDataModule":
        """Construct a module from a ``[user_id, item_id, ts]`` reviews frame."""
        self = cls(dm_cfg)
        ds_cfg = DatasetConfig(
            min_user_interactions=dm_cfg.min_user_interactions,
            max_hist_len=dm_cfg.max_hist_len,
        )
        train_df, val_df, test_df = filter_and_split(reviews_df, ds_cfg)
        user2id, item2id = build_id_maps([train_df, val_df, test_df])
        self.user2id = user2id
        self.item2id = item2id
        self._num_users = len(user2id)
        self._num_items = len(item2id)
        self._train_df = apply_id_maps(train_df, user2id, item2id)
        self._val_df = apply_id_maps(val_df, user2id, item2id)
        self._test_df = apply_id_maps(test_df, user2id, item2id)
        return self

    def build(self, cache_dir: Optional[str] = None) -> None:
        """Construct train examples, ``user_seen``, and eval candidates.

        Training examples are always (re)built — they are deterministic and
        cheap. Evaluation candidates are reused from ``cache_dir`` when a stored
        cache matches the current config version; otherwise they are sampled
        fresh and (if ``cache_dir`` is given) saved. This is the recommended
        entry point: it yields training data AND reproducible eval negatives in
        one call.
        """
        assert self._train_df is not None and self._val_df is not None and self._test_df is not None

        train_hist = self._chronological_histories(self._train_df)
        # Cap defensively; filter_and_split already enforces max_hist_len.
        max_len = self.cfg.max_hist_len
        train_hist = {u: h[-max_len:] for u, h in train_hist.items()}

        self._user_seen = self._build_user_seen()
        self._train_examples = self._build_train_examples(train_hist)

        # Reuse cached eval candidates when the version matches; else sample + save.
        if cache_dir is not None and self._restore_eval_from_cache(cache_dir):
            return

        sampler = NegativeSampler(
            self._num_items, strategy=self.cfg.neg_strategy, seed=self.cfg.seed
        )
        val_target = self._single_target_per_user(self._val_df)
        test_target = self._single_target_per_user(self._test_df)
        # Cap extended (test) histories so they never exceed train's max_hist_len.
        val_hist = train_hist
        test_hist = {
            u: h[-max_len:]
            for u, h in self._extend_histories(train_hist, val_target).items()
        }
        self._eval["val"] = self._build_eval_split(val_target, val_hist, sampler)
        self._eval["test"] = self._build_eval_split(test_target, test_hist, sampler)

        if cache_dir is not None:
            self.save_cache(cache_dir)

    def _build_train_examples(self, train_hist: Dict[int, List[int]]) -> List[Dict[str, object]]:
        """Next-item train pairs (label=1) plus optional sampled negatives (label=0)."""
        examples: List[Dict[str, object]] = []
        # A dedicated sampler (seed+1) so train negatives never perturb the eval
        # RNG stream, keeping eval candidates reproducible regardless of n_train_negatives.
        neg_sampler: Optional[NegativeSampler] = None
        if self.cfg.n_train_negatives > 0:
            neg_sampler = NegativeSampler(
                self._num_items, strategy=self.cfg.neg_strategy, seed=self.cfg.seed + 1
            )
        for user_idx in sorted(train_hist.keys()):
            hist = train_hist[user_idx]
            seen = self._user_seen.get(user_idx, set())
            for t in range(1, len(hist)):
                history = list(hist[:t])
                examples.append(
                    {
                        "user_idx": user_idx,
                        "history": history,
                        "target_idx": hist[t],
                        "label": 1,
                    }
                )
                if neg_sampler is not None:
                    negs = neg_sampler.sample(self.cfg.n_train_negatives, exclude=seen)
                    for neg in negs.tolist():
                        examples.append(
                            {
                                "user_idx": user_idx,
                                "history": list(history),
                                "target_idx": int(neg),
                                "label": 0,
                            }
                        )
        return examples

    def _restore_eval_from_cache(self, cache_dir: str) -> bool:
        """Load eval candidates from ``cache_dir`` if the version matches."""
        version = self._version()
        restored: Dict[str, Dict[int, Dict[str, object]]] = {}
        for split in ("val", "test"):
            candidates = load_candidates(
                os.path.join(cache_dir, f"candidates_{split}.npz"), version
            )
            if candidates is None:
                return False
            self._attach_histories(cache_dir, split, candidates)
            restored[split] = candidates
        self._eval = restored
        return True

    def _build_eval_split(
        self,
        target: Dict[int, int],
        histories: Dict[int, List[int]],
        sampler: NegativeSampler,
    ) -> Dict[int, Dict[str, object]]:
        eval_pairs: List[Tuple[int, int]] = sorted(target.items())
        candidates = build_eval_candidates(
            eval_pairs, self._user_seen, sampler, self.cfg.n_eval_negatives
        )
        for user_idx, entry in candidates.items():
            entry["history"] = list(histories.get(user_idx, []))
        return candidates

    @staticmethod
    def _chronological_histories(df: pd.DataFrame) -> Dict[int, List[int]]:
        out: Dict[int, List[int]] = {}
        ordered = df.sort_values(["user_idx", "ts"])
        for user_idx, group in ordered.groupby("user_idx", sort=False):
            out[int(user_idx)] = [int(i) for i in group["item_idx"].tolist()]
        return out

    @staticmethod
    def _single_target_per_user(df: pd.DataFrame) -> Dict[int, int]:
        out: Dict[int, int] = {}
        ordered = df.sort_values(["user_idx", "ts"])
        for user_idx, group in ordered.groupby("user_idx", sort=False):
            out[int(user_idx)] = int(group["item_idx"].tolist()[-1])
        return out

    @staticmethod
    def _extend_histories(
        base: Dict[int, List[int]], appended: Dict[int, int]
    ) -> Dict[int, List[int]]:
        out: Dict[int, List[int]] = {u: list(h) for u, h in base.items()}
        for user_idx, item_idx in appended.items():
            out.setdefault(user_idx, [])
            out[user_idx] = out[user_idx] + [item_idx]
        return out

    def _build_user_seen(self) -> Dict[int, Set[int]]:
        seen: Dict[int, Set[int]] = {}
        for df in (self._train_df, self._val_df, self._test_df):
            assert df is not None
            for user_idx, item_idx in zip(df["user_idx"].tolist(), df["item_idx"].tolist()):
                seen.setdefault(int(user_idx), set()).add(int(item_idx))
        return seen

    def train_examples(self) -> List[Dict[str, object]]:
        """Return the list of next-item training examples."""
        return self._train_examples

    def eval_examples(self, split: str) -> Dict[int, Dict[str, object]]:
        """Return the eval candidate dict for ``split`` in {"val", "test"}."""
        if split not in {"val", "test"}:
            raise ValueError(f"split must be 'val' or 'test', got {split!r}")
        return self._eval[split]

    @property
    def num_users(self) -> int:
        return self._num_users

    @property
    def num_items(self) -> int:
        return self._num_items

    def _version(self) -> Dict[str, object]:
        return cache_version(
            strategy=self.cfg.neg_strategy,
            seed=self.cfg.seed,
            n_negatives=self.cfg.n_eval_negatives,
            num_items=self._num_items,
            num_users=self._num_users,
        )

    def save_cache(self, cache_dir: str) -> None:
        """Persist id maps and eval candidates keyed by the cache version."""
        os.makedirs(cache_dir, exist_ok=True)
        version = self._version()

        with open(os.path.join(cache_dir, "id_maps.json"), "w") as f:
            json.dump(
                {
                    "user2id": {str(k): v for k, v in self.user2id.items()},
                    "item2id": {str(k): v for k, v in self.item2id.items()},
                    "num_users": self._num_users,
                    "num_items": self._num_items,
                    "config": asdict(self.cfg),
                },
                f,
            )

        for split in ("val", "test"):
            entries = self._eval[split]
            save_candidates(
                os.path.join(cache_dir, f"candidates_{split}.npz"), entries, version
            )
            histories = {str(u): list(e["history"]) for u, e in entries.items()}
            with open(os.path.join(cache_dir, f"history_{split}.json"), "w") as f:
                json.dump(histories, f)

    def _attach_histories(
        self, cache_dir: str, split: str, candidates: Dict[int, Dict[str, object]]
    ) -> None:
        """Attach cached per-user histories to ``candidates`` for ``split``."""
        hist_path = os.path.join(cache_dir, f"history_{split}.json")
        histories: Dict[str, List[int]] = {}
        if os.path.exists(hist_path):
            with open(hist_path) as f:
                histories = json.load(f)
        for user_idx, entry in candidates.items():
            entry["history"] = [int(i) for i in histories.get(str(user_idx), [])]

    def load_cache(self, cache_dir: str) -> bool:
        """Restore id maps and eval candidates if the version matches.

        Works on a FRESH instance: the id maps (and the ``num_users`` /
        ``num_items`` they define) are restored *before* the version check, so a
        cache can be loaded without first calling :meth:`from_reviews`. Note this
        restores eval candidates only — for training data use
        ``from_reviews(...).build(cache_dir=...)``. Returns ``False`` (no state
        mutated) when the cache is missing or its version does not match.
        """
        id_maps_path = os.path.join(cache_dir, "id_maps.json")
        if not os.path.exists(id_maps_path):
            return False

        with open(id_maps_path) as f:
            payload = json.load(f)
        num_users = int(payload["num_users"])
        num_items = int(payload["num_items"])
        # Build the candidate version from the cached counts so a fresh instance
        # (num_users/num_items == 0) can still match.
        version = cache_version(
            strategy=self.cfg.neg_strategy,
            seed=self.cfg.seed,
            n_negatives=self.cfg.n_eval_negatives,
            num_items=num_items,
            num_users=num_users,
        )
        restored: Dict[str, Dict[int, Dict[str, object]]] = {}
        for split in ("val", "test"):
            candidates = load_candidates(
                os.path.join(cache_dir, f"candidates_{split}.npz"), version
            )
            if candidates is None:
                return False
            self._attach_histories(cache_dir, split, candidates)
            restored[split] = candidates

        self.user2id = {k: int(v) for k, v in payload["user2id"].items()}
        self.item2id = {k: int(v) for k, v in payload["item2id"].items()}
        self._num_users = num_users
        self._num_items = num_items
        self._eval = restored
        return True
