# amazon_ranking

Unified dataset layer for implicit-feedback **ranking** on the Amazon
datasets (Beauty / Video_Games). The dataset layer is pure `numpy` + `pandas`;
the **DIN model** (`src/din.py`) + end-to-end runner (`run_din.py`) sit on top
and add a `torch` dependency. See `RESULTS.md` for the DIN-on-Beauty leaderboard
(sampled_auc 0.6338, T4).

## Purpose

Turn raw `[user_id, item_id, ts]` review frames into everything a sequence
ranking model (e.g. DIN) needs:

- chronological per-user train histories and next-item `(history, target, label)`
  training examples — positives plus, when `n_train_negatives > 0`, sampled
  label-0 negatives (so the data is directly usable for binary-CTR / pairwise
  training, not just positives);
- leakage-free, reproducible evaluation candidates (`[positive] + N negatives`)
  for `val` and `test`, where negatives exclude every item the user has ever
  interacted with;
- a persisted cache so eval negatives are sampled once and reused.

Preprocessing (user filtering, leave-one-out split, contiguous id maps) is
reused from `tiger_semantic_id.src.data`.

## The 6-tier cache contract

This layer satisfies the eval-candidate caching contract so that negatives are
deterministic and never re-sampled across runs:

1. `cache_version(strategy, seed, n_negatives, num_items, num_users)` — the
   identity of a cache; any field change invalidates it.
2. `save_candidates(path, candidates, version)` — write candidates + version to
   an `.npz`.
3. `load_candidates(path, version)` — return candidates only on a version match,
   else `None` (missing file → `None`), signalling a rebuild.
4. `NegativeSampler(seed=...)` — seeded RNG so a given config reproduces the
   exact same negatives.
5. `SequenceRankingDataModule.save_cache(cache_dir)` — persist id maps + eval
   candidates keyed by the version.
6. `SequenceRankingDataModule.load_cache(cache_dir)` — restore eval candidates +
   id maps (eval-only); returns `True` only on a version match, else `False`.
   Prefer `build(cache_dir=...)`, which consults this cache and also builds the
   training examples in one call.

## Usage

**Recommended path** — `build(cache_dir=...)` always (re)builds the (cheap,
deterministic) training examples and *reuses* the cached eval candidates when the
version matches, otherwise samples them fresh and saves. This avoids the trap of
restoring negatives but losing training data (or vice-versa):

```python
import pandas as pd
from amazon_ranking.src.datamodule import DataModuleConfig, SequenceRankingDataModule

reviews_df = ...  # columns: [user_id, item_id, ts]
cfg = DataModuleConfig(
    max_hist_len=20, n_eval_negatives=100, n_train_negatives=1,
    neg_strategy="uniform", seed=0,
)

dm = SequenceRankingDataModule.from_reviews(reviews_df, cfg)
dm.build(cache_dir="cache/amazon_beauty")   # builds train data + reuses/saves eval negatives

train = dm.train_examples()          # list of {"user_idx","history","target_idx","label"}
test = dm.eval_examples("test")      # {user_idx: {"positive","candidates","history"}}
```

`load_cache(cache_dir)` is a lower-level helper that restores **eval candidates +
id maps only** (it works on a fresh instance without `from_reviews`); it does NOT
restore training examples. Use it when you only need to score a trained model.

Metrics over a single ranked candidate list live in
`amazon_ranking.src.metrics` (`hit_at_k`, `recall_at_k`, `ndcg_at_k`,
`mrr_at_k`, `sampled_auc`, `mean_metrics`).

## Running DIN

```bash
# Colab T4 (downloads to local SSD; result JSON to Drive):
python -m amazon_ranking.run_din --dataset Beauty --data-dir /content/amazon \
    --out /content/drive/MyDrive/colab/data/amazon/din_Beauty_results.json
# Local CPU smoke (synthetic, no download):
python -m amazon_ranking.run_din --synthetic --epochs 8 --out /tmp/din_synth.json
```

`src/din.py` exposes `DIN`, `train_din`, `score_candidates`, `rank_candidates`,
`evaluate_ranking`. Seed torch **before** constructing `DIN` for reproducible init
(`train_din`'s seed governs only batch order / optimizer). See `RESULTS.md`.

## Caveats

- Test/val histories are capped to `max_hist_len` (test history = train history
  with the val item appended, then capped), so all histories share one max length.
- `NegativeSampler` is deterministic per `(seed, call order)`; `build()` iterates
  users in sorted order. The eval cache makes evaluation reproducible regardless.
  Train negatives use a separate sampler (`seed + 1`) so they never perturb the
  eval negative stream.
- `sampled_auc` returns `NaN` for an empty negative set (AUC undefined);
  `mean_metrics` ignores NaNs when averaging.

## Running tests

From the repo root:

```bash
./venv/bin/python -m pytest amazon_ranking/tests -q
```
