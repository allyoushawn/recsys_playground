# Trail: experiment_dcn_ple_rtm — MAE &lt; 0.8 push

**Date:** 2026-03-28  
**Notebook:** `notebooks/ad_hoc/experiment_dcn_ple_rtm.ipynb`  
**Goal:** Test **MAE &lt; 0.8** on the same held-out split as the notebook (`train_test_split(..., random_state=42)`).  
**Colab:** `tests-challenges-kids-edge.trycloudflare.com` (SSH OK, Tesla T4).

## Decisions (lead)

- **Primary goal:** MAE only; R² reported as diagnostic (`RTM_GOAL_R2` unchanged).
- **Round 6 (N–P):** Huber loss in `train_ple_mtl` (matches prior winning recipe spirit); strict **y=1 / y=5** masks; **N** = P90/P10 percentile routing on general head (train-only thresholds); **O** = same PLE checkpoint as N + **tail_strict** router (class 0 = rating 1, 1 = 2–4, 2 = 5); **P** = wider PLE (embed 48, d_model 96), 60 epochs, tail_strict router.
- **Code:** `rating_to_router_class_tail_strict`, `train_router(..., router_label_mode='tail_strict')`, `regression_loss='huber'` in `train_ple_mtl`.

## Round log

| Round | Status | Notes |
|-------|--------|--------|
| 6 | pending | Requires Colab run after `git pull` of `main`; cache key `round_6_results.json` under `CACHE_DIR`. |

## Leaderboard

_Append after runtime with comparability `canonical` unless operational downgrade._
