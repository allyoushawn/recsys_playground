# Trail: experiment_dcn_ple_rtm — MAE &lt; 0.8 push

**Date:** 2026-03-28  
**Notebook:** `notebooks/ad_hoc/experiment_dcn_ple_rtm.ipynb`  
**Goal:** Test **MAE &lt; 0.8** on the same held-out split as the notebook (`train_test_split(..., random_state=42)`).  
**Colab:** `gotta-daughters-totals-workshops.trycloudflare.com` (SSH OK, Tesla T4).

## Decisions (lead)

- **Primary goal:** MAE only; R² reported as diagnostic (`RTM_GOAL_R2` unchanged).
- **Round 6 (N–P):** Huber loss in `train_ple_mtl` (matches prior winning recipe spirit); strict **y=1 / y=5** masks; **N** = P90/P10 percentile routing on general head (train-only thresholds); **O** = same PLE checkpoint as N + **tail_strict** router (class 0 = rating 1, 1 = 2–4, 2 = 5); **P** = wider PLE (embed 48, d_model 96), 60 epochs, tail_strict router.
- **Code:** `rating_to_router_class_tail_strict`, `train_router(..., router_label_mode='tail_strict')`, `regression_loss='huber'` in `train_ple_mtl`.
- **Round 7:** Hypothesis **S** (multi-task PLE+CE) rejected **OVER_BUDGET**; **Q** and **R** accepted and run (canonical).
- **Round 8:** Hypothesis **U** (AdamW + warmup + cosine) rejected **OVER_BUDGET**; **T** and **V** accepted and run (canonical). Config: `SKIP_ROUND_7=True`, `SKIP_ROUND_8=False`. **T** = J-style PLE + legacy router, **masked L1** training. **V** = J-graph, dropout 0.22, weight decay 5e-4, ~55% PLE/router epochs, SmoothL1.
- **Round 9:** Hypothesis **W** rejected **TECHNICALLY_UNSOUND** (no J checkpoint); **U** and **X** accepted and run (canonical). Config: `SKIP_ROUND_8=True`, `SKIP_ROUND_9=False`.

## Round log

| Round | Status | Notes |
|-------|--------|--------|
| 6 | completed | Colab success, ~11.74 min; cache `round_6_results.json` under `CACHE_DIR`. **Config S (repro):** `SKIP_ROUND_1-5 True`. |
| 7 | completed | Colab success, ~5.3 min; **S** not run (OVER_BUDGET); **Q**, **R** canonical. |
| 8 | completed | Colab success, ~5.11 min; **U** not run (OVER_BUDGET); **T**, **V** canonical. Config: `SKIP_ROUND_7=True`, `SKIP_ROUND_8=False`. |
| 9 | completed | Colab success, ~6.43 min; **W** not run (TECHNICALLY_UNSOUND); **U**, **X** canonical. Config: `SKIP_ROUND_8=True`, `SKIP_ROUND_9=False`. |

## Leaderboard

_Append after runtime with comparability `canonical` unless operational downgrade._

| Exp | MAE | R² | sigma_ratio | Comparability | Notes |
|-----|-----|-----|-------------|---------------|-------|
| N | 0.9415 | -0.2154 | 0.805 | canonical | Round 6; config S (`SKIP_ROUND_1-5 True`). |
| O | 0.8786 | -0.2770 | 0.824 | canonical | Round 6; config S. |
| P | 0.8842 | -0.3144 | 0.851 | canonical | Round 6; config S. |
| Q | 0.8804 | -0.1683 | 0.737 | canonical | Round 7. |
| R | 0.8905 | -0.1582 | 0.748 | canonical | Round 7. |
| T | 0.8763 | -0.4797 | 0.866 | canonical | Round 8; masked L1, J-style PLE + legacy router. |
| V | 0.8727 | -0.4705 | 0.863 | canonical | Round 8; J-graph, dropout 0.22, wd 5e-4, SmoothL1. |
| U | 0.8654 | -0.4503 | 0.841 | canonical | Round 9. |
| X | 0.8740 | -0.4782 | 0.846 | canonical | Round 9. |

**Global best:** **U** — MAE **0.8654** (round 9). Prior notebook-history best **J** — MAE **0.8707** (cached round 4). **U** −0.0053 MAE vs **J**. Round 8 **V** / **T** remain on table above; round 9 **X** MAE 0.8740 (+0.0086 vs **U** on this run).

### Round 6 — results

| Exp | MAE | R² | sigma_ratio | comparability |
|-----|-----|-----|-------------|---------------|
| N | 0.9415 | -0.2154 | 0.805 | canonical |
| O | 0.8786 | -0.2770 | 0.824 | canonical |
| P | 0.8842 | -0.3144 | 0.851 | canonical |

_Runtime: success, ~11.74 min. Goal MAE &lt; 0.8 not met; best this round O (0.8786) vs notebook best J (0.8707)._

### Round 7 — results

| Exp | MAE | R² | sigma_ratio | comparability |
|-----|-----|-----|-------------|---------------|
| Q | 0.8804 | -0.1683 | 0.737 | canonical |
| R | 0.8905 | -0.1582 | 0.748 | canonical |

_Runtime: success, ~5.3 min. Goal MAE &lt; 0.8 not met; best this round Q (0.8804) vs notebook best J (0.8707). Hypothesis S rejected OVER_BUDGET (critic); only Q and R executed._

### Round 8 — results

| Exp | MAE | R² | sigma_ratio | comparability |
|-----|-----|-----|-------------|---------------|
| T | 0.8763 | -0.4797 | 0.866 | canonical |
| V | 0.8727 | -0.4705 | 0.863 | canonical |

_Runtime: success, ~5.11 min. Goal MAE &lt; 0.8 not met; global best remains J (0.8707). **V** −0.0036 MAE vs **T** on this run; **V** +0.0020 vs J; **T** +0.0056 vs J. Hypothesis **U** rejected OVER_BUDGET (critic); only **T** and **V** executed._

### Round 9 — results

| Exp | MAE | R² | sigma_ratio | comparability |
|-----|-----|-----|-------------|---------------|
| U | 0.8654 | -0.4503 | 0.841 | canonical |
| X | 0.8740 | -0.4782 | 0.846 | canonical |

_Runtime: success, ~6.43 min. Goal MAE &lt; 0.8 not met. **U** MAE 0.8654 vs **J** 0.8707 (−0.0053 MAE); **X** MAE 0.8740 (+0.0086 MAE vs **U**). Hypothesis **W** rejected TECHNICALLY_UNSOUND (critic); only **U** and **X** executed._
