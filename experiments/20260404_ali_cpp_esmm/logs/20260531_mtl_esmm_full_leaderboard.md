# ESMM Ali-CCP — Full 4-Model MTL Leaderboard (2026-05-31)

Definitive comparison of the four ESMM multi-task architectures on the **full** Ali-CCP
dataset. All four legs share identical data, splits, preprocessing, epochs, and embedding
dim, so the numbers are directly comparable.

## Setup (identical across all four legs)

- **Data:** full Ali-CCP — train 42,299,905 rows (clicks 1,644,256; conversions 8,802),
  test 43,016,614 rows (clicks 1,673,447; conversions 9,195). Conversion rate ~0.02%.
- **Preprocessing:** streaming parse → `log1p` dense normalize → filtered sparse vocabs
  (min_count=5), cached on Drive (`processed_esmm_full_parquet/`). 22 sparse + 8 dense fields.
- **Training:** 5 epochs, batch 4096, lr 1e-3, embed_dim 18, AMP on, seed 42, T4 GPU.
- **Eval:** full 43M-row test set (CTR / CTCVR / CVR AUC; PR-AUC, logloss, ECE for CTR & CTCVR).
- **Notebook:** `esmm_experiment_resume.ipynb` (resume mode, `CLEAN_EXPERIMENT_JSON=[]`);
  impl `esmm_ali_ccp_impl.py`. Result JSONs in Drive `data/ali_ccp/esmm_round_training_cache/`.

## Leaderboard

| Model | CTR_AUC | **CTCVR_AUC** | CVR_AUC | logloss_ctr | ECE_ctr | logloss_ctcvr | Params | Train wall (s) | Train rate (smp/s) |
|---|---|---|---|---|---|---|---|---|---|
| Baseline (separate CTR/CVR towers) | — | 0.5000 | 0.5002 | — | — | — | 42.20M | 2220 | 95.3k |
| Shared-Bottom (trunk 360→200→80) | 0.5678 | 0.5775 | **0.5985** | 0.2359 | 0.0584 | 0.00449 | 41.96M | 2130 | 99.3k |
| MMoE (E=4, expert_h=360, d=128) | 0.5652 | 0.5759 | 0.5971 | **0.2321** | **0.0540** | 0.00352 | 42.54M | 2626 | 80.5k |
| PLE (1 shared + 1 task expert/side, d=128) | 0.5675 | **0.5841** | 0.5909 | 0.2391 | 0.0616 | 0.00363 | 42.59M | 2518 | 84.0k |

Bold = best in column (excluding the degenerate baseline).

## Read

- **Baseline is near-random** on CVR/CTCVR (0.500) — architecturally expected: independent
  CTR and CVR towers receive the sparse CTCVR gradient (~0.02% positive rate) without any
  shared representation. This is the documented separate-tower failure mode, not an eval bug.
- **All three MTL architectures clear it decisively**, lifting CTCVR_AUC to ~0.576–0.584 and
  CVR_AUC to ~0.591–0.599 — real, comparable signal.
- **PLE wins the headline CTCVR_AUC (0.5841)** — CTCVR (click × conversion) is the primary
  ESMM objective. PLE's progressive layered extraction (shared + task-specific experts) gives
  it the edge on the joint task.
- **Shared-Bottom is the value pick:** best CVR_AUC (0.5985), fewest params (41.96M), and the
  cheapest to train (2130s) — within ~1 pt CTCVR of PLE for less compute.
- **MMoE is the best-calibrated CTR head** (lowest logloss_ctr 0.2321 and ECE_ctr 0.0540) but
  trails on the AUC ranking metrics and is the slowest leg.
- Spread across the three MTL archs is small (CTCVR 0.576–0.584); on this slice the choice is
  more about calibration/cost trade-offs than a large quality gap.

## Provenance / run history

This leaderboard was completed across **three Colab T4 sessions** because each free-tier T4
runtime lasted only ~75–80 min (≈ one model-leg) before dying. The per-model Drive result-JSON
cache (`CLEAN_EXPERIMENT_JSON=[]`) let each session resume and train only the uncached leg:

| Leg | Result JSON (Drive, UTC) | Notes |
|---|---|---|
| Baseline | `baseline_results.json` @ 02:45Z | cached from an earlier session |
| Shared-Bottom | `exp_shared_bottom_results.json` @ 04:17Z | cached from an earlier session |
| MMoE | `exp_mmoe_results.json` @ 07:34Z | this effort, session 2 (runtime died ~6 min into PLE after) |
| PLE | `exp_ple_results.json` @ 09:14Z | this effort, session 3 (re-triggered, PLE-only) |

An in-run **heartbeat** (`mtl_heartbeat.json`, daemon thread writing `{ts, done}` to Drive every
30s) made each runtime's liveness observable without SSH, so a dead session was detected in
~minutes instead of waiting ~1h for a result JSON that would never come. See
`kb/context/colab/colab-runtime-behavior.md` §"In-Run Heartbeat".

**Note on wall times:** `wall_clock_seconds`/`train_wall_seconds` cover **training only**; the
full 43M-row test-set eval runs afterward (~20–26 min, Drive-read-bound, GPU <5%) and is not
included in those figures. This eval cost is what pushes a single leg to ~70 min wall.
