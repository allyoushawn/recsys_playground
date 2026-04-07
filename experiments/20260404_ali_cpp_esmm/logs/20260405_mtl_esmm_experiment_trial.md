# MTL ESMM (MMoE/PLE) — trial log

**Notebook:** `experiments/20260404_ali_cpp_esmm/20260404_esmm_experiment.ipynb`  
**Goal:** CTCVR_AUC > 0.5917 (baseline K canonical 0.5917)

## Leaderboard

| experiment | CTCVR_AUC | CVR_AUC | CTR_AUC | wall_train_s | comparability | notes |
|------------|-----------|---------|---------|--------------|---------------|-------|
| R6_K_ref | 0.4873 | 0.4929 | 0.5000 | ~2717 | non_comparable | Divergence vs canonical K 0.5917; possible cache/data/seed. |
| R6_SharedBottom | 0.5774 | 0.6005 | 0.6042 | ~2082 | canonical | |
| R6_MMoE | 0.5949 | 0.6412 | 0.6038 | ~2171 | canonical | Round 1 run. |
| R6_PLE | — | — | — | — | non_comparable | Training crashed (BCE domain); fix applied in notebook after run; full metrics pending. Follow-up run INFRA (broken pipe / bad handshake); PLE re-run not completed. |
| R3_R6_K_ref | 0.5000 | 0.5000 | 0.6081 | ~1968 | canonical | Legacy CTCVR_AUC 0.5000; vs Round 4 K cache note in runtime (suspicious). |
| R3_R6_SharedBottom | 0.5740 | 0.6059 | 0.6040 | ~1868 | canonical | Legacy CTCVR_AUC 0.5740. |
| R3_R6_MMoE | **0.6164** | 0.6620 | 0.6086 | ~2120 | canonical | Legacy CTCVR_AUC 0.6164; best CTCVR_AUC this runtime; +0.0215 vs prior leaderboard R6_MMoE 0.5949. |
| R3_R6_PLE | — | — | — | — | non_comparable | Papermill aborted: AcceleratorError CUDA device-side assert in GradScaler.step; no metrics. |
| R3_R7 | — | — | — | — | non_comparable | Not reached (runtime failed during R6 PLE; K_ref/PLE legs for R7 not run). |

---

## Round 1

### Planner

- **A:** Eval metrics.
- **B:** SharedBottom / MMoE / PLE + `model_ctor`.
- **C:** Rejected — `LOW_EXPECTED_IMPACT`; defer λ.

### Critic

- Accepted **A**, **B**.
- `budget_ok`: false (~150 min estimated).

### Code-change

- Round 6 cell: `ESMM_SharedBottom`, `ESMM_MMoE`, `ESMM_PLE`, streaming multitask eval, `train_esmm_parquet_rowgroups` `model_ctor`.
- Second change: clamp `p_ctcvr` + `nan_to_num` on PLE level2 (fix CUDA BCE under AMP).

### Runtime

- **Log:** `20260406_esmm_r1_papermill_retry1` (attempt with completion before PLE crash).

| run | CTCVR_AUC | CVR_AUC | CTR_AUC | wall_train_s | comparability |
|-----|-----------|---------|---------|--------------|---------------|
| R6_K_ref | 0.4873 | 0.4929 | 0.5000 | ~2717 | non_comparable |
| R6_SharedBottom | 0.5774 | 0.6005 | 0.6042 | ~2082 | canonical |
| R6_MMoE | 0.5949 | 0.6412 | 0.6038 | ~2171 | canonical |
| R6_PLE | — | — | — | — | non_comparable (crash; no full metrics) |

- **Follow-up:** INFRA failure (broken pipe, bad handshake); partial logs only; PLE re-run not completed.

### Observations

- R6_K_ref CTCVR_AUC 0.4873 vs baseline K canonical 0.5917 (−0.1044 on CTCVR_AUC).
- R6_SharedBottom CTCVR_AUC 0.5774 vs R6_K_ref 0.4873 (+0.0901 on CTCVR_AUC).
- R6_MMoE CTCVR_AUC 0.5949 vs R6_SharedBottom 0.5774 (+0.0175 on CTCVR_AUC); R6_MMoE CVR_AUC 0.6412 vs R6_SharedBottom 0.6005 (+0.0407 on CVR_AUC).
- R6_MMoE CTCVR_AUC 0.5949 vs baseline K canonical 0.5917 (+0.0032 on CTCVR_AUC), under canonical comparability for SharedBottom/MMoE rows only.
- R6_K_ref wall_train ~2717 s vs R6_SharedBottom ~2082 s (+635 s); R6_MMoE ~2171 s vs R6_SharedBottom ~2082 s (+89 s).

---

## Round 6 rerun — root cause for “broken” K_ref (2026-04-06)

**Likely cause:** Round 6 passed `K_EARLY_STOP_MAX_WALL_SECONDS` / related caps through `_R6_TRAIN_KW`. On Colab, if those were set for dev (e.g. 900s throughput legs in Round 5), Experiment K itself uses `None` in the committed notebook — but any **local edit** to the config cell would truncate Round 6 training. **~2717 s train wall vs ~7094 s canonical** and **CTR_AUC ≈ 0.5** match **severely under-trained** ESMM (constant-ish predictions).

**Fix in notebook (Round 6 cell):** `_R6_TRAIN_KW` now sets `max_wall_seconds`, `max_optimizer_steps`, `max_batches_per_epoch`, and `max_row_groups_per_epoch` to **`None` always**, so MTL legs always run **full 5 epochs** regardless of `K_EARLY_STOP_*`.

**Eval cross-check:** After multitask metrics, each leg also runs `evaluate_esmm_ctcvr_streaming_parquet` and `evaluate_esmm_cvr_streaming_parquet` (same as Round 4 K) and stores `CTCVR_AUC_legacy_r4` / `CVR_AUC_legacy_r4`.

**Remote run:** Notebook synced via `scp` to host `milwaukee-homes-rate-expected.trycloudflare.com`; `round_6_results.json` removed if present; `papermill` started in background with log `/tmp/papermill_esmm_mtl.log` on the Colab VM. **Poll with:**  
`ssh … 'tail -f /tmp/papermill_esmm_mtl.log'`  
Round 4 cache on that runtime reports **K: CTCVR_AUC=0.5851** (for comparison after Round 6 finishes).

---

## Round 2

### Planner

- **D:** R7 K_ref + PLE.
- **E:** λ grid — rejected `OVER_BUDGET`.
- **F:** cosine — rejected `OVER_BUDGET`.

### Critic

- Accepted **D** only.
- `budget_ok`: false.
- `execution_route`: `code_change_then_runtime`.

### Code-change

- New Round 7 cell: writes `round_7_results.json`; legs **K_ref** + **PLE**; `train` kwargs mirror R6 (no `K_EARLY_STOP` caps); multitask + legacy eval.

### Runtime

- **Status:** failed **INFRA**.
- **Cause:** hostname `milwaukee-homes-rate-expected.trycloudflare.com` DNS **NXDOMAIN** (tunnel expired). No `scp`, no `papermill`.
- **Action required:** user must supply a fresh trycloudflare hostname and re-run runtime.

### Results

| run | CTCVR_AUC | CVR_AUC | CTR_AUC | wall_train_s | comparability |
|-----|-----------|---------|---------|--------------|---------------|
| — | — | — | — | — | N/A (runtime did not execute) |

### Observations

- Round 2 produced no new metric rows; leaderboard unchanged vs prior round.

---

## Round 3 (runtime-only)

**Context:** Lead skipped planner, critic, and code-change; delegated **experiment-runtime** only to run Round 6 + Round 7 after deleting `round_6` / `round_7` JSON caches. **Host:** `farms-kathy-todd-difficulties.trycloudflare.com`.

### Planner / Critic / Code-change

- Skipped (not run this round).

### Runtime

- **Status:** failed (**CODE**).
- **Failure:** `PapermillExecutionError` Round 6 **PLE** — `AcceleratorError` CUDA device-side assert in `GradScaler.step` during `train_esmm_parquet_rowgroups`.
- **Round 6 completed legs before crash:**
  - **K_ref:** CTCVR_AUC 0.5000, CVR_AUC 0.5000, CTR_AUC 0.6081, legacy CTCVR 0.5000, wall ~1968 s — note suspicious vs Round 4 K.
  - **SharedBottom:** CTCVR_AUC 0.5740, CVR_AUC 0.6059, CTR_AUC 0.6040, legacy 0.5740, wall ~1868 s.
  - **MMoE:** CTCVR_AUC 0.6164, CVR_AUC 0.6620, CTR_AUC 0.6086, legacy 0.6164, wall ~2120 s — best CTCVR_AUC this run.
  - **PLE:** crash; no metrics.
- **Round 7:** not reached (K_ref and PLE for R7 not run).
- **`round_6_results.json`:** not written (papermill aborted).
- **Elapsed:** ~235 min to failure.

### Results

| run | CTCVR_AUC | CVR_AUC | CTR_AUC | wall_train_s | comparability |
|-----|-----------|---------|---------|--------------|---------------|
| R3_R6_K_ref | 0.5000 | 0.5000 | 0.6081 | ~1968 | canonical |
| R3_R6_SharedBottom | 0.5740 | 0.6059 | 0.6040 | ~1868 | canonical |
| R3_R6_MMoE | 0.6164 | 0.6620 | 0.6086 | ~2120 | canonical |
| R3_R6_PLE | — | — | — | — | non_comparable |
| R3_R7 | — | — | — | — | non_comparable |

### Observations

- R3_R6_K_ref CTCVR_AUC 0.5000 vs R6_K_ref Round 1 CTCVR_AUC 0.4873 (+0.0127 on CTCVR_AUC).
- R3_R6_SharedBottom CTCVR_AUC 0.5740 vs R3_R6_K_ref 0.5000 (+0.0740 on CTCVR_AUC); R3_R6_SharedBottom CVR_AUC 0.6059 vs R3_R6_K_ref 0.5000 (+0.1059 on CVR_AUC).
- R3_R6_MMoE CTCVR_AUC 0.6164 vs R3_R6_SharedBottom 0.5740 (+0.0424 on CTCVR_AUC); R3_R6_MMoE CVR_AUC 0.6620 vs R3_R6_SharedBottom 0.6059 (+0.0561 on CVR_AUC).
- R3_R6_MMoE CTCVR_AUC 0.6164 vs prior leaderboard R6_MMoE 0.5949 (+0.0215 on CTCVR_AUC).
- R3_R6_MMoE CTCVR_AUC 0.6164 vs goal baseline line CTCVR_AUC > 0.5917 (+0.0247 margin on CTCVR_AUC vs threshold 0.5917).
- R3_R6_K_ref wall_train ~1968 s vs R3_R6_SharedBottom ~1868 s (−100 s); R3_R6_MMoE ~2120 s vs R3_R6_SharedBottom ~1868 s (+252 s).
- `round_6_results.json` absent; PLE leg and Round 7 produced no comparable metric rows.
