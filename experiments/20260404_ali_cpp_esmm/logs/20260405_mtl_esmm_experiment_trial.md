# MTL ESMM (MMoE/PLE) — trial log

**Notebook:** `experiments/20260404_ali_cpp_esmm/20260404_esmm_experiment.ipynb`  
**Goal:** CTCVR_AUC > 0.5917 (baseline K canonical 0.5917)

## Leaderboard

| experiment | CTCVR_AUC | CVR_AUC | CTR_AUC | wall_train_s | comparability | notes |
|------------|-----------|---------|---------|--------------|---------------|-------|
| R5_PLE | — | — | — | — | non_comparable | Failed CODE: CUDA BCE domain assert (`input_val` out of [0,1]) at `scaler.step(optimizer)`; no metrics; `exit_code: 1`. |
| R5_MMoE | 0.6096 | 0.6756 | 0.6055 | ~2932.9 | canonical | Success on retry 1 (attempt 0 INFRA broken pipe); Tesla T4; mmoe_params 42538012. |
| R5_SharedBottom | 0.5925 | 0.6298 | 0.6076 | ~1956.7 | canonical | Success `exit_code: 0`; retries 0; Tesla T4 (~14.6 GB). |
| R6_K_ref | 0.4873 | 0.4929 | 0.5000 | ~2717 | non_comparable | Divergence vs canonical K 0.5917; possible cache/data/seed. |
| R6_SharedBottom | 0.5774 | 0.6005 | 0.6042 | ~2082 | canonical | |
| R6_MMoE | 0.5949 | 0.6412 | 0.6038 | ~2171 | canonical | Round 1 run. |
| R6_PLE | — | — | — | — | non_comparable | Training crashed (BCE domain); fix applied in notebook after run; full metrics pending. Follow-up run INFRA (broken pipe / bad handshake); PLE re-run not completed. |
| R6_PLE_fix | 0.5000 | 0.5000 | 0.5000 | ~1719 | operational_downgrade | Papermill exit 0 ~49 min; PLE no crash; CTCVR/CVR/CTR AUC all 0.5000; subagent noted likely constant preds / masking. Host `remarks-reviewing-amp-verse.trycloudflare.com`. |
| R7_PLE_fp32 | 0.5896 | 0.6178 | 0.6084 | ~2388.5 | canonical | Papermill exit 0 ~60 min after fp32 PLE forward + level stabilizers; log `experiments/20260404_ali_cpp_esmm/logs/papermill_r7_ple_fp32.log`. Same host. |
| R3_R6_K_ref | 0.5000 | 0.5000 | 0.6081 | ~1968 | canonical | Legacy CTCVR_AUC 0.5000; vs Round 4 K cache note in runtime (suspicious). |
| R3_R6_SharedBottom | 0.5740 | 0.6059 | 0.6040 | ~1868 | canonical | Legacy CTCVR_AUC 0.5740. |
| R3_R6_MMoE | **0.6164** | 0.6620 | 0.6086 | ~2120 | canonical | Legacy CTCVR_AUC 0.6164; best CTCVR_AUC this runtime; +0.0215 vs prior leaderboard R6_MMoE 0.5949. |
| R3_R6_PLE | — | — | — | — | non_comparable | Papermill aborted: AcceleratorError CUDA device-side assert in GradScaler.step; no metrics. |
| R3_R7 | — | — | — | — | non_comparable | Not reached (runtime failed during R6 PLE; K_ref/PLE legs for R7 not run). |
| R4_baseline | 0.6114 | 0.6458 | — | ~1961 | canonical | 2026-04-08 Colab; Tesla T4; full-data 5 epochs; SHARED_BOTTOM/MMoE/PLE skipped; `scp` notebook + `esmm_ali_ccp_impl.py`; host `ant-gordon-downloading-employed.trycloudflare.com`. |

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

---

## Round 4 (runtime-only — baseline)

**Context:** Lead delegated **experiment-runtime** to execute the notebook end-to-end on Colab with a fresh tunnel. **Host:** `ant-gordon-downloading-employed.trycloudflare.com`.

### Planner / Critic / Code-change

- Skipped (not run this round).

### Runtime

- **Status:** success (`exit_code: 0`).
- **Wall clock (papermill):** ~49.4 min.
- **GPU:** Tesla T4 (~14.6 GB).
- **Sync:** `scp` of `experiments/20260404_ali_cpp_esmm/20260404_esmm_experiment.ipynb` and `experiments/20260404_ali_cpp_esmm/esmm_ali_ccp_impl.py` to `/content/drive/MyDrive/colab/recsys_playground/recsys_playground`; no git sync.
- **Notebook:** `[SKIP_GIT_REPO_SYNC]` path respected; pip install cell exit 0; device **cuda**.
- **Config this run:** **BASELINE** leg only; SHARED_BOTTOM / MMoE / PLE not executed.
- **Output (remote):** `experiments/20260404_ali_cpp_esmm/20260404_esmm_experiment_output.ipynb`.
- **Retries:** 0.

### Results

| run | CTCVR_AUC | CVR_AUC | CTR_AUC | wall_train_s | comparability |
|-----|-----------|---------|---------|--------------|---------------|
| R4_baseline | 0.6114 | 0.6458 | — | ~1961 | canonical |

**Auxiliary (cell log):** train_samples ≈ 211,499,525; train_throughput ≈ 108,305 samples/s; model_params ≈ 42,204,292; final train loss ≈ 0.1460.

### Observations

- R4_baseline CTCVR_AUC 0.6114 vs goal line CTCVR_AUC > 0.5917 (+0.0197 margin vs threshold 0.5917).
- R4_baseline CTCVR_AUC 0.6114 vs R3_R6_MMoE CTCVR_AUC 0.6164 (−0.0050 on CTCVR_AUC); legs differ (single baseline track vs Round 6 MMoE).
- R4_baseline CVR_AUC 0.6458 vs R3_R6_MMoE CVR_AUC 0.6620 (−0.0162 on CVR_AUC).
- R4_baseline wall_train ~1961 s vs R3_R6_MMoE ~2120 s (−159 s).

### SCRIBE_OUTPUT (mechanical)

```yaml
round: 4
goal_achieved: true
goal_metric: CTCVR_AUC
goal_threshold: 0.5917
best_this_round:
  experiment: R4_baseline
  CTCVR_AUC: 0.6114
leaderboard_rows_added: 1
scribe_doc_path: experiments/20260404_ali_cpp_esmm/logs/20260405_mtl_esmm_experiment_trial.md
```

---

## Round 5 (runtime-only — operational toggles)

**Context:** Lead skipped planner, critic, and code-change; runtime executed three sequential Colab runs on host `puts-pubs-flag-streams.trycloudflare.com`. Synced `experiments/20260404_ali_cpp_esmm/20260404_esmm_experiment.ipynb` and `experiments/20260404_ali_cpp_esmm/esmm_ali_ccp_impl.py` via `scp` before each run.

### Planner / Critic / Code-change

- Skipped (not run this round).

### Runtime

- **R5_PLE (`CLEAN_EXPERIMENT_JSON=['ple']`):** failed (**CODE**), papermill ~19.3 min, Tesla T4. After PLE epoch 1/5 (~loss 0.1628), CUDA BCE assert `input_val >= zero && input_val <= one` in `Loss.cu` surfaced as device-side assert at `scaler.step(optimizer)` in `esmm_ali_ccp_impl.py`; no CTCVR/CVR/CTR AUC printed; `exit_code: 1`.
- **R5_MMoE (`CLEAN_EXPERIMENT_JSON=['mmoe']`):** success (`exit_code: 0`) on **retry 1** after first attempt failed **INFRA** (SSH broken pipe mid-epoch 1). Successful run ~72.9 min, Tesla T4. Metrics: CTCVR_AUC 0.6096, CVR_AUC 0.6756, CTR_AUC 0.6055, wall_train_s ~2932.9, `mmoe_params` 42538012.
- **R5_SharedBottom (`CLEAN_EXPERIMENT_JSON=['shared_bottom']`):** success (`exit_code: 0`), retries 0, Tesla T4 (~14.6 GB). Metrics: CTCVR_AUC 0.5925, CVR_AUC 0.6298, CTR_AUC 0.6076, wall_train_s ~1956.7.

### Results

| run | CTCVR_AUC | CVR_AUC | CTR_AUC | wall_train_s | comparability |
|-----|-----------|---------|---------|--------------|---------------|
| R5_PLE | — | — | — | — | non_comparable |
| R5_MMoE | 0.6096 | 0.6756 | 0.6055 | ~2932.9 | canonical |
| R5_SharedBottom | 0.5925 | 0.6298 | 0.6076 | ~1956.7 | canonical |

### Observations

- R5_MMoE CTCVR_AUC 0.6096 vs R5_SharedBottom 0.5925 (+0.0171 on CTCVR_AUC); R5_MMoE CVR_AUC 0.6756 vs R5_SharedBottom 0.6298 (+0.0458 on CVR_AUC); R5_MMoE CTR_AUC 0.6055 vs R5_SharedBottom 0.6076 (−0.0021 on CTR_AUC).
- R5_MMoE CTCVR_AUC 0.6096 vs goal threshold 0.5917 (+0.0179 margin); R5_SharedBottom CTCVR_AUC 0.5925 vs threshold 0.5917 (+0.0008 margin).
- R5_MMoE CTCVR_AUC 0.6096 vs R4_baseline 0.6114 (−0.0018 on CTCVR_AUC); R5_MMoE CVR_AUC 0.6756 vs R4_baseline 0.6458 (+0.0298 on CVR_AUC).
- R5_MMoE CTCVR_AUC 0.6096 vs R3_R6_MMoE 0.6164 (−0.0068 on CTCVR_AUC); R5_MMoE CVR_AUC 0.6756 vs R3_R6_MMoE 0.6620 (+0.0136 on CVR_AUC).
- R5_MMoE wall_train ~2932.9 s vs R5_SharedBottom ~1956.7 s (+976.2 s).

### SCRIBE_OUTPUT (mechanical)

```yaml
round: 5
leaderboard_rows_added: 3
goal_achieved: true
goal_metric: CTCVR_AUC
goal_threshold: 0.5917
best_new_comparable_row_this_batch:
  experiment: R5_MMoE
  CTCVR_AUC: 0.6096
scribe_doc_path: experiments/20260404_ali_cpp_esmm/logs/20260405_mtl_esmm_experiment_trial.md
```

---

## Round 6–7 (lead: PLE CUDA BCE assert + degenerate 0.5 AUC)

**Context:** Fix PLE path under AMP (BCE domain assert) and follow up when multitask metrics collapsed to chance AUC.

### Code summary (`esmm_ali_ccp_impl.py`, local)

- **`_esmm_multitask_bce_from_probs`:** float32 cast, clamp, `nan_to_num` for multitask BCE inside `train_esmm_parquet_rowgroups` (AMP-safe).
- **`ESMM_PLE`:** `nan_to_num` + clamp on level1/level2; on CUDA, forward runs nested `autocast(enabled=False)` so the PLE body executes in fp32 for stability.

### Runtime

- **Colab host:** `remarks-reviewing-amp-verse.trycloudflare.com`
- **Run A — R6_PLE_fix:** `papermill` exit 0, ~49 min wall; PLE trained without crash. Metrics: CTCVR_AUC = CVR_AUC = CTR_AUC = 0.5000; `wall_train_s` ~1719. Retry used remote `python3 -m papermill` after `pip install papermill`.
- **Run B — R7_PLE_fp32:** After fp32 PLE forward + level stabilizers; `papermill` exit 0, ~60 min. **CTCVR_AUC=0.5896, CVR_AUC=0.6178, CTR_AUC=0.6084**; `wall_train_s` ~2388.5. Log: `experiments/20260404_ali_cpp_esmm/logs/papermill_r7_ple_fp32.log`.

### Results

| run | CTCVR_AUC | CVR_AUC | CTR_AUC | wall_train_s | comparability |
|-----|-----------|---------|---------|--------------|---------------|
| R6_PLE_fix | 0.5000 | 0.5000 | 0.5000 | ~1719 | operational_downgrade |
| R7_PLE_fp32 | 0.5896 | 0.6178 | 0.6084 | ~2388.5 | canonical |

### Observations

- R7_PLE_fp32 CTCVR_AUC 0.5896 vs goal threshold 0.5917 (−0.0021 on CTCVR_AUC).
- R7_PLE_fp32 CTCVR_AUC 0.5896 vs R5_MMoE CTCVR_AUC 0.6096 (−0.0200 on CTCVR_AUC); R7_PLE_fp32 CVR_AUC 0.6178 vs R5_MMoE CVR_AUC 0.6756 (−0.0578 on CVR_AUC); R7_PLE_fp32 CTR_AUC 0.6084 vs R5_MMoE CTR_AUC 0.6055 (+0.0029 on CTR_AUC).
- R7_PLE_fp32 CTCVR_AUC 0.5896 vs R5_SharedBottom CTCVR_AUC 0.5925 (−0.0029 on CTCVR_AUC); R7_PLE_fp32 CVR_AUC 0.6178 vs R5_SharedBottom CVR_AUC 0.6298 (−0.0120 on CVR_AUC).
- R7_PLE_fp32 CTCVR_AUC 0.5896 vs R6_PLE_fix CTCVR_AUC 0.5000 (+0.0896 on CTCVR_AUC); R7_PLE_fp32 CVR_AUC 0.6178 vs R6_PLE_fix 0.5000 (+0.1178 on CVR_AUC); R7_PLE_fp32 CTR_AUC 0.6084 vs R6_PLE_fix 0.5000 (+0.1084 on CTR_AUC).
- R7_PLE_fp32 `wall_train_s` ~2388.5 s vs R6_PLE_fix ~1719 s (+669.5 s).

### SCRIBE_OUTPUT (mechanical)

```yaml
round: 7
doc_path: experiments/20260404_ali_cpp_esmm/logs/20260405_mtl_esmm_experiment_trial.md
current_best:
  experiment: R3_R6_MMoE
  metric_name: CTCVR_AUC
  value: 0.6164
goal_achieved: false
observations_added:
  - "R7_PLE_fp32 CTCVR_AUC 0.5896 vs goal 0.5917: −0.0021"
  - "R7_PLE_fp32 vs R5_MMoE CTCVR_AUC: −0.0200; vs R5_SharedBottom CTCVR_AUC: −0.0029"
  - "R7_PLE_fp32 vs R6_PLE_fix (0.5 AUCs): +0.0896 CTCVR_AUC; wall_train +669.5 s"
leaderboard_rows_appended:
  - experiment: R6_PLE_fix
    metrics:
      CTCVR_AUC: 0.5000
      CVR_AUC: 0.5000
      CTR_AUC: 0.5000
      wall_train_s: 1719
    comparability: operational_downgrade
  - experiment: R7_PLE_fp32
    metrics:
      CTCVR_AUC: 0.5896
      CVR_AUC: 0.6178
      CTR_AUC: 0.6084
      wall_train_s: 2388.5
    comparability: canonical
auto_unblock_notes: []
```
