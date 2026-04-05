# Experiment Trail Log — ESMM Reproduction

| Field | Value |
|-------|-------|
| **Notebook** | `notebooks/ad_hoc/20260404_esmm_experiment.ipynb` |
| **Goal metric** | CVR_AUC > 0.65 |
| **Date started** | 2026-04-04 |

---

## Leaderboard

| Rank | Experiment | Round | CVR_AUC | CTCVR_AUC | Wall clock (s) | Comparability |
|------|-----------|-------|---------|-----------|----------------|---------------|
| 1 | K | 4 | 0.6158 | 0.5917 | 7094 | canonical |
| 2 | J | 4 | 0.5869 | — | 337 | canonical |
| 3 | G | 3 | 0.5841 | — | 977 | canonical |
| 4 | A | 1 | 0.5622 | 0.5450 | 87 | canonical |
| 5 | H | 3 | 0.5537 | — | 89 | canonical |
| 6 | D | 2 | 0.5255 | 0.5084 | 2169 | canonical |

**Current best: K — CVR_AUC 0.6158 (goal not achieved)**

---

## Round 1

### Proposals

| ID | Description | Category | Requires code change |
|----|-------------|----------|---------------------|
| A | Faithful BASE CVR model with paper-exact architecture (Embed(18) → field-wise sum pool → concat dense → MLP 360→200→80→1, ReLU, sigmoid, BCE, Adam) trained on clicked-only impressions | Architecture changes | Yes |
| B | Same BASE architecture as A but increase SAMPLE_SIZE from 5M to 20M | Data quality and quantity | Yes |
| C | Same BASE architecture as A but apply pos_weight=19 in BCEWithLogitsLoss | Loss function alignment | Yes |

### Critic summary

- **Accepted:** A (requires code change)
- **Rejected:** B (OVER_BUDGET — loading 20M raw rows is I/O-bound at 15–20 min alone), C (LOW_EXPECTED_IMPACT — pos_weight does not improve AUC ranking; paper used standard BCE)
- **Execution route:** code_change_then_runtime

### Results

| Experiment | CVR_AUC | CTCVR_AUC | Wall clock (s) | Comparability |
|------------|---------|-----------|----------------|---------------|
| A | 0.5622 | 0.5450 | 87 | canonical |

- **Runtime status:** success
- **Elapsed:** 7.5 min
- **Preflight fixes:** fixed `total_mem` → `total_memory` attribute in `torch.cuda.get_device_properties()` diagnostic print
- **Retries used:** 1

### Observations

- A: CVR_AUC 0.5622, below goal of 0.65 (gap −0.0878).
- A: CTCVR_AUC 0.5450, both towers producing near-random ranking signal on first run.
- A: Wall clock 87 s indicates training completed quickly; compute budget is not a constraint at current sample size.

---

## Round 2

### Proposals

| ID | Description | Category | Requires code change |
|----|-------------|----------|---------------------|
| D | ESMM two-tower (CTR+CVR) with shared embeddings and entire-space multi-task loss pCTCVR=pCTR×pCVR | Loss function alignment | Yes |
| E | Increase SAMPLE_SIZE from 5M to 15M impressions | Data quality and quantity | Yes |
| F | Apply pos_weight in BCEWithLogitsLoss for CVR head | Training procedure | Yes |

### Critic summary

- **Accepted:** D (requires code change)
- **Rejected:** E (OVER_BUDGET — 15M parse + ESMM training reaches 25–30 min; also confounds comparison with A), F (REDUNDANT — identical to Round 1 hypothesis C, already rejected)
- **Execution route:** code_change_then_runtime

### Results

| Experiment | CVR_AUC | CTCVR_AUC | Wall clock (s) | Comparability |
|------------|---------|-----------|----------------|---------------|
| D | 0.5255 | 0.5084 | 2169 | canonical |

- **Runtime status:** success
- **Elapsed:** 44.5 min
- **Preflight fixes:** none
- **Retries used:** 0

### Observations

- D: CVR_AUC 0.5255, regression of −0.0367 vs A (0.5622); ESMM multi-task formulation underperformed single-task BASE.
- D: CTCVR_AUC 0.5084, near random; joint pCTCVR=pCTR×pCVR prediction not learning meaningful signal.
- D: Training loss decreased (0.1884→0.0854) indicating model fits training data but does not generalize to CVR evaluation set.
- D: Wall clock 2169 s (~36 min) vs A 87 s — 25× slower due to training on all 5M impressions (entire-space) rather than clicked-only subset.
- D: Data sparsity likely root cause — only ~1,446 conversion positives in 5M impressions, severely limiting CVR tower learning.

---

## Round 3 (FINAL)

### Proposals

| ID | Description | Category | Requires code change |
|----|-------------|----------|---------------------|
| G | Increase data to 15M and run BASE CVR model | Data quality and quantity | Yes |
| H | Replace BCE with focal loss (gamma=2, alpha=0.25) on BASE CVR model at current 5M | Loss function alignment | Yes |
| I | Increase to 10M with cosine LR and doubled epochs on BASE model | Training procedure | Yes |

### Critic summary

- **Accepted:** G (requires code change), H (requires code change)
- **Rejected:** I (REDUNDANT — partially redundant with G on data-scaling axis)
- **Execution route:** code_change_then_runtime

### Results

| Experiment | CVR_AUC | CTCVR_AUC | Wall clock (s) | Comparability |
|------------|---------|-----------|----------------|---------------|
| G | 0.5841 | — | 977 | canonical |
| H | 0.5537 | — | 89 | canonical |

- **Runtime status:** success
- **Elapsed:** 23.6 min
- **Preflight fixes:** none
- **Retries used:** 0

### Observations

- G: 15M data improved CVR_AUC from 0.5622 to 0.5841 (+0.0219), confirming data volume as primary lever.
- G: 15M data yielded ~4,300 conversions (3× the 5M's ~1,446) — still far below paper's 18k.
- H: Focal loss (0.5537) underperformed standard BCE (0.5622) on 5M data.
- Overall: all results well below paper's BASE (0.6600), attributable to using ≤15M vs 84M impressions.

---

## Final Summary

Goal CVR_AUC > 0.65 was **not achieved** after 3 rounds. Best result: experiment G with CVR_AUC 0.5841 (gap −0.0659 to goal). Primary bottleneck identified: dataset size (≤15M vs paper's 84M impressions).

**Post–Round 4 update (2026-04-05):** Best CVR_AUC is now **K** at **0.6158** (gap −0.0342 to goal 0.65). See **Round 4 — executed** below.

---

## Round 4 — status (resume)

**Target:** Experiments **J** (BASE on full split + freq filter + batch 4096 + log1p) and **K** (ESMM, 5 epochs) using Parquet under `DATA_DIR/processed_esmm_full_parquet/`.

**2026-04-04 runtime attempt (earth-radius-controlling-switch):** **INFRA failure** — SSH / Cloudflare tunnel dropped mid-run (`Connection closed by remote host`, then `websocket: bad handshake`). Papermill had **not reached Round 4**; it was still executing earlier round cells when the session ended. No J/K metrics recorded.

**Preserved on Colab Drive (if unchanged):**

- `…/ali_ccp/processed_esmm_parsed_samples/` — cached 5M (and 15M if built) Parquet; notebook reloads from here.
- `…/ali_ccp/processed_esmm_full_parquet/` — full train/test Parquet after one successful full parse (avoids re-parsing raw CSV).
- `…/ali_ccp/esmm_round_training_cache/round_*.json` — delete a file to force re-run that round only.

**To resume:** Reconnect Colab + `cloudflared` tunnel, then re-run papermill (or open the notebook and **Run all**). Earlier rounds skip automatically when the JSON caches exist on Drive. Local stdout capture: `experiments/logs/20260404_esmm_r4_papermill_stdout.log`.

**2026-04-05 runtime attempt (`inch-utility-leading-spatial.trycloudflare.com`):** **INFRA** — SSH dropped during **Round 4 Cell 13** while parsing full split (`sample_skeleton_train` tqdm; no Parquet cache for `sample=full` yet). **Rounds 1–3** completed from **cached** JSON on Drive (printed: A CVR_AUC=0.5526; D CVR_AUC=0.5339; G CVR_AUC=0.5727; H CVR_AUC=0.5550 — values reflect remote cache, may differ slightly from earlier canonical trail rows). **J/K not finished.** Log: `experiments/logs/20260405_esmm_runtime_papermill.log`.

**Next run:** Refresh tunnel hostname if rotated → **scp** latest `20260404_esmm_experiment.ipynb` → **papermill**. Optional: in Config set `CLEAN_ROUND_RESULT_JSON = []` (keep caches for R1–3); only use cleanup flags if you need to invalidate artifacts. Once `parsed_train_rows_full.parquet` exists under `processed_esmm_full_parquet/`, Round 4 skips the long raw parse.

---

## Memory plan (OpenAI + Gemini consult, 2026-04-04)

**Problem:** Round 4 held full train/test DataFrames plus full float64-style tensor copies (~51GB RAM ceiling on Colab).

**Consultant alignment:** Both recommended (1) stop materializing full 42M-row training tensors for ESMM, (2) use compact dtypes, (3) serialize J then K without overlapping giant tensors, (4) stream or chunk from Parquet. OpenAI: store int32 on host, cast to `long` at batch for `nn.Embedding`. Gemini: same + IterableDataset / mmap if needed.

**Implemented in `20260404_esmm_experiment.ipynb`:**

| Change | Rationale |
|--------|-----------|
| `encode_and_tensorize` → int32 sparse, `torch.from_numpy` float32 labels | ~halves sparse host RAM |
| `BASEModel` / `ESMMModel` `forward`: `sparse_x = sparse_x.long()` | PyTorch Embedding expects long indices |
| Round 4: write `r4_norm_train/test.parquet` under `processed_esmm_full_parquet/` with `row_group_size=1_000_000`, then `del` DataFrames | Drops duplicate in-RAM frame before K |
| J: `read_parquet(..., filters=[('click','==',1)])` | Avoids loading full train into pandas for BASE |
| K: `train_esmm_parquet_rowgroups` — shuffle row-group order each epoch, tensorize per row group only | Peak RAM ≈ largest row group + chunk tensors, not full 42M |
| Test: single `encode_and_tensorize` for all rows; `evaluate_esmm_cvr_indexed` with boolean mask | Removes duplicate full-test tensorization |

**Not done (defer if still OOM):** Polars rewrite, `torch.load(mmap=True)` checkpoints, true multi-worker IterableDataset.

---

## Round 4 — executed (2026-04-05)

### Proposals

| ID | Description | Category | Requires code change |
|----|-------------|----------|---------------------|
| J | BASE on full split with frequency filter, batch 4096, log1p transforms; Parquet pipeline under `DATA_DIR/processed_esmm_full_parquet/` | Data quality / training setup | Yes (notebook) |
| K | ESMM (CTR+CVR), 5 epochs, row-group–wise training from Parquet | Architecture / training procedure | Yes (notebook) |
| L | Optional LR schedule (cosine/step), warmup, weight decay on BASE via `train_model` + `R4_BASE_*` toggles | Training procedure | Yes |

### Critic summary

- **Accepted:** J, K, L (L = optional LR schedule on BASE; defaults preserve prior J behavior)
- **Execution route:** code_change_then_runtime

### Code change (this round)

- `train_model` extended with optional weight decay, cosine/step LR, warmup; Round 4 `R4_BASE_*` toggles added.
- **Runtime:** defaults used → **L not activated**; run comparable to J recipe without LR-schedule variant.

### Results

| Experiment | CVR_AUC | CTCVR_AUC | Wall clock (s) | Comparability |
|------------|---------|-----------|----------------|---------------|
| J | 0.5869 | — | 337 | canonical |
| K | 0.6158 | 0.5917 | 7094 | canonical |

- **Runtime status:** success
- **Elapsed:** ~218 min (217.96)
- **Preflight fixes:** GPU preflight: no Critical issues (Tesla T4); papermill PTY sandbox issue — re-ran with full permissions
- **Retries used:** 1
- **Log reference:** `experiments/logs/20260405_esmm_papermill_inch_spatial.log`

### Observations

- K: CVR_AUC 0.6158 vs prior leaderboard best G 0.5841 (+0.0317).
- J: CVR_AUC 0.5869 vs G 0.5841 (+0.0028).
- K: CVR_AUC 0.6158 vs J 0.5869 (+0.0289).
- K: CTCVR_AUC 0.5917 recorded; J: CTCVR_AUC not reported in metrics payload (null).
- K: wall_clock_seconds 7094 vs J 337.
- Goal CVR_AUC > 0.65: best observed 0.6158 remains below threshold (mechanical: goal not achieved).
