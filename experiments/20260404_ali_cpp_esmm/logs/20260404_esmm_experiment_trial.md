# Experiment Trial Log — ESMM Reproduction

| Field | Value |
|-------|-------|
| **Notebook** | `experiments/20260404_ali_cpp_esmm/20260404_esmm_experiment.ipynb` |
| **Goal metric** | CVR_AUC > 0.65 |
| **Date started** | 2026-04-04 |
| **Last updated** | 2026-04-05 |
| **Canonical copy** | Same history is also summarized in the **final markdown cell** of the notebook (see `notebook-conventions.md`). This file is the repo-side trial log. |
| **Throughput sub-study** | Merged here from the former `20260405_20260404_esmm_experiment_trial.md`; goal `samples_per_sec > 45000` on capped K′ legs. **Default training (Round 4 K)** matches R2-optimized: fused ESMM, manual batches, AMP + prefetch on CUDA, batch 4096. |

**Execution policy (Cursor / Colab):** Sync the notebook with **`scp` + `papermill`** only — **do not** rely on `git push` to Colab. The notebook defaults to **`SKIP_GIT_REPO_SYNC = True`** so `git reset --hard` does not replace the synced file. See `.cursor/skills/run-notebook-on-colab/SKILL.md` and `compatibility.md` §5.

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

## Throughput study (2026-04-05)

Sub-study on the same notebook; logs consolidated from the former dated throughput-only file.

**Goal:** `samples_per_sec > 45000` (where measured on K′ legs with `max_wall_seconds` cap).

### Throughput leaderboard (samples/s)

| experiment | samples_per_sec | wall_clock_s | comparability | notes |
|------------|-----------------|--------------|-----------------|--------|
| K_prime_baseline | 35604 | 902 | canonical | amp=False, prefetch=False |
| K_prime_throughput | 35487 | 900 | canonical | amp=True, prefetch=True; cuda peak ~0.81 GiB (~869730877 B) |
| K_prime_r2_baseline | 33133 | 902 | canonical | batch 4096, DataLoader, amp=False, prefetch=False, manual=False |
| K_prime_r2_optimized | 78415 | 901 | canonical | batch 4096, manual=True, amp=True, prefetch=True |
| K_prime_r2_batch8192 | 138470 | 902 | canonical | batch 8192, manual=True, amp=True, prefetch=True, no OOM |
| K_prime_r3_compile | 124268 | 908 | canonical | batch 8192, manual, AMP, prefetch, torch.compile active, arrow false, early_stop max_wall_seconds |
| K_prime_r3_pyarrow | 126543 | 901 | canonical | batch 8192, manual, AMP, prefetch, compile off, arrow_used true |

**Notebook default (post-merge):** `train_esmm_parquet_rowgroups` defaults match **K_prime_r2_optimized** (manual batches, AMP + prefetch on CUDA, fused `ESMMModel`, batch still set by caller — Round 4 K uses 4096). Pass `use_manual_batches=False`, `use_amp=False`, `prefetch_row_groups=False` for legacy DataLoader + FP32 behavior.

### Throughput round A — T+U+V

**Runtime:** success, ~31.4 min; log `experiments/logs/20260405_esmm_throughput_papermill.log`. Fresh run after deleting `round_5_results.json` on Colab.

| id | hypothesis (one line) |
|----|------------------------|
| T | Enable AMP (mixed precision) for throughput. |
| U | Add data-loader / input prefetch for throughput. |
| V | Batch size 4096 configuration for throughput. |

**Critic:** All accepted (T, U, V). Combined **`K_prime_throughput`** = AMP + prefetch.

| leg | samples_per_sec | wall_s | early_stop | amp | prefetch |
|-----|-----------------|--------|------------|-----|----------|
| K_prime_baseline | 35604 | 902 | max_wall_seconds | false | false |
| K_prime_throughput | 35487 | 900 | max_wall_seconds | true | true |

### Throughput round B — W+X+Y

**Runtime:** success, ~46.5 min; log `experiments/logs/20260405_esmm_r2_papermill.log`.

| id | hypothesis (one line) |
|----|------------------------|
| W | Fused ESMM embedding. |
| X | Manual batching for throughput. |
| Y | Batch size 8192 leg. |

**Critic:** All accepted (W, X, Y).

| leg | samples_per_sec | wall_s | batch | manual | amp | prefetch |
|-----|-----------------|--------|-------|--------|-----|----------|
| K_prime_r2_baseline | 33133 | 902 | 4096 | false | false | false |
| K_prime_r2_optimized | 78415 | 901 | 4096 | true | true | true |
| K_prime_r2_batch8192 | 138470 | 902 | 8192 | true | true | true |

### Throughput round C — Z+AA (+ AB rejected)

**Runtime:** success, ~31.6 min; log `experiments/logs/20260405_esmm_r3_papermill.log`.

| id | hypothesis (one line) |
|----|------------------------|
| Z | Enable `torch.compile` for throughput. |
| AA | Use PyArrow-backed / accelerated data path for throughput. |
| AB | (not executed; critic rejected as REDUNDANT vs Z.) |

**Critic:** Accepted Z, AA; rejected AB.

| leg | samples_per_sec | wall_s | batch | compile_active | arrow_used | early_stop |
|-----|-----------------|--------|-------|----------------|------------|------------|
| K_prime_r3_compile | 124268 | 908 | 8192 | true | false | max_wall_seconds |
| K_prime_r3_pyarrow | 126543 | 901 | 8192 | false | true | max_wall_seconds |

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

*Historical infra notes below; J/K **completed** — see **Round 4 — executed** for metrics.*

**Target:** Experiments **J** (BASE on full split + freq filter + batch 4096 + log1p) and **K** (ESMM, 5 epochs) using Parquet under `DATA_DIR/processed_esmm_full_parquet/`.

**2026-04-04 runtime attempt (earth-radius-controlling-switch):** **INFRA failure** — SSH / Cloudflare tunnel dropped mid-run (`Connection closed by remote host`, then `websocket: bad handshake`). Papermill had **not reached Round 4**; it was still executing earlier round cells when the session ended. No J/K metrics recorded.

**Preserved on Colab Drive (if unchanged):**

- `…/ali_ccp/processed_esmm_parsed_samples/` — cached 5M (and 15M if built) Parquet; notebook reloads from here.
- `…/ali_ccp/processed_esmm_full_parquet/` — full train/test Parquet after one successful full parse (avoids re-parsing raw CSV).
- `…/ali_ccp/esmm_round_training_cache/round_*.json` — delete a file to force re-run that round only.

**To resume:** Reconnect Colab + `cloudflared` tunnel, then re-run papermill (or open the notebook and **Run all**). Earlier rounds skip automatically when the JSON caches exist on Drive. Local stdout capture: `experiments/logs/20260404_esmm_r4_papermill_stdout.log`.

**2026-04-05 runtime attempt (`inch-utility-leading-spatial.trycloudflare.com`):** **INFRA** — SSH dropped during **Round 4 Cell 13** while parsing full split (`sample_skeleton_train` tqdm; no Parquet cache for `sample=full` yet). **Rounds 1–3** completed from **cached** JSON on Drive (printed: A CVR_AUC=0.5526; D CVR_AUC=0.5339; G CVR_AUC=0.5727; H CVR_AUC=0.5550 — values reflect remote cache, may differ slightly from earlier canonical trail rows). **J/K not finished.** Log: `experiments/logs/20260405_esmm_runtime_papermill.log`.

**Next run:** Refresh tunnel hostname if rotated → **scp** latest `experiments/20260404_ali_cpp_esmm/20260404_esmm_experiment.ipynb` → **papermill**. Optional: in Config set `CLEAN_ROUND_RESULT_JSON = []` (keep caches for R1–3); only use cleanup flags if you need to invalidate artifacts. Once `parsed_train_rows_full.parquet` exists under `processed_esmm_full_parquet/`, Round 4 skips the long raw parse.

---

## Memory plan (OpenAI + Gemini consult, 2026-04-04)

**Problem:** Round 4 held full train/test DataFrames plus full float64-style tensor copies (~51GB RAM ceiling on Colab).

**Consultant alignment:** Both recommended (1) stop materializing full 42M-row training tensors for ESMM, (2) use compact dtypes, (3) serialize J then K without overlapping giant tensors, (4) stream or chunk from Parquet. OpenAI: store int32 on host, cast to `long` at batch for `nn.Embedding`. Gemini: same + IterableDataset / mmap if needed.

**Implemented in `experiments/20260404_ali_cpp_esmm/20260404_esmm_experiment.ipynb`:**

| Change | Rationale |
|--------|-----------|
| `encode_and_tensorize` → int32 sparse, `torch.from_numpy` float32 labels | ~halves sparse host RAM |
| `BASEModel` / `ESMMModel` `forward`: `sparse_x = sparse_x.long()` | PyTorch Embedding expects long indices |
| Round 4: write `r4_norm_train/test.parquet` under `processed_esmm_full_parquet/` with `row_group_size=1_000_000`, then `del` DataFrames | Drops duplicate in-RAM frame before K |
| J: `read_parquet(..., filters=[('click','==',1)])` | Avoids loading full train into pandas for BASE |
| K: `train_esmm_parquet_rowgroups` — shuffle row-group order each epoch, tensorize per row group only | Peak RAM ≈ largest row group + chunk tensors, not full 42M |
| Test: single `encode_and_tensorize` for all rows; `evaluate_esmm_cvr_indexed` with boolean mask | Removes duplicate full-test tensorization |

**Later (throughput, 2026-04-05):** Fused single-table `ESMMModel` embedding, optional manual batches, AMP + row-group prefetch; defaults align with **K_prime_r2_optimized** (batch 4096 for Round 4 K unless overridden).

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

---

## Notebook setup & artifacts (reference)

| Topic | Default / location |
|--------|-------------------|
| **DATA_DIR** | `/content/drive/MyDrive/colab/data/ali_ccp` (Colab) |
| **Sample rounds 1–3** | `SAMPLE_SIZE = 5_000_000` → `processed_esmm_parsed_samples/` |
| **Round 4 full split** | `processed_esmm_full_parquet/` — `parsed_train_rows_full.parquet`, `parsed_test_rows_full.parquet`, `r4_norm_train/test.parquet` |
| **Round caches** | `esmm_round_training_cache/round_{1…5}_results.json` — delete one file to force re-run that round only |
| **Freq-filter vocab cache** | `r4_filtered_sparse_vocab.pkl` — `load_or_build_sparse_vocabs_filtered_parquet` loads when train Parquet mtime/row count + `SPARSE_COLS` + `min_count` match; else full scan then write pickle. Toggle: `CLEAN_R4_VOCAB_CACHE`, `FORCE_REBUILD_R4_VOCAB` |
| **Git on Colab** | `SKIP_GIT_REPO_SYNC = True` → no fetch/reset if `recsys_playground/` exists; `FORCE_GIT_SYNC=1` env forces sync to `origin/main` |
| **K early-stop (Round 4)** | `K_EARLY_STOP_MAX_WALL_SECONDS` etc. default **`None`** (full 5-epoch K) |
| **Round 5 K′** | Throughput benchmarks vs legacy baselines; **`R5_K_PRIME_MAX_WALL_SECONDS = 900`** per leg unless changed; optional R3 legs (`torch.compile`, PyArrow) via `R5_RUN_R3_LEGS` |
| **Round 4 J** | 10 epochs, batch 4096, `R4_BASE_*` LR/WD (default constant LR, wd=0) |
| **Round 4 K** | 5 epochs, **batch 4096**, row-group streaming from `R4_NORM_TRAIN`; **`train_esmm_parquet_rowgroups` defaults:** manual batches, AMP + prefetch on CUDA, fused `ESMMModel` (legacy: pass `use_manual_batches=False`, `use_amp=False`, `prefetch_row_groups=False`) |

**Throughput / encode:** `encode_and_tensorize_fast` (and optional Arrow path), categorical tables; `train_esmm_parquet_rowgroups` returns `train_meta` (`samples_per_sec`, `early_stop_reason`, …). Optional print after K in Round 4 for throughput summary.

---

## Round 5 — Throughput / tooling (`K′`)

Notebook **Round 5** runs additional K′ legs (legacy DataLoader baseline, batch-8192, optional `torch.compile` / PyArrow) with **dev wall cap** by default. Full tables and logs live under **Throughput study** above.

**Note:** A short **papermill** run with all `round_*.json` present skips training and only replays caches (~tens of seconds locally); metrics printed are whatever is stored on Drive (may differ slightly from canonical J/K row above, e.g. ~0.62 / ~0.585 CTCVR in one cache-skipped readout).

---

## Recent runtime — experiment-runtime (2026-04-05)

| Field | Value |
|-------|-------|
| **Host** | `emerging-hobby-conservative-planned.trycloudflare.com` (Cloudflare tunnel; hostname expires when Colab session ends) |
| **Sync** | `scp` notebook → `…/recsys_playground/recsys_playground/experiments/20260404_ali_cpp_esmm/` |
| **Command** | Remote `papermill` → `20260404_esmm_experiment_output.ipynb` |
| **Outcome** | Success ~79 s wall when rounds 1–5 **cache-skipped** on Drive |
| **GPU** | Tesla T4 |

For a **full re-train**, clear relevant `round_n_results.json` (and optionally vocab/normalize flags in Config) on Drive before re-running.

---
