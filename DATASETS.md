# Datasets — Capability & Cache Reference

**Single source of truth for which datasets are experiment-ready, for which models, and how their
data is cached.** Read this before starting any modeling work or wiring a new dataset — it saves you
from re-auditing the notebooks.

- **Last audited:** 2026-05-30 (from the pipeline code, not from a live Drive mount).
- **Caveat:** Drive *file existence* is **not** verified here — paths below are the literals the code
  reads/writes. If a cache file is missing on Drive, the pipeline rebuilds it (see the flags).

---

## TL;DR readiness matrix

| Dataset | Task it actually solves | Experiment-ready for | Status |
|---|---|---|---|
| **Ali-CCP** | CTR / CVR / CTCVR (click + purchase labels) | ESMM, SharedBottom, MMoE, PLE, Wide&Deep, DeepFM, DCNv2 | ✅ **Ready now** |
| **Amazon Video Games** | Rating regression (1–5 stars) | rating-regression MLP/MF/NeuMF, DCNv2, PLE (regression heads) | ⚠️ **Partial** — not a CTR/ranking/DIN pipeline |
| **Amazon Beauty (TIGER)** | Generative semantic-ID retrieval (seq2seq next-item) | RQ-VAE + seq2seq SemanticID, LLM-SID finetune | ⚠️ **Partial** — not wired to DCN/DIN/DeepFM |

**Bottom line:** Ali-CCP is the only clean benchmark dataset today. Both Amazon datasets are usable
for their own task but are **not** a clean CTR/ranking/DIN suite yet. See
[What to build next](#what-to-build-next).

---

## Model → dataset routing

Use this when choosing where to run a model.

| Model family | Use this dataset | Why |
|---|---|---|
| Wide & Deep / DeepFM / DCNv2 | **Ali-CCP** | real CTR labels + full streaming feature pipeline |
| ESMM / MMoE / PLE (multi-task) | **Ali-CCP** | click + purchase = native CTR/CVR/CTCVR multitask |
| DIN (target-attention CTR) | **not ready** | needs an Amazon **negative-sampling ranking datamodule** (history + target item + label, sampled negatives, Recall@K / NDCG@K / sampled-AUC). TIGER's `data.py` already provides leave-one-out history splits to build on; the negative-sampling ranking wrapper is the missing piece. |
| SASRec / BERT4Rec / GRU4Rec / TIGER | **Amazon Beauty / Video_Games** | user histories + leave-one-out splits fit sequential/generative retrieval; Ali-CCP does not |

---

## The standard cache contract

Every dataset pipeline *should* provide these tiers so reruns are cheap and reproducible. This is the
target contract — the per-dataset tables below show how far each dataset meets it.

| Tier | What it is | Why it matters |
|---|---|---|
| **1. Raw cache** | Original downloaded files on Drive | avoid re-downloading multi-GB sources |
| **2. Processed cache** | Parsed + normalized data ready to train (e.g. Parquet) | avoid re-parsing every run |
| **3. Vocab / id-map cache** | Sparse-feature vocab or user/item → int maps | stable feature IDs across runs |
| **4. Split cache** | Persisted train / val / test (esp. leave-one-out) | reproducible, comparable evaluation |
| **5. Negative-sample cache** | Pre-sampled negatives for ranking / DIN / sequential models | fair, repeatable ranking metrics |
| **6. Result cache** | Per-experiment metric JSON | skip already-finished legs on rerun |

Legend: ✅ present & reused · 🔁 present but **rebuilt every run** · ❌ missing · — not applicable.

---

## 1. Ali-CCP — ✅ ready

The production pipeline lives in reusable modules; notebooks are thin orchestration layers.

- **Pipeline code:** `experiments/20260404_ali_cpp_esmm/esmm_ali_ccp_impl.py` (data I/O, streaming
  parse, vocab, dense normalization, ESMM/SharedBottom/MMoE/PLE models, training, multitask eval) and
  `experiments/20260519_wide_deep_deepfm_dcn/new_models_impl.py` (Wide&Deep, DeepFM, DCNv2).
- **Orchestration notebooks:** `experiments/20260404_ali_cpp_esmm/20260404_esmm_experiment.ipynb`,
  `experiments/20260519_wide_deep_deepfm_dcn/20260519_model_comparison.ipynb`.
- **Separate early exploratory notebook:** `notebooks/ad_hoc/experiment_aliccp_dataset.ipynb`
  (self-contained, `LabelEncoder` + `MinMaxScaler`, single-task `MLPClassifier`; **does not** use the
  modules above or the disk caches below — do not treat it as the production path).

**Drive layout** (base `DATA_DIR = /content/drive/MyDrive/colab/data/ali_ccp`):

```
ali_ccp/
  sample_train.tar, sample_test.tar                 # raw archives
  sample_skeleton_{train,test}.csv                  # extracted raw
  common_features_{train,test}.csv                  # extracted raw
  processed_esmm_full_parquet/
    parsed_{train,test}_rows_full.parquet           # parsed (pre-normalization)
    preprocessed_{train,test}.parquet               # normalized, training-ready
    preprocessed_sparse_vocab.pkl                   # sparse vocab (id 0 = UNK, min_count=5)
  esmm_round_training_cache/                         # ESMM/MTL result JSONs
    baseline_results.json, exp_shared_bottom_results.json, exp_mmoe_results.json, exp_ple_results.json
  classic_models_cache/                              # classic-model result JSONs
    wide_deep_results.json, deepfm_results.json, dcnv2_results.json
```

(A local SSD copy `preprocessed_test_local.parquet` at `/content/` is made once for faster eval in
the classic-models notebook.)

**Cache contract status:**

| Tier | Status | Detail |
|---|---|---|
| 1. Raw | ✅ | tars + extracted CSVs on Drive |
| 2. Processed | ✅ | `parsed_*` then normalized `preprocessed_*` Parquet, reused if present |
| 3. Vocab | ✅ | `preprocessed_sparse_vocab.pkl`, validated against train Parquet path/mtime/rowcount/cols/min_count |
| 4. Split | ✅ | Ali-CCP ships a predefined train/test split (separate `*_train` / `*_test` Parquet) |
| 5. Neg-sample | — | not applicable — explicit `click`/`purchase` labels, no sampling needed |
| 6. Result | ✅ | per-model JSON; an existing JSON skips that training leg |

**Reuse vs reprocess:** reuses Drive caches by default; reprocesses **only** when a cache is missing
or a flag forces it:

| Flag | Default | Effect when set |
|---|---|---|
| `CLEAN_PREPROCESSED_PARQUET` | `False` | deletes `preprocessed_{train,test}.parquet` (ESMM nb implements removal) |
| `CLEAN_PREPROCESSED_VOCAB_CACHE` | `False` | deletes `preprocessed_sparse_vocab.pkl` (ESMM nb) |
| `FORCE_REBUILD_PREPROCESSED_VOCAB` | `False` | rescans train Parquet, ignores PKL |
| `CLEAN_EXPERIMENT_JSON` | `[...]` (ESMM) / `[]` (classic) | removes the mapped result JSONs at startup |
| `force_rewrite` (ad_hoc nb) | `False` | re-extracts tar archives |

> ⚠️ The classic-models notebook *defines* `CLEAN_PREPROCESSED_PARQUET` /
> `CLEAN_PREPROCESSED_VOCAB_CACHE` but its config cell only runs cleanup for `CLEAN_EXPERIMENT_JSON`
> — setting the parquet/vocab clean flags there is a no-op. Use the ESMM notebook to rebuild caches.

**Feature schema:** 23 sparse string columns (`'101'`…`'150_14'`), 8 dense columns (prefixed `D`),
labels `click` + `purchase`. Dense normalization is `log1p(|x|)·sign(x)`. Parsing and normalization
stream in chunks (default 500k rows); training reads one Parquet row group at a time.

**Metrics:** `CTR_AUC`, `CVR_AUC` (clicked-only), `CTCVR_AUC`. The multitask legs
(SharedBottom / MMoE / PLE) additionally report PR-AUC, logloss, and ECE **for CTR and CTCVR only** —
the CVR (clicked-only) leg reports ROC-AUC only. The classic-models notebook reports the three AUCs.

> ✅ **Resolved — fixed 2026-05-30, re-validated 2026-05-31.** The classic-models notebook
> (`20260519_model_comparison.ipynb`) previously copied the *parsed, un-normalized* test Parquet to
> the local SSD and evaluated on it, while models train on the `log1p`-normalized
> `preprocessed_train.parquet` — a train/eval feature mismatch that invalidated the Wide&Deep /
> DeepFM / DCNv2 AUCs. The notebook now normalizes first and copies the normalized
> `preprocessed_test.parquet` to the local SSD. **All three caches were regenerated on the full 43M-row
> eval** and are sane: Wide&Deep CTCVR-AUC 0.651, DeepFM 0.623, DCNv2 0.651 (CTR 0.61–0.62,
> CVR 0.67–0.69). See `experiments/20260519_wide_deep_deepfm_dcn/logs/20260531_classic_models_revalidated.md`.
> (ESMM/MTL legs were unaffected — they evaluate on the normalized test Parquet directly.)

---

## 2. Amazon Video Games — ⚠️ partial (rating regression, notebooks only)

**This is a rating-regression project, not a CTR/ranking/DIN pipeline.** All code is inline in
notebooks; there is **no reusable `.py` module** and **no processed-data cache**.

- **Notebooks:** `notebooks/ad_hoc/experiment_amazon_review_game.ipynb`,
  `notebooks/ad_hoc/regression_target_experiment.ipynb`,
  `notebooks/ad_hoc/experiment_dcn_ple_rtm.ipynb`.
- **Summary doc:** `experiments/regression_target_optimization.md`.

**Drive layout:**

```
data/amazon_review_game/
  Video_Games.jsonl.gz                               # raw reviews (Amazon Reviews 2023), cached
data/experiment_cache/dcn_ple_rtm/
  round_1_results.json … round_6_results.json        # only the DCN/PLE notebook caches results
```

(`WORK_DIR = /content/drive/MyDrive/colab/amazon_review_game`; raw URL:
`https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/review_categories/Video_Games.jsonl.gz`.)

**Cache contract status:**

| Tier | Status | Detail |
|---|---|---|
| 1. Raw | ✅ | `.jsonl.gz` cached on Drive; `force_rewrite=False` reuses it |
| 2. Processed | ❌ | parsed to in-memory pandas every run; no Parquet cache |
| 3. Vocab / id-map | ❌ | `LabelEncoder` for user/item rebuilt in memory every run |
| 4. Split | ❌ | `train_test_split(test_size=0.2, random_state=42)` in memory every run (deterministic but not persisted) |
| 5. Neg-sample | ❌ | none (regression task) |
| 6. Result | ⚠️ | only `experiment_dcn_ple_rtm.ipynb` caches `round_*.json`; the other two keep results in memory |

**Reuse vs reprocess:** only the raw download is cached. **All preprocessing re-runs in memory each
session.** `SAMPLE_SIZE = 200_000` by default; `NOTEBOOK_SMOKE=1` shrinks to 5k.

**Models (all inline):** `RecMLP`, `RecMLPv2`, `BiasedMF`, `NeuMF`, `*WithFeatures`,
`RecMLPClassHead`, `RecMLPOrdinalHead` (regression nb); `DCNv2RecModel`, `PLE3TaskRecModel`,
`PLE3TaskRecModelDCN`, router models (DCN/PLE nb).

**Metrics:** regression `MSE/RMSE/MAE/R²` + diagnostics (`sigma_ratio`, calibration slope); optional
post-hoc `Hit@10/MRR@10/NDCG@10` computed from regression scores. **No AUC / CTR metrics.**

> ⚠️ **Known blocker (documented):** on the 200k sample (~83k users, 39k items, ~2.4
> interactions/user) the models collapse to predicting the mean (`σ_pred ≪ σ_true`, calibration
> slope < 0.47, negative R²). The dual goal `R² > 0.05 AND MAE < 0.85` was **not met** across 13
> experiments. The conclusion in `regression_target_optimization.md`: mean collapse is fundamental to
> this sparsity, not fixable by loss/architecture alone. **Rating regression on this slice is a dead
> end** — for ranking work, reframe to implicit-feedback CTR/ranking (see roadmap).

---

## 3. Amazon Beauty / TIGER SemanticID — ⚠️ partial (generative retrieval)

A self-contained semantic-ID project. It has the cleanest reusable data module of the Amazon work
(`tiger_semantic_id/src/data.py`) but the **prep notebook wipes and rebuilds everything every run**,
and it is **not** wired to any DCN/DIN/DeepFM runner.

- **Module:** `tiger_semantic_id/src/data.py` (parse, `load_reviews_df`, `load_meta_df`,
  `filter_and_split` = leave-one-out, `build_id_maps`, `apply_id_maps`, `save_mappings`).
  Ranking metrics `recall_at_k` / `ndcg_at_k` live in `tiger_semantic_id/src/seq2seq.py`.
- **Notebooks:** `notebooks/tiger_semantic_id/TIGER_SemanticID_data_preparation.ipynb`,
  `TIGER_SemanticID.ipynb`, plus LLM-finetune / EDA notebooks.
- **Datasets supported:** `Beauty` (2014 SNAP 5-core) and `Video_Games` (Amazon 2023); selected via
  `Config.dataset_name` (default `'Beauty'`).

**Drive layout** (base `WORK_DIR = /content/drive/MyDrive/colab/tiger_semantic_id`):

```
tiger_semantic_id/
  data_preparation/
    data/        reviews_Beauty_5.json.gz, meta_Beauty.json.gz  (or Video_Games equivalents)
    artifacts/   user2id.json, item2id.json, {train,val,test}_df.pkl,
                 items.pkl, config.json, item_texts.json, item_embeddings.pt
  rq_vae_building/artifacts/   # downstream RQ-VAE + seq2seq outputs
```

**Cache contract status:**

| Tier | Status | Detail |
|---|---|---|
| 1. Raw | 🔁 | re-downloaded every prep run (the folder is deleted first — see below) |
| 2. Processed | 🔁 | `items.pkl`, `item_embeddings.pt` saved, but wiped + rebuilt every prep run |
| 3. Id-map | 🔁 | `user2id.json` / `item2id.json` saved, but rebuilt every prep run |
| 4. Split | 🔁 | leave-one-out `{train,val,test}_df.pkl` saved, but rebuilt every prep run |
| 5. Neg-sample | ❌ | no negative sampling anywhere in `tiger_semantic_id/` |
| 6. Result | ⚠️ | generative `recall_at_k`/`ndcg_at_k` exist in `seq2seq.py` but are **not invoked** in the prep/main notebooks; no metrics file written |

> ⚠️ **Reprocess every run (incl. raw):** the prep notebook starts with
> `if os.path.exists(DATA_PREP_DIR): shutil.rmtree(DATA_PREP_DIR)` then recreates it — this wipes
> **both** the raw downloads (`data/`) and the processed artifacts (`artifacts/`), so each prep run
> re-downloads *and* reprocesses from scratch. Artifacts are only stable *after* a successful prep
> run and until the next one. (`TIGER_SemanticID.ipynb` similarly `rmtree`s `rq_vae_building`.)

**Evaluation style:** generative semantic-ID retrieval (seq2seq next-SID), not sampled-negative
ranking. No DIN/DCN/DeepFM wiring exists, by design.

---

## What to build next

The three datasets are enough for phase-one learning/exploration, but **not** yet a clean benchmark
suite. Build in this order (each item requires Colab/GPU/Drive + the actual data — do **not** attempt
blind on a CPU-only box). See the agent self-exploration task folder linked in the repo notes for the
detailed, sequenced backlog.

1. **Promote the Ali-CCP data path into a clean importable dataset layer.** Today the data I/O is
   reusable but lives mixed with models in `esmm_ali_ccp_impl.py`. Split it into a `datasets/aliccp`
   module (parse → normalize → vocab → row-group loader) so non-ESMM models import it directly.
2. **Ali-CCP smoke-test runner** for Wide&Deep / DeepFM / DCNv2 (tiny `max_row_groups`, asserts AUC
   computes and caches write). Lets any agent sanity-check the pipeline in minutes.
3. ~~**Amazon sequence datamodule** (Beauty/Video_Games): leave-one-out split + persisted
   negative-sampling cache + ranking metrics.~~ ✅ **Done — see [`amazon_ranking/`](./amazon_ranking/).**
   Pure numpy/pandas, 18 passing unit tests; provides `(history, target, label)` train examples
   (with optional sampled negatives), leakage-free eval candidates with a version-keyed cache, and
   Recall@K / NDCG@K / sampled-AUC metrics. This was the missing tier-5 cache and the DIN prerequisite.
4. **Implement DIN** on the `amazon_ranking` datamodule (history + target item + label, target
   attention). Build the torch model on top of the existing CPU-tested data layer.
5. **Fair comparison:** DIN vs DCN/DeepFM under sampled-ranking on the same Amazon datamodule.

Also adopt the cache contract above for the Amazon pipelines: stop the TIGER prep `rmtree` from
wiping a valid cache (add a reuse check / version key), and add a processed-Parquet + id-map cache to
the Amazon rating notebooks so preprocessing stops re-running in memory every session.
