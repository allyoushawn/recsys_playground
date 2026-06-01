# Ali-CCP Classic Models — Re-validated After Eval-Normalization Fix (2026-05-31)

Backlog **E0**. The classic-model comparison (`20260519_model_comparison.ipynb`) had a
train/eval feature-normalization mismatch: training used `log1p`-normalized dense features but
eval scored on the **un-normalized** test Parquet, so the cached AUCs were invalid. The fix
(copy the **normalized** `preprocessed_test.parquet` to local SSD before eval) was applied; this
run regenerates all three result JSONs with the corrected eval on the full 43M-row test set.

## Setup

- Full Ali-CCP: train 42.3M rows, **eval 43,016,614 rows** (the complete test set).
- Reuses the ESMM preprocessed Parquet + vocab cache (`processed_esmm_full_parquet/`).
- T4 GPU; per-model result cache in `data/ali_ccp/classic_models_cache/`.
- Ran across 2 Colab sessions (W&D + DeepFM in session 1; DCNv2 resumed in session 2 after the
  first runtime died ~84 min in) — per-model cache + in-run heartbeat made the resume reliable.

## Results (corrected eval)

| Model | CTR_AUC | CTCVR_AUC | CVR_AUC | Params | Train wall (s) |
|---|---|---|---|---|---|
| Wide & Deep | 0.6191 | **0.6508** | **0.6863** | 41.96M | 425 |
| DeepFM | 0.6108 | 0.6231 | 0.6722 | 44.52M | 445 |
| DCNv2 | **0.6232** | 0.6506 | 0.6800 | 42.50M | 457 |

(`train_wall` is training only; the full 43M-row eval adds ~20 min per model.)

## Read

- **Eval fix validated.** All CTCVR-AUCs are ~0.62–0.65 — sane, non-degenerate, and above the
  ~0.61 ESMM reference. The previously cached values (computed on un-normalized eval features) were
  invalid and have been replaced.
- **Wide & Deep and DCNv2 are ~tied** (CTCVR 0.651) and lead; **DeepFM trails** (0.623).
- **Notable:** these single-task classic models *outperform* the ESMM MTL variants on CTCVR-AUC
  (ESMM best = PLE 0.5841, see `experiments/20260404_ali_cpp_esmm/logs/20260531_mtl_esmm_full_leaderboard.md`).
  On Ali-CCP with full normalized eval, the classic CTR architectures are strong CTCVR rankers —
  the ESMM multi-task framing's advantage is in *jointly* modeling CTR+CVR with the conversion-given-click
  decomposition, not raw CTCVR-AUC at this scale/epoch budget.

## Provenance

Result JSONs (Drive `data/ali_ccp/classic_models_cache/`): `wide_deep_results.json` (14:15Z),
`deepfm_results.json` (14:44Z), `dcnv2_results.json` (15:43Z). Notebook:
`20260519_model_comparison.ipynb` (eval-normalization fix in the prep cell; heartbeat added).
