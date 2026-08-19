# aliccp_mtl_experiments

Self-contained Ali-CCP multi-task-learning (MTL) experiment folder, built for a public
blog article's code links. It is an **additive clone**: every file here was copied out
of two existing experiments in this repo (see Provenance below) rather than edited in
place, so the original notebooks and `.py` files are untouched.

## What's in this folder

| File | Contents |
|---|---|
| `models.py` | The 10 in-scope ESMM-family MTL architectures: `ESMMModel`, `ESMMModel_Wide`, `ESMM_SharedBottom`, `ESMM_SharedBottomWide`, `ESMM_MMoE`, `ESMM_MMoE_Wide`, `ESMM_PLE`, `ESMM_PLE_Wide`, `ESMM_PLE_Cross`, `ESMM_PLE_WideCross`, plus their shared helper classes (`_ESMMExpertMLP`, `_ESMMGate`, `_ESMM_PLELevel`, `_ESMM_PLETower`, `_ESMMCrossNet`) and the `_init_linear` init helper. |
| `single_task_models.py` | The 3 classic single-task ranking baselines: `WideAndDeepModel`, `DeepFMModel`, `DCNv2Model`, verbatim. |
| `harness.py` | Shared trainer (`train_esmm_parquet_rowgroups`), loss (`_esmm_multitask_bce_from_probs`), evaluator (`evaluate_esmm_multitask_streaming_parquet` + its metric deps `binary_pr_auc` / `binary_bce_log_loss` / `expected_calibration_error`), and `count_parameters`. All 13 models above plug into this one trainer/evaluator pair via `model_ctor` / `model_ctor_kwargs`. |
| `run_experiments.py` | The runner: loads Ali-CCP data via `datasets/aliccp/`, trains + evaluates all 13 models at the matched 1-epoch protocol, and writes the leaderboard. |
| `run_experiments.ipynb` | Thin Colab bootstrap (Drive mount + repo clone + deps) that calls `run_experiments.main()`. All actual logic lives in the `.py` files, not the notebook. |
| `results/` | Output: `aliccp_leaderboard.json` (machine-readable) and `aliccp_leaderboard.md` (human-readable table), sorted by CTCVR AUC descending. Written incrementally after each model finishes. Also `aliccp_leaderboard_progress.json`, a per-config checkpoint (written atomically after each model completes) that makes the script resumable — see note below. |

## Dependency: `datasets/aliccp/`

This folder does **not** contain a data layer. `run_experiments.py` and `harness.py`
import the repo's canonical Ali-CCP package, `datasets/aliccp/`, for:

- `datasets.aliccp.data`: `ensure_full_split_parquet_streaming`,
  `load_or_build_sparse_vocabs_filtered_parquet`, `stream_normalize_parquet` — raw
  Ali-CCP CSV -> parsed Parquet -> filtered sparse vocabs -> normalized train/test Parquet.
  (`run_experiments.py`.)
- `datasets.aliccp.encode`: `encode_and_tensorize_fast`, `encode_and_tensorize_arrow`,
  `_precompute_sparse_encode_tables` — used internally by `harness.py`.

Both use a repo-root-relative import (`from datasets.aliccp.data import ...` /
`from datasets.aliccp.encode import ...`) — the one place in this folder where that
import style is used, since here we are a *consumer* of that package's output rather
than part of it. `models.py` and `single_task_models.py` only import each other and
`harness.py` via plain local imports (`from models import ...`).

(Until 2026-07-20 this dependency was a standalone sibling folder,
`aliccp_data_preparation/`, built for this same article; it was migrated into
`datasets/aliccp/` after a compatibility audit — see that package's `CHANGE_LOG.md`.
No behavior changed for this folder; only the import path did.)

## How to run

**Standalone script** (from inside this folder, or anywhere with it on `sys.path`):

```bash
cd aliccp_mtl_experiments
python run_experiments.py
```

By default it looks for Ali-CCP data under
`/content/drive/MyDrive/colab/data/ali_ccp` (same path the original ESMM/classic-models
experiments used, for cache reuse). Override with environment variables for a
non-Colab / local run:

```bash
ALICCP_DATA_DIR=/path/to/ali_ccp ALICCP_PROCESSED_DIR=/path/to/processed python run_experiments.py
```

**Notebook (Colab):** open `run_experiments.ipynb`, run all cells. It mounts Drive,
clones/updates the repo, installs the small dependency set (`torch`, `pandas`, `numpy`,
`scikit-learn`, `pyarrow`), then calls `run_experiments.main()`.

**Resumable:** the script is safe to rerun after an interruption (e.g. a free-tier Colab
disconnect). Each of the 13 configs is checkpointed to `results/aliccp_leaderboard_progress.json`
as soon as it finishes; rerunning `python run_experiments.py` (or re-executing the notebook)
reads that file first and only trains configs not yet marked complete there, picking up where
it left off instead of starting over.

## Protocol

1 epoch, batch_size=4096, embed_dim=18, seed=42, full 43M-row normalized test Parquet
for eval, no LR schedule (constant lr, harness default). This matches the published
single-protocol leaderboard at
`knowledge_base/projects/agent_self_exploration/20260607_ple_fix/leaderboard_1ep_overall.md`
so results from this folder are directly comparable to it.

## Provenance / faithfulness notes

- `models.py` is a byte-identical clone of lines 178-733 of
  `experiments/20260404_ali_cpp_esmm/esmm_ali_ccp_impl.py` (as of the 20260609
  AdaOrder round), covering only the classic ESMM/SharedBottom/MMoE/PLE (+Wide/Cross)
  family. It intentionally excludes everything from `ESMM_PLE_AdaOrderCross` onward in
  that file (the NDM/ESCM2/EGEAN/DCMT/AdaOrderCross/TaskCross/EPNetGate family) — a
  separate, later study not part of this article.
- `single_task_models.py` is a byte-identical full-file clone of
  `experiments/20260519_wide_deep_deepfm_dcn/new_models_impl.py`.
- `harness.py` clones `train_esmm_parquet_rowgroups`, `_esmm_multitask_bce_from_probs`,
  `evaluate_esmm_multitask_streaming_parquet` (+ its metric deps), and
  `count_parameters` from `esmm_ali_ccp_impl.py`. Two deliberate deviations from the
  source:
  1. The `isinstance(model, ESMM_PLE): use_amp=False` special case is preserved
     **exactly** — PLE is numerically unstable (CUDA BCE domain asserts) under AMP/fp16.
  2. The `hasattr(model, 'compute_egean_loss'/'compute_dcmt_loss'/'compute_escm2_loss')`
     custom-loss routing and the `track_grad_snr`/`GradSNRTracker` instrumentation were
     dropped, including their downstream branches in `_prepare_row_group_tensors` and
     `_step_batch`. None of the 13 models in this folder expose those `compute_*_loss`
     methods or set `track_grad_snr=True`, so those branches were dead code here — they
     belong to the same out-of-scope family excluded from `models.py`.
- `run_experiments.py` is new code (not cloned), written to drive the 13 models above
  through the harness at the matched protocol and produce the leaderboard.
