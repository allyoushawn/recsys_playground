"""Ali-CCP MTL leaderboard runner.

Trains and evaluates the 13 model architectures in this folder (3 classic single-task
baselines from `single_task_models.py` + 10 MTL architectures from `models.py`) on the
full Ali-CCP dataset via the shared harness in `harness.py`, and writes a leaderboard.
21 configs run in total: the 13 architectures once each at seed=42 (main sweep), plus
an 8-config seed-diagnostic sweep (see below).

Protocol (matched to the published single-protocol leaderboard so these runs are
directly comparable — see `/Users/fox/knowledge_base/projects/agent_self_exploration/
20260607_ple_fix/leaderboard_1ep_overall.md`):
  1 epoch, batch_size=4096, embed_dim=18, seed=42 (RANDOM_STATE) for the main sweep,
  full 43M-row normalized test Parquet for eval, no LR schedule (constant lr, harness
  default).

Seed diagnostic: ESMMModel and ESMMModel_Wide are additionally re-run at 4 extra seeds
each (17, 101, 202, 303), giving 5 total seeds per architecture. This checks whether
the two-tower family's occasional chance-level collapse (CTCVR/CVR AUC ≈ 0.50) is a
real seed-dependent phenomenon now that `harness.py` actually calls `torch.manual_seed`
/ `torch.cuda.manual_seed_all` before model construction (previously it did not, despite
every result being labeled seed=42 — see `results/v1_unseeded/README.md`).

Data comes from `datasets/aliccp/` (parsed/normalized Parquet + filtered sparse
vocabs), which this script imports with repo-root-relative imports — the one place in
this folder where that import style is used, since here we are a *consumer* of that
package's output rather than part of it. (Migrated 2026-07-20 from the standalone
`aliccp_data_preparation/` folder, which now lives at `datasets/aliccp/` — see its
CHANGE_LOG.md.)

Usage:
    python run_experiments.py

Resumable: after each config finishes, its result is checkpointed to
results/aliccp_leaderboard_progress.json. Rerunning this script (e.g. after a Colab
disconnect) reads that file first and skips any config already marked complete there.

Outputs:
    results/aliccp_leaderboard.json
    results/aliccp_leaderboard.md
    results/aliccp_leaderboard_progress.json  (per-config checkpoint; enables resume)
"""
import json
import math
import os
import sys
import time

# --- Make both the repo root (for datasets.aliccp) and this folder itself
# --- (for models / single_task_models / harness) importable, regardless of cwd.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_THIS_DIR)
for _p in (_REPO_ROOT, _THIS_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch

from datasets.aliccp.data import (
    ensure_full_split_parquet_streaming,
    load_or_build_sparse_vocabs_filtered_parquet,
    stream_normalize_parquet,
)

from models import (
    ESMMModel, ESMMModel_Wide,
    ESMM_SharedBottom, ESMM_SharedBottomWide,
    ESMM_MMoE, ESMM_MMoE_Wide,
    ESMM_PLE, ESMM_PLE_Wide,
    ESMM_PLE_Cross, ESMM_PLE_WideCross,
)
from single_task_models import WideAndDeepModel, DeepFMModel, DCNv2Model
from harness import (
    train_esmm_parquet_rowgroups,
    evaluate_esmm_multitask_streaming_parquet,
    count_parameters,
)

# --------------- Protocol (matched to leaderboard_1ep_overall.md) ---------------

EPOCHS = 1
BATCH_SIZE = 4096
EMBED_DIM = 18
RANDOM_STATE = 42

# --------------- Ali-CCP feature schema (fixed for this dataset) ---------------

SPARSE_COLS = ['101', '121', '122', '124', '125', '126', '127', '128', '129',
               '205', '206', '207', '210', '216', '508', '509', '702', '853',
               '301', '109_14', '110_14', '127_14', '150_14']
DENSE_COLS = ['109_14', '110_14', '127_14', '150_14', '508', '509', '702', '853']
DENSE_FEAT_COLS = ['D' + c for c in DENSE_COLS]

# --------------- I/O paths ---------------
# Same convention as experiments/20260519_wide_deep_deepfm_dcn/20260519_model_comparison.ipynb
# (Google Drive path, since the full Ali-CCP parse/normalize is expensive and other
# experiments in this repo already cache their output there). Override via env vars
# for a non-Colab / local run.

DATA_DIR = os.environ.get('ALICCP_DATA_DIR', '/content/drive/MyDrive/colab/data/ali_ccp')
PROCESSED_FULL_DIR = os.environ.get(
    'ALICCP_PROCESSED_DIR', os.path.join(DATA_DIR, 'processed_esmm_full_parquet'))
PREPROCESSED_TRAIN = os.path.join(PROCESSED_FULL_DIR, 'preprocessed_train.parquet')
PREPROCESSED_TEST = os.path.join(PROCESSED_FULL_DIR, 'preprocessed_test.parquet')
PREPROCESSED_SPARSE_VOCAB_CACHE = os.path.join(PROCESSED_FULL_DIR, 'preprocessed_sparse_vocab.pkl')

VOCAB_SCAN_ROWS_PER_BATCH = 200_000
NORM_STREAM_BATCH_ROWS = 500_000

RESULTS_DIR = os.path.join(_THIS_DIR, 'results')
LEADERBOARD_JSON = os.path.join(RESULTS_DIR, 'aliccp_leaderboard.json')
LEADERBOARD_MD = os.path.join(RESULTS_DIR, 'aliccp_leaderboard.md')
# Per-config checkpoint (dict keyed by friendly name -> result + completion flag). Written
# atomically after every config finishes so a Colab disconnect never loses more than the
# in-flight config, and read at startup so a restarted session skips configs already done.
LEADERBOARD_PROGRESS_JSON = os.path.join(RESULTS_DIR, 'aliccp_leaderboard_progress.json')

# --------------- The 21 configs: (model_class, ctor_kwargs, friendly_name, seed_override) ---
# Wide/Cross variants subclass their plain counterpart and forward **kwargs into it, so an
# empty kwargs dict already reproduces the plain counterpart's architecture shape (hidden
# dims / expert counts / d_model) — only num_cross_layers needs to be stated explicitly for
# the two Cross variants.
#
# seed_override is None for the 13 main-sweep entries (use the global RANDOM_STATE, same
# behavior as before this field existed) or an explicit int for the 8 diagnostic entries
# appended below.

CONFIGS = [
    # --- Single-task baselines (3) ---
    (WideAndDeepModel, dict(deep_dims=(360, 200, 80)), 'WideAndDeepModel', None),
    (DeepFMModel, dict(dnn_dims=(360, 200, 80)), 'DeepFMModel', None),
    (DCNv2Model, dict(num_cross_layers=3, deep_dims=(360, 200, 80)), 'DCNv2Model', None),
    # --- Plain MTL (4) ---
    (ESMMModel, dict(hidden_dims=(360, 200, 80)), 'ESMMModel', None),
    (ESMM_SharedBottom, dict(trunk_dims=(360, 200, 80)), 'ESMM_SharedBottom', None),
    (ESMM_MMoE, dict(num_experts=4, expert_hidden=360, d_model=128), 'ESMM_MMoE', None),
    (ESMM_PLE, dict(d_model=128, expert_hidden=256, num_shared_experts=1, num_task_experts=1), 'ESMM_PLE', None),
    # --- Wide / Cross MTL (6) ---
    (ESMMModel_Wide, dict(), 'ESMMModel_Wide', None),
    (ESMM_SharedBottomWide, dict(), 'ESMM_SharedBottomWide', None),
    (ESMM_MMoE_Wide, dict(), 'ESMM_MMoE_Wide', None),
    (ESMM_PLE_Wide, dict(), 'ESMM_PLE_Wide', None),
    (ESMM_PLE_Cross, dict(num_cross_layers=3), 'ESMM_PLE_Cross', None),
    (ESMM_PLE_WideCross, dict(num_cross_layers=3), 'ESMM_PLE_WideCross', None),
]

# --- Two-tower collapse diagnostic (8) ---
# One of the 13 main-sweep results (ESMMModel_Wide) came back at chance-level
# CTCVR≈0.50/CVR≈0.50 in the v1 (unseeded) run — see results/v1_unseeded/README.md. Now
# that harness.py actually seeds model init, re-run ESMMModel + ESMMModel_Wide across 4
# additional seeds each; combined with each architecture's seed=42 run in the main sweep
# above, that's 5 total seeds per architecture — enough to check whether the two-tower
# family's collapse is seed-dependent now that seeding is actually fixed.
DIAGNOSTIC_SEEDS = [17, 101, 202, 303]
for _s in DIAGNOSTIC_SEEDS:
    CONFIGS.append((ESMMModel, dict(hidden_dims=(360, 200, 80)), f'ESMMModel_seed{_s}', _s))
    CONFIGS.append((ESMMModel_Wide, dict(), f'ESMMModel_Wide_seed{_s}', _s))

# --- Seed-robustness scan for the leaderboard podium (12) ---
# The v2 podium spacing (MMoE 0.6672 > PLE_WideCross 0.6652 > MMoE_Wide 0.6620) sits inside
# the historically observed ±0.004-0.005 init/seed noise band, and the v1→v2 podium order
# flipped under a (then-uncontrolled) init change. Re-run the top contenders — the champion,
# the runner-up, and the best single-task model — across the same 4 extra seeds so the article
# can report mean ± std per model and whether the podium ordering is seed-stable at all.
for _s in DIAGNOSTIC_SEEDS:
    CONFIGS.append((ESMM_MMoE, dict(num_experts=4, expert_hidden=360, d_model=128), f'ESMM_MMoE_seed{_s}', _s))
    CONFIGS.append((ESMM_PLE_WideCross, dict(num_cross_layers=3), f'ESMM_PLE_WideCross_seed{_s}', _s))
    CONFIGS.append((DCNv2Model, dict(num_cross_layers=3, deep_dims=(360, 200, 80)), f'DCNv2Model_seed{_s}', _s))

assert len(CONFIGS) == 33, f'expected 33 configs, got {len(CONFIGS)}'


def load_data():
    """Parse -> vocab -> normalize, all cached/skip-if-present via the sibling data-prep module."""
    os.makedirs(PROCESSED_FULL_DIR, exist_ok=True)
    print(f'[data] Ensuring parsed full-split Parquet under {PROCESSED_FULL_DIR} ...')
    p_train, p_test = ensure_full_split_parquet_streaming(
        DATA_DIR, PROCESSED_FULL_DIR, SPARSE_COLS, DENSE_COLS, DENSE_FEAT_COLS)

    print('[data] Loading (or building) filtered sparse vocabs ...')
    vocabs, sparse_cardinalities = load_or_build_sparse_vocabs_filtered_parquet(
        p_train, SPARSE_COLS, min_count=5,
        cache_path=PREPROCESSED_SPARSE_VOCAB_CACHE,
        force_rebuild=False,
        vocab_scan_rows_per_batch=VOCAB_SCAN_ROWS_PER_BATCH,
    )

    if not (os.path.isfile(PREPROCESSED_TRAIN) and os.path.isfile(PREPROCESSED_TEST)):
        print('[data] Streaming dense log1p -> normalized Parquet ...')
        stream_normalize_parquet(
            p_train, PREPROCESSED_TRAIN, SPARSE_COLS, DENSE_FEAT_COLS,
            norm_stream_batch_rows=NORM_STREAM_BATCH_ROWS)
        stream_normalize_parquet(
            p_test, PREPROCESSED_TEST, SPARSE_COLS, DENSE_FEAT_COLS,
            norm_stream_batch_rows=NORM_STREAM_BATCH_ROWS)
    else:
        print('[data] Normalized train/test Parquet already present; reusing.')

    return vocabs, sparse_cardinalities


def _sort_key(row):
    v = row.get('CTCVR_AUC')
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return float('-inf')
    return v


def _fmt(x, digits=4):
    if isinstance(x, float) and not math.isnan(x):
        return f'{x:.{digits}f}'
    return '—'


def render_markdown(results_sorted):
    lines = [
        '# Ali-CCP MTL Leaderboard',
        '',
        f'Protocol: {EPOCHS} epoch, batch_size={BATCH_SIZE}, embed_dim={EMBED_DIM}, '
        f'seed={RANDOM_STATE} for the main sweep (see Seed column per row; diagnostic '
        f'rows vary), full test Parquet eval. Sorted by CTCVR AUC descending.',
        '',
        '| # | Model | Seed | CTCVR AUC | CVR AUC | CTR AUC | CTCVR PR-AUC | logloss (CTCVR) | ECE (CTCVR) | Params | Runtime (s) |',
        '|---|---|---|---|---|---|---|---|---|---|---|',
    ]
    for i, r in enumerate(results_sorted, 1):
        lines.append(
            f"| {i} | `{r['model']}` | {r.get('seed', RANDOM_STATE)} | {_fmt(r.get('CTCVR_AUC'))} | "
            f"{_fmt(r.get('CVR_AUC'))} | "
            f"{_fmt(r.get('CTR_AUC'))} | {_fmt(r.get('CTCVR_PR_AUC'))} | "
            f"{_fmt(r.get('logloss_ctcvr'))} | {_fmt(r.get('ECE_ctcvr'))} | "
            f"{r.get('params', 0):,} | {r.get('runtime_s', 0):.1f} |"
        )
    return '\n'.join(lines) + '\n'


def _write_results(results):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    results_sorted = sorted(results, key=_sort_key, reverse=True)
    with open(LEADERBOARD_JSON, 'w') as f:
        json.dump({
            'protocol': {
                'epochs': EPOCHS,
                'batch_size': BATCH_SIZE,
                'embed_dim': EMBED_DIM,
                'seed': RANDOM_STATE,
                'sparse_cols': SPARSE_COLS,
                'dense_feat_cols': DENSE_FEAT_COLS,
            },
            'results': results_sorted,
        }, f, indent=2)
    with open(LEADERBOARD_MD, 'w') as f:
        f.write(render_markdown(results_sorted))
    return results_sorted


def _load_progress():
    """Read the per-config checkpoint file if present; {} if absent or unreadable."""
    if not os.path.isfile(LEADERBOARD_PROGRESS_JSON):
        return {}
    try:
        with open(LEADERBOARD_PROGRESS_JSON) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        print(f'[checkpoint] Warning: could not read {LEADERBOARD_PROGRESS_JSON} ({e}); '
              f'starting with an empty checkpoint.')
        return {}


def _save_progress(progress):
    """Atomically persist the per-config checkpoint dict (temp file + os.replace), so a
    process death mid-write can never leave a corrupt/partial checkpoint file behind."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    tmp_path = LEADERBOARD_PROGRESS_JSON + '.tmp'
    with open(tmp_path, 'w') as f:
        json.dump(progress, f, indent=2)
    os.replace(tmp_path, LEADERBOARD_PROGRESS_JSON)


def run_all(vocabs, sparse_cardinalities):
    progress = _load_progress()
    if progress:
        done = [n for n, r in progress.items() if r.get('completed')]
        print(f'[checkpoint] Loaded {LEADERBOARD_PROGRESS_JSON}: '
              f'{len(done)}/{len(CONFIGS)} config(s) already completed -> {done}')

    results = []
    total = len(CONFIGS)
    for i, (model_cls, ctor_kwargs, name, seed_override) in enumerate(CONFIGS, 1):
        cached = progress.get(name)
        if cached is not None and cached.get('completed'):
            print(f'[{i}/{total}] {name}: already completed in checkpoint '
                  f'(CTCVR={_fmt(cached.get("CTCVR_AUC"))}); skipping retrain.')
            results.append({k: v for k, v in cached.items() if k != 'completed'})
            _write_results(results)
            continue

        seed_used = seed_override if seed_override is not None else RANDOM_STATE
        print(f'\n{"=" * 70}\n{name}  ({model_cls.__name__}, kwargs={ctor_kwargs}, '
              f'seed={seed_used})\n{"=" * 70}')
        t0 = time.time()
        model, losses, train_meta = train_esmm_parquet_rowgroups(
            PREPROCESSED_TRAIN, vocabs, sparse_cardinalities, SPARSE_COLS, DENSE_FEAT_COLS,
            epochs=EPOCHS, batch_size=BATCH_SIZE, embed_dim=EMBED_DIM, seed=seed_used,
            model_ctor=model_cls, model_ctor_kwargs=ctor_kwargs,
        )
        metrics = evaluate_esmm_multitask_streaming_parquet(
            model, PREPROCESSED_TEST, vocabs, SPARSE_COLS, DENSE_FEAT_COLS,
        )
        invalid_metrics = [
            m for m in ('CTCVR_AUC', 'CVR_AUC', 'CTR_AUC')
            if metrics.get(m) is None or math.isnan(metrics.get(m))
        ]
        if invalid_metrics:
            raise ValueError(
                f'[{name}] Invalid (None or NaN) metric(s) {invalid_metrics} in evaluation '
                f'result {metrics}; refusing to checkpoint this config as completed.'
            )
        runtime_s = time.time() - t0
        n_params = count_parameters(model)
        print(f'[{name}] CTCVR_AUC={metrics["CTCVR_AUC"]:.4f}  CVR_AUC={metrics["CVR_AUC"]:.4f}  '
              f'CTR_AUC={metrics["CTR_AUC"]:.4f}  params={n_params:,}  runtime={runtime_s:.1f}s')

        result = {
            'model': name,
            'model_class': model_cls.__name__,
            'seed': seed_used,
            'CTCVR_AUC': metrics.get('CTCVR_AUC'),
            'CVR_AUC': metrics.get('CVR_AUC'),
            'CTR_AUC': metrics.get('CTR_AUC'),
            'CTCVR_PR_AUC': metrics.get('CTCVR_PR_AUC'),
            'CTR_PR_AUC': metrics.get('CTR_PR_AUC'),
            'logloss_ctcvr': metrics.get('logloss_ctcvr'),
            'logloss_ctr': metrics.get('logloss_ctr'),
            'ECE_ctcvr': metrics.get('ECE_ctcvr'),
            'ECE_ctr': metrics.get('ECE_ctr'),
            'params': int(n_params),
            'runtime_s': round(runtime_s, 1),
            'train_wall_seconds': round(float(train_meta.get('train_wall_seconds', 0.0)), 1),
            'samples_per_sec': round(float(train_meta.get('samples_per_sec', 0.0)), 1),
        }
        results.append(result)

        # Durable per-config checkpoint: atomic write (temp file + os.replace) so a
        # mid-write crash/disconnect can't corrupt it, and a restarted session can skip
        # this config next time via the 'completed' flag.
        progress[name] = dict(result, completed=True)
        _save_progress(progress)
        print(f'[{i}/{total}] {name} done: CTCVR={_fmt(result.get("CTCVR_AUC"))} (checkpointed)')

        # Also refresh the human-facing leaderboard after every config so partial
        # progress survives a crash/preemption.
        _write_results(results)

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return results


def main():
    print(f'Device: {"cuda" if torch.cuda.is_available() else "cpu"}')
    print(f'[data] DATA_DIR={DATA_DIR}')
    vocabs, sparse_cardinalities = load_data()
    print(f'[data] {len(SPARSE_COLS)} sparse fields, cardinalities={sparse_cardinalities}')

    results = run_all(vocabs, sparse_cardinalities)
    results_sorted = _write_results(results)

    print(f'\nWrote {LEADERBOARD_JSON}')
    print(f'Wrote {LEADERBOARD_MD}')
    print('\n' + render_markdown(results_sorted))
    return results_sorted


if __name__ == '__main__':
    main()
