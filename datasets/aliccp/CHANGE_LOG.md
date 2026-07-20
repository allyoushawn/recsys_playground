# CHANGE_LOG — aliccp_data_preparation → datasets/aliccp migration

## Context

This folder was built 2026-07-17 as a clean, dedicated Ali-CCP data pipeline for a
public github.io article on MTL work (`knowledge_base/projects/career_skill_building/
building_github_io/20260715_recsys_mtl/`). It started as an additive clone of the
original `datasets/aliccp/` (in production use since 2026-05-31), extended with new
tar-extraction + MD5-checksum-verification code (`extract.py`) that didn't exist
anywhere in the repo before, and hardened by a Codex code review (fixed a partial-
extraction-mistaken-for-complete bug and added a safe-extraction filter).

`aliccp_mtl_experiments/` — the folder that produced the article's full 33-run
leaderboard — has depended on this folder (not `datasets/aliccp/`) since it was built,
via `from aliccp_data_preparation.data import ...` / `from aliccp_data_preparation.encode
import ...`.

## The plan

Replace `datasets/aliccp/` with this folder's content, since this is now the more
correct, better-tested, better-documented version of the same data layer. Concretely:

1. Delete `datasets/aliccp/` (5 files: README.md, __init__.py, data.py, encode.py,
   smoke.py — safely recoverable via git history, already committed).
2. Move this folder (`aliccp_data_preparation/`, now 7 files including this changelog)
   to `datasets/aliccp/`, replacing what was deleted.
3. Update the two import lines in `aliccp_mtl_experiments/harness.py` and
   `run_experiments.py` from `aliccp_data_preparation.X` to `datasets.aliccp.X` (that
   package name stops existing after step 2, so these must change or that folder breaks
   entirely — not just a partial break, a complete one, since these are its core data
   dependencies).

## Compatibility audit before executing

Before touching anything, diffed this folder against `datasets/aliccp/` file-by-file
(manual pass + an independent Codex review) to make sure nothing that depends on the
OLD folder's exact API would silently break.

**Confirmed clean, no risk:**
- `encode.py`: byte-identical except a docstring.
- All 17 shared constants: identical values, including exact list order (verified —
  no silent default/filename/column-list drift).
- 8 of 9 same-name functions: identical executable logic.
- `smoke.py`, `__init__.py` (aside from the fix below): equivalent behavior, only
  import style / messaging differs.

**Intentionally dropped, confirmed zero real callers repo-wide:**
- `parse_raw_ali_ccp()` — in-memory parser, superseded by the streaming version this
  folder already uses (`parse_raw_ali_ccp_streaming_writes`).
- `load_or_parse_ali_ccp()` — in-memory load-or-parse wrapper.
- A legacy `esmm_cache/`-fallback branch inside `ensure_full_split_parquet_streaming`
  (function itself kept; only the fallback-copy-from-legacy-dir path removed). This is
  an accepted behavior change — it only mattered for a pre-standardization cache
  location that predates this whole pipeline.

**Found missing and restored (this is the actual fix in this changeset), because they
have real callers that would otherwise break:**
- `_sample_tag_for_cache()` — private helper, restored to `data.py`. Found via Codex
  review: `experiments/20260404_ali_cpp_esmm/esmm_ali_ccp_impl.py` line 24 imports it
  **by name, explicitly** (`from datasets.aliccp.data import (..., _sample_tag_for_cache,
  ...)`). Without it, `import esmm_ali_ccp_impl` — and everything that imports
  `esmm_ali_ccp_impl` (10+ files: the ESMM/AdaOrder-paper notebooks, `20260519_model_
  comparison.ipynb`, `tests/test_ndm_models.py`) — would fail immediately with
  `ImportError` at module-import time, not just when some rarely-used function is
  called.
- `parquet_split_summary()`, aliased `summarize_split` — restored to `data.py`, and
  re-added to `__init__.py`'s re-export list (matches the original package's public
  surface). Found via manual diff: `experiments/20260404_ali_cpp_esmm/
  20260404_esmm_experiment.ipynb` line 325 calls `summarize_split(p_train, p_test,
  summary_batch_rows=STREAM_PARSE_CHUNK_ROWS)` — a diagnostic print (row counts,
  click/conversion totals), not something affecting training output, but real working
  code that would otherwise raise `NameError` if that cell is ever re-run.

Both restorations are verbatim copies of the original implementations (not
reinvented), verified by direct comparison against `datasets/aliccp/data.py`, and
functionally re-tested after restoration (AST parse + live import + call).

## What this means for `aliccp_mtl_experiments/`

Nothing in this changeset touches that folder's model/harness/experiment code. Once
the two import lines noted above are updated post-move, its behavior is unchanged —
none of the dropped-or-restored symbols were ever in its dependency chain
(`ensure_full_split_parquet_streaming`, `load_or_build_sparse_vocabs_filtered_parquet`,
`stream_normalize_parquet`, `_precompute_sparse_encode_tables`,
`encode_and_tensorize_arrow`, `encode_and_tensorize_fast` — none overlap with what was
dropped or restored here).
