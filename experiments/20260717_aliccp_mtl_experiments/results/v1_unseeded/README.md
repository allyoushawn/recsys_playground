# v1 (unseeded) — archived reference results

These are real, measured results from the first full 13-model run of
`run_experiments.py` (protocol: 1 epoch, batch_size=4096, embed_dim=18,
`seed=42` label). At the time of this run, `harness.py`'s
`train_esmm_parquet_rowgroups` never actually called `torch.manual_seed` (or
`torch.cuda.manual_seed_all`) before model construction — so despite every
result being labeled `seed=42`, model initialization was drawn from the
uncontrolled global RNG state.

That means these numbers are **not properly seed-controlled or guaranteed
reproducible**, even though the run itself and the reported metrics are real.

The harness bug is now fixed (`torch.manual_seed(seed)` +
`torch.cuda.manual_seed_all(seed)` before model construction, in
`harness.py`). This archive is kept for **methodology-comparison reference**
(before/after fixing the seeding bug) — not as the canonical leaderboard
numbers. The canonical, properly-seeded results live back at
`results/aliccp_leaderboard.json` / `results/aliccp_leaderboard.md` from the
rerun.

Files:
- `aliccp_leaderboard.json` — full protocol + per-model results (moved as-is from `results/`)
- `aliccp_leaderboard.md` — rendered leaderboard table (moved as-is from `results/`)

Note: no `aliccp_leaderboard_progress.json` checkpoint file was present in
`results/` at archive time (only the two files above existed), so there is
nothing to archive for it.
