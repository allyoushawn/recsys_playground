---
name: iterative-experiment
description: >-
  Iteratively improves ML experiment notebooks by proposing solutions,
  implementing them, running on Colab, and analyzing results across multiple
  rounds. Stops when the goal metric is met or max rounds reached. Use when the
  user says "iterate on experiments", "improve metrics iteratively",
  "run experiment rounds", or wants to systematically try multiple approaches
  to hit a metric target.
---

# Iterative Experiment

Autonomous experiment loop: analyze results → propose K solutions → implement →
run on Colab → check goal → repeat. Delegates execution to `run-notebook-on-colab`.

## Inputs

Gather from the user message or ask if missing:

| Input | Required | Default | Example |
|-------|----------|---------|---------|
| Notebook path | Yes | — | `notebooks/ad_hoc/experiment_foo.ipynb` |
| Goal metric + threshold | Yes | — | `Hit@50 > 0.2` |
| Max rounds | No | 3 | `3` |
| Solutions per round (K) | No | 3 | `3` |
| Colab hostname | Ask in Phase 0 | — | `xxx.trycloudflare.com` |

If the user gives a short notebook name, resolve it by searching `notebooks/`.

## Dependencies

This skill delegates Colab execution to **run-notebook-on-colab**.
Read [run-notebook-on-colab/SKILL.md](../run-notebook-on-colab/SKILL.md)
before starting — you will follow its Phase 3 (scp → papermill → poll) for
every execution step.

This skill uses **gpu-review** to audit GPU utilization before every Colab run.
Read [gpu-review/SKILL.md](../gpu-review/SKILL.md) and its
`references/gpu-patterns.md` at startup — you will follow its workflow before
each execution step.

For improvement ideas, consult [strategies.md](strategies.md).

## Workflow

### Phase 0: Setup

1. **Read the notebook.** Identify:
   - Model architecture (MLP, MF, NeuMF, etc.)
   - Loss function (MSE, BPR, cross-entropy, etc.)
   - Data pipeline (dataset size, sparsity, train/test split method)
   - Evaluation metrics and current results (if cells have output)

2. **Print a baseline summary** for the user:
   > Current setup: [model] with [loss], [N] interactions, [U] users,
   > [I] items. [metric] = [value]. Goal: [metric] > [threshold].

3. **Establish Colab connection** (once, reused for all rounds):
   - Ask the user for the Colab hostname (same prompt as
     run-notebook-on-colab Phase 2).
   - Verify SSH connectivity.
   - Store `<HOSTNAME>` for all subsequent rounds.

4. **GPU review** — Run the gpu-review workflow on the notebook:
   - Detect framework, audit device handling against `references/gpu-patterns.md`.
   - Fix any **Critical** findings (missing device detection, hardcoded CPU,
     tensors on wrong device) before proceeding.
   - Log **Performance** findings; apply quick wins (e.g., `pin_memory=True`)
     but defer heavy changes (mixed precision) unless they don't risk breakage.
   - Print the GPU review report for the user.

5. **Run baseline** (if the notebook has no results yet):
   - Follow run-notebook-on-colab Phase 3 to execute the notebook as-is.
   - Parse and record baseline metrics from the papermill log.

### Phase 1–N: Iteration Rounds

Repeat for up to `max_rounds`:

#### Step 1 — Analyze

- Read the latest results (from papermill log output).
- Identify what worked, what didn't, and why.
- Consult [strategies.md](strategies.md) for improvement ideas.
- Focus on the highest-impact lever not yet tried (see priority order
  in strategies.md).

#### Step 2 — Propose

- Propose K solutions, each with a one-line rationale.
- Print them for the user before implementing:
  > **Round N proposals:**
  > 1. [Solution] — [rationale]
  > 2. [Solution] — [rationale]
  > 3. [Solution] — [rationale]
- Each solution should be independent and testable in isolation.
- Build on the best-performing experiment from the previous round.

#### Step 3 — Implement

- **Shared utilities cell**: If the round introduces new model classes,
  loss functions, or dataset classes, add them to the shared definitions
  cell (or create one if it doesn't exist). Update in-place.
- **Round cell**: Add a new code cell for this round. The cell must:
  - Run all K experiments sequentially.
  - For each experiment: print experiment name, train, evaluate, print
    metrics and wall-clock time.
  - End with a summary table comparing all experiments in the round.
  - Include `time.time()` timing per experiment.

Example round cell output format:
```
============================================================
EXPERIMENT D: MLP + BPR + Multi-Neg (K=5)
============================================================
  Epoch 1/20 BPR loss=0.4853
  ...
  >> Hit@50=0.1460, MRR@50=0.0271, NDCG@50=0.0509 (110s)

============================================================
ROUND 2 SUMMARY (vs best so far: A=0.154)
============================================================
Experiment                       Hit@50   MRR@50  NDCG@50
-----------------------------------------------------------
D: MultiNeg(K=5)                 0.1460   0.0271   0.0509
E: PopNeg                        0.0240   0.0024   0.0066
F: Large+CosineLR                0.1360   0.0240   0.0460
```

#### Step 4 — GPU Review

Before sending to Colab, run the gpu-review workflow on the notebook:

1. Re-audit the notebook (new cells from Step 3 may introduce GPU issues).
2. Fix any **Critical** findings — these would cause silent CPU fallback
   or device mismatch errors on Colab.
3. Apply safe **Performance** fixes (`pin_memory`, `non_blocking`,
   `torch.no_grad` in eval). Skip mixed precision unless the user
   requested it or prior rounds showed it was stable.
4. If fixes were applied, briefly note them:
   > GPU review: fixed [N] issues (e.g., added `.to(device)` on new
   > tensor in Experiment D).

#### Step 5 — Execute

Follow **run-notebook-on-colab Phase 3** using the stored `<HOSTNAME>`:
1. `scp` the notebook to Colab.
2. Run via `papermill` (background, poll until done or 10 min).
3. If papermill fails, follow the error-fix loop from run-notebook-on-colab
   (fix cell → re-scp → re-run, up to 3 retries).

#### Step 6 — Check stopping criteria

- Parse the round summary table from the papermill log output.
- **If any experiment meets the goal** → stop, proceed to Final.
- **If max rounds reached** → stop, proceed to Final.
- **Otherwise** → next round (go to Step 1).

### Final: Summary

Print two things:

1. **Overall leaderboard** — all experiments from all rounds, sorted by
   the goal metric (best first):
   ```
   OVERALL RESULTS (ALL ROUNDS)
   ============================================================
   Experiment                       Hit@50   MRR@50  NDCG@50
   -----------------------------------------------------------
   G: NeuMF+BPR                     0.1720   0.0391   0.0657
   A: MLP+BPR                       0.1520   0.0345   0.0584
   ...
   ```

2. **Key findings** — 3–5 bullet points:
   - What was the single biggest improvement and why
   - What didn't work and why
   - What to try next if the goal wasn't met

## Notebook Convention

- **Cells 0–K**: Data loading, preprocessing, config. Untouched across rounds.
- **Cell K+1**: Shared utilities (model classes, loss functions, dataset
  classes, eval function, training loop). Updated in-place when new
  components are needed.
- **Cell K+2**: Round 1 experiments (appended by this skill).
- **Cell K+3**: Round 2 experiments (appended by this skill).
- ...and so on.

Each round cell is self-contained — it can reference models/functions from
the shared utilities cell but defines its own experiment logic.

## Experiment Naming

Use sequential letters across rounds:
- Round 1: A, B, C
- Round 2: D, E, F
- Round 3: G, H, I

This makes the overall leaderboard easy to read.

## Edge Cases

| Situation | Handling |
|-----------|----------|
| Goal met in Round 1 | Stop early, still print full summary |
| Papermill error | Follow run-notebook-on-colab error-fix loop (3 retries) |
| Colab disconnects mid-run | Detect SSH failure, ask user for new hostname, resume |
| No goal metric specified | Ask: "What metric should I optimize and what's the target?" |
| Notebook has no results yet | Run baseline in Phase 0 first |
| All K experiments regress | Note regression in analysis, try fundamentally different approaches next round |
