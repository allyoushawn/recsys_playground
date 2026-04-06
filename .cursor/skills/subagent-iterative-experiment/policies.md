# Policies — subagent iterative experiment

## Runtime budget

- **30 minutes** wall-clock per round for remote execution (papermill on Colab), unless the user overrides.
- On timeout, runtime reports `failure_class: TIMEOUT`. Lead follows [Auto-unblock](#auto-unblock-policy).

## Critic and replanning

- If `needs_replan: true`, planner rewrites proposals (same round number until a valid set exists).
- If `needs_replan` is true for **3 consecutive critic cycles** (same experiment run / same round context): lead escalates to user. If no user response per [Auto-unblock](#auto-unblock-policy), **stop the loop cleanly**; scribe records the reason.

## Analyzer cadence (adaptive)

- Default: run **experiment-analyzer** every **2** rounds.
- After consecutive regressions or failures: every **1** round.
- After stable improvements: relax toward every **3** rounds.

## Change taxonomy

Three tiers — use consistently across runtime, code-change, and lead routing.

### Scientific changes

**Never** auto-apply. Require user approval for intent.

Includes: model architecture, loss function, sampling strategy, hyperparameter **intent**, evaluation methodology, data slice (substantive).

### Operational downgrades

Lead may apply after user has had a chance to respond; if no response, use [Auto-unblock](#auto-unblock-policy).

Includes: batch size reduction (e.g. OOM), epoch count reduction (e.g. timeout), disabling a feature for stability, changing training precision mode when it materially affects training.

Implemented via **experiment-code-change**, not runtime silent edits to experiment semantics.

### Execution-safe changes

**experiment-runtime** may apply autonomously during preflight and limited retries.

Includes: papermill compatibility, **mechanical** GPU fixes per `gpu-review` **Critical** only (e.g. missing `.to(device)`, hardcoded CPU), import path fixes, notebook path fixes, log path fixes, remote sync fixes.

**Not** execution-safe at runtime: **Performance** row items from `gpu-patterns.md` that can change training behavior (e.g. mixed precision, `pin_memory`) — treat as operational downgrade if applied deliberately.

Runtime classifies failures with `failure_is_execution_safe`. **Operational-downgrade** and **scientific** failures are reported to the lead; runtime does not change batch size, epochs, architecture, loss, or data slice autonomously.

## Auto-unblock policy

**Framing:** Logical policy for the lead in the current session — not a guaranteed OS-level timer. If the user replies before the lead proceeds, user choice wins. If the environment cannot wait, use the safest allowed fallback immediately after surfacing options once.

### Operational-downgrade failures (OOM, TIMEOUT, etc.)

1. Lead surfaces options to the user.
2. If no user response: lead invokes **experiment-code-change** with a **predefined** safe downgrade (e.g. halve batch size for OOM, halve epochs for timeout).
3. Re-run **experiment-runtime**.
4. Scribe: `AUTO-UNBLOCK: [action] [reason] [original error]`.
5. Leaderboard row: comparability `operational_downgrade`.

### Critic 3-strike

- Do not auto-generate new proposals.
- Stop the loop cleanly; scribe: `AUTO-UNBLOCK: loop stopped after 3 consecutive critic rejections. Awaiting user direction.`

### Scientific failures

- Do not auto-fix. Stop cleanly; scribe: `AUTO-UNBLOCK: loop stopped, scientific fix needed.`

## Leaderboard comparability

Each leaderboard row MUST include:

| Tag | Meaning |
|-----|---------|
| `canonical` | Ran as planned; no auto-unblock downgrades. |
| `operational_downgrade` | Ran after AUTO-UNBLOCK operational change (document what changed). |
| `non_comparable` | Conditions make direct comparison unreliable (scribe applies mechanically when lead flags). |

Scribe does not interpret strategy; it records tags per lead/runtime flags and AUTO-UNBLOCK records.

## Scribe log path

Lead derives once at setup:

```text
experiments/logs/YYYYMMDD_<notebook_stem>_trial.md
```

Example: `notebooks/ad_hoc/experiment_foo.ipynb` on 2026-03-25 → `experiments/logs/20260325_experiment_foo_trial.md`.

Create `experiments/logs/` if missing. Reuse the same path for all rounds of that run.
