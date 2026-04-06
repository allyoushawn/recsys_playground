# Validation checklist — subagent iterative experiment

## Dry-run (no Colab)

- [ ] Lead can assemble a Handoff block matching [contracts.md](contracts.md).
- [ ] Invoke `/experiment-planner` with fake `round_history`; output validates against PLANNER_OUTPUT schema.
- [ ] Invoke `/experiment-plan-critic` with sample planner output; `execution_route` matches OR of `requires_code_change`.
- [ ] With mixed accept (one `requires_code_change: true`), route is `code_change_then_runtime`.
- [ ] With all `requires_code_change: false`, route is `runtime_only`.
- [ ] Critic `needs_replan: true` triggers replan path in lead (no code-change until resolved).

## Notebook / code-change

- [ ] `/experiment-code-change` edits only shared utilities + new round cell per [notebook-conventions.md](notebook-conventions.md).
- [ ] Cache guard pattern present in new round cell; `round_N_results.json` naming correct.

## Runtime (real Colab)

- [ ] SSH + `nvidia-smi` from run-notebook-on-colab Phase 2.
- [ ] Papermill completes or fails with classified `failure_class`.
- [ ] Runtime does not change batch size / epochs / architecture without lead → code-change path.
- [ ] Preflight applies at most **Critical** GPU fixes per policies.

## Scribe

- [ ] Log file created at `experiments/logs/YYYYMMDD_<stem>_trial.md`.
- [ ] Append-only; no strategy “we should” language in observations.
- [ ] Leaderboard rows include `comparability`.
- [ ] After AUTO-UNBLOCK operational fix, tag `operational_downgrade`.

## Analyzer

- [ ] Output has no experiment proposals and no “next action” directives.
- [ ] Cadence: every 2 rounds default; tightens on regression per policies.

## Regression

- [ ] [iterative-experiment/SKILL.md](../iterative-experiment/SKILL.md) unchanged.
- [ ] Original iterative-experiment skill still runnable standalone.
