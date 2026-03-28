# Example handoffs — subagent iterative experiment

Synthetic round illustrating payloads between lead and subagents.

## Setup (lead, Phase 0)

- `notebook_path`: `notebooks/ad_hoc/experiment_foo.ipynb`
- `scribe_doc_path`: `experiments/logs/20260325_experiment_foo_trail.md`
- `goal_metric`: `Hit@50`, `goal_threshold`: `0.20`, `goal_operator`: `>`

---

## Analyzer (optional, cadence)

**Handoff in:** `scribe_doc` (empty or prior rounds), `strategies_reference` (link to iterative-experiment/strategies.md), `round_history`, `trigger_reason: "round 2, default cadence"`

**Sample ANALYZER_OUTPUT:**

```yaml
as_of_round: 2
trigger_reason: "every 2 rounds"
categories_tried: ["Architecture", "Loss Function"]
categories_untried: ["Data Quality", "Training Procedure", "Evaluation Methodology", "Negative Sampling", "Regularization"]
categories_underexplored: ["Training Procedure"]
local_opt_warning: false
local_opt_detail: ""
avoid_list: []
distance_to_goal:
  current_best: 0.152
  target: 0.20
  gap: 0.048
  trend: "improving"
```

---

## Planner

**Handoff in:** `run_state`, `round_history`, `analyzer_pack` (YAML above), `strategies_reference`

**Sample PLANNER_OUTPUT:**

```yaml
round: 2
hypotheses:
  - id: "D"
    description: "NeuMF GMF+MLP vs current MLP"
    category: "Architecture Changes"
    rationale: "MLP best 0.152; NeuMF adds linear path per strategies.md"
    estimated_runtime_class: "medium"
    requires_code_change: true
  - id: "E"
    description: "Train on full 500k interactions"
    category: "Data Quality and Quantity"
    rationale: "Data category untried; analyzer lists as untried"
    estimated_runtime_class: "long"
    requires_code_change: true
  - id: "F"
    description: "Cosine LR on current MLP"
    category: "Training Procedure"
    rationale: "Training category underexplored"
    estimated_runtime_class: "medium"
    requires_code_change: false
```

---

## Critic

**Handoff in:** `planner_output`, `round_history`, `runtime_budget: 30`, `scribe_learnings` (optional)

**Sample CRITIC_OUTPUT:**

```yaml
round: 2
accepted:
  - id: "D"
    requires_code_change: true
  - id: "F"
    requires_code_change: false
rejected:
  - id: "E"
    reason_code: "OVER_BUDGET"
    explanation: "Full dataset alone ~25min; exceeds 30min cap with D"
total_estimated_runtime: "20min"
budget_ok: true
execution_route: "code_change_then_runtime"
needs_replan: false
```

Lead: because `execution_route` is `code_change_then_runtime`, invoke **experiment-code-change** then **experiment-runtime**.

---

## Code-change

**Handoff in:** `accepted_hypotheses` (D, F), `notebook_path`, `round_number: 2`, `existing_shared_cell_summary`

**Sample CODE_CHANGE_OUTPUT:**

```yaml
round: 2
files_changed:
  - path: "notebooks/ad_hoc/experiment_foo.ipynb"
    cells_modified: ["shared_utilities", "new_round_2"]
    summary: "Added NeuMF; round 2 cell for D (NeuMF+BPR) and F (MLP+BPR+cosine)"
new_classes_added: ["NeuMF"]
cache_invalidation_needed: false
```

---

## Runtime

**Handoff in:** `notebook_path`, `hostname`, `ssh_config` (from run-notebook-on-colab skill), `round_number`

**Sample RUNTIME_OUTPUT (success):**

```yaml
round: 2
status: "success"
elapsed_minutes: 18
preflight_fixes: ["Critical GPU: .to(device) for tensor in NeuMF path"]
failure_class: null
failure_is_execution_safe: null
retries_used: 0
metrics:
  - experiment: "D"
    Hit@50: 0.172
    MRR@50: 0.039
    NDCG@50: 0.066
    wall_clock_seconds: 480
    comparability: "canonical"
  - experiment: "F"
    Hit@50: 0.148
    MRR@50: 0.028
    NDCG@50: 0.049
    wall_clock_seconds: 540
    comparability: "canonical"
error_trace: ""
log_reference: "papermill experiment_foo_output.ipynb"
```

---

## Scribe

**Handoff in:** `runtime_output`, `critic_output`, `planner_output`, `round_number`, `scribe_doc_path`, `prior_best: {experiment: A, Hit@50: 0.152}`

**Sample markdown append (excerpt):**

```markdown
## Round 2 — 2026-03-25

### Proposed (planner)
- D: NeuMF …
- E: full data … (rejected)
- F: Cosine LR …

### Critic
- Accepted: D, F | Rejected: E (OVER_BUDGET)

### Results
| Exp | Hit@50 | MRR@50 | NDCG@50 | time | comparability |
|-----|--------|--------|---------|------|---------------|
| D   | 0.172  | 0.039  | 0.066   | 480s | canonical     |
| F   | 0.148  | 0.028  | 0.049   | 540s | canonical     |

### Observations (evidence only)
- D: +0.020 Hit@50 vs A (0.172 vs 0.152)
- F: below A on Hit@50 (0.148 vs 0.152)
```

**Sample SCRIBE_OUTPUT:**

```yaml
round: 2
doc_path: "experiments/logs/20260325_experiment_foo_trail.md"
current_best:
  experiment: "D"
  metric_name: "Hit@50"
  value: 0.172
goal_achieved: false
observations_added:
  - "D +0.020 Hit@50 vs A (0.172 vs 0.152)"
  - "F Hit@50 0.148 below A 0.152"
leaderboard_rows_appended:
  - experiment: "D"
    metrics: {Hit@50: 0.172, MRR@50: 0.039, NDCG@50: 0.066}
    comparability: "canonical"
auto_unblock_notes: []
```

---

## AUTO-UNBLOCK example (OOM)

After user timeout, code-change halves batch; scribe adds:

```text
AUTO-UNBLOCK: batch_size 512→256 | OOM CUDA | trace: ...
```

Leaderboard row for that experiment: `comparability: operational_downgrade`.
