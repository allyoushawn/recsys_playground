# Handoff contracts — subagent iterative experiment

Parent Agent (lead) pastes these fields into each subagent invocation. Subagents have **no** prior chat history.

## Handoff block (minimum)

```yaml
experiment_id: "<string>"
round: <int>
notebook_path: "<repo-relative .ipynb>"
goal_metric: "<name>"
goal_threshold: <number or string>
goal_operator: ">" | ">=" | "<" | "<="
run_state:
  best_experiment: "<letter|null>"
  best_metric_value: <float|null>
  critic_replan_streak: <int>
  last_analyzer_round: <int|null>
scribe_doc_path: "<from policies.md convention>"
hostname: "<colab trycloudflare host>"  # runtime only
```

Optional: `round_history`, `analyzer_pack`, `planner_output`, `critic_output`, `code_change_output`, `runtime_output`, `ssh_config` (see run-notebook-on-colab skill).

---

## PLANNER_OUTPUT

```yaml
round: <int>
hypotheses:
  - id: "<letter>"
    description: "<one line>"
    category: "<strategies.md category>"
    rationale: "<evidence-grounded>"
    estimated_runtime_class: "short" | "medium" | "long"
    requires_code_change: <bool>
```

Rules: no file edits; no duplicate of prior runs per `round_history`.

---

## CRITIC_OUTPUT

```yaml
round: <int>
accepted:
  - id: "<letter>"
    requires_code_change: <bool>  # pass-through from planner; do not modify
rejected:
  - id: "<letter>"
    reason_code: "REDUNDANT" | "OVER_BUDGET" | "CONTRADICTS_EVIDENCE" | "TECHNICALLY_UNSOUND" | "LOW_EXPECTED_IMPACT"
    explanation: "<one sentence>"
total_estimated_runtime: "<e.g. 22min>"
budget_ok: <bool>
execution_route: "code_change_then_runtime" | "runtime_only"
# execution_route = code_change_then_runtime iff ANY accepted requires_code_change
needs_replan: <bool>
```

---

## CODE_CHANGE_OUTPUT

```yaml
round: <int>
files_changed:
  - path: "<path>"
    cells_modified: ["shared_utilities", "new_round_N", ...]
    summary: "<short>"
new_classes_added: ["<names>"]
cache_invalidation_needed: <bool>
cache_files_to_delete: ["<optional list>"]
```

---

## RUNTIME_OUTPUT

```yaml
round: <int>
status: "success" | "failed" | "timeout"
elapsed_minutes: <float>
preflight_fixes: ["<strings>"]
failure_class: null | "INFRA" | "DEPENDENCY" | "CODE" | "DATA" | "OOM" | "TIMEOUT"
failure_is_execution_safe: null | <bool>
retries_used: <0-3>
possibly_hung: <bool>  # optional: true if no log progress >3min while process alive
metrics:  # if success
  - experiment: "<letter>"
    Hit@50: <float>
    MRR@50: <float>
    NDCG@50: <float>
    wall_clock_seconds: <int>
    comparability: "canonical" | "operational_downgrade" | "non_comparable"
error_trace: "<if failed>"
log_reference: "<path or terminal id>"
```

---

## SCRIBE_OUTPUT

```yaml
round: <int>
doc_path: "<path>"
current_best:
  experiment: "<letter>"
  metric_name: "<goal_metric>"
  value: <float>
goal_achieved: <bool>
observations_added:
  - "<evidence-only bullet>"
leaderboard_rows_appended:
  - experiment: "<letter>"
    metrics: { ... }
    comparability: "canonical" | "operational_downgrade" | "non_comparable"
auto_unblock_notes: ["<optional>"]
```

---

## ANALYZER_OUTPUT

```yaml
as_of_round: <int>
trigger_reason: "<string>"
categories_tried: ["<strings>"]
categories_untried: ["<strings>"]
categories_underexplored: ["<strings>"]
local_opt_warning: <bool>
local_opt_detail: "<string>"
avoid_list:
  - strategy: "<string>"
    reason: "<evidence>"
distance_to_goal:
  current_best: <float>
  target: <float>
  gap: <float>
  trend: "improving" | "flat" | "declining"
```

No experiment proposals; no “do X next” recommendations.

---

## Reason codes (critic)

| Code | Use |
|------|-----|
| REDUNDANT | Too similar to prior run |
| OVER_BUDGET | Exceeds 30min aggregate or single hypothesis |
| CONTRADICTS_EVIDENCE | Conflicts with scribe/strategies evidence |
| TECHNICALLY_UNSOUND | Infeasible / incompatible |
| LOW_EXPECTED_IMPACT | Marginal gain vs cost |

## Failure classes (runtime)

| Class | Typical handling |
|-------|------------------|
| INFRA | Escalate / retry SSH |
| DEPENDENCY | Execution-safe fix → retry if import/path only |
| CODE | If logic bug → lead → code-change; not runtime semantic edit |
| DATA | Sync/path only → execution-safe; else escalate |
| OOM | Operational downgrade via lead/code-change |
| TIMEOUT | Operational downgrade or stop |
