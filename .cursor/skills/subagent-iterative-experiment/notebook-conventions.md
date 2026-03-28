# Notebook conventions (experiment notebooks)

Referenced by **experiment-code-change** and the orchestration skill. Aligns with [iterative-experiment/SKILL.md](../iterative-experiment/SKILL.md).

## Cell layout

| Region | Purpose | Editable by code-change? |
|--------|---------|---------------------------|
| Cells 0–K | Data load, preprocessing, `CACHE_DIR`, config | **No** |
| Shared utilities cell | Model classes, losses, datasets, eval, training helpers | **Yes** (append new defs) |
| Round 1, 2, … cells | One cell per round, early-exit cache guard | **Yes** (add new round cell only) |

Do **not** edit prior round cells after they are committed. Add a new cell for each new round.

## `CACHE_DIR`

Define once in setup (config cell):

```python
import os
CACHE_DIR = '/content/drive/MyDrive/colab/data/experiment_cache'
os.makedirs(CACHE_DIR, exist_ok=True)
```

Adjust path if your Colab layout differs; keep consistent across runs.

## Experiment letters

- Round 1: A, B, C  
- Round 2: D, E, F  
- Round 3: G, H, I  
- Continue alphabetically per round.

## Early-exit cache guard (round cell)

Replace `N` with round number; replace experiment keys with letters for that round.

```python
import json, os
_ROUND_N_CACHE = os.path.join(CACHE_DIR, 'round_N_results.json')

if not os.path.exists(_ROUND_N_CACHE):
    # ===================== ROUND N =====================
    results = {}
    # --- Experiment X ---
    # ... train, eval ...
    results['X'] = {'Hit@50': ..., 'MRR@50': ..., 'NDCG@50': ...}
    with open(_ROUND_N_CACHE, 'w') as f:
        json.dump(results, f)
    # Print round summary table
else:
    with open(_ROUND_N_CACHE) as f:
        results = json.load(f)
    print(f'ROUND N: SKIPPED (cached)')
    for name, m in results.items():
        print(f"  {name}: Hit@50={m['Hit@50']:.4f}")
```

## Round cell requirements

Inside the guard (first run):

1. Run all **accepted** experiments for this round sequentially.
2. For each: print header, train, evaluate, print metrics and wall-clock (`time.time()`).
3. End with a **summary table** comparing experiments in the round.
4. Write `round_N_results.json` only after all experiments in the round complete.

## Cache invalidation

If the **shared utilities cell** changes (new/changed model or loss used by prior rounds), prior `round_*_results.json` may be stale. Set `cache_invalidation_needed: true` in code-change output and list which cache files users should delete to force re-runs.

## Papermill / Colab

Follow [run-notebook-on-colab/compatibility.md](../run-notebook-on-colab/compatibility.md). **experiment-runtime** applies execution-safe fixes; code-change should not rely on brittle `%` / `!` magics in new cells.
