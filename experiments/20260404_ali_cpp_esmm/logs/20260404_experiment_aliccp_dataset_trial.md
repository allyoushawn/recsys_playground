# Experiment Trail Log — AliCCP Dataset

| Field | Value |
|-------|-------|
| **Notebook** | `notebooks/ad_hoc/experiment_aliccp_dataset.ipynb` |
| **Goal metric** | AUC > 0.55 |
| **Date started** | 2026-04-04 |

---

## Leaderboard

| Rank | Experiment | Round | AUC | Accuracy | F1 | Wall clock (s) | Comparability |
|------|-----------|-------|-----|----------|----|----------------|---------------|
| 1 | A | 1 | 0.5393 | 0.9996 | 0.0000 | 334 | canonical |

**Current best: A — AUC 0.5393 (goal not achieved)**

---

## Round 1

### Proposals

| ID | Description | Category | Requires code change |
|----|-------------|----------|---------------------|
| A | Execute the notebook end-to-end to train the existing MLP (BCE, Adam, 15 epochs, 500k rows) and record test Accuracy, F1, and goal metric AUC. | Loss function alignment | No |

### Critic summary

- **Accepted:** A (runtime_only)
- **Rejected:** none
- **Estimated runtime:** 25 min
- **Budget OK:** yes
- **Execution route:** runtime_only

### Results

| Experiment | AUC | Accuracy | F1 | Wall clock (s) | Comparability |
|------------|-----|----------|----|----------------|---------------|
| A | 0.5393 | 0.9996 | 0.0000 | 334 | canonical |

- **Runtime status:** success
- **Elapsed:** 5.56 min
- **Preflight fixes:** converted `le.classes_` numpy-array membership check to set-based O(1) lookup
- **Retries used:** 1

### Observations

- A: AUC 0.5393, below goal of 0.55 (gap −0.0107).
- A: Accuracy 0.9996 with F1 0.0000 — model predicts all-negative, consistent with extreme class imbalance (positive rate ≈ 0.043%).
- A: AUC > 0.5 confirms the model captures some ranking signal despite degenerate classification output.
