# Improvement Strategies

Consult this catalog when proposing solutions in each round. Strategies are
ordered by typical impact — try higher-impact changes first.

## Priority Order

1. Loss function alignment
2. Data quality and quantity
3. Architecture changes
4. Training procedure
5. Evaluation methodology
6. Negative sampling
7. Regularization

---

## 1. Loss Function Alignment

**Highest impact.** If the training loss doesn't match the evaluation metric,
fix this first.

| Symptom | Fix |
|---------|-----|
| Ranking metrics = 0 but train loss decreases | Switch from MSE/regression to BPR or sampled softmax |
| Classification metrics poor with MSE loss | Switch to cross-entropy |
| Regression metrics poor with ranking loss | Switch to MSE or Huber loss |

### Ranking losses

- **BPR (Bayesian Personalized Ranking)**: Pairwise — score(pos) > score(neg).
  Simple, robust, good default for ranking.
- **Sampled softmax**: Cross-entropy over 1 positive vs K negatives.
  Richer gradient per batch but can underperform BPR on sparse data.
- **Margin loss**: `max(0, margin - (pos_score - neg_score))`. Alternative to
  BPR, sometimes more stable.

### Evidence from experiments

BPR took Hit@50 from 0.000 to ~0.15 on Amazon Video Games (200k interactions).
Sampled softmax (100 neg) underperformed BPR on the same data (0.114 vs 0.152).

---

## 2. Data Quality and Quantity

**Second highest impact.** More data usually helps more than any model change.

| Symptom | Fix |
|---------|-----|
| Very sparse (<3 interactions/user avg) | Increase sample size, use full dataset |
| Noisy labels (e.g., rating=1 treated as positive) | Filter to rating ≥ 3 as positives |
| Cold-start users dominate test set | Filter to users with ≥ N interactions |

### Pitfalls

- Filtering to active users reduces dataset size — this can hurt more than it
  helps if the total dataset is small. On Amazon Video Games, filtering to
  users with ≥5 interactions cut data from 200k to 85k and degraded Hit@50
  from 0.152 to 0.076.
- More data helps but with diminishing returns. 200k→500k improved Hit@50
  from 0.152 to 0.160 (modest).

---

## 3. Architecture Changes

| Symptom | Fix |
|---------|-----|
| MLP plateau with BPR loss | Try NeuMF (GMF + MLP paths combined) |
| MF underperforms MLP | Add non-linear interactions (MLP on top) |
| Model too large for sparse data | Reduce embedding dim, use simpler model |

### Architecture ladder (increasing complexity)

1. **Matrix Factorization** (dot product + bias) — fewest parameters, good
   baseline for sparse data.
2. **MLP** (concat embeddings → feedforward) — captures non-linear interactions.
3. **NeuMF** (MF path + MLP path merged) — combines linear and non-linear.
   Best performer in our experiments (0.172 vs 0.152 for MLP).
4. **Two-tower** — separate user/item encoders with dot-product scoring.
   Good for large-scale production.
5. **Attention-based** (e.g., SASRec for sequential) — if temporal order matters.

### Evidence from experiments

NeuMF (GMF dim=32 + MLP dim=32) achieved Hit@50=0.172, beating MLP (0.152)
and MF (0.128) on Amazon Video Games.

---

## 4. Training Procedure

| Symptom | Fix |
|---------|-----|
| Loss still decreasing at last epoch | Train longer (more epochs) |
| Loss diverges or spikes | Reduce learning rate |
| Loss plateaus early | Increase learning rate, add warmup |
| Train loss low but test metrics degrade | Early stopping, reduce LR later |

### Hyperparameter ranges to try

| Param | Conservative | Aggressive |
|-------|-------------|------------|
| Learning rate | 0.0005–0.001 | 0.002–0.01 |
| Embedding dim | 32 | 64–128 |
| Hidden dim | 64 | 128–256 |
| Epochs | 10–20 | 30–50 |
| Batch size | 512–1024 | 2048–4096 |
| Weight decay | 1e-5 | 1e-4 |

### LR scheduling

- **Cosine annealing**: Good default, decays smoothly.
- **Step decay**: Reduce by 0.5 every N epochs.
- **Warmup + decay**: Helps with large learning rates.

### Evidence from experiments

Larger model (embed=128, hidden=256) + cosine LR + 40 epochs did not beat
the simpler model (embed=64, hidden=128) + 20 epochs. Overfitting on sparse
data negated the capacity gains (Hit@50: 0.118 vs 0.152).

---

## 5. Evaluation Methodology

Check that the evaluation setup is reasonable before blaming the model.

| Symptom | Fix |
|---------|-----|
| Ranking metrics implausibly low for all models | Reduce candidate pool (score 1000 random + positives, not all items) |
| Hit@K = 0 for all K | Check that test items exist in the item vocabulary |
| Metrics vary wildly between runs | Increase eval user sample, fix random seed |
| Most eval users have 1 test interaction | Use leave-one-out split instead of random |

### Sampled evaluation

Scoring all N items per user is expensive and can make metrics look
artificially low. Standard practice: score the positive item(s) against
99–999 random negatives. This gives faster eval and more interpretable metrics.

---

## 6. Negative Sampling

| Symptom | Fix |
|---------|-----|
| BPR loss converges but metrics plateau | Try multiple negatives per positive (K=3–5) |
| Model doesn't learn to distinguish similar items | Use in-batch negatives |
| Metrics collapse after changing sampling | Revert — hard negatives can destabilize training |

### Approaches

- **Uniform random** (default): Simple, stable. Good starting point.
- **Multiple negatives (K>1)**: K random negatives per positive. Moderate
  improvement, linear cost increase.
- **Popularity-weighted**: Sample negatives ∝ item_freq^0.75. Theory says
  harder negatives give better gradients, but in practice this often hurts
  on sparse data.
- **In-batch negatives**: Use other users' positives in the same batch as
  negatives. Free hard negatives, works well with large batch sizes.
- **Hard negative mining**: Score all items, pick near-boundary negatives.
  Expensive, risk of collapse.

### Evidence from experiments

- Multi-neg (K=5) was neutral (0.146 vs 0.152 baseline).
- Popularity-weighted negatives severely hurt (0.032 vs 0.152). The hard
  negatives overwhelmed the weak embeddings on sparse data.

---

## 7. Regularization

| Symptom | Fix |
|---------|-----|
| Train loss much lower than test metric | Add dropout (0.1–0.3), increase weight decay |
| Embedding norms growing large | Add L2 regularization on embeddings |
| Model memorizes frequent users/items | Add embedding dropout |

### Techniques

- **Dropout** (0.1–0.3 in MLP layers)
- **Weight decay** (1e-5 to 1e-4 in Adam)
- **Embedding L2 norm** (add `λ * (||u||² + ||i||²)` to loss)
- **Early stopping** (monitor validation metric, stop after patience epochs)

---

## Strategy Selection Heuristic

When proposing K solutions for a round, pick from different categories to
maximize diversity. Don't propose 3 architecture changes in the same round.

Good round composition:
- 1 from the highest-priority untried category
- 1 architectural change
- 1 training/data change

Bad round composition:
- 3 variations of the same idea (e.g., K=3, K=5, K=10 negatives)
