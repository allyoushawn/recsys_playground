# Amazon Video Games Rating Prediction — Experiment Summary

**Dataset:** Amazon Reviews 2023 — Video Games (200k interactions, 83k users, 39k items, ratings 1–5)
**Task:** Regression (minimize MAE)
**Baseline:** MLP (embed=32, hidden=64) + MSE loss, 10 epochs → **MAE = 1.005**

## Results (15 experiments, 5 rounds)

| Rank | Experiment | MAE | RMSE | R² | Round |
|------|-----------|-----|------|----|-------|
| 1 | M: MeanFeats + Huber + Sigmoid + 50ep + LR=5e-4 | **0.874** | 1.353 | -0.073 | 5 |
| 2 | I: MLP + MeanFeats + Huber + Sigmoid | 0.881 | 1.371 | -0.102 | 3 |
| 3 | O: NeuMF + MeanFeats + Huber + Sigmoid | 0.897 | 1.397 | -0.145 | 5 |
| 4 | N: RichFeats (mean, count, std) + Huber + Sigmoid | 0.898 | 1.373 | -0.105 | 5 |
| 5 | L: LargeMLP + MeanFeats + L1 + WD | 0.901 | 1.372 | -0.105 | 4 |
| 6 | J: MeanFeats + L1Loss | 0.911 | 1.319 | -0.021 | 4 |
| 7 | K: Residual + MeanFeats + Huber | 0.914 | 1.349 | -0.067 | 4 |
| 8 | G: Sigmoid[1,5] + Huber | 0.926 | 1.424 | -0.190 | 3 |
| 9 | C: Huber + 30ep + CosineLR | 0.936 | 1.293 | 0.020 | 1 |
| — | **Baseline** | **1.005** | **1.301** | **0.007** | 0 |
| 11 | H: NeuMF + Huber + Sigmoid | 1.011 | 1.522 | -0.358 | 3 |
| 12 | A: MSE + 30ep + CosineLR | 1.017 | 1.304 | 0.002 | 1 |
| 13 | F: Huber + NormRatings + Clamp | 1.025 | 1.295 | 0.016 | 2 |
| 14 | B: LargeMLP + 30ep + CosineLR | 1.036 | 1.341 | -0.055 | 1 |
| 15 | E: BiasedMF + Huber + Clamp | 1.401 | 1.585 | -0.475 | 2 |
| 16 | D: Huber + Clamp (hard) + WD | 3.172 | 3.430 | -5.902 | 2 |

**Total improvement: 13.0%** (1.005 → 0.874)

## What worked

1. **Huber loss over MSE** (Round 1, +7% MAE). Robust to outlier ratings; effectively optimizes MAE for large residuals. This was the single biggest lever.
2. **User/item mean-rating features** (Round 3, +6% MAE). Giving the model explicit user and item bias signals mattered more than any architecture change.
3. **Sigmoid-bounded output** `1 + 4·σ(x)` (Round 3). Smooth bounding to [1, 5] with full gradient flow. Improved MAE at a slight RMSE cost.
4. **Patient optimization** (Round 5, +0.7%). 50 epochs at LR=5e-4 with cosine annealing squeezed out the final gains.

## What didn't work

1. **Hard clamping** `clamp(1,5)` killed training — zeroes gradients when initial outputs fall outside the clamp range (D: MAE=3.17).
2. **Larger/deeper models** (B, H, O) consistently overfitted on sparse data (~2.4 interactions/user).
3. **Rating normalization to [0,1]** compressed the target range too much, hurting discrimination (F).
4. **Residual learning** (K) added no value — the mean features already captured biases.
5. **Extra count/std features** (N) were noise; the model couldn't leverage them effectively.

## Key takeaways for DL regression

- **Match the loss to the metric.** Huber/L1 for MAE; MSE for RMSE. This matters more than architecture.
- **Feature engineering > architecture complexity.** Explicit bias features beat NeuMF, deeper MLPs, and BiasedMF.
- **Bound outputs smoothly.** Use sigmoid scaling, never hard clamp — gradient flow is non-negotiable.
- **Respect data sparsity.** Keep models small when interactions-per-entity are low. Capacity without signal = overfitting.
- **R² will be low.** Individual rating prediction from IDs alone is inherently noisy. Collaborative filtering captures biases, not the full variance of human preferences.

## Regression-to-Mean (RTM) Investigation — Complete Results

**Hypothesis:** Negative R² indicates models collapse to predicting the mean; σ_pred ≪ σ_true confirms regression-to-mean.

**Goal:** R² > 0.05 AND MAE < 0.85 — **NOT MET** after 13 experiments across 5 rounds.

### Metrics used in RTM optimization

- **σ_ratio** = std(predictions) / std(true ratings). Target 1.0. &lt; 1 means predictions are compressed toward the mean (regression to the mean); the model does not spread outputs enough to match the true rating variance.
- **Cal.Slope** = slope of the line when regressing actual ratings on predictions (predicted vs actual). Target 1.0. &lt; 1 means under-spread: low predictions are not low enough, high not high enough; &gt; 1 would mean overconfident spread.
- **Per-rating buckets:** For each true rating (1–5), we computed count, mean prediction, std of predictions, and MAE. This shows whether the model outputs similar values regardless of true rating (mean collapse).

Diagnostic also included: calibration scatter (predicted vs actual, y=x line, fit slope), residual plot (residual vs predicted), and prediction vs true histograms.

### Diagnostic (Round 1)

| Metric | Value |
|--------|-------|
| σ_true | 1.306 |
| σ_pred | 0.881 |
| σ_ratio | 0.675 (target 1.0) |
| Cal. slope | 0.424 (target 1.0) |

Per-rating bucket: True 1s predicted ~3.8, true 5s predicted ~4.5. Severe compression toward the mean — the model barely distinguishes low vs high ratings.

### What each RTM round tested (interventions)

- **Round 2 — Breaking mean collapse:**  
  **P** Variance-preserving loss (Huber + λ|σ_y − σ_pred|): increased σ_ratio (0.79) but **worsened R²** (-0.21); optimizing spread directly hurt accuracy.  
  **Q** Embedding noise N(0, 0.1) during training: no real gain.  
  **R** Z-score targets + unbounded output + Huber(δ=0.5): best R² in round 2 (-0.059), no sigmoid compression.

- **Round 3 — Alternative framings:**  
  **S** 5-class cross-entropy + low-temperature softmax (τ=0.5) at inference, expectation as prediction: **best MAE overall (0.853)**; classification avoids mean-seeking bias of regression.  
  **T** Ordinal regression (CORAL-style cumulative logits): strong R² (-0.047), good calibration.  
  **U** Focal-style 2× weight on ratings 1 and 5: slightly better σ_ratio but worse R² (-0.17); up-weighting tails did not fix collapse.

- **Round 4 — Addressing sparsity:**  
  **V** SVD-initialized embeddings + residual from item mean: **hurt** (MAE 0.95, R² -0.14); SVD on this sparse matrix did not help.  
  **W** Learnable temperature sigmoid 1+4·σ(x/τ): small gain.  
  **X** Heteroscedastic NLL (predict μ and log σ): **best R² (-0.033)**; learning uncertainty discouraged mean collapse without forcing spread artificially.

- **Round 5 — Refinements:**  
  **Y** Ordinal + SVD init: **worst R² (-0.32)**; SVD + ordinal combined badly.  
  **Z** Z-score + 60 epochs + lower LR (3e-4): strong (R² -0.049, MAE 0.868).  
  **AA** 5-class CE + τ=0.3 (sharper): good MAE (0.854), worse R² than S.

### RTM Leaderboard (sorted by R² desc)

| Rank | Experiment | MAE | R² | σ_ratio | Cal.Slope | Round |
|------|-----------|-----|-----|---------|-----------|-------|
| 1 | X: Heteroscedastic NLL | 0.891 | -0.033 | 0.655 | 0.464 | 4 |
| 2 | T: Ordinal (CORAL) | 0.882 | -0.047 | 0.670 | 0.454 | 3 |
| 3 | Z: Z-score+60ep+LR3e-4 | 0.868 | -0.049 | 0.644 | 0.462 | 5 |
| 4 | R: Z-score+Unbounded+Huber | 0.871 | -0.059 | 0.657 | 0.449 | 2 |
| 5 | Q: EmbedNoise+Huber+Sigmoid | 0.875 | -0.079 | 0.670 | 0.432 | 2 |
| 6 | W: TempSigmoid | 0.876 | -0.085 | 0.671 | 0.427 | 4 |
| 7 | M: Baseline best (diagnostic) | 0.876 | -0.089 | 0.675 | 0.424 | — |
| 8 | S: 5-class CE+τ=0.5 | **0.853** | -0.112 | 0.697 | 0.412 | 3 |
| 9 | AA: 5-class CE+τ=0.3 | 0.854 | -0.139 | 0.713 | 0.390 | 5 |
| 10 | V: SVD+Residual | 0.949 | -0.140 | 0.720 | 0.369 | 4 |
| 11 | U: FocalWeight(1,5)+Huber | 0.860 | -0.170 | 0.723 | 0.380 | 3 |
| 12 | P: VarPreserve+Huber+Sigmoid | 0.889 | -0.214 | 0.785 | 0.345 | 2 |
| 13 | Y: Ordinal+SVD | 0.898 | -0.316 | 0.841 | 0.288 | 5 |

### RTM Investigation — What we learned

1. **Mean collapse is fundamental, not fixable by loss/architecture alone.** With ~2.4 interactions/user, user embeddings are virtually untrained. The model learns item popularity bias + global mean, not individual preferences.

2. **Heteroscedastic NLL (X) achieved the best R² (-0.033)** by forcing the model to learn uncertainty, indirectly discouraging mean collapse. Still negative R², but closest to zero.

3. **Classification framing (S, AA) achieved the best MAE (0.853)** — CE loss doesn't have the mean-seeking bias of regression losses. Low-temperature softmax (τ=0.5) at inference makes the model commit to a rating rather than hedging. R² is worse because the model predicts modes, not calibrated continuous values.

4. **R² and MAE are in tension.** Experiments that improve R² (less mean collapse) tend to hurt MAE, and vice versa. The Pareto front is: S (best MAE, 0.853) vs X (best R², -0.033).

5. **Variance-preserving loss (P) backfired:** it raised σ_ratio (0.79) but worsened R² (-0.21). Directly penalizing low prediction spread distorted the loss and hurt accuracy.

6. **SVD initialization hurt performance** (V, Y). The sparse interaction matrix doesn't provide meaningful SVD factors at this sparsity level; Ordinal+SVD (Y) was the worst experiment (R² -0.32).

7. **Z-score + unbounded output (R, Z)** helped R² without forcing spread artificially; Z with longer training (60 ep) and lower LR was among the best (R² -0.049).

8. **σ_ratio remained 0.64–0.84 across all experiments** (target: 1.0). Even the most aggressive interventions only partially addressed mean collapse.

9. **Calibration slope stayed below 0.47** everywhere. The model fundamentally cannot rank users' preferences from IDs alone at this sparsity.

### Bug fixes during investigation

- **cal_slope computation:** Fixed `np.polyfit(preds, y_true, 1)[1]` → `[0]` — was reporting intercept instead of slope in experiment cells.
- **Variable shadowing:** Fixed `mu, log_sigma = model_x(...)` in heteroscedastic training loop overwriting the global `mu` scalar.

## Next steps to explore

- Text features from reviews (pretrained language model encodings) — likely the only path to R² > 0 at this sparsity
- Temporal features (timestamp, recency)
- Non-DL baselines (SVD, gradient-boosted trees) as a ceiling check
- Ensemble: S (best MAE) + X (best R²) for Pareto improvement
