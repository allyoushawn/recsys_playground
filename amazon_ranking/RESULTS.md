# amazon_ranking — DIN Results

Results from `run_din.py` (DIN on the `SequenceRankingDataModule`), backlog item **E4**.

## DIN on Amazon Beauty (2026-05-31, Colab T4)

End-to-end run: `python -m amazon_ranking.run_din --dataset Beauty --epochs 10 --embed-dim 32 --batch-size 1024`.

**Data** (Amazon Beauty 5-core, SNAP): 198,502 reviews → 22,363 users, 12,101 items,
232,518 train examples (leave-one-out split; 1 sampled negative per positive; 100 eval negatives).

**Model:** DIN (local-activation-unit target attention, masked-softmax interest pooling),
embed_dim=32, BCE, Adam lr=1e-3, seed=0, device=cuda, wall 74.5s.

| Metric | Value |
|---|---|
| sampled_auc | **0.6338** |
| recall@5 / @10 / @20 | 0.179 / 0.279 / 0.406 |
| ndcg@5 / @10 / @20 | 0.118 / 0.150 / 0.182 |
| mrr@5 / @10 / @20 | 0.098 / 0.111 / 0.119 |
| BCE loss (epoch 1 → 10) | 0.689 → 0.489 |

**DoD (E4):** non-degenerate sampled AUC > 0.5 on Beauty → **PASS** (0.6338). Training loss
decreases monotonically; ranking metrics are well above random (random recall@10 over 101
candidates ≈ 0.10). Result JSON: Drive `data/amazon/din_Beauty_results.json`.

## Reproduce

```bash
# Colab T4 (data downloads to local SSD; result JSON to Drive):
python -m amazon_ranking.run_din --dataset Beauty \
    --data-dir /content/amazon \
    --out /content/drive/MyDrive/colab/data/amazon/din_Beauty_results.json

# Local CPU smoke test (synthetic, no download):
python -m amazon_ranking.run_din --synthetic --epochs 8 --out /tmp/din_synth.json
```

## E5 — DIN vs DCN / DeepFM / MeanPool, fair comparison (2026-05-31, Colab T4)

`python -m amazon_ranking.run_din --dataset Beauty --models din,dcn,deepfm,meanpool --epochs 10 --embed-dim 32 --batch-size 1024`.
All four models trained + evaluated on **one shared `SequenceRankingDataModule` build** — identical
splits, negatives, eval candidates, and metrics (the fairness contract). Beauty: 22,363 users /
12,101 items / 232,518 train examples; 100 eval negatives; device=cuda; total wall 213s.

Sequence handling: DIN pools history with **target attention**; the baselines pool history with a
**mask-aware mean** then model interactions over `[interest, target]` (MeanPool = MLP, DeepFM =
FM+DNN, DCN = cross-net+DNN). So the comparison isolates attention vs mean-pooling + interaction style.

| Model | sampled_auc | recall@10 | ndcg@10 | mrr@10 | recall@20 | params | wall (s) |
|---|---|---|---|---|---|---|---|
| MeanPool | **0.6397** | 0.2837 | 0.1523 | 0.1127 | 0.4177 | 398,305 | 36.0 |
| DCN | 0.6383 | **0.2871** | **0.1524** | 0.1118 | **0.4212** | 396,065 | 42.3 |
| DIN | 0.6338 | 0.2790 | 0.1498 | 0.1108 | 0.4061 | 402,986 | 53.9 |
| DeepFM | 0.6052 | 0.2242 | 0.1152 | 0.0824 | 0.3531 | 395,810 | 43.1 |

(Sorted by sampled_auc. Bold = best in column. Random recall@10 over 101 candidates ≈ 0.10 — all
models are well above random.) Result JSON: Drive `data/amazon/ranking_Beauty_results.json`.

### Read
- **DIN's target attention gives no advantage over mean pooling on Beauty.** MeanPool, DCN, and DIN
  are statistically tight (sampled_auc 0.634–0.640; recall@10 0.279–0.287). On this dataset/config the
  attention machinery doesn't pay off — likely because Beauty histories are short and the next-item
  signal is dominated by simple co-occurrence rather than which past item to attend to.
- **DCN is the best all-rounder** — top recall@10/@20 and ndcg@10, fewer params than DIN, faster.
- **DeepFM clearly lags** (sampled_auc 0.605, recall@10 0.224). With only two fields
  `[interest, target]` its second-order FM term reduces to a single `<interest, target>` dot product
  — too little interaction capacity here.
- **Takeaway:** for short-history implicit-feedback ranking on Beauty, a mean-pool MLP or DCN is a
  better quality/cost trade-off than DIN. Hypothesis to test: attention should help more on a
  longer-history dataset (Video_Games) — tested below.

**DoD (E5):** one leaderboard table, same dataset / negatives / metrics, committed to the repo → **met**.

## E5 cross-dataset — same comparison on Video_Games (2026-05-31, Colab T4)

`--dataset Video_Games --models din,dcn,deepfm,meanpool` (identical config + one shared datamodule).
Video_Games is larger and denser: **117,742 users / 80,844 items / 1,211,476 train examples** (vs
Beauty's 22,363 / 12,101 / 232,518). The 814 MB 2023 dump was loaded with the **streaming** reviews
reader (`reviews_io.py`) — projecting to `[user_id, item_id, ts]` at parse time avoids the OOM that
eager full-record parsing would cause. Total wall 1546s.

| Model | sampled_auc | recall@10 | ndcg@10 | mrr@10 | recall@20 | wall (s) |
|---|---|---|---|---|---|---|
| DCN | **0.8334** | **0.6220** | **0.3984** | **0.3291** | **0.7507** | 259 |
| MeanPool | 0.8321 | 0.6172 | 0.3938 | 0.3245 | 0.7454 | 228 |
| DIN | 0.8274 | 0.6041 | 0.3735 | 0.3021 | 0.7374 | 296 |
| DeepFM | 0.8191 | 0.5839 | 0.3635 | 0.2953 | 0.7171 | 264 |

### Cross-dataset read (the real finding)
- **AUCs are far higher than Beauty** (~0.82–0.83 vs ~0.60–0.64): Video_Games has a much stronger
  sequential next-item signal (denser interactions per user).
- **But the model ranking is identical to Beauty: DCN ≈ MeanPool > DIN > DeepFM.** DIN's target
  attention is **3rd on both datasets** — the hypothesis that attention would pay off on the
  longer-history dataset is **not supported**. DCN (cross-network over `[mean-interest, target]`) is
  the consistent winner; plain mean-pool is a hair behind at the lowest cost.
- **Practical conclusion:** for this implicit-feedback next-item ranking setup (≤20-length histories,
  1 sampled negative, 100 eval negatives), **DCN or even a mean-pool MLP beats DIN** — attention's
  added cost isn't justified. DeepFM is consistently last (2-field FM is too thin).
- **Caveat / next probe:** both runs cap `max_hist_len=20`. Attention's potential edge may only show
  with much longer, noisier histories (raise the cap) or richer item features (side info) — a
  follow-up if the question matters. The current evidence (2 datasets, same ranking) is a solid
  "attention is not worth it here."
