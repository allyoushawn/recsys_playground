MMoE (Multi-gate Mixture of Experts) on Census‑Income (KDD)
==========================================================

This baseline implements MMoE and optional stacked MMoE (ML‑MMoE) for two binary tasks:
- Task 1 (income): predict income >50K
- Task 2 (never_married): predict marital‑status == "Never married"

Prerequisites
- Python 3.10+
- Install deps for this subproject:
  - `python -m pip install -r ple_experiment/requirements.txt`
- Prepare data artifacts (see `ple_experiment/README.md`) or run:
  - `python ple_experiment/prepare_census_income.py --output_dir ./data/census_kdd`

Train
```
python ple_experiment/train_mmoe.py \
  --data_dir ./data/census_kdd \
  --out_dir ./runs/mmoe_census \
  --epochs 15 \
  --batch_size 4096 \
  --num_workers 4 \
  --lr 2e-3 \
  --weight_decay 1e-4 \
  --d_model 128 \
  --expert_hidden 256 \
  --num_experts 4 \
  --num_levels 1 \
  --dropout 0.1 \
  --w_income 1.0 \
  --w_never_married 1.0 \
  --use_pos_weight true \
  --grad_clip 1.0 \
  --mixed_precision true \
  --seed 42
```

What gets saved under `--out_dir`
- `config.json`: exact hyperparameters (includes inferred `d_in`)
- `metrics.jsonl`: one JSON line per epoch with train/val metrics
- `best.pt`: best model checkpoint by validation Combined AUC
- `last.pt`: last model checkpoint
- `test_report.json`: metrics on the test set using the best model

Model summary
- Experts: `num_experts` shared MLPs producing `d_model` features
- Gates: one per task, softmax over experts; fused vectors routed to task towers
- Towers: per‑task MLPs `d_model → d_model//2 → 1`
- ML‑MMoE: when `--num_levels > 1`, stack layers; level j>1 gates use the task’s level (j−1) fused vector as selector; experts remain shared (consume input `x` per level)

Parity with PLE
- Uses the same dataloaders, batch size, d_model, hidden sizes, optimizer, scheduler, and training length to enable fair comparison with PLE.

Notes
- Mixed precision (AMP) is enabled by default on CUDA.
- Pos‑class weighting can be enabled via `--use_pos_weight` (computed from train labels).
- Combined AUC is the mean of the two task AUCs and drives model selection/early stopping.

