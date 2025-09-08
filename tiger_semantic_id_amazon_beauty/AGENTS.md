# Project: TIGER SemanticID on Amazon Beauty

## Goal
- Implement the Semantic ID pipeline via RQ‑VAE and a compact seq2seq transformer for generative retrieval on the Amazon Beauty 5‑core dataset; produce metrics and visualizations validating paper claims.

## Structure (planned)
- `tiger_semantic_id_amazon_beauty/`
  - `src/` — pipeline modules (data, rqvae, seq2seq, eval, utils)
  - `tests/` — unit tests (data mapping, RQ‑VAE shapes/usage)
  - `README.md` — overview + Colab usage
  - `requirements.txt` — pinned deps
- `notebooks/tiger_semantic_id_amazon_beauty/`
  - `TIGER_SemanticID_AmazonBeauty.ipynb` — end‑to‑end Colab notebook

## Path & Colab Setup
- Notebook installs deps with: `!pip -q install -r tiger_semantic_id_amazon_beauty/requirements.txt`
- Add `tiger_semantic_id_amazon_beauty/src` to `sys.path` in the notebook for imports.
- Use `/content/data` for downloads and `/content/artifacts` for saved models/outputs in Colab.

## Dev Setup (venv — do this first)
- Always activate the repo's virtual environment before running any commands or notebooks.
  - macOS/Linux (zsh/bash): `source venv/bin/activate`
  - Alternatives: `source .venv/bin/activate` or `source vevn/bin/activate` (if named differently)
  - Verify: `python -V && which python`
  - Deactivate: `deactivate`
- Install project dependencies from repo root: `python -m pip install -r tiger_semantic_id_amazon_beauty/requirements.txt`
- Agents: assume the venv is active for all shell commands and Python runs.

## Dependencies (pin standard versions)
- torch, torchvision, torchaudio (Colab CUDA build)
- sentence-transformers, transformers
- pandas, numpy, scikit-learn, tqdm
- matplotlib, umap-learn
- optional: orjson
- tests/dev: pytest

## Dataset
- Amazon Product Reviews, Beauty 5‑core from SNAP:
  - `reviews_Beauty_5.json.gz`
  - `meta_Beauty.json.gz`
- Parse JSON lines to DataFrames:
  - Reviews: `reviewerID`, `asin`, `unixReviewTime`
  - Metadata: `asin`, `title`, `brand`, `category`, `price`
- Clean text; extract leaf category from category paths.

## Preprocessing
- Filter users with ≥ 5 interactions.
- Sort by timestamp.
- Leave‑one‑out splits: last→test, last‑1→valid, rest→train.
- Cap train histories to 20.
- Build contiguous integer IDs for users/items; persist mappings.
- Save split indices and mappings to `/content/artifacts`.

## Content Embeddings
- Build item text: “{title}. Brand: {brand}. Category: {category}. {Price: $X}” (omit missing parts).
- Encode with `SentenceTransformer("sentence-t5-base")` in batches (with tqdm).
- Save `item_embeddings.pt` (float32, shape `[num_items, 768]`).

## RQ‑VAE Semantic IDs
- Model config:
  - latent_dim=32, levels=3, codebook_size=256, beta=0.25
  - encoder MLP: [512, 256, 128] → latent
  - decoder MLP: [128, 256, 512] → recon
  - Residual vector quantization across levels with k‑means init per level (first batch).
  - Loss: MSE recon + per‑level vector‑quantization terms (stop‑gradient).
- Training:
  - Optimizer: Adagrad(lr=0.4), batch_size=1024, epochs≈50 (downscalable for smoke tests).
  - Track per‑level code usage (target ≥ 80%).
- Semantic IDs:
  - Compute (c1,c2,c3) per item; resolve collisions with c4 ∈ {0,1,2,…}, else c4=0.
  - Save: `semantic_ids.npy` ([num_items, 4], int16), `sid_to_items.json`, `item_to_sid.json`.
  - Print collision stats (#collisions, max c4).

## Visualizations
- c1 ↔ category: bar chart of category distribution per c1.
- Hierarchy: for selected c1, stacked bars by c2 to show refinement.
- Optional UMAP: scatter on embeddings colored by c1 and by top categories.
- Save figures to `/content/artifacts/figs`.

## Sequence Construction
- Token vocab:
  - Semantic vocab size = 4*256 = 1024 (index as `pos*256 + code`).
  - User tokens: hash raw users into 2000 IDs.
  - Special tokens: `<PAD>=0`, `<BOS>`, `<EOS>`; offset others to avoid collisions with PAD.
- Training samples:
  - Input: `[<USER_ID_TOKEN>, c1,c2,c3,c4, c1,c2,c3,c4, … up to max_hist_len]`
  - Target: next item’s 4 semantic tokens (teacher forcing).
- Build PyTorch Datasets/DataLoaders with masks/padding.

## Seq2Seq Model
- Compact `nn.Transformer`:
  - d_model=128, nhead=6, enc_layers=4, dec_layers=4, dim_ff=1024, dropout=0.1.
  - Token embeddings + positional encodings (sinusoidal or learned).
- Loss: cross‑entropy over each of the 4 output tokens.
- Optim: Adam(lr=1e‑2), optional inverse‑sqrt schedule; gradient clipping.
- Train up to 20k steps with early stopping on val NDCG@10 (evaluate every N steps).
- Save best `seq2seq.pt`.

## Decoding & Evaluation
- Beam search (beam=10) for 4 tokens.
- Map decoded IDs → items via `sid_to_items`; handle multiple candidates sharing (c1,c2,c3).
- Metrics: Recall@5/10, NDCG@5/10; invalid‑ID rate among top‑K.
- Save per‑user metrics CSV; print concise summary.

## Ablations
- Random IDs: assign random 4‑codes; retrain seq2seq briefly; evaluate.
- LSH IDs: 4 codewords via SimHash (8 hyperplanes/codeword); retrain briefly; evaluate.
- Compare (RQ‑VAE vs LSH vs Random) on Recall/NDCG; save table.

## Mini Cold‑Start Probe
- Remove 5% of test items from training (unseen).
- Train RQ‑VAE & seq2seq on remaining; generate IDs for all with trained RQ‑VAE.
- Evaluate Recall@K allowing ε=0.1 unseen cap among top‑K; save results.

## Artifacts
- `/content/artifacts/`:
  - `item_embeddings.pt`, `rqvae.pt`, `codebooks.pt`
  - `semantic_ids.npy`, `sid_maps.json`
  - `seq2seq.pt`, `vocab.json`
  - `metrics_main.csv`, `metrics_ablation.csv`, `metrics_coldstart.csv`
  - `figs/*.png`

## Quality Bar
- Runs on Colab GPU; artifacts saved; at least one plot showing c1↔category alignment.
- Comparison table (RQ‑VAE vs Random vs LSH) on Recall/NDCG.
- Invalid‑ID rate ≤ ~2% for top‑10 on small runs.
- Cold‑start probe with non‑zero Recall for unseen items.

## Tests (minimal)
- Data: mapping integrity (round‑trip user/item ID maps), split correctness (leave‑one‑out), sequence tokenization shape/padding.
- RQ‑VAE: encoder/decoder output shapes, codebook usage non‑zero, codes in valid range.

## Config Knobs (for smoke tests)
- Reduce epochs/steps/batches; smaller d_model/ff; subset items/users to keep runtime low.

## Next Steps
1) Scaffold `src/`, `tests/`, `README.md`, `requirements.txt` under `tiger_semantic_id_amazon_beauty/`.
2) Create `notebooks/tiger_semantic_id_amazon_beauty/TIGER_SemanticID_AmazonBeauty.ipynb` with sections 0–11 implemented.
3) Add minimal tests and pin requirements; validate smoke‑test run in Colab.
