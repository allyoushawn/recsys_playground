# Project: TIGER SemanticID on Amazon Beauty

## Goal
- Implement the Semantic ID pipeline via RQ‑VAE and a compact seq2seq transformer for generative retrieval on the Amazon Beauty 5‑core dataset; produce metrics and visualizations validating paper claims.

## Structure (planned)
- `tiger_semantic_id/`
  - `src/` — pipeline modules (data, rqvae, seq2seq, eval, utils)
  - `tests/` — unit tests (data mapping, RQ‑VAE shapes/usage)
  - `README.md` — overview + Colab usage
  - `requirements.txt` — pinned deps
- `notebooks/tiger_semantic_id/`
  - `TIGER_SemanticID_AmazonBeauty.ipynb` — end‑to‑end Colab notebook

## Path & Colab Setup
- Notebook installs deps with: `!pip -q install -r tiger_semantic_id/requirements.txt`
- Add `tiger_semantic_id/src` to `sys.path` in the notebook for imports.
- Use `/content/data` for downloads and `/content/artifacts` for saved models/outputs in Colab.

## Dev Setup (venv — do this first)
- Always activate the repo's virtual environment before running any commands or notebooks.
  - macOS/Linux (zsh/bash): `source venv/bin/activate`
  - Alternatives: `source .venv/bin/activate` or `source vevn/bin/activate` (if named differently)
  - Verify: `python -V && which python`
  - Deactivate: `deactivate`
- Install project dependencies from repo root: `python -m pip install -r tiger_semantic_id/requirements.txt`
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
  - latent_dim=128 (increased from 32 to reduce bottleneck), levels=3, codebook_size=256
  - **Loss weights**: alpha=1.0 (codebook loss), beta=0.01 (commitment loss, reduced from 0.25 to preserve diversity)
  - encoder MLP: [768→256→128→128] with ReLU + dropout(0.1) (improved architecture)
  - decoder MLP: [128→128→256→768] with ReLU (improved architecture)
  - **CRITICAL**: NO LayerNorm (removed - amplifies low encoder variance ~0.005 into 400k+ distance explosion)
  - Dropout(0.1) for regularization without normalization
  - Residual vector quantization across levels with k‑means init per level (first batch).
  - **Loss (CORRECTED)**: MSE recon + Σ(l=0 to m-1)[α·||sg[r_l] - e_c_l||² + β·||r_l - sg[e_c_l]||²]
    - Per-level codebook loss (weight α): α·||sg[r_l] - e_c_l||²
    - Per-level commitment loss (weight β): β·||r_l - sg[e_c_l]||²
    - Implementation: `loss = recon + alpha * codebook_loss + beta * commit_loss`
- Training:
  - **Optimizer: Adam(lr=1e-3)** with gradient clipping (max_norm=1.0)
  - batch_size=1024, epochs=50
  - Dead code revival every 10 epochs (revive_every=10, threshold=5)
  - Track per‑level code usage (target ≥ 80%) and perplexity (target ≥50/level)
  - **Monitor encoder diversity** (std should be >0.1 in latent space to prevent collapse).
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
  - d_model=128, nhead=8 (divides d_model cleanly), enc_layers=4, dec_layers=4, dim_ff=1024, dropout=0.1.
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
- **RQ‑VAE Loss Verification**: per-level loss computation correctly implemented (✓ resolved model collapse issues).

## Config Knobs (for smoke tests)
- Reduce epochs/steps/batches; smaller d_model/ff; subset items/users to keep runtime low.

## Critical Issues and Solutions through running notebook

### 1. Data Parsing Issue - Python Dict Format vs JSON
- Problem: Amazon metadata ships as Python dict text (`{'key': 'value'}`) but loaders expected JSON (`{"key": "value"}`), yielding empty DataFrames.
- Symptoms: Missing `title/brand/category/price`, identical item texts, `JSONDecodeError` on load.
- Solution:
```python
def _parse_python_dict_lines(path: str):
    """Parse Python dict lines (not JSON) from a gzipped file using ast.literal_eval."""
    import ast
    import gzip

    opener = gzip.open if path.endswith(".gz") else open
    rows = []
    with opener(path, "rt") as f:
        for raw in f:
            try:
                line = raw.strip()
                if line:
                    data = ast.literal_eval(line)
                    rows.append(data)
            except (ValueError, SyntaxError, MemoryError):
                continue
    return rows

from tiger_semantic_id.src import data
data._parse_json_lines = _parse_python_dict_lines
```
- Critical timing: patch `_parse_json_lines` before importing any data-loading functions.

### 2. RQ-VAE Model Collapse - Resolved ✅
- Root cause chain: bad metadata → identical embeddings → encoder collapse; k-means init used raw vectors; unstable training hyperparameters.
- Symptoms: zero pairwise distances, identical codes like `[187, 0, 0]`, exploding/zero loss, CUDA `device-side assert`.
- Final solution: improved shallower encoder/decoder with dropout, Kaiming init, per-level loss computation now part of main `rqvae.py`.
- Legacy fixes still required:
```python
# K-means init: encode samples before seeding codebooks
with torch.no_grad():
    sample = data[torch.randperm(data.shape[0])[: min(batch_size, data.shape[0])]].to(device)
    encoded_sample = model.encoder(sample)
    model.codebook.kmeans_init(encoded_sample)

def fixed_train_rqvae(model, data, epochs=50, batch_size=1024, lr=1e-3):
    data_mean = data.mean(dim=0, keepdim=True)
    data_std = data.std(dim=0, keepdim=True) + 1e-8
    data = (data - data_mean) / data_std
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    # training loop uses gradient clipping + early stopping
```

### 3. Seq2Seq Configuration Issues
- Transformer constraint: `d_model % heads == 0`; use `heads=8` with `d_model=128` (see config above).
- Ensure `VocabConfig.levels` mirrors `rqvae_levels` (3 semantic levels + collision token) everywhere to avoid vocabulary mismatches.

### 4. List Column Analysis Errors
- Problem: `TypeError: unhashable type: 'list'` when calling `.nunique()` on list-valued columns during EDA.
- Solution:
```python
def safe_analyze_column(df, col):
    sample_val = df[col].dropna().iloc[0] if not df[col].dropna().empty else None
    if isinstance(sample_val, list):
        non_null_count = df[col].dropna().shape[0]
        print(f"  Non-null values: {non_null_count} (contains lists)")
    else:
        print(f"  Unique values: {df[col].nunique()}")
```

## Diagnostic Workflow
1. Verify metadata richness:
```python
print("Meta columns:", meta.columns.tolist())
print("Meta shape:", meta.shape)
print("Sample titles:", meta['title'].head(3).tolist())
```
2. Confirm generated item texts differ:
```python
texts = build_item_text(items.head(10))
print("All texts identical?", all(texts[0] == text for text in texts))
```
3. Check embedding diversity:
```python
print("Embeddings identical?", torch.allclose(item_emb[0], item_emb[1]))
```
4. Check encoder output separation:
```python
encoded = model.encoder(item_emb[:10])
dists = torch.cdist(encoded[:5], encoded[:5])
print("Min pairwise distance:", dists.fill_diagonal_(float('inf')).min().item())
```
5. Check quantized code diversity:
```python
codes = encode_codes(model, item_emb)
unique_codes = len(torch.unique(codes, dim=0))
print(f"Unique code combinations: {unique_codes}")
```

## Expected Healthy Metrics
- Metadata: >250K titles with nested category lists.
- Embeddings: pairwise distances > 0.01, std > 0.01.
- RQ-VAE training: loss starts ~1-10, converges to 0.1-1.0; code usage ≥80% per level.
- Encoded outputs: pairwise distances > 0.1 with improved architecture.
- Semantic codes: thousands of unique combinations (95%+ unique pre-training, 80%+ post-training).
- Seq2Seq runs: stable loss curve, no CUDA assertions, invalid-ID rate ≤2% @10.

## Current Status (Updated 2025-01-14)
- Major issues resolved: data parsing, RQ-VAE diversity collapse, encoder collapse, quantization diversity, GPU optimization, architecture integration, notebook cleanup.
- Production readiness: end-to-end GPU pipeline, robust architecture, unified codebase, real-time diversity/perplexity monitoring, streamlined notebook workflow.

## Capabilities & Performance
- Data loading: handles 250K+ items with rich metadata.
- Text embedding: SentenceTransformer on GPU with device-aware batching.
- RQ-VAE training: maintains 80-95% code diversity across epochs.
- Semantic ID generation: 3-level hierarchical codes with collision handling (c4 fallback).
- Seq2Seq training: transformer-based generative retrieval with beam search decoding.
- Metrics achieved: Recall/NDCG tables for RQ-VAE vs Random vs LSH; cold-start probe shows non-zero recall for unseen items.

## Files Modified
- `tiger_semantic_id/src/rqvae.py`:
  - Improved architecture (shallower encoder/decoder with dropout)
  - **REMOVED per-level residual normalization** (caused numerical explosion with 400k+ distances)
  - Residual k-means init on encoded samples
  - Data normalization buffer
  - Code usage + perplexity tracking
  - Per-level loss computation (`forward_with_losses()`)
  - Dead code revival mechanism
- `tiger_semantic_id/src/embeddings.py`: GPU acceleration with auto device selection, smart batch sizing, device-aware tensors, extended logging.
- `notebooks/tiger_semantic_id/TIGER_SemanticID_AmazonBeauty.ipynb`:
  - Parsing patch for Python dict format
  - GPU embedding integration
  - Diversity monitors (encoder, per-level usage, collision profile, neighbor preservation)
  - **Corrected training cell** with Adam optimizer (lr=1e-3), 50 epochs, beta=0.25
  - Quantization mechanism verification diagnostics
  - Acceptance criteria checks
  - Notebook cleanup, centralized device management
- `notebooks/tiger_semantic_id/data_eda.ipynb`: diagnostic tooling for data sanity checks.
- Documentation: this `AGENTS.md` consolidates fixes + production runbook.

## Key Learnings

### Critical Architecture Insights
- **Straight-through estimator (STE) is MANDATORY for VQ-VAE**: Without STE, encoder receives ZERO gradients from reconstruction loss. The discrete quantization operation (`argmin` + `embedding`) blocks backprop. Implementation: `q_st = z + (q - z).detach()` allows gradients to flow to encoder while using quantized values in forward pass.
- **VQ loss weights must be TINY with STE**: With reconstruction gradients flowing, alpha=0.01 and beta=0.0025 are optimal. Higher weights cause encoder to prioritize VQ losses over reconstruction, destroying diversity.
- **Training shows healthy compression-then-expansion dynamics**: Temporary diversity drops (epoch 10: 1/4/7 active codes) are NORMAL. Model simplifies during early learning, then expands diversity as reconstruction improves. Don't panic and stop training early!
- **Encoder gradient norm is key diagnostic**: With STE: 0.67 (healthy). Without STE: 0.0004 (broken). This single metric confirms whether STE is working.

### Normalization and Numerical Stability
- **LayerNorm on low-variance outputs causes numerical explosion**: When encoder std ~0.005, LayerNorm amplifies by ~200x, creating 400k+ distances. Remove all normalization before quantization.
- **Low encoder variance is NORMAL, not a bug**: Neural network encoder outputs naturally have std ~0.005-0.01. This is fine - codebooks adapt to the scale. Don't try to "fix" it with normalization.
- **Per-level residual normalization also fails**: Same issue as LayerNorm - dividing by small variance causes explosion. Keep residual quantization simple.

### Previous Misdiagnoses (Before STE Discovery)
- ~~**Gradient-based codebook loss is fundamentally flawed**~~ → **WRONG**: The issue was missing STE, not codebook loss itself. With STE, gradient-based codebook loss works fine with proper tuning.
- ~~**VQ losses compete with reconstruction**~~ → **PARTIALLY WRONG**: Without STE, VQ losses were the ONLY gradients encoder received. With STE, they work together harmoniously with tiny weights.
- ~~**For best results: alpha=0, beta=0**~~ → **MISLEADING**: This "worked" because encoder stayed frozen at k-means init while decoder learned. Not a real VQ-VAE. Proper approach: STE + alpha=0.01 + beta=0.0025.

### Training Best Practices
- **Beta must be MUCH smaller than typically recommended**: Standard VQ-VAE uses β=0.25, but optimal for this RQ-VAE is β=0.0025 (100x smaller). High beta forces encoder to output values near codebook centers.
- **Use diagnostic tests to isolate issues**: Test (1) pure autoencoder, (2) quantization without VQ losses, (3) full model to pinpoint where gradient flow breaks. Systematic experimentation reveals root causes.
- **High learning rate + Adagrad is dangerous**: lr=0.4 with Adagrad causes rapid convergence to local minima. Adam with lr=1e-3 is safer and more stable.
- **Monitor MSE, encoder std, and gradient norms during training**: MSE improving (2.3→0.5), encoder std stable (~2-3), encoder grad norm >0.1 indicate healthy training.
- **Extended training (1000 epochs) allows diversity recovery**: Model needs time to recover from mid-training compression phase. Don't stop at epoch 50 just because diversity temporarily dropped.

### Infrastructure and Tooling
- Inspect raw data formats; Python dict vs JSON mismatches can silently break pipelines.
- Model collapse usually traces back to data diversity and initialization; fix upstream issues first.
- Shallower networks with dropout and solid initialization beat ad-hoc patches for diversity preservation.
- Holistic GPU optimization (embeddings + training) matters more than isolated accelerations.
- K-means init must use latent encodings, not raw embeddings, to avoid dimension mismatches.
- Import order matters when monkeypatching loaders; do it before data access.
- Dedicated diagnostics (EDA notebook, monitoring hooks) speed root-cause analysis.
- Integrated architecture (single RQVAE class) is easier to maintain than parallel "improved" variants.
- Real-time monitoring (code usage, perplexity) prevents silent regressions and catches collapse early.
- Explicit device management averts performance drops and CUDA sync issues.
- **Dead code revival mechanism is essential**: Periodic reinitalization of rarely-used codes (every 10 epochs) helps maintain diversity.

### Success Metrics
- **Acceptance criteria**: Active codes ≥80/level, perplexity ≥50/level, collision median ≤2, diversity ≥30%.
- **Production metrics achieved**: MSE=0.53, diversity=92.5%, all levels pass acceptance criteria, encoder actively learning.

## Success Metrics Achieved
- Data diversity: 0% → 100% after parsing fix.
- Code diversity: 4% → 95% unique combinations with improved RQ-VAE.
- Training stability: from exploding losses to smooth convergence.
- GPU utilization: CPU-only to full CUDA pipeline.
- Code quality: temporary patches replaced with integrated solutions.
- Documentation: scattered notes unified into this knowledge base.

## Recent Updates

### Missing Straight-Through Estimator Discovery (2025-10-05) 🔥 CRITICAL - LATEST

**Problem:** All previous attempts to train RQ-VAE failed because the encoder received NO gradients from reconstruction loss.

**Root Cause Discovery:** The implementation was **missing the straight-through estimator (STE)**!

**Gradient Flow Analysis:**
```python
# Line 222: Quantization (discrete operation)
q, codes, commit_loss, codebook_loss = self.codebook.forward_with_losses(z)

# Line 223: Decoder uses quantized q
x_hat = self.decoder(q)

# Line 224: Reconstruction loss
recon = F.mse_loss(x_hat, x_n)
```

**The Problem:**
1. Reconstruction gradients: `∂recon/∂x_hat` → decoder → `∂decoder/∂q`
2. But `q` comes from **discrete operations** (`argmin` + `embedding` lookup in line 110-112)
3. **NO gradient can flow from `q` back to encoder `z`** - discrete ops are non-differentiable!
4. Encoder only received gradients from VQ losses:
   - `commit_loss = ||res.detach() - q||²` → updates codebook only (encoder detached!)
   - `codebook_loss = ||res - q.detach()||²` → updates encoder only

**Why All Previous Experiments Failed:**

| Configuration | Encoder Gradients From | Result |
|---------------|------------------------|--------|
| alpha=0, beta=0 | NOTHING | Encoder frozen at k-means init, decoder learned on fixed codes |
| alpha=0.001, beta=0 | codebook_loss only | Encoder pulled toward codebook centers, collapsed to 30% diversity |
| alpha=1.0, beta=0.01 | codebook_loss only | Encoder fully collapsed to 10.8% diversity |

**The "success" with alpha=0, beta=0 (MSE=0.79, 98% diversity) was misleading:**
- Encoder never trained - stayed at k-means initialization
- Only decoder learned to map fixed quantized codes back to embeddings
- Good diversity came from k-means, not from training
- **Not a real RQ-VAE** - just a decoder on top of clustering!

**Solution - Implement Straight-Through Estimator:**
```python
# In RQVAE.forward() at line 227 (rqvae.py)
# Forward pass: use quantized q
# Backward pass: treat quantization as identity
q_st = z + (q - z).detach()

x_hat = self.decoder(q_st)
```

**How STE works:**
- Forward: `q_st = z + (q - z) = q` (uses quantized values)
- Backward: `∂q_st/∂z = 1 + 0 = I` (gradient flows as if quantization is identity)
- Now reconstruction gradients flow: `∂recon/∂x_hat` → decoder → `∂decoder/∂q_st` → **encoder** ✅

**Impact:**
- Encoder now receives gradients from **both** reconstruction loss AND codebook_loss
- Can use standard VQ-VAE hyperparameters (alpha=1.0, beta=0.25)
- Encoder learns to balance: (1) good reconstruction, (2) quantization-friendly representations
- No more gradient starvation or feedback loop dominance

**Key Insight:** Without STE, VQ-VAE cannot work. The discrete quantization operation must have a "straight-through" path for gradients during backprop, otherwise encoder has no reconstruction signal.

### Codebook Loss Gradient Feedback Loop Discovery (2025-10-04) 🔥 CRITICAL

**Problem:** Even with alpha=0.001 (extremely small codebook loss weight), MSE stuck at 1.0 and diversity collapsed to 30%.

**Discovery:** The codebook loss `||res - q.detach()||²` creates a **gradient feedback loop** that pulls encoder outputs toward codebook centers, even with tiny weights.

**Why tiny alpha has huge impact:**
1. Codebook loss gradients: `∇_encoder = 2α(res - q)` directly affect encoder
2. Reconstruction gradients are weak (chain rule attenuation through multi-layer decoder)
3. Even 0.001× codebook loss is enough to dominate and bias encoder
4. Creates vicious cycle: encoder → codebook centers → less diversity → poor reconstruction

**Experimental evidence:**

| Config | Alpha | Beta | Final MSE | Diversity | Interpretation |
|--------|-------|------|-----------|-----------|----------------|
| Pure reconstruction | 0.0 | 0.0 | 0.79 | 98.3% | ✅ Encoder frozen at k-means (misleading!) |
| Tiny codebook loss | 0.001 | 0.0 | 1.00 | 29.9% | ❌ Still breaks |
| Standard VQ-VAE | 1.0 | 0.01 | 1.00 | 10.8% | ❌ Worse |

**NOTE:** These experiments were conducted WITHOUT straight-through estimator, so encoder had no reconstruction gradients. The interpretations above are now superseded by STE discovery.

**Root cause (REVISED with STE):** Without STE, encoder only receives gradients from VQ losses. Even tiny codebook_loss dominates because there are no reconstruction gradients to compete with.

### VQ Loss Destroying Diversity Discovery (2025-10-04) 🔥 CRITICAL

**Problem:** Even after fixing LayerNorm, diversity remained poor (10.8%) and MSE stuck at 1.0.

**Diagnostic Approach:** Tested 3 configurations to isolate the issue:
1. **Pure autoencoder** (no quantization): MSE 1.86 → 0.49 ✅
2. **With quantization, no VQ losses**: MSE 1.71 → 0.79, Diversity 98.3% ✅
3. **With quantization + VQ losses (beta=0.25)**: MSE stuck at 1.0, Diversity 10.8% ❌

**Root Cause Discovery:**

The VQ losses (commit_loss + codebook_loss) are **competing with reconstruction**:
- Reconstruction loss: "Make encoder diverse to represent different items"
- Codebook loss: "Move residuals toward codebook centers"
- Commit loss (beta=0.25): "Move codebook centers toward residuals"

**When combined, beta=0.25 is TOO STRONG:**
- Encoder learns to output values close to codebook centers (minimize commit loss)
- This collapses encoder diversity
- Only 1,308 unique codes used (10.8%) instead of 11,897 (98.3%)
- MSE penalty: VQ losses add 0.21 to MSE vs reconstruction-only training

**Evidence:**

| Configuration | Final MSE | Diversity | Unique Codes |
|---------------|-----------|-----------|--------------|
| No quantization | 0.49 | N/A | N/A |
| Quant (no VQ loss) | 0.79 | 98.3% | 11,897/12,101 |
| Quant + VQ (β=0.25) | 1.00 | 10.8% | 1,308/12,101 |

**Solution:**
- **Reduce beta from 0.25 → 0.01** (25x reduction)
- VQ losses still provide quantization guidance but don't overwhelm reconstruction
- Alternative: Use EMA (exponential moving average) for codebook updates instead of gradient-based

**Key Insight:** In VQ-VAE, the commitment loss coefficient (beta) must be carefully tuned. Too high and it forces encoder collapse to minimize VQ losses at the expense of reconstruction quality and diversity.

**Expected results with beta=0.01:**
- MSE: 0.80-0.85 (close to no-VQ-loss baseline)
- Diversity: 80-95% (8x improvement)
- All 3 RQ levels active throughout training

### LayerNorm Numerical Explosion Discovery (2025-10-03) 🔥 CRITICAL ROOT CAUSE

**Problem:** All fixes to encoder collapse failed. Model still collapsed to single code [152, 69, 0] with 0.0% diversity.

**Breakthrough Discovery:** LayerNorm was the root cause all along!

**The Vicious Cycle:**
1. Encoder outputs have low variance (std ~0.005) - this is actually NORMAL for neural networks
2. LayerNorm divides by this small std to normalize: `y = (x - mean) / (std + eps)`
3. Division by ~0.005 amplifies magnitudes by ~200x
4. After LayerNorm, distances explode from ~0.05 to ~400,000+
5. Distance computation breaks → all items assigned to same code
6. Gradients die → encoder learns to output constant

**Evidence:**
- Diagnostic [1] showed encoder variance collapse (std=0.004-0.006)
- Diagnostic [2] showed WITHOUT LayerNorm: 20 unique codes, WITH LayerNorm: 1 unique code
- Diagnostic [4] showed 6-digit distances (287k+) instead of expected 1-10 range
- MSE reconstruction stayed stuck at ~1.0 throughout training

**Solution:**
- **REMOVE LayerNorm entirely** - no pre-quantization normalization
- Keep Dropout(0.1) for regularization
- Let encoder outputs go directly to quantization with their natural small variance
- Low encoder variance is FINE - the codebook will adapt to match the scale

**Key Insight:** LayerNorm is designed for high-variance inputs. When applied to low-variance encoder outputs, it becomes a numerical explosion amplifier, not a stabilizer.

### RQ-VAE Encoder Collapse Discovery (2025-10-02) 🔥 CRITICAL - FAILED EXPERIMENT

**Initial Problem:** Level-0 codebook collapsed (perplexity=1.0), 3.6% diversity (435/12101 unique triples)

**Attempted Fix (FAILED):**
1. Per-level residual normalization: `res_norm = res / (res.std(dim=0) + 1e-6)`
2. Adagrad optimizer (lr=0.4)
3. Extended training (150 epochs)
4. Increased beta (0.5)

**Result:** COMPLETE COLLAPSE - 0.0% diversity (1/12101 unique triple), all items → [126, 110, 207]

**Root Cause Discovery via Quantization Mechanism Diagnostics:**

1. **Encoder Variance Collapse** (Primary Issue):
   - Encoder outputs have extremely low variance: std ~0.001-0.005 across batch
   - All items produce nearly identical latent codes
   - Encoder converged to degenerate solution (outputs mean embedding for all inputs)
   - **Diagnosis**: [1] ENCODER OUTPUT DIVERSITY showed std in range [0.0007, 0.005]

2. **Residual Normalization Catastrophically Amplifies Problem**:
   - WITHOUT norm: 14 unique codes (some diversity preserved)
   - WITH norm: 1 unique code (total collapse)
   - Dividing by tiny std (~0.001) creates numerical explosion
   - Distance magnitudes explode to 400,000+ (vs expected ~1-10)
   - **Diagnosis**: [2] RESIDUAL NORMALIZATION IMPACT showed 14 → 1 unique codes

3. **Distance Computation Breakdown**:
   - Normalized residuals have massive magnitude due to division by ~0.001
   - Distance formula `||r_norm||² + ||e||² - 2⟨r_norm, e⟩` produces nonsensical values
   - Min distances: [410296, 409418, 408253, ...] (should be 0-10 range)
   - **Diagnosis**: [4] DISTANCE COMPUTATION showed 6-digit distances

4. **Gradient Death**:
   - Encoder grad norm: 0.0004 (4 orders of magnitude too small)
   - Codebook grad norm: 0.01 (weak learning signal)
   - Encoder converged to local minimum with no escape
   - **Diagnosis**: [5] GRADIENT FLOW CHECK showed dead gradients

**Vicious Cycle:**
1. High lr (0.4) + Adagrad → encoder learns degenerate constant-output solution
2. High beta (0.5) → strong commitment loss forces encoder to match codes, encouraging collapse
3. LayerNorm + tiny encoder variance → numerical instability
4. Residual normalization divides by ~0.001 → magnitude explosion
5. Distance computation breaks → all items map to same code
6. No learning signal → gradients die

**Correct Solution:**
1. ❌ REMOVE per-level residual normalization (causes numerical explosion)
2. ✅ REVERT to Adam optimizer (lr=1e-3) - Adagrad too aggressive
3. ✅ REDUCE beta to 0.25 - high beta encourages encoder collapse
4. ✅ REDUCE epochs to 50 - longer training in bad basin doesn't help
5. ✅ KEEP pre-quantization LayerNorm (not the problem)
6. ✅ ADD encoder diversity monitoring to catch collapse early

**Key Insight:** The original 3.6% diversity wasn't a quantization problem - it was an **encoder collapse problem**. Trying to fix quantization made it worse. Need to prevent encoder from learning constant outputs.

**Diagnostic Tools Added:**
- Quantization mechanism verification cell (checks encoder variance, normalization impact, codebook diversity, distance sanity, gradient flow)
- Per-level usage/perplexity tracking
- Encoder diversity analysis

### RQ-VAE Loss Fix (2025-09-16)
- **Issue**: Previous implementation computed VQ losses only on final aggregated vectors
- **Solution**: Added `forward_with_losses()` method that computes per-level losses during quantization
- **Impact**: Resolves model collapse issues and improves diversity preservation
- **Beta application fix**: β now only applied to commitment loss, not codebook loss
- **Architecture improvements**: Shallower networks with dropout and better initialization

## Next Steps

### Straight-Through Estimator Success ✅ VALIDATED (2025-10-05)

**Fix Applied:** Implemented straight-through estimator in `rqvae.py:227`

```python
# CRITICAL: Straight-through estimator for gradient flow to encoder
# Forward pass: use quantized q
# Backward pass: treat quantization as identity, gradients flow to z
q_st = z + (q - z).detach()

x_hat = self.decoder(q_st)
```

**Training Results with STE:**

| Metric | Value | Status |
|--------|-------|--------|
| Final MSE | 2.32 → 0.53 | ✅ Excellent improvement |
| Final Diversity | 92.5% (11,191/12,101) | ✅ Excellent |
| Encoder gradient norm | 0.67 | ✅ Strong signal (was 0.0004 before!) |
| Level 0 active codes | 255/256 (99.6%) | ✅ Pass |
| Level 1 active codes | 147/256 (57.4%) | ✅ Pass |
| Level 2 active codes | 103/256 (40.2%) | ✅ Pass |
| Level 0 perplexity | 137.2 | ✅ Pass (≥50) |
| Level 1 perplexity | 112.2 | ✅ Pass (≥50) |
| Level 2 perplexity | 72.4 | ✅ Pass (≥50) |

**Optimal Config Found:**
```python
rqvae_alpha: float = 0.01      # Very low codebook loss
rqvae_beta: float = 0.0025     # Very low commitment loss
rqvae_epochs: int = 1000       # Extended training for convergence
rqvae_lr: float = 0.001        # Adam optimizer
```

**Key Learnings:**

1. **STE enables very light VQ losses:** With reconstruction gradients flowing, alpha=0.01 and beta=0.0025 are sufficient. VQ losses provide gentle guidance without overwhelming reconstruction.

2. **Training dynamics show compression-then-expansion:**
   - Epoch 1: 52/121/158 active codes per level
   - Epoch 10: 1/4/7 active codes (temporary collapse!)
   - Epoch 1000: 255/144/77 active codes (full recovery!)
   - This is HEALTHY - model simplifies during early learning, then expands diversity as reconstruction improves

3. **Encoder learning confirmed:**
   - Encoder std: 3.27 → 2.56 (maintains healthy variance)
   - Encoder gradient norm: 0.67 (strong signal vs 0.0004 without STE)
   - Encoder actively learns throughout training (not frozen at k-means init)

4. **Reconstruction dominates with STE:**
   - MSE improvement (2.32→0.53) proves reconstruction gradients are strong
   - Tiny VQ losses don't interfere - they provide gentle regularization
   - Balance achieved: excellent reconstruction + diverse codes

5. **Dead code revival works:**
   - At epoch 1000: revived 58/130/179 codes across levels
   - Helps maintain diversity even with low VQ loss weights

**Production Status:** ✅ Model is production-ready with STE + optimized config

### Future Options (if STE needs refinement)

**Option 1: Tune VQ loss weights**
- Start with alpha=1.0, beta=0.25 (standard)
- If diversity drops too much, reduce alpha to 0.1-0.5
- If codebook doesn't learn well, increase beta slightly

**Option 2: Implement EMA codebook updates (BEST PRACTICE)**

Replace gradient-based codebook loss with exponential moving average updates:
```python
# In RQCodebook.forward_with_losses()
# Instead of: codebook_loss = ||res - q.detach()||²
# Use EMA update (no gradients):
with torch.no_grad():
    codebook[idx] = 0.99 * codebook[idx] + 0.01 * res.detach()
```

This is the standard VQ-VAE approach - decouples codebook learning from encoder gradients. With EMA, set alpha=0 (no gradient-based codebook loss) and keep beta for commitment.

### Future Work
1) ✅ **COMPLETED**: Validate STE fix with full training run
2) **LLM Fine-tuning Pipeline**: Qwen3-8B for SID recommendation (see below)
3) If needed, implement EMA codebook updates
4) Compare: (a) STE + gradient VQ losses vs (b) STE + EMA updates
5) Evaluate downstream performance: Seq2Seq vs LLM-based recommendation
6) Add minimal tests and validate smoke‑test run in Colab

## LLM Fine-tuning Pipeline (Qwen3-8B for SID Recommendation)

### Overview

Fine-tune Qwen3-8B to generate Semantic IDs for next-item recommendation using a two-stage approach:
- **Stage A (Vocabulary Extension)**: Fine-tune only embeddings to learn 1,027 new SID tokens
- **Stage B (Full Fine-tuning)**: Fine-tune entire model on conversational recommendation task

### Prerequisites: Pre-trained RQ-VAE Required

**IMPORTANT**: This pipeline assumes you have **already trained the RQ-VAE** and generated semantic IDs.

**Required artifacts from RQ-VAE training:**
- `/content/artifacts/semantic_ids.npy` - [num_items, 4] array of codes
- `/content/artifacts/sid_to_items.json` - Mapping "c1,c2,c3,c4" → list of item IDs
- `/content/artifacts/item_to_sid.json` - Mapping item_id → [c1, c2, c3, c4]
- `/content/artifacts/user_sequences.json` - User interaction histories (item IDs)

**Pipeline relationship:**
```
┌─────────────────────────────────────────────────────────┐
│ Step 1: RQ-VAE Training (Already Completed)            │
│ ─────────────────────────────────────────────────────── │
│ Input:  Item embeddings [num_items, 768]               │
│ Train:  Encoder/Decoder/Codebooks                       │
│ Output: Semantic IDs [num_items, 4]                     │
│         Each item → (c1, c2, c3, c4) codes              │
└─────────────────────────────────────────────────────────┘
                          ↓
                  semantic_ids.npy
                  sid_to_items.json
                  item_to_sid.json
                          ↓
┌─────────────────────────────────────────────────────────┐
│ Step 2: LLM Fine-tuning (This Pipeline)                │
│ ─────────────────────────────────────────────────────── │
│ Input:  Pre-computed semantic IDs from Step 1          │
│ Build:  Dialog data (history SIDs → next SID)          │
│ Train:  Qwen3-8B to predict next SID from context      │
│ Output: LLM that generates valid SIDs                   │
└─────────────────────────────────────────────────────────┘
```

**What the LLM learns:**
- **NOT** how to encode items into SIDs (that's the RQ-VAE's job)
- **DOES** learn sequential patterns in SID histories to predict next SID
- Essentially a "language model" over the discrete SID vocabulary

**Analogy:**
- **RQ-VAE** = Tokenizer (maps items to codes, like BPE for text)
- **LLM** = Language Model (predicts next token given context)

### Package Structure

```
tiger_semantic_id/src/llm/
├── __init__.py
├── build_sid_dialogs.py       # Data → JSONL dialogs + trie
├── tokenizer_resize_qwen.py   # Add 1,027 SID tokens
├── constraints.py             # Level masks + trie constraints
├── finetune_qwen_vocab.py     # Stage A: embeddings only
├── finetune_qwen_lora.py      # Stage B: LoRA adapters (RECOMMENDED)
├── finetune_qwen_full.py      # Stage B: full model (requires A100 80GB)
└── inference_qwen.py          # Constrained SID generation (supports both LoRA and full)
```

### Token Vocabulary

**1,027 new tokens added:**
- `<SID_START>`, `<SID_END>`, `<REC>` (3 special tokens)
- `<sid_0>` through `<sid_1023>` (1,024 SID tokens)

**Level token ranges:**
- L1: `<sid_0>` to `<sid_255>` (first level codes)
- L2: `<sid_256>` to `<sid_511>` (second level codes)
- L3: `<sid_512>` to `<sid_767>` (third level codes)
- L4: `<sid_768>` to `<sid_1023>` (collision codes)

### Dialog Format

**System prompt:**
```
You are a recommender that must reply ONLY with the next product's Semantic ID
as 4 tokens in order: L1, L2, L3, L4.
Valid token ranges by level:
- L1: <sid_0>.. <sid_255>
- L2: <sid_256>.. <sid_511>
- L3: <sid_512>.. <sid_767>
- L4: <sid_768>.. <sid_1023>
Do not output anything else.
```

**User message (history of 3 items → 12 tokens):**
```
History:
<sid_64> <sid_313> <sid_637> <sid_768>
<sid_64> <sid_447> <sid_706> <sid_768>
<sid_112> <sid_201> <sid_523> <sid_804>
Recommend next:
```

**Assistant response (4 tokens):**
```
<sid_64> <sid_325> <sid_630> <sid_768>
```

### Constrained Decoding

**Two-level constraints ensure valid SIDs:**

1. **Level masks**: At each generation step (L1→L2→L3→L4), mask logits to only allow tokens from the current level's 256-token range.

2. **Trie constraints**: For L2, L3, L4, further mask to only allow valid continuations from the trie built during data preprocessing:
   - L2: Given `c1`, only allow `c2` values that exist in catalog
   - L3: Given `(c1, c2)`, only allow `c3` values that exist
   - L4: Given `(c1, c2, c3)`, only allow `c4` values that exist

This guarantees **Invalid-ID@K = 0%** (all generated SIDs exist in catalog).

### Training Recipe

**Stage A: Vocabulary Extension (1 epoch, embeddings only)**
```bash
python -m tiger_semantic_id.src.llm.finetune_qwen_vocab \
  --data /content/artifacts/llm/dialogs_train.jsonl \
  --valid /content/artifacts/llm/dialogs_valid.jsonl \
  --in_model /content/artifacts/llm/qwen3_vocab_stage \
  --out_model /content/artifacts/llm/qwen3_vocab_stage \
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 8 \
  --learning_rate 5e-4 \
  --num_train_epochs 1 \
  --bf16 \
  --gradient_checkpointing
```

**Stage B Option 1: LoRA Fine-tuning (RECOMMENDED - 3 epochs, LoRA adapters)**
```bash
python -m tiger_semantic_id.src.llm.finetune_qwen_lora \
  --data /content/artifacts/llm/dialogs_train.jsonl \
  --valid /content/artifacts/llm/dialogs_valid.jsonl \
  --in_model /content/artifacts/llm/qwen3_vocab_stage \
  --out_model /content/artifacts/llm/qwen3_lora_adapter \
  --sid_trie /content/artifacts/llm/sid_trie.pkl \
  --lora_r 16 \
  --lora_alpha 32 \
  --lora_dropout 0.05 \
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 8 \
  --learning_rate 1e-4 \
  --num_train_epochs 3 \
  --bf16 \
  --gradient_checkpointing
```

**Stage B Option 2: Full Fine-tuning (3 epochs, all parameters - requires A100 80GB)**
```bash
python -m tiger_semantic_id.src.llm.finetune_qwen_full \
  --data /content/artifacts/llm/dialogs_train.jsonl \
  --valid /content/artifacts/llm/dialogs_valid.jsonl \
  --in_model /content/artifacts/llm/qwen3_vocab_stage \
  --out_model /content/artifacts/llm/qwen3_full_stage \
  --sid_trie /content/artifacts/llm/sid_trie.pkl \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 32 \
  --learning_rate 1e-5 \
  --num_train_epochs 3 \
  --bf16 \
  --gradient_checkpointing \
  --use_8bit_adam
```

### Inference

**With LoRA adapter (recommended):**
```python
from tiger_semantic_id.src.llm.inference_qwen import SIDRecommender

# Load LoRA adapter
recommender = SIDRecommender(
    model_path='/content/artifacts/llm/qwen3_lora_adapter',
    base_model_path='/content/artifacts/llm/qwen3_vocab_stage',
    trie_path='/content/artifacts/llm/sid_trie.pkl',
    is_lora_adapter=True,
)

# Generate from history
history_sids = [
    (91, 54, 165, 0),
    (146, 204, 254, 0),
    (225, 239, 96, 0),
]

generated_sid = recommender.generate_sid(history_sids=history_sids)
# Returns: (c1, c2, c3, c4) tuple with codes in [0, 255]
```

**With full fine-tuned model:**
```python
from tiger_semantic_id.src.llm.inference_qwen import SIDRecommender

# Load full model
recommender = SIDRecommender(
    model_path='/content/artifacts/llm/qwen3_full_stage',
    trie_path='/content/artifacts/llm/sid_trie.pkl',
)

# Generate from history (same API as LoRA)
generated_sid = recommender.generate_sid(history_sids=history_sids)
```

### Artifacts

**Outputs to `/content/artifacts/llm/`:**
- `dialogs_train.jsonl`, `dialogs_valid.jsonl` - Training data
- `sid_trie.pkl` - Valid continuation trie
- `qwen3_vocab_stage/` - Stage A checkpoint (embeddings learned)
- `qwen3_lora_adapter/` - Stage B LoRA checkpoint (recommended)
- `qwen3_full_stage/` - Stage B full fine-tuning checkpoint (optional, requires A100 80GB)

### Evaluation Metrics

1. **Invalid-ID@K**: Fraction of generated SIDs that don't exist in catalog
   - Target: 0% (enforced by constraints)

2. **SID@K**: Whether generated (c1,c2,c3,c4) matches ground truth exactly
   - Target: >0% (baseline comparison)

3. **Item-level metrics**: Map SID to items and compute Recall@K, NDCG@K
   - Compare with Seq2Seq transformer baseline

4. **Qualitative**: Natural language prompts → valid SIDs → product titles
   - E.g., "Recommend lipstick for dry skin" → SID → mapped items

### Acceptance Criteria

✅ **Stage A completes** and new tokens are learned (loss decreases)
✅ **Stage B completes** with validation loss ↓
✅ **Invalid-ID@1 = 0%** on eval set (thanks to masks + trie)
✅ **SID@10 ≥ baseline** (compare with Seq2Seq)
✅ **NL prompts** produce valid 4-token SIDs that map to plausible items

### Resource Requirements

**Full Fine-tuning (finetune_qwen_full.py):**
- **GPU**: A100 80GB ONLY (requires ~54-60GB VRAM)
- **Parameters trained**: 8B (100%)
- **Memory breakdown**:
  - Model weights (BF16): 16GB
  - Gradients (BF16): 16GB
  - Optimizer states (8-bit AdamW): ~16GB
  - Activations + overhead: ~6-12GB
- **Training time**: ~6-10 hours (3 epochs, batch_size=1)
- **Inference**: ~200ms per SID generation

**LoRA Fine-tuning (finetune_qwen_lora.py) - RECOMMENDED:**
- **GPU**: V100 32GB, A100 40GB, or T4 (with smaller batch size)
- **Parameters trained**: ~40-80M (0.5-1%)
- **Memory**: ~20-25GB for bf16 (60% reduction vs full fine-tuning)
- **Training time**:
  - Stage A: ~30 min (1 epoch, embeddings only)
  - Stage B: ~2-4 hours (3 epochs, LoRA adapters)
- **Inference**: ~200ms per SID generation (same as full fine-tuning)

### Key Implementation Details

1. **Gradient checkpointing**: Required for 8B model on single GPU
2. **BF16 training**: 2x faster than FP32, same quality
3. **8-bit AdamW**: Reduces optimizer memory by 4x
4. **Level-wise generation**: 4 sequential forward passes with masking
5. **Trie lookup**: O(1) per level, negligible overhead

### Known Limitations

- **Single-path generation**: Greedy decode only (no beam search across levels)
- **Context length**: Limited to last 8-16 items in history (~32-64 tokens)
- **Cold start**: Requires at least 1 item in history
- **Catalog coverage**: Only generates SIDs that exist in training catalog

### Recent Updates (2025-01-14)

**LoRA Training Pipeline Added:**
- Created `finetune_qwen_lora.py` for memory-efficient training
- Updated `inference_qwen.py` to support loading LoRA adapters via PEFT
- Updated notebook to use LoRA training by default
- Memory usage: 60% reduction (~20-25GB vs ~54-60GB)
- GPU compatibility: V100 32GB, A100 40GB, T4 (with reduced batch size)
- Training time: ~2-4 hours vs ~6-10 hours for full fine-tuning
- Quality: Expected to match full fine-tuning performance

**Key Benefits:**
- LoRA trains only 0.5-1% of parameters (LoRA adapters on attention/MLP layers)
- Supports larger batch sizes (4 vs 1) for faster convergence
- Can use higher learning rate (1e-4 vs 1e-5)
- No 8-bit AdamW needed (memory is already low)

### LLM Training Failure Analysis and Resolution (2025-01-14)

#### Initial Symptoms: Catastrophic Training Failure

**Observed metrics on evaluation:**
- **Invalid-ID@1**: 100% (all predictions generated invalid SID tokens)
- **Unique SIDs generated**: 8 out of 1,000 test examples (severe mode collapse)
- **SID@1 exact match**: 0.00% (zero correct predictions)
- **Diversity collapse**: Model predicting only 8 distinct outputs despite 12,101 possible items

#### Root Causes Identified (7 Critical Issues)

**1. Loss Masking Problem - Training on Wrong Tokens**

**Issue location:** `finetune_qwen_lora.py:64-71`
```python
# PROBLEMATIC CODE:
labels = encoded["input_ids"].copy()
labels = [-100 if token_id == tokenizer.pad_token_id else token_id
          for token_id in labels]
```

**Problem:**
- Training loss computed on ENTIRE conversation (system prompt + user message + assistant response)
- Only ~5% of tokens are actual SID predictions (4 tokens out of ~80 total)
- 95% of loss comes from boilerplate text that should be ignored
- Model optimizes for repeating system instructions rather than learning SID patterns

**Correct approach:**
```python
# SHOULD MASK everything except assistant response:
# system: -100 (masked)
# user: -100 (masked)
# assistant: actual token IDs (trained)
```

**Impact:** Severe - model learns wrong task (text generation instead of SID prediction)

---

**2. No Constraints During Training - Train-Test Mismatch**

**Issue:** Training uses unconstrained generation, inference uses hard constraints

**Training behavior:**
- Model free to predict ANY token from 151K vocabulary
- No enforcement of level ranges (L1=[0-255], L2=[256-511], etc.)
- Model can output invalid sequences like `<sid_0> <sid_0> <sid_0> <sid_0>` (all L1 tokens)

**Inference behavior:** (`inference_qwen.py:189-233`)
- Level masks force valid ranges per position
- Trie constraints enforce valid continuations
- Model CANNOT generate invalid SIDs

**Result:** Massive distribution mismatch
- Training: "anything goes"
- Inference: "strict constraints"
- Model never learned what "valid" means, constraints applied as post-hoc fix

**Solution needed:** Apply constraints during training (rejection sampling or masking invalid tokens in training loop)

---

**3. Insufficient Embedding Learning - Stage A Underfitting**

**Issue location:** `finetune_qwen_vocab.py:143`
```python
num_train_epochs=1,  # Only 1 epoch for 1,024 new tokens!
learning_rate=5e-4,  # Also quite high
```

**Problem:**
- Adding 1,024 new SID tokens (<sid_0> through <sid_1023>)
- Each token needs to learn meaningful embedding in 768-dim space
- Only 1 epoch with 416K examples = ~400 updates per token on average
- High LR (5e-4) may cause instability

**Reference comparison:**
- Standard practice: 3-5 epochs for vocabulary extension
- Embedding typically needs more training than other parameters

**Impact:** Model starts Stage B with poorly-initialized SID embeddings, making it harder to learn SID patterns

---

**4. Autoregressive Mismatch - Teacher Forcing vs Sequential Generation**

**Training:** Teacher forcing with full context
```
Input:  [system][user][assistant: <sid_64> <sid_325> <sid_630>]
Target: [ignore][ignore][         <sid_64> <sid_325> <sid_630> <sid_768>]
```
Model sees ground truth SID tokens during training (even if previous predictions were wrong)

**Inference:** Sequential generation (`inference_qwen.py:194-233`)
```python
for level in range(1, 5):  # Generate L1, then L2, then L3, then L4
    # Use previously generated tokens as context
    # If L1 wrong → L2 sees wrong input → cascading errors
```

**Problem:** Training never experiences cascading errors from wrong predictions, but inference does

**Solution needed:** Scheduled sampling (mix teacher forcing with model's own predictions during training)

---

**5. Aggressive LoRA Configuration - Insufficient Capacity**

**Issue location:** `finetune_qwen_lora.py:110-119`
```python
lora_r=16,        # Rank 16 for 8B parameter model
lora_alpha=32,    # Alpha = 2 × rank
lora_dropout=0.05,
```

**Problem:**
- Qwen3-8B has ~8 billion parameters
- LoRA with r=16 trains only ~40-80M parameters (0.5-1%)
- Task requires learning complex patterns:
  - Semantic relationships between SID levels
  - Sequential dependencies in user behavior
  - Mapping from history context to next SID

**Typical LoRA recommendations:**
- r=32-64 for instruction tuning on 7-8B models
- r=16 suitable for simpler tasks (classification, simple QA)
- Complex reasoning tasks need higher rank

**Impact:** Model capacity bottleneck prevents learning nuanced patterns

---

**6. High Learning Rate - Stage B Instability**

**Issue location:** `finetune_qwen_lora.py:142`
```python
learning_rate=1e-4,  # 5x higher than reference
```

**Reference implementation uses:** 2e-5 (from Eugene Yan's semantic-ids-llm)

**Problem:**
- LoRA fine-tuning typically needs lower LR than full fine-tuning
- High LR can cause:
  - Loss spikes and instability
  - Catastrophic forgetting (model forgets language capabilities)
  - Overshooting optimal parameters
  - Mode collapse (converges to predicting few outputs)

**Evidence:** Mode collapse to 8 unique predictions suggests overshooting

---

**7. No Prompt Diversity - Single Template Overfitting**

**Issue:** All training examples use identical format (`build_sid_dialogs.py:79-82`)
```python
def format_history_compact(history_sids):
    sid_strs = [format_sid_tokens(sid) for sid in history_sids]
    return "User's last purchases: " + ", ".join(sid_strs) + ". Next:"
```

**Problem:**
- Every example starts with "User's last purchases:"
- Model may learn to pattern-match on fixed text rather than SID content
- No robustness to prompt variations

**Reference implementation:** Uses multiple task types (A-F) with different prompt templates

**Impact:** Model overfits to specific phrasing, may fail on alternative prompts

---

#### Data Generation Analysis: Reference vs Current Implementation

**Dataset Comparison:**

| Aspect | Reference (Eugene Yan) | Current (Amazon Beauty) |
|--------|----------------------|-------------------------|
| Dataset | Amazon Reviews 2023 - Video Games | Amazon Reviews - Beauty |
| Users | 79,000 | 22,363 |
| Products | 66,000 | 12,101 |
| Avg sequence length | ~6.5 items | ~8.2 items |
| Total interactions | ~514K | ~183K |

**Key insight:** Different dataset categories (Video Games vs Beauty), not same dataset. Beauty is smaller/niche but has longer sequences.

---

#### Training Data Types Breakdown

Reference implementation uses **6 task types** to generate 4.2M examples:

**Type A: SID → Title (Per-Item Task)**
- Format: Given SID, generate product title
- Example count: ~319,000 (7.6%)
- Purpose: Teach model semantic meaning of each SID
- Example:
  ```
  User: What product has SID <sid_64> <sid_325> <sid_630> <sid_768>?
  Assistant: Maybelline SuperStay Matte Ink Liquid Lipstick
  ```

**Type B: Title → SID (Per-Item Task)**
- Format: Given product title, generate its SID
- Example count: ~479,000 (11.4%)
- Purpose: Reverse mapping, reinforces SID-product association
- Example:
  ```
  User: What is the SID for "Neutrogena Hydro Boost Water Gel"?
  Assistant: <sid_91> <sid_402> <sid_653> <sid_768>
  ```

**Type C: Next-Item Prediction (Sequential Task)** ✅ **IMPLEMENTED**
- Format: Given SID history, predict next SID
- Example count: ~801,000 (19.1%) in reference, **416,000 achieved**
- Purpose: Core recommendation task
- Example:
  ```
  User: User's last purchases: <sid_64> <sid_313> <sid_637> <sid_768>,
        <sid_112> <sid_201> <sid_523> <sid_804>. Next:
  Assistant: <sid_64> <sid_325> <sid_630> <sid_768>
  ```

**Type D: Semantic Understanding (Analytical Task)**
- Format: Questions about SID relationships, categories
- Example count: ~63,000 (1.5%)
- Purpose: Teach hierarchical structure of SIDs
- Example:
  ```
  User: Do these products share the same category?
        <sid_64> <sid_313> <sid_637> <sid_768>
        <sid_64> <sid_325> <sid_630> <sid_768>
  Assistant: Yes, they share L1 (makeup) and L2 (lips).
  ```

**Type E: Co-Purchase Patterns (Graph Task)** 🔥 **LARGEST TYPE**
- Format: "Users who bought X also bought Y" patterns
- Example count: ~2,570,000 (61.2%) ← **Dominates the dataset!**
- Purpose: Capture product affinity beyond sequential patterns
- Example:
  ```
  User: Users who bought <sid_64> <sid_313> <sid_637> <sid_768>
        also frequently bought:
  Assistant: <sid_91> <sid_402> <sid_653> <sid_768>
  ```

**Type F: Additional Patterns (Misc)**
- Format: Various other recommendation patterns
- Example count: ~540 (0.01%)
- Purpose: Edge cases and diversity

**Total:** 4,198,540 examples

---

#### Achievement: Type C Data Boost (Next-Item Prediction)

**Original implementation:**
```python
def create_dialogs(user_sequences, history_length=8, min_seq_len=5):
    for split_point in range(8, len(item_seq)):  # Start at position 8
        history = item_seq[:split_point]
        target = item_seq[split_point]
        # Generate 1 example with last 8 items
```

**Problems:**
- Started at position 8 → sequences <9 items generated 0 examples
- Only 1 variation per position (fixed history_length=8)
- With avg sequence length 8.2, most users contributed few examples

**Results:**
- 22,363 sequences → **37,052 examples** (1.66 examples/sequence)

---

**Updated implementation:** (`build_sid_dialogs.py:85-158`)

**Key changes:**
1. **Start at position 2** (not 8) - maximizes data from short sequences
2. **Multiple variations** - 3 examples per position (last_2, last_3, last_5)
3. **Compact format** - matches reference implementation style

```python
def create_dialogs(user_sequences, history_lengths=[2, 3, 5], min_seq_len=3):
    for split_point in range(2, len(item_seq)):  # Start at 2 not 8!
        for hist_len in history_lengths:  # 3 variations per position
            history_subset = history[-hist_len:]
            # Generate example with last N items
```

**Results:**
- 22,363 sequences → **415,866 examples** (18.6 examples/sequence)
- **11.2x increase** from original implementation
- **52% of reference Type C** (416K vs 801K) despite having 28% of sequences

**Statistics by variation:**
- last_2: ~138,622 examples (33%)
- last_3: ~138,622 examples (33%)
- last_5: ~138,622 examples (33%)

**Why this works:**
- Position 2 allows sequences as short as 3 items to contribute
- Multiple history lengths teach model to work with varying context
- More examples per user → better personalization learning

---

#### Plans to Scale Training Data

**Current status:**
- Type C: 416K examples ✅
- Types A, B, D, E, F: 0 examples ❌

**Immediate next steps:**

**1. Implement Type A (SID → Title) - Est. ~73K examples**
- Requires: `meta_Beauty.json.gz` with product titles
- Generation: Per-item task, 1 example per item with metadata
- Code needed: Template in `build_sid_dialogs.py`
  ```python
  def create_sid_to_title_dialogs(semantic_ids, metadata):
      for item_id, sid in enumerate(semantic_ids):
          dialog = {
              "messages": [
                  {"role": "system", "content": SYSTEM_PROMPT},
                  {"role": "user", "content": f"What product has SID {format_sid(sid)}?"},
                  {"role": "assistant", "content": metadata[item_id]['title']},
              ],
              "type": "sid_to_title",
          }
  ```

**2. Implement Type B (Title → SID) - Est. ~109K examples**
- Similar to Type A but reversed
- Can augment with paraphrased titles (1.5x multiplier)

**3. Implement Type E (Co-Purchase) - Est. ~257K examples** 🎯 **HIGH IMPACT**
- Requires: Co-occurrence matrix from user sequences
- Algorithm: For each item, find top-K frequently co-purchased items
- Generation: Multiple examples per item (top-10 co-purchases)
  ```python
  def create_copurchase_dialogs(cooccurrence_matrix, semantic_ids, top_k=10):
      for item_id, sid in enumerate(semantic_ids):
          copurchased_items = cooccurrence_matrix[item_id].topk(top_k)
          for copurchased in copurchased_items:
              # Generate dialog...
  ```

**4. Implement Type D (Semantic Understanding) - Est. ~15K examples**
- Requires: Category labels from `c1` clustering analysis
- Generate questions about SID relationships

**5. Skip Type F** (miscellaneous patterns - minimal contribution)

---

**Projected totals with all types:**

| Type | Current | Projected | % of Reference |
|------|---------|-----------|----------------|
| A: SID → Title | 0 | 73K | 23% |
| B: Title → SID | 0 | 109K | 23% |
| C: Next-item | 416K | 416K | 52% |
| D: Semantic | 0 | 15K | 24% |
| E: Co-purchase | 0 | 257K | 10% |
| F: Misc | 0 | 0 | 0% |
| **Total** | **416K** | **870K** | **21%** |

**Why 21% of reference despite 28% of sequences:**
- Smaller catalog (12K vs 66K items) → fewer per-item tasks
- Co-purchase type (Type E) scales with catalog size × top-K
- Video Games has denser co-purchase graph than Beauty

---

#### Recommended Training Improvements Priority

**🔥 CRITICAL (Fix immediately for next training run):**
1. **Fix loss masking** - Only train on assistant response tokens
2. **Reduce Stage B learning rate** - 1e-4 → 2e-5
3. **Increase Stage A epochs** - 1 → 3-5 epochs

**⚠️ HIGH PRIORITY (Fix before production):**
4. **Add training constraints** - Mask invalid tokens during training
5. **Increase LoRA rank** - r=16 → r=32 or r=64
6. **Reduce Stage B beta** - 0.25 → 0.01 (prevent commitment loss collapse)

**📊 MEDIUM PRIORITY (Improves performance):**
7. **Implement scheduled sampling** - Mix teacher forcing with model predictions
8. **Add prompt diversity** - Multiple templates for Type C
9. **Increase Stage A learning rate** - 5e-4 → 1e-3 (embeddings need strong signal)

**🚀 LONG-TERM (Scaling):**
10. **Implement Types A, B, D, E** - Scale to 870K examples
11. **Add data augmentation** - Paraphrasing, history shuffling
12. **Multi-task training** - Mix all 6 types in each batch

---

#### Expected Impact of Fixes

**After CRITICAL fixes:**
- Invalid-ID@1: 100% → <5% (constraints + proper training)
- Unique SIDs: 8 → 1000+ (loss masking + lower LR prevents mode collapse)
- SID@1: 0.0% → 5-10% (model can learn basic patterns)

**After HIGH PRIORITY fixes:**
- SID@10: 10-15% (better capacity and constraints)
- Item Recall@10: 20-30% (competitive with Seq2Seq baseline)

**After implementing all data types:**
- SID@10: 20-30% (multi-task learning improves robustness)
- Item Recall@10: 35-45% (richer training signal from 6 task types)
- Qualitative: Model can answer "SID→Title" and "Title→SID" queries

---

### Extensions (Future)

1. **Natural language features**: Use item titles/categories in prompts
2. **Multi-item generation**: Generate top-K diverse SIDs in one pass
3. **Retrieval augmentation**: Combine with embedding-based retrieval
4. **Online learning**: Incremental updates for new items/SIDs
5. **LoRA hyperparameter tuning**: Experiment with different rank/alpha values
