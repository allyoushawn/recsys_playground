# TIGER SemanticID Implementation - Agent Knowledge Base

This document contains critical learnings and troubleshooting guidance for agents working on the TIGER SemanticID Amazon Beauty implementation.

## Overview

TIGER SemanticID is a generative retrieval system that uses RQ-VAE to create semantic IDs for items, then trains a seq2seq transformer for next-item prediction. This implementation uses the Amazon Beauty dataset.

## Critical Issues and Solutions

### 1. Data Parsing Issue - Python Dict Format vs JSON

**Problem**: The Amazon metadata files contain Python dictionary format (`{'key': 'value'}`) but the code expects JSON format (`{"key": "value"}`). This causes all metadata loading to fail, resulting in empty DataFrames.

**Symptoms**:
- Empty metadata DataFrame with only `item_id` and `category_leaf` columns
- Missing expected columns: `title`, `description`, `categories`, `brand`, `price`
- All item texts generated identically
- `JSONDecodeError: Expecting property name enclosed in double quotes`

**Solution**:
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
                    data = ast.literal_eval(line)  # Use ast.literal_eval instead of json.loads
                    rows.append(data)
            except (ValueError, SyntaxError, MemoryError):
                continue
    return rows

# Apply the fix BEFORE importing data functions
from tiger_semantic_id_amazon_beauty.src import data
data._parse_json_lines = _parse_python_dict_lines
```

**Critical Timing**: The patch must be applied BEFORE any data loading functions are imported, not after.

### 2. RQ-VAE Model Collapse - Multiple Root Causes

**Problem**: All items getting identical semantic codes, leading to failed training and CUDA assertion errors.

**Root Cause Chain**:
1. **Data parsing failure** → Empty metadata → Identical texts → Identical embeddings → Encoder collapse
2. **Tensor dimension mismatch** in k-means initialization
3. **Training instability** due to poor initialization and high learning rates

**Symptoms**:
- All encoded embeddings identical: `pairwise distances = 0.0000`
- All RQ-VAE codes identical: `[187, 0, 0]` for all items
- Training loss explodes to billions or drops to exactly 0.0000
- CUDA assertion error: `device-side assert triggered`
- Vocabulary out-of-bounds errors in seq2seq training

**Solution - Multi-part Fix**:

#### Part 1: Data Loading (see above)
Apply the Python dict parser fix first.

#### Part 2: RQ-VAE K-means Initialization Fix
```python
# In rqvae.py kmeans_init method - fix tensor dimension mismatch
with torch.no_grad():
    sample = data[torch.randperm(data.shape[0])[: min(batch_size, data.shape[0])]].to(device)
    # CRITICAL: Encode sample first to get correct latent dimension
    encoded_sample = model.encoder(sample)
    model.codebook.kmeans_init(encoded_sample)  # Use encoded, not raw embeddings
```

#### Part 3: Training Stability Improvements
```python
def fixed_train_rqvae(model, data, epochs=50, batch_size=1024, lr=1e-3):  # Lower LR
    # Fix 1: Normalize input data
    data_mean = data.mean(dim=0, keepdim=True)
    data_std = data.std(dim=0, keepdim=True) + 1e-8
    data = (data - data_mean) / data_std
    
    # Fix 2: Proper weight initialization
    def init_weights(m):
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
    model.apply(init_weights)
    
    # Fix 3: Use Adam instead of Adagrad, lower learning rate
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    
    # ... rest of training with gradient clipping and early stopping
```

### 3. Seq2Seq Configuration Issues

**Problem**: `embed_dim must be divisible by num_heads` error in transformer.

**Solution**: Ensure `d_model` is divisible by `heads`. Change `seq2seq_heads: int = 6` to `seq2seq_heads: int = 8` (since 128 ÷ 8 = 16).

**Problem**: Vocabulary level mismatch between RQ-VAE config and seq2seq config.

**Solution**: Use `levels=cfg.rqvae_levels` (3) consistently, not hardcoded `levels=4`.

### 4. List Column Analysis Errors

**Problem**: `TypeError: unhashable type: 'list'` when analyzing `categories` column containing lists.

**Solution**:
```python
def safe_analyze_column(df, col):
    sample_val = df[col].dropna().iloc[0] if not df[col].dropna().empty else None
    is_list_column = isinstance(sample_val, list)
    
    if is_list_column:
        # Can't use nunique() on lists - use non_null_count instead
        non_null_count = df[col].dropna().shape[0]
        print(f"  Non-null values: {non_null_count} (contains lists)")
    else:
        print(f"  Unique values: {df[col].nunique()}")
```

## Diagnostic Workflow

When debugging RQ-VAE issues, follow this order:

1. **Check data loading**: Verify metadata has rich, diverse content
   ```python
   print("Meta columns:", meta.columns.tolist())
   print("Meta shape:", meta.shape)
   print("Sample titles:", meta['title'].head(3).tolist())
   ```

2. **Check text generation**: Ensure item texts are diverse
   ```python
   texts = build_item_text(items.head(10))
   print("All texts identical?", all(texts[0] == text for text in texts))
   ```

3. **Check embeddings**: Verify Sentence-T5 outputs are diverse
   ```python
   print("Embeddings identical?", torch.allclose(item_emb[0], item_emb[1]))
   ```

4. **Check encoder output**: Ensure encoder doesn't collapse inputs
   ```python
   encoded = model.encoder(item_emb[:10])
   dists = torch.cdist(encoded[:5], encoded[:5])
   print("Min pairwise distance:", dists.fill_diagonal_(float('inf')).min().item())
   ```

5. **Check quantization**: Verify codes are diverse after RQ-VAE
   ```python
   codes = encode_codes(model, item_emb)
   unique_codes = len(torch.unique(codes, dim=0))
   print(f"Unique code combinations: {unique_codes}")
   ```

## Expected Healthy Metrics

- **Metadata**: 250K+ unique titles, categories with nested lists
- **Embeddings**: Pairwise distances > 0.01, std > 0.01  
- **RQ-VAE Training**: Loss starts ~1-10, decreases to 0.1-1.0 range
- **Encoded diversity**: Pairwise distances > 0.1
- **Semantic codes**: Thousands of unique combinations, not all identical
- **Seq2seq training**: Stable loss curve, no CUDA assertions

## Files Modified

- `notebooks/tiger_semantic_id_amazon_beauty/TIGER_SemanticID_AmazonBeauty.ipynb`: Added parser fix
- `tiger_semantic_id_amazon_beauty/src/rqvae.py`: Fixed k-means initialization
- Created `data_eda.ipynb`: Diagnostic notebook for data analysis

## Key Learnings

1. **Data format assumptions can be wrong**: Always inspect raw data files
2. **Model collapse has upstream causes**: Fix data issues before model issues  
3. **Tensor dimension mismatches propagate**: K-means init must use encoded dimensions
4. **Import order matters**: Apply patches before importing functions that use them
5. **Diagnostic tools are essential**: Create EDA notebooks to isolate issues

This knowledge base should help future agents quickly identify and resolve similar issues in the TIGER SemanticID pipeline.