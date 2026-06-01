# datasets.aliccp — model-free Ali-CCP dataset layer

Reusable Ali-CCP data layer extracted from `experiments/20260404_ali_cpp_esmm/esmm_ali_ccp_impl.py`
(backlog **E1**) so **any** model — not just ESMM — can build the dataset without importing torch
or model code.

## Two tiers

- **`data.py` — torch-free.** Raw-CSV parse → chunked Parquet, dense `log1p` normalization, filtered
  sparse-vocab build/load (+ pickle cache), full-split orchestration, path constants, file helpers.
  Imports only stdlib + numpy + pandas + (lazy) pyarrow. **Importing `datasets.aliccp` /
  `datasets.aliccp.data` does not import torch** — verified in CI-style checks. Use this from any
  framework (torch, JAX, sklearn, …) to produce the normalized Parquet + vocabs.
- **`encode.py` — torch tensorizers, no model code.** `encode_and_tensorize{,_fast,_arrow}`,
  `build_sparse_vocabs`, `_precompute_sparse_encode_tables`. Turns normalized Parquet rows + vocabs
  into `torch` batches. Depends on torch but on **no model class**, so a non-ESMM model can build
  training batches via `from datasets.aliccp.encode import ...` without pulling ESMM.

## Usage

```python
from datasets.aliccp.data import (
    ensure_full_split_parquet_streaming, load_or_build_sparse_vocabs_filtered_parquet,
    stream_normalize_parquet, COMMON_FEATURES_TRAIN,
)
# 1) parse + normalize (torch-free)
p_train, p_test = ensure_full_split_parquet_streaming(DATA_DIR, PROCESSED_DIR, SPARSE_COLS, DENSE_COLS, DENSE_FEAT_COLS)
vocabs, cards = load_or_build_sparse_vocabs_filtered_parquet(p_train, SPARSE_COLS, min_count=5, cache_path=VOCAB_PKL)

# 2) tensorize batches (torch, but no model import)
from datasets.aliccp.encode import encode_and_tensorize_arrow
# ... iterate row groups -> encode_and_tensorize_arrow(...) -> feed any model
```

## Back-compat

`esmm_ali_ccp_impl.py` re-exports everything here (`from datasets.aliccp.data import *` /
`encode import *`), so existing ESMM/classic notebooks (`from esmm_ali_ccp_impl import *`) are
unaffected — they still see the same symbols.

See `kb/projects/agent_self_exploration/20260530_recsys_dataset_standardization/` for the extraction
rationale and the E2 smoke runner that exercises this layer end-to-end on real data.
