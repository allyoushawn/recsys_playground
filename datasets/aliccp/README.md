# datasets.aliccp

Ali-CCP dataset preparation pipeline: archive verification + extraction, raw-CSV
parsing, Parquet I/O, sparse-vocabulary building, dense normalization, and torch
tensorization for the [Ali-CCP](https://tianchi.aliyun.com/dataset/408) (Alibaba
Click and Conversion Prediction) dataset used in multi-task CTR/CVR modeling (e.g.
ESMM). This is the repo's canonical Ali-CCP data layer — `experiments/20260404_ali_cpp_esmm/
esmm_ali_ccp_impl.py` and `aliccp_mtl_experiments/` both depend on it.

Every module uses plain local imports (`from data import ...`, `from encode import
...`) rather than relying on the rest of the repo being on `sys.path`, so this folder
can also be copied out and used standalone. See `CHANGE_LOG.md` for the folder's
history (it started as a separate `aliccp_data_preparation/` package built for a blog
post, then replaced this location on 2026-07-20 after a compatibility audit).

## Files

| File | Purpose |
|---|---|
| `extract.py` | Locates the raw `sample_train` / `sample_test` archives (`.tar.gz`, `.tgz`, or bare `.tar`), verifies an MD5 checksum if a `.md5` file is present, and extracts them. |
| `data.py` | Torch-free: raw-CSV parsing to chunked Parquet, dense `log1p` normalization, frequency-filtered sparse-vocab build/load (with a pickle cache), path constants, file-discovery helpers. Imports only stdlib + numpy + pandas + (lazy) pyarrow. |
| `encode.py` | Torch tensorizers only (`encode_and_tensorize`, `encode_and_tensorize_fast`, `encode_and_tensorize_arrow`, `build_sparse_vocabs`, `_precompute_sparse_encode_tables`). Depends on `torch` but no model code — imported explicitly so `data.py` alone never pulls in torch. |
| `smoke.py` | End-to-end proof script: builds one real training batch from the processed Parquet + vocab using only `data.py` + `encode.py`, and asserts no model code was imported along the way. |
| `__init__.py` | Re-exports `data.py`'s constants/functions at the package level, for use as `datasets.aliccp.<name>`. |

## Getting the raw data

Download from the Tianchi dataset page: **https://tianchi.aliyun.com/dataset/408**
(sign-in required). You need up to 4 files:

| File | Notes |
|---|---|
| `sample_train.tar.gz` | ~4.1GB compressed. Some mirrors instead serve an already-decompressed bare `sample_train.tar` (~19GB) or `.tgz` — `extract.py` looks for all three. |
| `sample_test.tar.gz` | ~4.7GB compressed (bare `.tar` ~22GB). Same handling as above. |
| `sample_train.tar.gz.md5` | Optional. If present next to the archive, `extract.py` verifies it before extracting. |
| `sample_test.tar.gz.md5` | Optional, same as above. |

Place whichever files you have under a single data directory (e.g. `./data/ali_ccp/`).
The `.md5` files are best-effort: if Tianchi or your mirror doesn't provide one,
`extract.py` prints a warning and extracts anyway — it will not hard-fail on a
checksum file you never had. A checksum *mismatch*, on the other hand, raises an error
rather than silently extracting a possibly-corrupted archive.

**Gotcha, confirmed against a real download**: checksum lookup is exact-name
(`<archive_path>.md5`), so a `.md5` file only verifies an archive with the *matching*
extension. A real-world case we hit: a bare `sample_train.tar` sitting next to a
`sample_train.tar.gz.md5` (checksum for the compressed form the mirror originally
distributed, but only the decompressed `.tar` was kept) — `extract.py` looks for
`sample_train.tar.md5`, doesn't find it, and silently falls back to the no-checksum
warning path even though a checksum file is right there under a different name. If you
want verification to actually run, make sure the `.md5` filename matches whichever
archive form you kept.

## Pipeline

Four steps, run in order:

1. **Extract + verify** (`extract.py: extract_archives`) — find the archives
   (`.tar.gz`, `.tgz`, or bare `.tar`), check MD5 if a matching `.md5` file is
   available, extract to the data directory. Skips extraction entirely if the raw
   CSVs are already present.
2. **Parse raw to Parquet** (`data.py: parse_raw_ali_ccp_streaming_writes`, usually via
   the `ensure_full_split_parquet_streaming` wrapper) — streams the Tianchi
   `sample_skeleton_*.csv` / `common_features_*.csv` files into chunked train/test
   Parquet, without ever materializing the full ~42M-row dataset as one DataFrame.
3. **Normalize + build vocab** (`data.py: load_or_build_sparse_vocabs_filtered_parquet`
   + `stream_normalize_parquet`) — a frequency-filtered (`min_count`) label-encoding
   vocabulary is built per sparse column directly from the parsed Parquet (cached to
   disk so repeat runs skip the scan), then dense features are `log1p`-normalized into
   the Parquet that training actually reads from. Both steps stream over Parquet
   row-group batches.
4. **Tensorize for training** (`encode.py: encode_and_tensorize_arrow` or
   `encode_and_tensorize_fast`) — turns a batch of normalized Parquet rows + the
   vocabs into `torch` tensors (`sparse`, `dense`, `label`) ready to feed any model.

```python
from extract import extract_archives
from data import (
    ALICCP_SPARSE_COLS as SPARSE, ALICCP_DENSE_COLS as DENSE, ALICCP_DENSE_FEAT_COLS as DENSEF,
    ensure_full_split_parquet_streaming, load_or_build_sparse_vocabs_filtered_parquet,
    stream_normalize_parquet,
)
from encode import _precompute_sparse_encode_tables, encode_and_tensorize_arrow

DATA_DIR = "./data/ali_ccp"
PROCESSED_DIR = "./data/ali_ccp/processed"

# 1) extract + verify
extract_archives(DATA_DIR)

# 2) parse raw -> Parquet
p_train, p_test = ensure_full_split_parquet_streaming(DATA_DIR, PROCESSED_DIR, SPARSE, DENSE, DENSEF)

# 3) build vocab (from the parsed, pre-normalization Parquet — vocab only touches sparse cols)...
vocabs, cardinalities = load_or_build_sparse_vocabs_filtered_parquet(
    p_train, SPARSE, min_count=5, cache_path=f"{PROCESSED_DIR}/sparse_vocab.pkl")
# ...then normalize dense features into the Parquet training actually reads from
stream_normalize_parquet(p_train, f"{PROCESSED_DIR}/preprocessed_train.parquet", SPARSE, DENSEF)
stream_normalize_parquet(p_test, f"{PROCESSED_DIR}/preprocessed_test.parquet", SPARSE, DENSEF)

# 4) tensorize a batch for training
import pyarrow as pa
import pyarrow.parquet as pq

enc_tables = _precompute_sparse_encode_tables(vocabs, SPARSE)
pf = pq.ParquetFile(f"{PROCESSED_DIR}/preprocessed_train.parquet")
batch = next(pf.iter_batches(batch_size=8192, columns=SPARSE + DENSEF + ["click", "purchase"]))
table = pa.Table.from_batches([batch])
sparse_t, dense_t, label_t = encode_and_tensorize_arrow(table, enc_tables, SPARSE, DENSEF, "click")
```

Once you have normalized Parquet + a vocab cache under `PROCESSED_DIR`, `smoke.py`
exercises steps 3-4 end to end as a single command:

```bash
python smoke.py --processed-dir ./data/ali_ccp/processed
```

## Dataset schema

23 sparse (categorical) columns and 8 dense (numeric, stored `log1p`-normalized as
`'D' + col`) columns — see `ALICCP_SPARSE_COLS` / `ALICCP_DENSE_COLS` /
`ALICCP_DENSE_FEAT_COLS` in `data.py`. Labels are `click` (CTR) and `purchase` (CVR);
rows where `click == 0` and `purchase == 1` are dropped during parsing (purchase
implies click in this dataset, so that combination is invalid).

This 23 sparse + 8 dense field list and the dual use of some field IDs as both a
categorical `feat` and a numeric `val` (`D…`) follow the community-standard Ali-CCP
preprocess recipe, not a fresh discovery of the raw CSVs. Primary references:

- **Parse / column recipe (including the sparse/dense split):**
  [torch-rechub `preprocess_ali_ccp.py`](https://github.com/datawhalechina/torch-rechub/blob/main/examples/ranking/data/ali-ccp/preprocess_ali_ccp.py)
- **Feature field ID → name mapping (user / item / combination / context):**
  [NVIDIA Merlin Ali-CCP `dataset.py`](https://github.com/NVIDIA-Merlin/models/blob/stable/merlin/datasets/ecommerce/aliccp/dataset.py)
