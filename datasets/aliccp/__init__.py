"""Ali-CCP dataset preparation package.

`datasets.aliccp` and `datasets.aliccp.data` are TORCH-FREE — they expose the
model-agnostic data layer (raw-CSV parsing, Parquet I/O, vocab building,
normalization). The torch tensorizers live in `datasets.aliccp.encode` and must be
imported explicitly (`from .encode import ...` when used as a package, or
`from encode import ...` when this folder is used standalone) so that importing this
package does not pull in torch. Archive discovery, MD5 checksum verification, and
extraction live in `datasets.aliccp.extract`.
"""
from .data import (  # noqa: F401
    SAMPLE_TRAIN_TAR,
    SAMPLE_TEST_TAR,
    TRAIN_CSV,
    VAL_CSV,
    TEST_CSV,
    SINGLE_CSV,
    SAMPLE_SKELETON_TRAIN,
    SAMPLE_SKELETON_TEST,
    COMMON_FEATURES_TRAIN,
    COMMON_FEATURES_TEST,
    ALICCP_SPARSE_COLS,
    ALICCP_DENSE_COLS,
    ALICCP_DENSE_FEAT_COLS,
    DEFAULT_STREAM_PARSE_CHUNK_ROWS,
    DEFAULT_VOCAB_SCAN_ROWS_PER_BATCH,
    DEFAULT_NORM_STREAM_BATCH_ROWS,
    DEFAULT_EVAL_TEST_BATCH_ROWS,
    find_file_recursive,
    parse_raw_ali_ccp_streaming_writes,
    build_sparse_vocabs_filtered_parquet,
    load_or_build_sparse_vocabs_filtered_parquet,
    stream_normalize_parquet,
    ensure_full_split_parquet_streaming,
    parquet_split_summary,
    summarize_split,
)
