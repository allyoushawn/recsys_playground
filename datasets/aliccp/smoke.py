"""E2 smoke / E1 end-to-end proof: build an Ali-CCP training batch via
``datasets.aliccp`` WITHOUT importing any model code.

Validates the decoupled dataset layer on real data: reads the cached normalized
Parquet + sparse vocab through ``datasets.aliccp.{data,encode}``, encodes a few
row groups into tensors, and asserts the batch is well-formed — then asserts that
no ESMM/model module was imported. This is the standalone proof that the data
layer is usable by any model (E1) and a fast preflight that the pipeline is sound (E2).

Usage (Colab, reusing the ESMM prep cache)::

    python -m datasets.aliccp.smoke \
        --processed-dir /content/drive/MyDrive/colab/data/ali_ccp/processed_esmm_full_parquet

Note: the first run rebuilds the sparse-vocab cache if absent/mismatched (a one-time
full-train scan, ~25 min on 42M rows); once `preprocessed_sparse_vocab.pkl` is written
the batch+encode assertions are sub-second.
"""

from __future__ import annotations

import argparse
import os
import sys
import time


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--processed-dir",
        default="/content/drive/MyDrive/colab/data/ali_ccp/processed_esmm_full_parquet",
        help="dir holding preprocessed_{train,test}.parquet + preprocessed_sparse_vocab.pkl",
    )
    p.add_argument("--batch-rows", type=int, default=8192)
    p.add_argument("--min-count", type=int, default=5)
    args = p.parse_args()

    import pyarrow as pa
    import pyarrow.parquet as pq
    import torch

    from datasets.aliccp.data import (
        ALICCP_DENSE_FEAT_COLS as DENSEF,
        ALICCP_SPARSE_COLS as SPARSE,
        load_or_build_sparse_vocabs_filtered_parquet,
    )
    from datasets.aliccp.encode import _precompute_sparse_encode_tables, encode_and_tensorize_arrow

    train_pq = os.path.join(args.processed_dir, "preprocessed_train.parquet")
    test_pq = os.path.join(args.processed_dir, "preprocessed_test.parquet")
    vocab_cache = os.path.join(args.processed_dir, "preprocessed_sparse_vocab.pkl")
    for f in (train_pq, test_pq):
        assert os.path.isfile(f), f"missing {f} — run the ESMM prep (parse+normalize) first"

    t0 = time.time()
    vocabs, cards = load_or_build_sparse_vocabs_filtered_parquet(
        train_pq, SPARSE, min_count=args.min_count, cache_path=vocab_cache
    )
    enc = _precompute_sparse_encode_tables(vocabs, SPARSE)

    # Read only the first batch of the NORMALIZED test Parquet (fast preflight).
    pf = pq.ParquetFile(test_pq)
    cols = SPARSE + DENSEF + ["click", "purchase"]
    batch = next(pf.iter_batches(batch_size=args.batch_rows, columns=cols))
    tbl = pa.Table.from_batches([batch])
    sparse_t, dense_t, label_t = encode_and_tensorize_arrow(tbl, enc, SPARSE, DENSEF, "click")

    b = sparse_t.shape[0]
    assert sparse_t.shape == (b, len(SPARSE)), sparse_t.shape
    assert dense_t.shape == (b, len(DENSEF)), dense_t.shape
    assert label_t.shape == (b,), label_t.shape
    assert torch.isfinite(dense_t).all(), "non-finite dense features"
    assert int(sparse_t.min()) >= 0, "negative sparse id"
    # The whole point of E1: the data path pulls in NO model code.
    leaked = [m for m in sys.modules if m.startswith("esmm_ali_ccp_impl")]
    assert not leaked, f"model module leaked into the data path: {leaked}"

    print(
        f"[smoke] OK in {time.time() - t0:.1f}s — built a real batch via datasets.aliccp "
        f"(B={b}, sparse={tuple(sparse_t.shape)}, dense={tuple(dense_t.shape)}, "
        f"vocab_fields={len(cards)}); NO model module imported.",
        flush=True,
    )


if __name__ == "__main__":
    main()
