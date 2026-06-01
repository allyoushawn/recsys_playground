"""Ali-CCP torch tensorizers: label-encode sparse features + build CPU tensors.

Extracted (E1) from `experiments/20260404_ali_cpp_esmm/esmm_ali_ccp_impl.py`. Contains
the torch-dependent encoding helpers only — NO model / training / evaluation code.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch

# --------------- Label encoding ---------------

def build_sparse_vocabs(df, sparse_cols):
    """Build label-encoding vocabularies from a DataFrame. Index 0 = UNK/unseen."""
    vocabs = {}
    cardinalities = []
    for col in sparse_cols:
        unique_vals = df[col].astype(str).unique()
        vocab = {v: i + 1 for i, v in enumerate(unique_vals)}
        vocabs[col] = vocab
        cardinalities.append(len(vocab))
    return vocabs, cardinalities


def encode_and_tensorize(df, vocabs, sparse_cols, dense_feat_cols, label_col):
    """Encode sparse features via vocabs, cast dense to float, return tensors.
    Sparse indices use int32 on CPU to halve RAM; models call .long() for Embedding."""
    sparse_arrays = []
    for col in sparse_cols:
        encoded = df[col].astype(str).map(vocabs[col]).fillna(0).astype(np.int32).values
        sparse_arrays.append(encoded)
    sparse_t = torch.from_numpy(np.column_stack(sparse_arrays).astype(np.int32))
    dense_t = torch.from_numpy(
        df[dense_feat_cols].apply(pd.to_numeric, errors='coerce')
        .fillna(0.0).values.astype(np.float32)
    )
    label_t = torch.from_numpy(df[label_col].values.astype(np.float32))
    return sparse_t, dense_t, label_t


def _precompute_sparse_encode_tables(vocabs, sparse_cols):
    """Per-column (categories_tuple, lookup_int32) for vectorized sparse encoding."""
    tables = {}
    for col in sparse_cols:
        v = vocabs[col]
        cats = tuple(v.keys())
        lookup = np.array([v[c] for c in cats], dtype=np.int32)
        tables[col] = (cats, lookup)
    return tables


def encode_and_tensorize_fast(df, enc_tables, sparse_cols, dense_feat_cols, label_col):
    """Same outputs as encode_and_tensorize; faster categorical path + contiguous dense."""
    sparse_arrays = []
    for col in sparse_cols:
        cats, lookup = enc_tables[col]
        c = pd.Categorical(df[col].astype(str), categories=cats)
        codes = c.codes.astype(np.int64, copy=False)
        enc = np.where(codes >= 0, lookup[codes], 0).astype(np.int32)
        sparse_arrays.append(enc)
    sparse_t = torch.from_numpy(np.column_stack(sparse_arrays).astype(np.int32, copy=False))
    dense_arr = (
        df[dense_feat_cols].apply(pd.to_numeric, errors='coerce').fillna(0.0).values.astype(np.float32)
    )
    dense_t = torch.from_numpy(np.ascontiguousarray(dense_arr))
    label_t = torch.from_numpy(df[label_col].values.astype(np.float32))
    return sparse_t, dense_t, label_t


def encode_and_tensorize_arrow(table, enc_tables, sparse_cols, dense_feat_cols, label_col):
    """Same tensor dtypes/shapes as encode_and_tensorize_fast; input is pyarrow.Table."""
    import pyarrow as pa
    import pyarrow.compute as pc

    sparse_arrays = []
    for col in sparse_cols:
        cats, lookup = enc_tables[col]
        pa_col = table.column(col).combine_chunks()
        if pa.types.is_string(pa_col.type) or pa.types.is_large_string(pa_col.type):
            try:
                str_np = pa_col.to_numpy(zero_copy_only=False)
            except Exception:
                str_np = np.array(pa_col.to_pylist(), dtype=object)
        else:
            str_np = pc.cast(pa_col, pa.large_string()).to_numpy(zero_copy_only=False)
        s = pd.Series(str_np, dtype=object).astype(str)
        c = pd.Categorical(s, categories=list(cats))
        codes = c.codes.astype(np.int64, copy=False)
        enc = np.where(codes >= 0, lookup[codes], 0).astype(np.int32)
        sparse_arrays.append(enc)
    sparse_t = torch.from_numpy(np.column_stack(sparse_arrays).astype(np.int32, copy=False))

    dense_stack = []
    for cname in dense_feat_cols:
        x = table.column(cname).combine_chunks()
        if pa.types.is_null(x.type):
            arr = np.zeros(table.num_rows, dtype=np.float32)
        elif pa.types.is_floating(x.type) or pa.types.is_integer(x.type):
            arr = np.asarray(x.to_numpy(zero_copy_only=False), dtype=np.float64)
        else:
            arr = np.asarray(pc.cast(x, pa.float64()).to_numpy(zero_copy_only=False), dtype=np.float64)
        arr = np.nan_to_num(arr, nan=0.0).astype(np.float32)
        dense_stack.append(arr.reshape(-1, 1))
    dense_t = torch.from_numpy(np.ascontiguousarray(np.hstack(dense_stack)))

    ycol = table.column(label_col).combine_chunks()
    if pa.types.is_floating(ycol.type) or pa.types.is_integer(ycol.type):
        yarr = np.asarray(ycol.to_numpy(zero_copy_only=False), dtype=np.float32)
    else:
        yarr = np.asarray(pc.cast(ycol, pa.float32()).to_numpy(zero_copy_only=False), dtype=np.float32)
    label_t = torch.from_numpy(np.ascontiguousarray(yarr))
    return sparse_t, dense_t, label_t
