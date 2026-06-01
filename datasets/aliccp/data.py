"""Ali-CCP dataset layer: raw-CSV parsing, Parquet I/O, vocab building, normalization.

Extracted (E1) from `experiments/20260404_ali_cpp_esmm/esmm_ali_ccp_impl.py` so any
(non-ESMM) model can use the Ali-CCP dataset WITHOUT importing torch/model code.
This module is TORCH-FREE: stdlib + numpy + pandas + (lazy) pyarrow only.
"""
from __future__ import annotations

import gc
import os
import pickle
import tarfile
from collections import Counter
from contextlib import nullcontext

import numpy as np
import pandas as pd
from tqdm import tqdm

# Defaults (match notebook Config cell)
DEFAULT_STREAM_PARSE_CHUNK_ROWS = 500_000
DEFAULT_VOCAB_SCAN_ROWS_PER_BATCH = 200_000
DEFAULT_NORM_STREAM_BATCH_ROWS = 500_000
DEFAULT_EVAL_TEST_BATCH_ROWS = 500_000


# --- Ali-CCP data / Parquet ---

SAMPLE_TRAIN_TAR = 'sample_train.tar'
SAMPLE_TEST_TAR = 'sample_test.tar'
TRAIN_CSV = 'ali_ccp_train.csv'
VAL_CSV = 'ali_ccp_val.csv'
TEST_CSV = 'ali_ccp_test.csv'
SINGLE_CSV = 'ali_ccp.csv'
SAMPLE_SKELETON_TRAIN = 'sample_skeleton_train.csv'
SAMPLE_SKELETON_TEST = 'sample_skeleton_test.csv'
COMMON_FEATURES_TRAIN = 'common_features_train.csv'
COMMON_FEATURES_TEST = 'common_features_test.csv'

# Canonical Ali-CCP feature schema (a property of the dataset, not of any model).
# Sparse = categorical id fields; dense = the numeric fields, stored normalized as
# 'D'+col after log1p. Any model consuming this dataset should use these.
ALICCP_SPARSE_COLS = ['101', '121', '122', '124', '125', '126', '127', '128', '129',
                      '205', '206', '207', '210', '216', '508', '509', '702', '853',
                      '301', '109_14', '110_14', '127_14', '150_14']
ALICCP_DENSE_COLS = ['109_14', '110_14', '127_14', '150_14', '508', '509', '702', '853']
ALICCP_DENSE_FEAT_COLS = ['D' + c for c in ALICCP_DENSE_COLS]


def _sample_tag_for_cache(sample_size):
    return 'full' if sample_size is None else str(sample_size)

def load_or_parse_ali_ccp(data_dir, sample_size, processed_dir, sparse_cols, dense_cols, dense_feat_cols):
    """Load Parquet from processed_dir if present; else parse raw Tianchi CSVs and save."""
    os.makedirs(processed_dir, exist_ok=True)
    tag = _sample_tag_for_cache(sample_size)
    p_train = os.path.join(processed_dir, f'parsed_train_rows_{tag}.parquet')
    p_test = os.path.join(processed_dir, f'parsed_test_rows_{tag}.parquet')
    # Reuse legacy Parquet from older notebook layout (…/data/esmm_cache/) if present.
    if tag == 'full' and not (os.path.isfile(p_train) and os.path.isfile(p_test)):
        _legacy = os.path.join(os.path.dirname(os.path.abspath(data_dir)), 'esmm_cache')
        _lt, _le = os.path.join(_legacy, 'ali_ccp_full_train.parquet'), os.path.join(_legacy, 'ali_ccp_full_test.parquet')
        if os.path.isfile(_lt) and os.path.isfile(_le):
            import shutil
            print(f'Copying full split from legacy {_legacy} -> {processed_dir}')
            shutil.copy2(_lt, p_train)
            shutil.copy2(_le, p_test)
    if os.path.isfile(p_train) and os.path.isfile(p_test):
        print(f'Loading cached parsed AliCCP (sample={tag}) from {processed_dir}')
        if sample_size is None:
            print('  Full split: not calling pd.read_parquet (avoids ~40GB+ RAM). Use Parquet paths in Round 4.')
            return None, None
        return pd.read_parquet(p_train), pd.read_parquet(p_test)
    print(f'No Parquet cache for sample={tag}; parsing raw Tianchi files (slow)...')
    if sample_size is None:
        parse_raw_ali_ccp_streaming_writes(
            data_dir, p_train, p_test, sample_size=None,
            sparse_cols=sparse_cols, dense_cols=dense_cols, dense_feat_cols=dense_feat_cols,
        )
        print('  Full Parquet written. Returning (None, None) to avoid read_parquet OOM.')
        return None, None
    df_tr, df_te = parse_raw_ali_ccp(
        data_dir,
        sample_size=sample_size,
        sparse_cols=sparse_cols,
        dense_cols=dense_cols,
        dense_feat_cols=dense_feat_cols,
    )
    if df_tr is not None and len(df_tr) > 0:
        print(f'Writing Parquet cache to {processed_dir} (snappy compression)...')
        df_tr.to_parquet(p_train, index=False, compression='snappy')
        df_te.to_parquet(p_test, index=False, compression='snappy')
    return df_tr, df_te

def _find_file_recursive(root, filename):
    direct = os.path.join(root, filename)
    if os.path.isfile(direct):
        return direct
    for dirpath, _, filenames in os.walk(root):
        if filename in filenames:
            return os.path.join(dirpath, filename)
    return None


find_file_recursive = _find_file_recursive

def _parse_feat_str(feat_str, sparse_cols, dense_cols):
    feat_dict = {}
    for fstr in feat_str.split('\x01'):
        if '\x02' not in fstr or '\x03' not in fstr:
            continue
        parts = fstr.split('\x02', 1)
        filed = parts[0]
        feat_val = parts[1]
        if '\x03' in feat_val:
            feat, val = feat_val.split('\x03', 1)
            if filed in sparse_cols:
                feat_dict[filed] = feat
            if filed in dense_cols:
                feat_dict['D' + filed] = val
    return feat_dict

def parse_raw_ali_ccp(
    data_dir, sample_size=None, sparse_cols=None, dense_cols=None, dense_feat_cols=None,
):
    common_train_path = _find_file_recursive(data_dir, COMMON_FEATURES_TRAIN)
    common_test_path = _find_file_recursive(data_dir, COMMON_FEATURES_TEST)
    skeleton_train_path = _find_file_recursive(data_dir, SAMPLE_SKELETON_TRAIN)
    skeleton_test_path = _find_file_recursive(data_dir, SAMPLE_SKELETON_TEST)
    if not all([common_train_path, common_test_path, skeleton_train_path, skeleton_test_path]):
        return None, None

    test_limit = (sample_size // 4) if sample_size else None

    needed_ids = set()
    for path, limit, mode in [
        (skeleton_train_path, sample_size, 'train'),
        (skeleton_test_path, test_limit, 'test'),
    ]:
        with open(path, 'r') as f:
            for i, line in enumerate(tqdm(f, desc=f'scan_skeleton_{mode}', leave=False)):
                if limit and i >= limit:
                    break
                parts = line.strip().split(',')
                if len(parts) >= 4:
                    needed_ids.add(parts[3])
    print(f'Unique common-feature IDs needed: {len(needed_ids):,}')

    common_feat = {}
    for path, mode in [(common_train_path, 'train'), (common_test_path, 'test')]:
        with open(path, 'r') as f:
            for line in tqdm(f, desc=f'common_features_{mode}', leave=False):
                parts = line.strip().split(',')
                if len(parts) < 3:
                    continue
                if parts[0] not in needed_ids:
                    continue
                feat_dict = _parse_feat_str(parts[2], sparse_cols, dense_cols)
                common_feat[parts[0]] = feat_dict
    print(f'Loaded {len(common_feat):,} common-feature entries')

    rows_train, rows_test = [], []
    for path, rows_out, limit, mode in [
        (skeleton_train_path, rows_train, sample_size, 'train'),
        (skeleton_test_path, rows_test, test_limit, 'test'),
    ]:
        with open(path, 'r') as f:
            for i, line in enumerate(tqdm(f, desc=f'sample_skeleton_{mode}', leave=False)):
                if limit and i >= limit:
                    break
                parts = line.strip().split(',')
                if len(parts) < 6:
                    continue
                click, purchase = parts[1], parts[2]
                if click == '0' and purchase == '1':
                    continue
                feat_dict = _parse_feat_str(parts[5], sparse_cols, dense_cols)
                feat_dict.update(common_feat.get(parts[3], {}))
                row = {'click': click, 'purchase': purchase}
                for k in sparse_cols + dense_feat_cols:
                    row[k] = feat_dict.get(k, '0')
                rows_out.append(row)
    df_train = pd.DataFrame(rows_train)
    df_test = pd.DataFrame(rows_test)
    return df_train, df_test


def parse_raw_ali_ccp_streaming_writes(
    data_dir,
    p_train_out,
    p_test_out,
    sample_size=None,
    chunk_rows=None,
    sparse_cols=None,
    dense_cols=None,
    dense_feat_cols=None,
):
    """Write full (or sampled) train/test Parquet in chunks — avoids 42M-row Python list + giant DataFrame."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    if chunk_rows is None:
        chunk_rows = DEFAULT_STREAM_PARSE_CHUNK_ROWS

    common_train_path = _find_file_recursive(data_dir, COMMON_FEATURES_TRAIN)
    common_test_path = _find_file_recursive(data_dir, COMMON_FEATURES_TEST)
    skeleton_train_path = _find_file_recursive(data_dir, SAMPLE_SKELETON_TRAIN)
    skeleton_test_path = _find_file_recursive(data_dir, SAMPLE_SKELETON_TEST)
    if not all([common_train_path, common_test_path, skeleton_train_path, skeleton_test_path]):
        raise FileNotFoundError('Missing raw Tianchi CSVs under DATA_DIR')

    test_limit = (sample_size // 4) if sample_size else None

    needed_ids = set()
    for path, limit, mode in [
        (skeleton_train_path, sample_size, 'train'),
        (skeleton_test_path, test_limit, 'test'),
    ]:
        with open(path, 'r') as f:
            for i, line in enumerate(tqdm(f, desc=f'scan_skeleton_{mode}', leave=False)):
                if limit and i >= limit:
                    break
                parts = line.strip().split(',')
                if len(parts) >= 4:
                    needed_ids.add(parts[3])
    print(f'Unique common-feature IDs needed: {len(needed_ids):,}')

    common_feat = {}
    for path, mode in [(common_train_path, 'train'), (common_test_path, 'test')]:
        with open(path, 'r') as f:
            for line in tqdm(f, desc=f'common_features_{mode}', leave=False):
                parts = line.strip().split(',')
                if len(parts) < 3:
                    continue
                if parts[0] not in needed_ids:
                    continue
                common_feat[parts[0]] = _parse_feat_str(parts[2], sparse_cols, dense_cols)
    print(f'Loaded {len(common_feat):,} common-feature entries')

    def skeleton_rows(path, limit, desc):
        with open(path, 'r') as f:
            for i, line in enumerate(tqdm(f, desc=desc, leave=False)):
                if limit and i >= limit:
                    break
                parts = line.strip().split(',')
                if len(parts) < 6:
                    continue
                click, purchase = parts[1], parts[2]
                if click == '0' and purchase == '1':
                    continue
                fd = _parse_feat_str(parts[5], sparse_cols, dense_cols)
                fd.update(common_feat.get(parts[3], {}))
                row = {'click': int(click), 'purchase': int(purchase)}
                for k in sparse_cols + dense_feat_cols:
                    row[k] = fd.get(k, '0')
                yield row

    def flush_write(buf, writer_holder, out_path, is_first_table):
        if not buf:
            return writer_holder, is_first_table
        df = pd.DataFrame(buf)
        buf.clear()
        table = pa.Table.from_pandas(df, preserve_index=False)
        del df
        if writer_holder[0] is None:
            writer_holder[0] = pq.ParquetWriter(out_path, table.schema, compression='snappy')
        writer_holder[0].write_table(table)
        del table
        gc.collect()
        return writer_holder, is_first_table

    def write_split(skel_path, limit, out_path, desc):
        buf, wh = [], [None]
        for row in skeleton_rows(skel_path, limit, desc):
            buf.append(row)
            if len(buf) >= chunk_rows:
                wh, _ = flush_write(buf, wh, out_path, False)
        flush_write(buf, wh, out_path, False)
        if wh[0] is not None:
            wh[0].close()
        gc.collect()
        print(f'Wrote {out_path}')

    write_split(skeleton_train_path, sample_size, p_train_out, 'stream_parse_train')
    del common_feat
    gc.collect()
    common_feat = {}
    for path, mode in [(common_train_path, 'train'), (common_test_path, 'test')]:
        with open(path, 'r') as f:
            for line in tqdm(f, desc=f'common_features_{mode}_re', leave=False):
                parts = line.strip().split(',')
                if len(parts) < 3 or parts[0] not in needed_ids:
                    continue
                common_feat[parts[0]] = _parse_feat_str(parts[2], sparse_cols, dense_cols)
    print(f'Reloaded {len(common_feat):,} common-feature entries for test split')
    write_split(skeleton_test_path, test_limit, p_test_out, 'stream_parse_test')
    del common_feat
    gc.collect()


def build_sparse_vocabs_filtered_parquet(
    parquet_path, sparse_cols, min_count=5, vocab_scan_rows_per_batch=DEFAULT_VOCAB_SCAN_ROWS_PER_BATCH,
):
    import pyarrow.parquet as pq

    vocabs, cardinalities = {}, []
    pf = pq.ParquetFile(parquet_path)
    for col in sparse_cols:
        c = Counter()
        for batch in pf.iter_batches(batch_size=vocab_scan_rows_per_batch, columns=[col]):
            for v in batch.column(0).to_pylist():
                c[str(v)] += 1
            del batch
        kept_vals = sorted([x for x, y in c.items() if y >= min_count], key=lambda x: -c[x])
        vocab = {v: i + 1 for i, v in enumerate(kept_vals)}
        vocabs[col] = vocab
        cardinalities.append(len(vocab))
        print(f'  {col}: {len(c)} unique, {len(kept_vals)} kept (>={min_count}), {len(c) - len(kept_vals)} filtered')
        del c
    return vocabs, cardinalities


def _try_load_filtered_vocab_cache(cache_path, parquet_path, sparse_cols, min_count):
    """Return (vocabs, cardinalities) if cache matches train Parquet + settings; else None."""
    import pyarrow.parquet as pq
    if not cache_path or not os.path.isfile(cache_path):
        return None
    ap = os.path.abspath(parquet_path)
    if not os.path.isfile(ap):
        return None
    try:
        with open(cache_path, 'rb') as f:
            payload = pickle.load(f)
    except Exception as e:
        print(f'[vocab cache] unreadable ({e}); rebuilding')
        return None
    meta, vocabs = payload.get('meta'), payload.get('vocabs')
    if not isinstance(meta, dict) or not isinstance(vocabs, dict):
        return None
    if meta.get('parquet_path') != ap:
        return None
    if list(meta.get('sparse_cols') or []) != list(sparse_cols):
        return None
    if int(meta.get('min_count', -1)) != int(min_count):
        return None
    cur_mtime = os.path.getmtime(ap)
    if meta.get('parquet_mtime') != cur_mtime:
        print('[vocab cache] train Parquet mtime changed; rebuilding vocabs')
        return None
    cur_rows = int(pq.ParquetFile(ap).metadata.num_rows)
    if int(meta.get('parquet_num_rows', -1)) != cur_rows:
        print('[vocab cache] train Parquet row count changed; rebuilding vocabs')
        return None
    cards = meta.get('cardinalities')
    if not cards or len(cards) != len(sparse_cols):
        return None
    for col in sparse_cols:
        if col not in vocabs:
            return None
    print(f'[vocab cache] loaded {cache_path} (skip {len(sparse_cols)} Parquet column scans)')
    return vocabs, list(cards)


def _save_filtered_vocab_cache(cache_path, vocabs, cardinalities, parquet_path, sparse_cols, min_count):
    import pyarrow.parquet as pq
    ap = os.path.abspath(parquet_path)
    os.makedirs(os.path.dirname(os.path.abspath(cache_path)) or '.', exist_ok=True)
    meta = {
        'version': 1,
        'parquet_path': ap,
        'parquet_mtime': os.path.getmtime(ap),
        'parquet_num_rows': int(pq.ParquetFile(ap).metadata.num_rows),
        'sparse_cols': list(sparse_cols),
        'min_count': int(min_count),
        'cardinalities': list(cardinalities),
    }
    tmp = cache_path + '.tmp'
    with open(tmp, 'wb') as f:
        pickle.dump({'meta': meta, 'vocabs': vocabs}, f, protocol=4)
    os.replace(tmp, cache_path)
    print(f'[vocab cache] wrote {cache_path}')


def load_or_build_sparse_vocabs_filtered_parquet(
    parquet_path, sparse_cols, min_count=5, cache_path=None, force_rebuild=False,
    vocab_scan_rows_per_batch=DEFAULT_VOCAB_SCAN_ROWS_PER_BATCH,
):
    """Same as build_sparse_vocabs_filtered_parquet but loads from disk when cache is valid."""
    if not force_rebuild and cache_path:
        loaded = _try_load_filtered_vocab_cache(cache_path, parquet_path, sparse_cols, min_count)
        if loaded is not None:
            return loaded
    vocabs, cards = build_sparse_vocabs_filtered_parquet(
        parquet_path, sparse_cols, min_count=min_count,
        vocab_scan_rows_per_batch=vocab_scan_rows_per_batch,
    )
    if cache_path:
        _save_filtered_vocab_cache(cache_path, vocabs, cards, parquet_path, sparse_cols, min_count)
    return vocabs, cards


def stream_normalize_parquet(
    in_path, out_path, sparse_cols, dense_feat_cols,
    norm_stream_batch_rows=DEFAULT_NORM_STREAM_BATCH_ROWS,
):
    import pyarrow as pa
    import pyarrow.parquet as pq

    cols = sparse_cols + dense_feat_cols + ['click', 'purchase']
    pf = pq.ParquetFile(in_path)
    writer = None
    for batch in pf.iter_batches(batch_size=norm_stream_batch_rows, columns=cols):
        df = batch.to_pandas()
        del batch
        for dc in dense_feat_cols:
            x = pd.to_numeric(df[dc], errors='coerce').fillna(0.0)
            df[dc] = np.log1p(np.abs(x)) * np.sign(x)
        df['click'] = df['click'].astype(int)
        df['purchase'] = df['purchase'].astype(int)
        table = pa.Table.from_pandas(df, preserve_index=False)
        del df
        if writer is None:
            writer = pq.ParquetWriter(out_path, table.schema, compression='snappy')
        writer.write_table(table)
        del table
        gc.collect()
    if writer is None:
        raise RuntimeError(f'No rows in {in_path}')
    writer.close()
    gc.collect()
    print(f'Normalized -> {out_path}')


def ensure_full_split_parquet_streaming(data_dir, processed_dir, sparse_cols, dense_cols, dense_feat_cols):
    """Paths to full train/test Parquet; streaming-parse if missing."""
    import shutil
    os.makedirs(processed_dir, exist_ok=True)
    p_train = os.path.join(processed_dir, 'parsed_train_rows_full.parquet')
    p_test = os.path.join(processed_dir, 'parsed_test_rows_full.parquet')
    if not (os.path.isfile(p_train) and os.path.isfile(p_test)):
        _legacy = os.path.join(os.path.dirname(os.path.abspath(data_dir)), 'esmm_cache')
        _lt, _le = os.path.join(_legacy, 'ali_ccp_full_train.parquet'), os.path.join(_legacy, 'ali_ccp_full_test.parquet')
        if os.path.isfile(_lt) and os.path.isfile(_le):
            print(f'Copying legacy full split -> {processed_dir}')
            shutil.copy2(_lt, p_train)
            shutil.copy2(_le, p_test)
    if os.path.isfile(p_train) and os.path.isfile(p_test):
        return p_train, p_test
    print('Streaming parse: raw Tianchi -> full Parquet (chunked)...')
    parse_raw_ali_ccp_streaming_writes(
        data_dir,
        p_train,
        p_test,
        sample_size=None,
        sparse_cols=sparse_cols,
        dense_cols=dense_cols,
        dense_feat_cols=dense_feat_cols,
    )
    return p_train, p_test


def parquet_split_summary(train_path, test_path, summary_batch_rows=None):
    """Row counts from Parquet metadata; click/conversion totals via chunked reads (no full-column load)."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    import pyarrow.compute as pc

    if summary_batch_rows is None:
        summary_batch_rows = DEFAULT_STREAM_PARSE_CHUNK_ROWS

    pf_t, pf_e = pq.ParquetFile(train_path), pq.ParquetFile(test_path)
    print(f'Parquet rows: train={pf_t.metadata.num_rows:,}, test={pf_e.metadata.num_rows:,}')
    for name, pth in [('train', train_path), ('test', test_path)]:
        pf = pq.ParquetFile(pth)
        clicks, conv = 0, 0
        for batch in pf.iter_batches(batch_size=summary_batch_rows, columns=['click', 'purchase']):
            clicks += int(pc.sum(pc.cast(batch.column('click'), pa.int64())).as_py() or 0)
            conv += int(pc.sum(pc.cast(batch.column('purchase'), pa.int64())).as_py() or 0)
            del batch
        print(f'  {name}: clicks={clicks:,}, conversions={conv:,}')

summarize_split = parquet_split_summary
