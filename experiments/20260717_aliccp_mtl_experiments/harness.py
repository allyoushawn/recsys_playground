"""Ali-CCP MTL training harness — shared trainer, loss, and evaluator.

Cloned from `experiments/20260404_ali_cpp_esmm/esmm_ali_ccp_impl.py`:
  - `train_esmm_parquet_rowgroups` (the trainer; generic over model_ctor/model_ctor_kwargs,
    already shared across the MTL heads in models.py and the single-task baselines in
    single_task_models.py)
  - `_esmm_multitask_bce_from_probs` (its loss function)
  - `evaluate_esmm_multitask_streaming_parquet` (primary eval: CTR/CTCVR/CVR AUC + PR-AUC +
    logloss + ECE) and its metric dependencies `binary_pr_auc`, `binary_bce_log_loss`,
    `expected_calibration_error`
  - `count_parameters`

Two deliberate deviations from the source file:
  1. The `isinstance(model, ESMM_PLE): use_amp=False` special case is preserved EXACTLY —
     load-bearing; PLE is numerically unstable (CUDA BCE domain asserts) under AMP/fp16.
  2. The `hasattr(model, 'compute_egean_loss'/'compute_dcmt_loss'/'compute_escm2_loss')`
     custom-loss routing and the `track_grad_snr`/`GradSNRTracker` instrumentation were
     DROPPED, including their downstream branches in `_prepare_row_group_tensors` and
     `_step_batch`. None of the 13 models trained by this folder (models.py,
     single_task_models.py) expose those `compute_*_loss` methods or set
     `track_grad_snr=True`, so those branches were always dead code for this study —
     they belong to the separate NDM/ESCM2/EGEAN/DCMT/AdaOrderCross family that is out
     of scope here (see models.py's module docstring).
"""
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, average_precision_score, log_loss
from torch.utils.data import DataLoader, TensorDataset

from datasets.aliccp.encode import (
    _precompute_sparse_encode_tables,
    encode_and_tensorize_arrow,
    encode_and_tensorize_fast,
)
from models import ESMMModel, ESMM_PLE

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')
if device.type == 'cuda':
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')

DEFAULT_EMBED_DIM = 18
# Matches DEFAULT_EVAL_TEST_BATCH_ROWS in datasets/aliccp/data.py (streaming chunk size for
# eval, not a subsampling cap — evaluate_esmm_multitask_streaming_parquet still covers the
# full test file). Defined locally since data.py is not part of the sibling contract.
DEFAULT_EVAL_TEST_BATCH_ROWS = 500_000
R5_COMPILE_MODE = "default"  # string mode for torch.compile(..., mode=R5_COMPILE_MODE)


# --------------- Eval metrics ---------------

def binary_pr_auc(labels, probs):
    '''Average precision (PR-AUC) for binary labels in {0,1}.'''
    y = np.asarray(labels).ravel()
    p = np.asarray(probs, dtype=np.float64).ravel()
    if len(np.unique(y)) < 2:
        return float('nan')
    return float(average_precision_score(y, p))


def binary_bce_log_loss(labels, probs):
    '''Sklearn log loss for binary probabilities (matches BCE on probabilities).'''
    y = np.asarray(labels).ravel()
    p = np.clip(np.asarray(probs, dtype=np.float64).ravel(), 1e-9, 1 - 1e-9)
    if len(np.unique(y)) < 2:
        return float('nan')
    return float(log_loss(y, p, labels=[0, 1]))


def expected_calibration_error(probs, labels, n_bins=15):
    '''ECE: mean |bin_confidence - bin_accuracy| weighted by bin mass.'''
    p = np.clip(np.asarray(probs, dtype=np.float64).ravel(), 1e-9, 1 - 1e-9)
    y = np.asarray(labels, dtype=np.float64).ravel()
    edges = np.linspace(0.0, 1.0, int(n_bins) + 1)
    ece = 0.0
    n = len(p)
    if n == 0:
        return float('nan')
    for i in range(int(n_bins)):
        lo, hi = edges[i], edges[i + 1]
        if i == int(n_bins) - 1:
            m = (p >= lo) & (p <= hi)
        else:
            m = (p >= lo) & (p < hi)
        cnt = int(m.sum())
        if cnt == 0:
            continue
        ece += (cnt / n) * abs(p[m].mean() - y[m].mean())
    return float(ece)


def evaluate_esmm_multitask_streaming_parquet(
    model, parquet_path, vocabs, sparse_cols, dense_feat_cols, batch_rows=None, ece_bins=15, eval_batch_rows=DEFAULT_EVAL_TEST_BATCH_ROWS,
):
    '''One streaming pass over test Parquet: CTR / CTCVR / CVR (clicked-only) AUC, PR-AUC, log loss, ECE.'''
    import pyarrow.parquet as pq
    if batch_rows is None:
        batch_rows = eval_batch_rows
    cols = sparse_cols + dense_feat_cols + ['click', 'purchase']
    model.eval()
    enc_tables = _precompute_sparse_encode_tables(vocabs, sparse_cols)
    pf = pq.ParquetFile(parquet_path)
    ctr_p, ctr_y = [], []
    ctcvr_p, ctcvr_y = [], []
    cvr_p, cvr_y = [], []
    for batch in pf.iter_batches(batch_size=batch_rows, columns=cols):
        df = batch.to_pandas()
        del batch
        sp, dn, _ = encode_and_tensorize_fast(df, enc_tables, sparse_cols, dense_feat_cols, 'purchase')
        y_click = torch.from_numpy(df['click'].values.astype(np.float32))
        y_pur = torch.from_numpy(df['purchase'].values.astype(np.float32))
        y_ctcvr = y_click * y_pur
        del df
        loader = DataLoader(
            TensorDataset(sp, dn, y_click, y_pur, y_ctcvr),
            batch_size=4096, shuffle=False,
        )
        with torch.no_grad():
            for spb, dnb, ycb, ypb, yccb in loader:
                spb, dnb = spb.to(device), dnb.to(device)
                pc, pv, pcc = model(spb, dnb)
                pc_np = pc.cpu().numpy()
                pv_np = pv.cpu().numpy()
                pcc_np = pcc.cpu().numpy()
                yc_np = ycb.numpy()
                yp_np = ypb.numpy()
                ycc_np = yccb.numpy()
                ctr_p.append(pc_np)
                ctr_y.append(yc_np)
                ctcvr_p.append(pcc_np)
                ctcvr_y.append(ycc_np)
                m = yc_np > 0.5
                if m.any():
                    cvr_p.append(pv_np[m])
                    cvr_y.append(yp_np[m])
        del sp, dn, y_click, y_pur, y_ctcvr
    ctr_p = np.concatenate(ctr_p)
    ctr_y = np.concatenate(ctr_y)
    ctcvr_p = np.concatenate(ctcvr_p)
    ctcvr_y = np.concatenate(ctcvr_y)
    cvr_p = np.concatenate(cvr_p) if cvr_p else np.array([], dtype=np.float32)
    cvr_y = np.concatenate(cvr_y) if cvr_y else np.array([], dtype=np.float32)
    out = {}
    # CTR
    if len(np.unique(ctr_y)) >= 2:
        out['CTR_AUC'] = float(roc_auc_score(ctr_y, ctr_p))
        out['CTR_PR_AUC'] = binary_pr_auc(ctr_y, ctr_p)
        out['logloss_ctr'] = binary_bce_log_loss(ctr_y, ctr_p)
        out['ECE_ctr'] = expected_calibration_error(ctr_p, ctr_y, n_bins=ece_bins)
    else:
        out['CTR_AUC'] = float('nan')
        out['CTR_PR_AUC'] = float('nan')
        out['logloss_ctr'] = float('nan')
        out['ECE_ctr'] = float('nan')
    # CTCVR
    if len(np.unique(ctcvr_y)) >= 2:
        out['CTCVR_AUC'] = float(roc_auc_score(ctcvr_y, ctcvr_p))
        out['CTCVR_PR_AUC'] = binary_pr_auc(ctcvr_y, ctcvr_p)
        out['logloss_ctcvr'] = binary_bce_log_loss(ctcvr_y, ctcvr_p)
        out['ECE_ctcvr'] = expected_calibration_error(ctcvr_p, ctcvr_y, n_bins=ece_bins)
    else:
        out['CTCVR_AUC'] = float('nan')
        out['CTCVR_PR_AUC'] = float('nan')
        out['logloss_ctcvr'] = float('nan')
        out['ECE_ctcvr'] = float('nan')
    # CVR clicked-only
    if cvr_p.size > 0 and len(np.unique(cvr_y)) >= 2:
        out['CVR_AUC'] = float(roc_auc_score(cvr_y, cvr_p))
    else:
        out['CVR_AUC'] = float('nan')
    return out


# --------------- ESMM streaming training ---------------

def _esmm_multitask_bce_from_probs(p_ctr, p_ctcvr, y_click, y_ctcvr, eps=1e-6):
    """BCE on probabilities with float32 + clamp; avoids CUDA asserts from NaN/Inf or (0,1) drift under AMP."""
    yc = torch.nan_to_num(y_click.float(), nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
    ycc = torch.nan_to_num(y_ctcvr.float(), nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
    pc = torch.nan_to_num(p_ctr.float(), nan=0.5, posinf=1.0, neginf=0.0).clamp(eps, 1.0 - eps)
    pcc = torch.nan_to_num(p_ctcvr.float(), nan=0.5, posinf=1.0, neginf=0.0).clamp(eps, 1.0 - eps)
    return nn.functional.binary_cross_entropy(pc, yc) + nn.functional.binary_cross_entropy(pcc, ycc)


def train_esmm_parquet_rowgroups(
    parquet_path, vocabs, field_cardinalities, sparse_cols, dense_feat_cols,
    epochs=5, batch_size=4096, lr=1e-3, seed=42,
    weight_decay=0.0,
    embed_dim=DEFAULT_EMBED_DIM,
    max_wall_seconds=None,
    max_optimizer_steps=None,
    max_batches_per_epoch=None,
    max_row_groups_per_epoch=None,
    use_amp=True,
    prefetch_row_groups=True,
    use_manual_batches=True,
    use_torch_compile=False,
    read_row_groups_as_arrow=False,
    model_ctor=None,
    model_ctor_kwargs=None,
):
    """One full pass over the file = one epoch; row-group order shuffled each epoch.

    Optional caps (None disables each): after each optimizer.step(), check limits and
    break with EARLY_STOP. Optimizer steps are cumulative across epochs; batch and
    row-group caps reset each epoch. Wall clock uses perf_counter from train start.

    use_amp: if True and CUDA, forward runs under autocast; BCE uses the float32 multitask
    helper. Default True (no-op on CPU). **Always treated as False for ESMM_PLE** (AMP caused
    CUDA BCE domain asserts and unstable half-precision in the deep gated stack).

    prefetch_row_groups: if True, overlap Parquet decode/tensor prep for the next row
    group with training on the current (ThreadPoolExecutor max_workers=1, depth 1). Default True.

    use_manual_batches: if True, shuffle each row group with torch.randperm and slice
    batch_size chunks without DataLoader. If False, use DataLoader (legacy path). Default True.

    use_torch_compile: if True, CUDA, and torch>=2.0, wrap the model with torch.compile;
    on failure prints and keeps eager. Warmup steps run before the timed train span.

    model_ctor: optional callable (field_cardinalities, num_dense, embed_dim) -> nn.Module.
        Default builds ESMMModel(..., **model_ctor_kwargs).

    model_ctor_kwargs: optional dict of extra kwargs forwarded into model_ctor (or default ESMMModel).

    read_row_groups_as_arrow: if True, decode row groups with pyarrow only (no full
    DataFrame); falls back to pandas with a one-time message on first failure.
    """
    import random
    import pyarrow.parquet as pq

    # torch.manual_seed was missing — model init previously drew from the unseeded global RNG despite the seed label.
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    _mkw = dict(model_ctor_kwargs or {})
    if model_ctor is None:
        model = ESMMModel(
            field_cardinalities, num_dense=len(dense_feat_cols), embed_dim=embed_dim, **_mkw,
        )
    else:
        model = model_ctor(field_cardinalities, len(dense_feat_cols), embed_dim, **_mkw)
    model.to(device)
    if isinstance(model, ESMM_PLE):
        if use_amp:
            print(
                '[train_esmm_parquet_rowgroups] ESMM_PLE: forcing use_amp=False '
                '(disable autocast/GradScaler for numerical stability).'
            )
        use_amp = False
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=weight_decay)
    use_amp_cuda = bool(use_amp and device.type == 'cuda')
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp_cuda)
    cols = sparse_cols + dense_feat_cols + ["click", "purchase"]
    pf = pq.ParquetFile(parquet_path)
    nrg = pf.num_row_groups
    if nrg == 0:
        raise ValueError(f"No row groups in {parquet_path}")
    n_train = int(pf.metadata.num_rows)
    enc_tables = _precompute_sparse_encode_tables(vocabs, sparse_cols)
    losses_out = []

    compiled_active = False
    if use_torch_compile:
        if torch.cuda.is_available():
            try:
                tv = torch.__version__.split('+')[0].split('.')
                major, minor = int(tv[0]), int(tv[1])
            except Exception:
                major, minor = 0, 0
            if (major, minor) >= (2, 0):
                try:
                    model = torch.compile(model, mode=R5_COMPILE_MODE)
                    compiled_active = True
                except Exception as e:
                    print(f'torch.compile failed ({e}); using eager ESMMModel.')
            else:
                print(f'torch.compile skipped: need torch>=2.0, got {torch.__version__}')
        else:
            print('torch.compile skipped: CUDA not available')

    arrow_fallback = [False]
    arrow_warned = {'printed': False}

    def _prepare_row_group_tensors(rg_idx):
        raw = pf.read_row_group(rg_idx, columns=cols)
        if read_row_groups_as_arrow and not arrow_fallback[0]:
            try:
                sp, dn, y_click = encode_and_tensorize_arrow(
                    raw, enc_tables, sparse_cols, dense_feat_cols, 'click')
                y_purchase = torch.from_numpy(
                    np.asarray(
                        raw.column('purchase').combine_chunks().to_numpy(zero_copy_only=False),
                        dtype=np.float32,
                    ))
                y_ctcvr = y_click * y_purchase
                del y_purchase
                return sp, dn, y_click, y_ctcvr
            except Exception as e:
                if not arrow_warned['printed']:
                    print(f'read_row_groups_as_arrow failed ({e}); falling back to pandas for remaining row groups.')
                    arrow_warned['printed'] = True
                arrow_fallback[0] = True
        sub = raw.to_pandas()
        sp, dn, y_click = encode_and_tensorize_fast(
            sub, enc_tables, sparse_cols, dense_feat_cols, 'click')
        y_purchase = torch.from_numpy(sub['purchase'].values.astype(np.float32))
        y_ctcvr = y_click * y_purchase
        del y_purchase
        del sub
        return sp, dn, y_click, y_ctcvr

    if compiled_active:
        # torch.compile warmup.
        try:
            sp0, dn0, yc0, ycc0 = _prepare_row_group_tensors(0)
            n0 = int(sp0.size(0))
            if n0 > 0:
                nw = min(int(batch_size), n0)
                for _ in range(3):
                    optimizer.zero_grad(set_to_none=True)
                    sp_b = sp0[:nw].to(device, non_blocking=True).long()
                    dn_b = dn0[:nw].to(device, non_blocking=True)
                    yc_b = yc0[:nw].to(device, non_blocking=True)
                    ycc_b = ycc0[:nw].to(device, non_blocking=True)
                    if use_amp_cuda:
                        with torch.amp.autocast('cuda', enabled=True):
                            p_ctr, _, p_ctcvr = model(sp_b, dn_b)
                        loss = _esmm_multitask_bce_from_probs(p_ctr, p_ctcvr, yc_b, ycc_b)
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        p_ctr, _, p_ctcvr = model(sp_b, dn_b)
                        loss = _esmm_multitask_bce_from_probs(p_ctr, p_ctcvr, yc_b, ycc_b)
                        loss.backward()
                        optimizer.step()
            del sp0, dn0, yc0, ycc0
        except Exception as e:
            print(f'torch.compile warmup failed ({e}); continuing training.')

    early_reason = None
    opt_steps = 0
    samples_total_run = 0
    t_train_all = time.perf_counter()
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()

    def _train_one_rg_tensors(sp, dn, y_click, y_ctcvr):
        nonlocal total_loss, n_batches, opt_steps, samples_this_epoch, batches_this_epoch
        nonlocal samples_total_run, early_reason

        def _step_batch(sp_b, dn_b, yc_b, ycc_b):
            nonlocal total_loss, n_batches, opt_steps, samples_this_epoch, batches_this_epoch
            nonlocal samples_total_run, early_reason
            optimizer.zero_grad(set_to_none=True)
            if use_amp_cuda:
                with torch.amp.autocast('cuda', enabled=True):
                    p_ctr, _, p_ctcvr = model(sp_b, dn_b)
                loss = _esmm_multitask_bce_from_probs(p_ctr, p_ctcvr, yc_b, ycc_b)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                p_ctr, _, p_ctcvr = model(sp_b, dn_b)
                loss = _esmm_multitask_bce_from_probs(p_ctr, p_ctcvr, yc_b, ycc_b)
                loss.backward()
                optimizer.step()
            total_loss += loss.item()
            n_batches += 1
            opt_steps += 1
            bs = int(sp_b.size(0))
            batches_this_epoch += 1
            samples_this_epoch += bs
            samples_total_run += bs
            if max_wall_seconds is not None and (time.perf_counter() - t_train_all) >= max_wall_seconds:
                early_reason = 'max_wall_seconds'
                return True
            if max_optimizer_steps is not None and opt_steps >= max_optimizer_steps:
                early_reason = 'max_optimizer_steps'
                return True
            if max_batches_per_epoch is not None and batches_this_epoch >= max_batches_per_epoch:
                early_reason = 'max_batches_per_epoch'
                return True
            return False

        if use_manual_batches:
            n = int(sp.size(0))
            perm = torch.randperm(n)
            for start in range(0, n, batch_size):
                idx = perm[start:start + batch_size]
                sp_b = sp[idx].to(device, non_blocking=True).long()
                dn_b = dn[idx].to(device, non_blocking=True)
                yc_b = y_click[idx].to(device, non_blocking=True)
                ycc_b = y_ctcvr[idx].to(device, non_blocking=True)
                if _step_batch(sp_b, dn_b, yc_b, ycc_b):
                    return True
            return False

        loader = DataLoader(
            TensorDataset(sp, dn, y_click, y_ctcvr),
            batch_size=batch_size, shuffle=True, pin_memory=(device.type == 'cuda'),
        )
        for sp_b, dn_b, yc_b, ycc_b in loader:
            sp_b = sp_b.to(device, non_blocking=True).long()
            dn_b = dn_b.to(device, non_blocking=True)
            yc_b = yc_b.to(device, non_blocking=True)
            ycc_b = ycc_b.to(device, non_blocking=True)
            if _step_batch(sp_b, dn_b, yc_b, ycc_b):
                return True
        return False

    for epoch in range(epochs):
        if early_reason:
            break
        rng = list(range(nrg))
        random.seed(seed + epoch)
        random.shuffle(rng)
        model.train()
        total_loss, n_batches = 0.0, 0
        t_epoch = time.perf_counter()
        samples_this_epoch = 0
        batches_this_epoch = 0
        rgs_this_epoch = 0
        if prefetch_row_groups and len(rng) > 0:
            with ThreadPoolExecutor(max_workers=1) as _rg_ex:
                _fut = _rg_ex.submit(_prepare_row_group_tensors, rng[0])
                for rg_i in range(len(rng)):
                    rg = rng[rg_i]
                    if max_row_groups_per_epoch is not None and rgs_this_epoch >= max_row_groups_per_epoch:
                        early_reason = 'max_row_groups_per_epoch'
                        break
                    rgs_this_epoch += 1
                    sp, dn, y_click, y_ctcvr = _fut.result()
                    if rg_i + 1 < len(rng):
                        _fut = _rg_ex.submit(_prepare_row_group_tensors, rng[rg_i + 1])
                    rg_stop = _train_one_rg_tensors(sp, dn, y_click, y_ctcvr)
                    del sp, dn, y_click, y_ctcvr
                    if rg_stop:
                        break
        else:
            for rg in rng:
                if max_row_groups_per_epoch is not None and rgs_this_epoch >= max_row_groups_per_epoch:
                    early_reason = 'max_row_groups_per_epoch'
                    break
                rgs_this_epoch += 1
                sp, dn, y_click, y_ctcvr = _prepare_row_group_tensors(rg)
                rg_stop = _train_one_rg_tensors(sp, dn, y_click, y_ctcvr)
                del sp, dn, y_click, y_ctcvr
                if rg_stop:
                    break
        avg = total_loss / max(n_batches, 1)
        losses_out.append(avg)
        dt_ep = time.perf_counter() - t_epoch
        denom = samples_this_epoch if samples_this_epoch > 0 else n_train
        sps_ep = denom / dt_ep if dt_ep > 0 else 0.0
        print(f"  Epoch {epoch+1}/{epochs}: loss={avg:.4f} ({n_batches} batches)  ({sps_ep:,.0f} samples/s)")
        if early_reason:
            print(f'EARLY_STOP: reason={early_reason}')
            break
    dt_all = time.perf_counter() - t_train_all
    sps_all = samples_total_run / dt_all if dt_all > 0 else 0.0
    print(f'  Throughput (train span): {sps_all:,.0f} samples/s  ({samples_total_run:,} samples in {dt_all:.1f}s)')
    train_meta = {
        'early_stop_reason': early_reason,
        'samples_total_run': int(samples_total_run),
        'train_wall_seconds': float(dt_all),
        'samples_per_sec': float(sps_all),
        'use_amp': use_amp_cuda,
        'prefetch_row_groups': bool(prefetch_row_groups),
        'use_manual_batches': bool(use_manual_batches),
        'batch_size': int(batch_size),
        'use_torch_compile': bool(use_torch_compile),
        'torch_compile_active': bool(compiled_active),
        'read_row_groups_as_arrow': bool(read_row_groups_as_arrow),
        'read_row_groups_arrow_used': bool(read_row_groups_as_arrow and not arrow_fallback[0]),
    }
    if device.type == 'cuda':
        train_meta['cuda_max_memory_allocated_bytes'] = int(torch.cuda.max_memory_allocated())
    return model, losses_out, train_meta


# --------------- Parameter count helper ---------------

def count_parameters(model):
    """Return total number of trainable parameters in a nn.Module."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
