"""Ali-CCP ESMM experiment: data I/O, models, training, and evaluation.

Extracted from `20260404_esmm_experiment.ipynb` for use by that notebook (orchestration only).
"""
from __future__ import annotations

from contextlib import nullcontext

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from datasets.aliccp.data import *      # noqa: F401,F403  (extracted data layer, E1)
from datasets.aliccp.encode import *    # noqa: F401,F403
# Underscore-prefixed names that `*` won't import but internal code still references.
from datasets.aliccp.data import (  # noqa: F401
    _find_file_recursive,
    _parse_feat_str,
    _try_load_filtered_vocab_cache,
    _save_filtered_vocab_cache,
    _sample_tag_for_cache,
)
from datasets.aliccp.encode import _precompute_sparse_encode_tables  # noqa: F401

DEFAULT_EMBED_DIM = 18


# --- Models, training, evaluation ---

import numpy as np
import math
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score, average_precision_score, log_loss
import time
from concurrent.futures import ThreadPoolExecutor

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')
if device.type == 'cuda':
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')

import pandas as pd

# --------------- BASEModel ---------------

class BASEModel(nn.Module):
    """Paper-exact BASE CVR tower: Embed(18) per field -> concat dense -> MLP 360->200->80->1."""

    def __init__(self, field_cardinalities, num_dense, embed_dim=18,
                 hidden_dims=(360, 200, 80)):
        super().__init__()
        self.embeddings = nn.ModuleList([
            nn.Embedding(card + 1, embed_dim) for card in field_cardinalities
        ])
        input_dim = len(field_cardinalities) * embed_dim + num_dense
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, sparse_x, dense_x):
        sparse_x = sparse_x.long()
        embs = [self.embeddings[i](sparse_x[:, i]) for i in range(sparse_x.size(1))]
        x = torch.cat(embs + [dense_x], dim=1)
        return self.mlp(x).squeeze(1)

# --------------- Training ---------------

def _train_base_lr_at_step(
    step, total_steps, base_lr, warmup_steps, lr_schedule,
    steps_per_epoch, lr_step_epochs, lr_step_gamma, cosine_min_lr_ratio,
):
    if warmup_steps > 0 and step < warmup_steps:
        return base_lr * float(step + 1) / float(warmup_steps)
    t = step - warmup_steps
    Tpost = max(1, total_steps - warmup_steps)
    progress = min(1.0, float(t) / float(Tpost))
    if lr_schedule == 'constant':
        return base_lr
    if lr_schedule == 'cosine':
        eta_min = base_lr * cosine_min_lr_ratio
        return eta_min + (base_lr - eta_min) * 0.5 * (1.0 + math.cos(math.pi * progress))
    if lr_schedule == 'step':
        if lr_step_epochs is None or lr_step_epochs <= 0:
            return base_lr
        period = max(1, int(lr_step_epochs) * steps_per_epoch)
        n_decays = t // period
        return base_lr * (float(lr_step_gamma) ** float(n_decays))
    raise ValueError(f'Unknown lr_schedule={lr_schedule!r} (use constant, cosine, step)')


def train_model(model, sparse_train, dense_train, y_train,
                epochs=10, batch_size=1024, lr=1e-3,
                weight_decay=0.0,
                lr_schedule='constant',
                warmup_steps=0,
                lr_step_epochs=3,
                lr_step_gamma=0.1,
                cosine_min_lr_ratio=0.01):
    """Train with BCEWithLogitsLoss + Adam. Returns per-epoch average losses.

    lr_schedule: 'constant' (default, legacy), 'cosine', or 'step' (decay every lr_step_epochs epochs).
    warmup_steps: linear warmup batches; 0 disables.
    """
    model.to(device)
    n_samples = len(sparse_train)
    steps_per_epoch = max(1, (n_samples + batch_size - 1) // batch_size)
    total_steps = epochs * steps_per_epoch

    optimizer = torch.optim.Adam(
        model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=weight_decay,
    )
    criterion = nn.BCEWithLogitsLoss()
    loader = DataLoader(
        TensorDataset(sparse_train, dense_train, y_train),
        batch_size=batch_size, shuffle=True,
    )
    losses = []
    global_step = 0
    for epoch in range(epochs):
        model.train()
        total_loss, n_batches = 0.0, 0
        t_epoch = time.perf_counter()
        for sp, dn, y in loader:
            lr_now = _train_base_lr_at_step(
                global_step, total_steps, lr, warmup_steps, lr_schedule,
                steps_per_epoch, lr_step_epochs, lr_step_gamma, cosine_min_lr_ratio,
            )
            for pg in optimizer.param_groups:
                pg['lr'] = lr_now

            sp, dn, y = sp.to(device), dn.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(sp, dn)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
            global_step += 1
        avg = total_loss / n_batches
        losses.append(avg)
        dt = time.perf_counter() - t_epoch
        sps = n_samples / dt if dt > 0 else 0.0
        print(f'  Epoch {epoch+1}/{epochs}: loss={avg:.4f}  ({sps:,.0f} samples/s)')
    return losses

# --------------- Evaluation ---------------

def evaluate_auc(model, sparse_test, dense_test, y_test, batch_size=2048):
    """Compute ROC-AUC. Returns (auc, predictions_array)."""
    model.eval()
    loader = DataLoader(
        TensorDataset(sparse_test, dense_test, y_test),
        batch_size=batch_size, shuffle=False,
    )
    all_preds, all_labels = [], []
    with torch.no_grad():
        for sp, dn, y in loader:
            sp, dn, y = sp.to(device), dn.to(device), y.to(device)
            preds = torch.sigmoid(model(sp, dn))
            all_preds.append(preds.cpu().numpy())
            all_labels.append(y.cpu().numpy())
    preds_arr = np.concatenate(all_preds)
    labels_arr = np.concatenate(all_labels)
    return roc_auc_score(labels_arr, preds_arr), preds_arr

# --------------- ESMMModel ---------------

class ESMMModel(nn.Module):
    """ESMM two-tower (CTR+CVR) with shared embeddings. pCTCVR = pCTR * pCVR."""

    def __init__(self, field_cardinalities, num_dense, embed_dim=18,
                 hidden_dims=(360, 200, 80)):
        super().__init__()
        self.field_cardinalities = list(field_cardinalities)
        self.embed_dim = embed_dim
        self.num_fields = len(field_cardinalities)
        offsets = []
        off = 0
        for card in field_cardinalities:
            offsets.append(off)
            off += int(card) + 1
        self.register_buffer(
            'field_offsets',
            torch.tensor(offsets, dtype=torch.long).view(1, -1),
        )
        total_vocab = off
        self.unified_emb = nn.Embedding(total_vocab, embed_dim)
        input_dim = self.num_fields * embed_dim + num_dense

        ctr_layers = []
        prev = input_dim
        for h in hidden_dims:
            ctr_layers.append(nn.Linear(prev, h))
            ctr_layers.append(nn.ReLU())
            prev = h
        ctr_layers.append(nn.Linear(prev, 1))
        self.ctr_tower = nn.Sequential(*ctr_layers)

        cvr_layers = []
        prev = input_dim
        for h in hidden_dims:
            cvr_layers.append(nn.Linear(prev, h))
            cvr_layers.append(nn.ReLU())
            prev = h
        cvr_layers.append(nn.Linear(prev, 1))
        self.cvr_tower = nn.Sequential(*cvr_layers)

    def forward(self, sparse_x, dense_x):
        idx = sparse_x.long() + self.field_offsets
        e = self.unified_emb(idx)
        x = torch.cat([e.flatten(1), dense_x], dim=1)
        p_ctr = torch.sigmoid(self.ctr_tower(x).squeeze(1)).clamp(1e-7, 1 - 1e-7)
        p_cvr = torch.sigmoid(self.cvr_tower(x).squeeze(1)).clamp(1e-7, 1 - 1e-7)
        p_ctcvr = (p_ctr * p_cvr).clamp(1e-7, 1 - 1e-7)
        return p_ctr, p_cvr, p_ctcvr


# --------------- ESMM variants (shared bottom / MMoE / PLE) ---------------

def _init_linear(layer: nn.Linear) -> None:
    nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)


class _ESMMExpertMLP(nn.Module):
    """Single hidden-layer expert: d_in -> hidden -> d_model (+ LayerNorm), PLE-style."""

    def __init__(self, d_in: int, hidden: int, d_model: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.fc1 = nn.Linear(d_in, hidden)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden, d_model)
        self.ln = nn.LayerNorm(d_model)
        _init_linear(self.fc1)
        _init_linear(self.fc2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        return self.ln(x)


class _ESMMGate(nn.Module):
    def __init__(self, d_selector: int, num_experts: int) -> None:
        super().__init__()
        self.linear = nn.Linear(d_selector, num_experts)
        _init_linear(self.linear)

    def forward(self, selector: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.linear(selector), dim=-1)


class _ESMM_PLELevel(nn.Module):
    """One PLE level: shared + task1 + task2 experts; three gates over all experts (ple_experiment/model.py)."""

    def __init__(
        self,
        d_in: int,
        d_model: int,
        expert_hidden: int,
        num_shared_experts: int,
        num_task_experts: int,
        dropout: float = 0.0,
        d_selector_t1=None,
        d_selector_t2=None,
        d_selector_shared=None,
    ) -> None:
        super().__init__()
        E_s = max(0, int(num_shared_experts))
        E_t = max(0, int(num_task_experts))
        self.shared_experts = nn.ModuleList(
            [_ESMMExpertMLP(d_in, expert_hidden, d_model, dropout) for _ in range(E_s)]
        )
        self.t1_experts = nn.ModuleList(
            [_ESMMExpertMLP(d_in, expert_hidden, d_model, dropout) for _ in range(E_t)]
        )
        self.t2_experts = nn.ModuleList(
            [_ESMMExpertMLP(d_in, expert_hidden, d_model, dropout) for _ in range(E_t)]
        )
        total_experts = E_s + E_t + E_t
        d_sel_t1 = d_selector_t1 if d_selector_t1 is not None else d_in
        d_sel_t2 = d_selector_t2 if d_selector_t2 is not None else d_in
        d_sel_sh = d_selector_shared if d_selector_shared is not None else d_in
        self.gate_t1 = _ESMMGate(d_sel_t1, total_experts)
        self.gate_t2 = _ESMMGate(d_sel_t2, total_experts)
        self.gate_shared = _ESMMGate(d_sel_sh, total_experts)

    def forward(self, x_expert, sel_t1, sel_t2, sel_shared):
        outs = []
        outs += [e(x_expert) for e in self.shared_experts]
        outs += [e(x_expert) for e in self.t1_experts]
        outs += [e(x_expert) for e in self.t2_experts]
        if len(outs) == 0:
            raise RuntimeError('PLELevel must have at least one expert')
        stacked = torch.stack(outs, dim=1)
        g_t1 = (self.gate_t1(sel_t1).unsqueeze(-1) * stacked).sum(dim=1)
        g_t2 = (self.gate_t2(sel_t2).unsqueeze(-1) * stacked).sum(dim=1)
        g_sh = (self.gate_shared(sel_shared).unsqueeze(-1) * stacked).sum(dim=1)
        return g_t1, g_t2, g_sh


class _ESMM_PLETower(nn.Module):
    def __init__(self, d_model: int, out_dim: int = 1) -> None:
        super().__init__()
        hidden = max(1, d_model // 2)
        self.fc1 = nn.Linear(d_model, hidden)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(hidden, out_dim)
        _init_linear(self.fc1)
        _init_linear(self.fc2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.fc1(x))
        return self.fc2(x)


class ESMM_SharedBottom(nn.Module):
    """ESMM with one shared trunk then separate CTR/CVR heads (same unified embeddings as ESMMModel)."""

    def __init__(
        self,
        field_cardinalities,
        num_dense,
        embed_dim=18,
        trunk_dims=(360, 200, 80),
    ):
        super().__init__()
        self.field_cardinalities = list(field_cardinalities)
        self.embed_dim = embed_dim
        self.num_fields = len(field_cardinalities)
        offsets, off = [], 0
        for card in field_cardinalities:
            offsets.append(off)
            off += int(card) + 1
        self.register_buffer('field_offsets', torch.tensor(offsets, dtype=torch.long).view(1, -1))
        self.unified_emb = nn.Embedding(off, embed_dim)
        nn.init.normal_(self.unified_emb.weight, 0.0, 0.01)
        input_dim = self.num_fields * embed_dim + num_dense
        trunk_layers = []
        prev = input_dim
        for h in trunk_dims:
            trunk_layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        self.shared_trunk = nn.Sequential(*trunk_layers)
        for m in self.shared_trunk.modules():
            if isinstance(m, nn.Linear):
                _init_linear(m)
        self.ctr_head = nn.Linear(prev, 1)
        self.cvr_head = nn.Linear(prev, 1)
        _init_linear(self.ctr_head)
        _init_linear(self.cvr_head)

    def forward(self, sparse_x, dense_x):
        idx = sparse_x.long() + self.field_offsets
        e = self.unified_emb(idx)
        x = torch.cat([e.flatten(1), dense_x], dim=1)
        h = self.shared_trunk(x)
        p_ctr = torch.sigmoid(self.ctr_head(h).squeeze(1)).clamp(1e-7, 1 - 1e-7)
        p_cvr = torch.sigmoid(self.cvr_head(h).squeeze(1)).clamp(1e-7, 1 - 1e-7)
        p_ctcvr = (p_ctr * p_cvr).clamp(1e-7, 1 - 1e-7)
        return p_ctr, p_cvr, p_ctcvr


class ESMM_MMoE(nn.Module):
    """Single-level MMoE: separate gates for CTR vs CVR over shared experts (same embedding front as ESMMModel)."""

    def __init__(
        self,
        field_cardinalities,
        num_dense,
        embed_dim=18,
        num_experts=4,
        expert_hidden=360,
        d_model=128,
        tower_hidden_ratio=0.5,
        dropout=0.0,
    ):
        super().__init__()
        self.field_cardinalities = list(field_cardinalities)
        self.embed_dim = embed_dim
        self.num_fields = len(field_cardinalities)
        offsets, off = [], 0
        for card in field_cardinalities:
            offsets.append(off)
            off += int(card) + 1
        self.register_buffer('field_offsets', torch.tensor(offsets, dtype=torch.long).view(1, -1))
        self.unified_emb = nn.Embedding(off, embed_dim)
        nn.init.normal_(self.unified_emb.weight, 0.0, 0.01)
        d_in = self.num_fields * embed_dim + num_dense
        E = int(num_experts)
        self.experts = nn.ModuleList(
            [_ESMMExpertMLP(d_in, expert_hidden, d_model, dropout) for _ in range(E)]
        )
        self.gate_ctr = _ESMMGate(d_in, E)
        self.gate_cvr = _ESMMGate(d_in, E)
        th = max(1, int(d_model * tower_hidden_ratio))
        self.ctr_tower = nn.Sequential(
            nn.Linear(d_model, th), nn.ReLU(), nn.Linear(th, 1),
        )
        self.cvr_tower = nn.Sequential(
            nn.Linear(d_model, th), nn.ReLU(), nn.Linear(th, 1),
        )
        for m in list(self.ctr_tower.modules()) + list(self.cvr_tower.modules()):
            if isinstance(m, nn.Linear):
                _init_linear(m)

    def forward(self, sparse_x, dense_x):
        idx = sparse_x.long() + self.field_offsets
        e = self.unified_emb(idx)
        x = torch.cat([e.flatten(1), dense_x], dim=1)
        expert_outs = torch.stack([ex(x) for ex in self.experts], dim=1)
        w_ctr = self.gate_ctr(x).unsqueeze(-1)
        w_cvr = self.gate_cvr(x).unsqueeze(-1)
        h_ctr = (w_ctr * expert_outs).sum(dim=1)
        h_cvr = (w_cvr * expert_outs).sum(dim=1)
        p_ctr = torch.sigmoid(self.ctr_tower(h_ctr).squeeze(1)).clamp(1e-7, 1 - 1e-7)
        p_cvr = torch.sigmoid(self.cvr_tower(h_cvr).squeeze(1)).clamp(1e-7, 1 - 1e-7)
        p_ctcvr = (p_ctr * p_cvr).clamp(1e-7, 1 - 1e-7)
        return p_ctr, p_cvr, p_ctcvr


class ESMM_PLE(nn.Module):
    """Two-level PLE for CTR (task1) and CVR (task2); selectors follow ple_experiment/model.py PLEModel."""

    def __init__(
        self,
        field_cardinalities,
        num_dense,
        embed_dim=18,
        d_model=128,
        expert_hidden=256,
        num_shared_experts=1,
        num_task_experts=1,
        dropout=0.0,
    ):
        super().__init__()
        self.field_cardinalities = list(field_cardinalities)
        self.embed_dim = embed_dim
        self.num_fields = len(field_cardinalities)
        offsets, off = [], 0
        for card in field_cardinalities:
            offsets.append(off)
            off += int(card) + 1
        self.register_buffer('field_offsets', torch.tensor(offsets, dtype=torch.long).view(1, -1))
        self.unified_emb = nn.Embedding(off, embed_dim)
        nn.init.normal_(self.unified_emb.weight, 0.0, 0.01)
        d_in = self.num_fields * embed_dim + num_dense
        ns, nt = int(num_shared_experts), int(num_task_experts)
        self.level1 = _ESMM_PLELevel(
            d_in, d_model, expert_hidden, ns, nt, dropout,
            d_selector_t1=d_in, d_selector_t2=d_in, d_selector_shared=d_in,
        )
        self.level2 = _ESMM_PLELevel(
            d_in, d_model, expert_hidden, ns, nt, dropout,
            d_selector_t1=d_model, d_selector_t2=d_model, d_selector_shared=d_model,
        )
        self.tower_ctr = _ESMM_PLETower(d_model, 1)
        self.tower_cvr = _ESMM_PLETower(d_model, 1)

    def forward(self, sparse_x, dense_x):
        # Deep PLE + LayerNorm under fp16 autocast can overflow; run forward in fp32 on CUDA.
        _ac = torch.amp.autocast('cuda', enabled=False) if sparse_x.is_cuda else nullcontext()
        with _ac:
            idx = sparse_x.long() + self.field_offsets
            e = self.unified_emb(idx)
            x = torch.cat([e.flatten(1), dense_x], dim=1)
            g1_t1, g1_t2, g1_sh = self.level1(x, x, x, x)
            g1_t1 = torch.nan_to_num(g1_t1, nan=0.0, posinf=1e4, neginf=-1e4).clamp(-50.0, 50.0)
            g1_t2 = torch.nan_to_num(g1_t2, nan=0.0, posinf=1e4, neginf=-1e4).clamp(-50.0, 50.0)
            g1_sh = torch.nan_to_num(g1_sh, nan=0.0, posinf=1e4, neginf=-1e4).clamp(-50.0, 50.0)
            g2_t1, g2_t2, g2_sh = self.level2(x, g1_t1, g1_t2, g1_sh)
            g2_t1 = torch.nan_to_num(g2_t1, nan=0.0, posinf=1e4, neginf=-1e4).clamp(-50.0, 50.0)
            g2_t2 = torch.nan_to_num(g2_t2, nan=0.0, posinf=1e4, neginf=-1e4).clamp(-50.0, 50.0)
            p_ctr = torch.sigmoid(self.tower_ctr(g2_t1).squeeze(1)).clamp(1e-7, 1 - 1e-7)
            p_cvr = torch.sigmoid(self.tower_cvr(g2_t2).squeeze(1)).clamp(1e-7, 1 - 1e-7)
            p_ctcvr = (p_ctr * p_cvr).clamp(1e-7, 1 - 1e-7)
            return p_ctr, p_cvr, p_ctcvr


# --------------- ESMM Training ---------------

def train_esmm(model, sparse_train, dense_train, y_click, y_purchase,
               epochs=10, batch_size=1024, lr=1e-3):
    """Entire-space multi-task loss: BCE(click, pCTR) + BCE(click*purchase, pCTCVR)."""
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-8)
    y_ctcvr = y_click * y_purchase
    loader = DataLoader(
        TensorDataset(sparse_train, dense_train, y_click, y_ctcvr),
        batch_size=batch_size, shuffle=True,
    )
    losses = []
    n_samples_esmm = len(sparse_train)
    for epoch in range(epochs):
        model.train()
        total_loss, n_batches = 0.0, 0
        t_epoch = time.perf_counter()
        for sp, dn, yc, ycc in loader:
            sp, dn, yc, ycc = sp.to(device), dn.to(device), yc.to(device), ycc.to(device)
            optimizer.zero_grad()
            p_ctr, _, p_ctcvr = model(sp, dn)
            loss = (nn.functional.binary_cross_entropy(p_ctr, yc) +
                    nn.functional.binary_cross_entropy(p_ctcvr, ycc))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
        avg = total_loss / n_batches
        losses.append(avg)
        dt = time.perf_counter() - t_epoch
        sps = n_samples_esmm / dt if dt > 0 else 0.0
        print(f'  Epoch {epoch+1}/{epochs}: loss={avg:.4f}  ({sps:,.0f} samples/s)')
    return losses

# --------------- ESMM Evaluation ---------------

def evaluate_esmm_cvr(model, sparse_test, dense_test, y_purchase, batch_size=2048):
    """CVR AUC: pCVR vs purchase labels on clicked-only data."""
    model.eval()
    loader = DataLoader(
        TensorDataset(sparse_test, dense_test, y_purchase),
        batch_size=batch_size, shuffle=False,
    )
    all_preds, all_labels = [], []
    with torch.no_grad():
        for sp, dn, y in loader:
            sp, dn = sp.to(device), dn.to(device)
            _, p_cvr, _ = model(sp, dn)
            all_preds.append(p_cvr.cpu().numpy())
            all_labels.append(y.numpy())
    preds_arr = np.concatenate(all_preds)
    labels_arr = np.concatenate(all_labels)
    return roc_auc_score(labels_arr, preds_arr), preds_arr


def evaluate_esmm_ctcvr(model, sparse_test, dense_test, y_ctcvr, batch_size=2048):
    """CTCVR AUC: pCTCVR vs (click & purchase) labels on all data."""
    model.eval()
    loader = DataLoader(
        TensorDataset(sparse_test, dense_test, y_ctcvr),
        batch_size=batch_size, shuffle=False,
    )
    all_preds, all_labels = [], []
    with torch.no_grad():
        for sp, dn, y in loader:
            sp, dn = sp.to(device), dn.to(device)
            _, _, p_ctcvr = model(sp, dn)
            all_preds.append(p_ctcvr.cpu().numpy())
            all_labels.append(y.numpy())
    preds_arr = np.concatenate(all_preds)
    labels_arr = np.concatenate(all_labels)
    return roc_auc_score(labels_arr, preds_arr), preds_arr

# --------------- Focal Loss ---------------

class FocalLoss(nn.Module):
    """Focal loss for class-imbalanced binary classification (Lin et al., 2017)."""

    def __init__(self, gamma=2.0, alpha=0.25):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits, targets):
        bce = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        p = torch.sigmoid(logits)
        pt = targets * p + (1 - targets) * (1 - p)
        alpha_t = targets * self.alpha + (1 - targets) * (1 - self.alpha)
        focal_weight = alpha_t * (1 - pt) ** self.gamma
        return (focal_weight * bce).mean()

# --------------- Training with Focal Loss ---------------

def train_model_focal(model, sparse_train, dense_train, y_train,
                      epochs=10, batch_size=1024, lr=1e-3, gamma=2.0, alpha=0.25):
    """Train with FocalLoss + Adam. Returns per-epoch average losses."""
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-8)
    criterion = FocalLoss(gamma=gamma, alpha=alpha)
    loader = DataLoader(
        TensorDataset(sparse_train, dense_train, y_train),
        batch_size=batch_size, shuffle=True,
    )
    losses = []
    n_samples_focal = len(sparse_train)
    for epoch in range(epochs):
        model.train()
        total_loss, n_batches = 0.0, 0
        t_epoch = time.perf_counter()
        for sp, dn, y in loader:
            sp, dn, y = sp.to(device), dn.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(sp, dn)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
        avg = total_loss / n_batches
        losses.append(avg)
        dt = time.perf_counter() - t_epoch
        sps = n_samples_focal / dt if dt > 0 else 0.0
        print(f'  Epoch {epoch+1}/{epochs}: loss={avg:.4f}  ({sps:,.0f} samples/s)')
    return losses


# --------------- ESMM eval streaming (full test — avoid giant pandas) ---------------

def evaluate_esmm_cvr_streaming_parquet(model, parquet_path, vocabs, sparse_cols, dense_feat_cols, batch_rows=None, eval_batch_rows=DEFAULT_EVAL_TEST_BATCH_ROWS):
    import pyarrow.parquet as pq
    if batch_rows is None:
        batch_rows = eval_batch_rows
    cols = sparse_cols + dense_feat_cols + ['click', 'purchase']
    model.eval()
    enc_tables = _precompute_sparse_encode_tables(vocabs, sparse_cols)
    pf = pq.ParquetFile(parquet_path)
    all_p, all_y = [], []
    for batch in pf.iter_batches(batch_size=batch_rows, columns=cols):
        df = batch.to_pandas()
        del batch
        m = (df['click'].values == 1)
        if not m.any():
            del df
            continue
        df_c = df.loc[m].reset_index(drop=True)
        del df
        sp, dn, y = encode_and_tensorize_fast(df_c, enc_tables, sparse_cols, dense_feat_cols, 'purchase')
        del df_c
        loader = DataLoader(TensorDataset(sp, dn, y), batch_size=4096, shuffle=False)
        with torch.no_grad():
            for spb, dnb, yb in loader:
                spb, dnb = spb.to(device), dnb.to(device)
                _, pc, _ = model(spb, dnb)
                all_p.append(pc.cpu().numpy())
                all_y.append(yb.numpy())
        del sp, dn, y
    preds = np.concatenate(all_p)
    labels = np.concatenate(all_y)
    return roc_auc_score(labels, preds), preds


def evaluate_esmm_ctcvr_streaming_parquet(model, parquet_path, vocabs, sparse_cols, dense_feat_cols, batch_rows=None, eval_batch_rows=DEFAULT_EVAL_TEST_BATCH_ROWS):
    import pyarrow.parquet as pq
    if batch_rows is None:
        batch_rows = eval_batch_rows
    cols = sparse_cols + dense_feat_cols + ['click', 'purchase']
    model.eval()
    enc_tables = _precompute_sparse_encode_tables(vocabs, sparse_cols)
    pf = pq.ParquetFile(parquet_path)
    all_p, all_y = [], []
    for batch in pf.iter_batches(batch_size=batch_rows, columns=cols):
        df = batch.to_pandas()
        del batch
        sp, dn, _ = encode_and_tensorize_fast(df, enc_tables, sparse_cols, dense_feat_cols, 'purchase')
        y_ct = torch.from_numpy(
            (df['click'].values * df['purchase'].values).astype(np.float32))
        del df
        loader = DataLoader(TensorDataset(sp, dn, y_ct), batch_size=4096, shuffle=False)
        with torch.no_grad():
            for spb, dnb, yb in loader:
                spb, dnb = spb.to(device), dnb.to(device)
                _, _, pct = model(spb, dnb)
                all_p.append(pct.cpu().numpy())
                all_y.append(yb.numpy())
        del sp, dn, y_ct

    preds = np.concatenate(all_p)
    labels = np.concatenate(all_y)
    return roc_auc_score(labels, preds), preds


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


# --------------- ESMM streaming training (Round 4 RAM) ---------------

# Consultants: avoid full 42M-row tensors; train from Parquet row groups + int32 host tensors.

R5_COMPILE_MODE = "default"  # string mode for torch.compile(..., mode=R5_COMPILE_MODE)


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
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-8)
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


def evaluate_esmm_cvr_indexed(model, sparse_all, dense_all, y_purchase_all, click_mask, batch_size=4096):
    """CVR AUC on clicked rows using a boolean mask over precomputed test tensors."""
    model.eval()
    sp_s = sparse_all[click_mask]
    dn_s = dense_all[click_mask]
    y_s = y_purchase_all[click_mask]
    loader = DataLoader(
        TensorDataset(sp_s, dn_s, y_s),
        batch_size=batch_size, shuffle=False,
    )
    all_preds, all_labels = [], []
    with torch.no_grad():
        for sp, dn, y in loader:
            sp, dn = sp.to(device), dn.to(device)
            _, p_cvr, _ = model(sp, dn)
            all_preds.append(p_cvr.cpu().numpy())
            all_labels.append(y.numpy())
    preds_arr = np.concatenate(all_preds)
    labels_arr = np.concatenate(all_labels)
    return roc_auc_score(labels_arr, preds_arr), preds_arr


# --------------- Frequency-filtered Vocabs (Round 4+) ---------------

def build_sparse_vocabs_filtered(df, sparse_cols, min_count=5):
    """Build label-encoding vocabularies with frequency filtering.
    IDs appearing fewer than min_count times map to index 0 (UNK)."""
    vocabs = {}
    cardinalities = []
    for col in sparse_cols:
        vals = df[col].astype(str)
        counts = vals.value_counts()
        total_unique = len(counts)
        kept = counts[counts >= min_count]
        vocab = {v: i + 1 for i, v in enumerate(kept.index)}
        vocabs[col] = vocab
        cardinalities.append(len(vocab))
        filtered = total_unique - len(kept)
        print(f'  {col}: {total_unique} unique, {len(kept)} kept (>={min_count}), {filtered} filtered')
    return vocabs, cardinalities

# --------------- Dense Feature Normalization (Round 4+) ---------------

def normalize_dense_features(df, dense_feat_cols):
    """Apply log1p normalization to dense features. Returns a copy with only
    dense_feat_cols transformed; all other columns are preserved as-is."""
    df_out = df.copy()
    for i, col in enumerate(dense_feat_cols):
        x = pd.to_numeric(df_out[col], errors='coerce').fillna(0.0)
        if i < 3:
            print(f'  {col} BEFORE: min={x.min():.4f}, max={x.max():.4f}, mean={x.mean():.4f}')
        x_norm = np.log1p(np.abs(x)) * np.sign(x)
        if i < 3:
            print(f'  {col} AFTER:  min={x_norm.min():.4f}, max={x_norm.max():.4f}, mean={x_norm.mean():.4f}')
        df_out[col] = x_norm
    return df_out