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
        # Match SharedBottom/MMoE/PLE: small-variance init. Default nn.Embedding init is N(0,1),
        # 100x larger; under AMP/fp16 the tower pre-activations overflow -> sigmoid(±inf) constant
        # output -> degenerate ~0.5000 AUC (D7 investigation, 2026-06-01).
        nn.init.normal_(self.unified_emb.weight, 0.0, 0.01)
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


class ESMMModel_Wide(ESMMModel):
    """R6: ESMMModel (two-tower ESMM) + Wide (linear) interaction term on the same embeddings,
    added to both CTR and CVR logits. Completes the 'interaction term for all MTL heads' sweep."""

    def __init__(self, field_cardinalities, num_dense, embed_dim=18, **kwargs):
        super().__init__(field_cardinalities, num_dense, embed_dim=embed_dim, **kwargs)
        input_dim = self.num_fields * self.embed_dim + num_dense
        self.wide_ctr = nn.Linear(input_dim, 1)
        self.wide_cvr = nn.Linear(input_dim, 1)
        _init_linear(self.wide_ctr)
        _init_linear(self.wide_cvr)

    def forward(self, sparse_x, dense_x):
        idx = sparse_x.long() + self.field_offsets
        e = self.unified_emb(idx)
        x = torch.cat([e.flatten(1), dense_x], dim=1)
        logit_ctr = self.ctr_tower(x).squeeze(1) + self.wide_ctr(x).squeeze(1)
        logit_cvr = self.cvr_tower(x).squeeze(1) + self.wide_cvr(x).squeeze(1)
        p_ctr = torch.sigmoid(logit_ctr).clamp(1e-7, 1 - 1e-7)
        p_cvr = torch.sigmoid(logit_cvr).clamp(1e-7, 1 - 1e-7)
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
    """One PLE extraction level, faithful to Tang et al., RecSys 2020 (§4.1 CGC / §4.2 PLE, Eqs 2-6).

    Holds shared + task1 + task2 expert pools with FAITHFUL CGC gate scoping:
      - task-k gate selects ONLY over [task-k experts + shared experts]  (Eq 3-4: W_g^k in R^{(m_k+m_s)xd});
        the OTHER task's experts are deliberately excluded (the defining CGC property).
      - the shared gate (used only between levels) selects over ALL experts (§4.2).
    Progressive routing (Eq 6): each expert group consumes its OWN previous-level fused output as
    input, which is also that gate's selector (so in_t1 == sel_t1, in_t2 == sel_t2, in_shared == sel_sh).
    The last level sets has_shared_gate=False (towers only need the two task fusions).
    """

    def __init__(
        self,
        d_in: int,
        d_model: int,
        expert_hidden: int,
        num_shared_experts: int,
        num_task_experts: int,
        dropout: float = 0.0,
        has_shared_gate: bool = True,
    ) -> None:
        super().__init__()
        E_s = max(0, int(num_shared_experts))
        E_t = max(0, int(num_task_experts))
        if E_s < 1 or E_t < 1:
            raise ValueError('PLELevel needs >=1 shared and >=1 task expert per task')
        self.shared_experts = nn.ModuleList(
            [_ESMMExpertMLP(d_in, expert_hidden, d_model, dropout) for _ in range(E_s)]
        )
        self.t1_experts = nn.ModuleList(
            [_ESMMExpertMLP(d_in, expert_hidden, d_model, dropout) for _ in range(E_t)]
        )
        self.t2_experts = nn.ModuleList(
            [_ESMMExpertMLP(d_in, expert_hidden, d_model, dropout) for _ in range(E_t)]
        )
        # Faithful CGC gate scopes (selector dim == this level's input dim d_in, Eq 6).
        self.gate_t1 = _ESMMGate(d_in, E_t + E_s)            # [task1 experts] + [shared]
        self.gate_t2 = _ESMMGate(d_in, E_t + E_s)            # [task2 experts] + [shared]
        self.has_shared_gate = bool(has_shared_gate)
        if self.has_shared_gate:
            self.gate_shared = _ESMMGate(d_in, E_s + E_t + E_t)  # all experts

    def forward(self, in_t1, in_t2, in_shared):
        # Each expert group runs on its own previous-level fused output (progressive routing).
        e_t1 = [e(in_t1) for e in self.t1_experts]
        e_t2 = [e(in_t2) for e in self.t2_experts]
        e_sh = [e(in_shared) for e in self.shared_experts]
        # task1 gate fuses [task1 + shared] experts; stack order must match gate output order.
        sel1 = torch.stack(e_t1 + e_sh, dim=1)
        g_t1 = (self.gate_t1(in_t1).unsqueeze(-1) * sel1).sum(dim=1)
        # task2 gate fuses [task2 + shared] experts.
        sel2 = torch.stack(e_t2 + e_sh, dim=1)
        g_t2 = (self.gate_t2(in_t2).unsqueeze(-1) * sel2).sum(dim=1)
        # shared gate (between levels only) fuses ALL experts.
        g_sh = None
        if self.has_shared_gate:
            sel_sh = torch.stack(e_sh + e_t1 + e_t2, dim=1)
            g_sh = (self.gate_shared(in_shared).unsqueeze(-1) * sel_sh).sum(dim=1)
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
        input_dim = self.num_fields * embed_dim + num_dense
        trunk_layers = []
        prev = input_dim
        for h in trunk_dims:
            trunk_layers += [nn.Linear(prev, h), nn.ReLU()]
            if dropout > 0:
                trunk_layers.append(nn.Dropout(dropout))  # R3: regularize the plain-MLP trunk
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


class ESMM_SharedBottomWide(nn.Module):
    """R4: ESMM_SharedBottom + a Wide (linear) term on the same concatenated embedding+dense input,
    added to BOTH the CTR and CVR logits (Wide&Deep style). The ONLY change vs ESMM_SharedBottom —
    which plateaus at CTCVR ~0.578 — is this explicit low-order interaction/memorization path. It is
    the same structure as the classic Wide&Deep that reached CTCVR 0.6508 on this exact pipeline, so
    it isolates 'explicit interaction term' as the lever that lifts plain-MLP ESMM toward the paper."""

    def __init__(self, field_cardinalities, num_dense, embed_dim=18, trunk_dims=(360, 200, 80), dropout=0.0, **kwargs):
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
        # R4c: match Wide&Deep's _make_mlp — ReLU on hidden layers only, final projection has NO ReLU
        # (a ReLU'd bottleneck feeding linear heads loses signal; this closed the last gap to ~0.65).
        trunk_layers = []
        prev = input_dim
        *hidden_dims, last_dim = trunk_dims
        for h in hidden_dims:
            trunk_layers += [nn.Linear(prev, h), nn.ReLU()]
            if dropout > 0:
                trunk_layers.append(nn.Dropout(dropout))
            prev = h
        trunk_layers.append(nn.Linear(prev, last_dim))  # final projection, no ReLU
        self.shared_trunk = nn.Sequential(*trunk_layers)
        prev = last_dim
        for m in self.shared_trunk.modules():
            if isinstance(m, nn.Linear):
                _init_linear(m)
        self.ctr_head = nn.Linear(prev, 1)        # deep head
        self.cvr_head = nn.Linear(prev, 1)
        self.wide_ctr = nn.Linear(input_dim, 1)   # wide (linear) head on raw embeddings+dense
        self.wide_cvr = nn.Linear(input_dim, 1)
        for m in (self.ctr_head, self.cvr_head, self.wide_ctr, self.wide_cvr):
            _init_linear(m)

    def forward(self, sparse_x, dense_x):
        idx = sparse_x.long() + self.field_offsets
        e = self.unified_emb(idx)
        x = torch.cat([e.flatten(1), dense_x], dim=1)
        h = self.shared_trunk(x)
        logit_ctr = self.ctr_head(h).squeeze(1) + self.wide_ctr(x).squeeze(1)
        logit_cvr = self.cvr_head(h).squeeze(1) + self.wide_cvr(x).squeeze(1)
        p_ctr = torch.sigmoid(logit_ctr).clamp(1e-7, 1 - 1e-7)
        p_cvr = torch.sigmoid(logit_cvr).clamp(1e-7, 1 - 1e-7)
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


class ESMM_MMoE_Wide(ESMM_MMoE):
    """R6: MMoE + Wide (linear) interaction term on the same embeddings, added to both CTR and CVR
    logits. Completes the 'interaction term for all MTL heads' sweep."""

    def __init__(self, field_cardinalities, num_dense, embed_dim=18, **kwargs):
        super().__init__(field_cardinalities, num_dense, embed_dim=embed_dim, **kwargs)
        input_dim = self.num_fields * self.embed_dim + num_dense
        self.wide_ctr = nn.Linear(input_dim, 1)
        self.wide_cvr = nn.Linear(input_dim, 1)
        _init_linear(self.wide_ctr)
        _init_linear(self.wide_cvr)

    def forward(self, sparse_x, dense_x):
        idx = sparse_x.long() + self.field_offsets
        e = self.unified_emb(idx)
        x = torch.cat([e.flatten(1), dense_x], dim=1)
        expert_outs = torch.stack([ex(x) for ex in self.experts], dim=1)
        w_ctr = self.gate_ctr(x).unsqueeze(-1)
        w_cvr = self.gate_cvr(x).unsqueeze(-1)
        h_ctr = (w_ctr * expert_outs).sum(dim=1)
        h_cvr = (w_cvr * expert_outs).sum(dim=1)
        logit_ctr = self.ctr_tower(h_ctr).squeeze(1) + self.wide_ctr(x).squeeze(1)
        logit_cvr = self.cvr_tower(h_cvr).squeeze(1) + self.wide_cvr(x).squeeze(1)
        p_ctr = torch.sigmoid(logit_ctr).clamp(1e-7, 1 - 1e-7)
        p_cvr = torch.sigmoid(logit_cvr).clamp(1e-7, 1 - 1e-7)
        p_ctcvr = (p_ctr * p_cvr).clamp(1e-7, 1 - 1e-7)
        return p_ctr, p_cvr, p_ctcvr


class ESMM_PLE(nn.Module):
    """Two-level PLE for CTR (task1) and CVR (task2), faithful to Tang et al., RecSys 2020 (Eqs 2-7):
    CGC gate scoping (task gate over [own+shared] only) + progressive separation (level-2 experts
    consume level-1 fused outputs, not the raw input)."""

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
        # Level-1 experts consume raw input x (d_in); level-2 experts consume level-1 fused outputs
        # (d_model) — progressive separation (Eq 6). Level 2 is the last level: no shared gate needed.
        self.level1 = _ESMM_PLELevel(
            d_in, d_model, expert_hidden, ns, nt, dropout, has_shared_gate=True,
        )
        self.level2 = _ESMM_PLELevel(
            d_model, d_model, expert_hidden, ns, nt, dropout, has_shared_gate=False,
        )
        self.tower_ctr = _ESMM_PLETower(d_model, 1)
        self.tower_cvr = _ESMM_PLETower(d_model, 1)

    def _ple_trunk(self, sparse_x, dense_x):
        """Shared 2-level PLE trunk → (g2_t1, g2_t2, x). Caller wraps in the fp32 autocast guard.
        Faithful progressive routing (Eq 6): level-1 experts see raw x; level-2 expert groups consume
        the matching level-1 fused outputs (task1←g1_t1, task2←g1_t2, shared←g1_sh).
        Used by ESMM_PLE.forward and the +interaction subclasses (Wide/Cross)."""
        idx = sparse_x.long() + self.field_offsets
        e = self.unified_emb(idx)
        x = torch.cat([e.flatten(1), dense_x], dim=1)
        g1_t1, g1_t2, g1_sh = self.level1(x, x, x)
        g1_t1 = torch.nan_to_num(g1_t1, nan=0.0, posinf=1e4, neginf=-1e4).clamp(-50.0, 50.0)
        g1_t2 = torch.nan_to_num(g1_t2, nan=0.0, posinf=1e4, neginf=-1e4).clamp(-50.0, 50.0)
        g1_sh = torch.nan_to_num(g1_sh, nan=0.0, posinf=1e4, neginf=-1e4).clamp(-50.0, 50.0)
        g2_t1, g2_t2, _ = self.level2(g1_t1, g1_t2, g1_sh)
        g2_t1 = torch.nan_to_num(g2_t1, nan=0.0, posinf=1e4, neginf=-1e4).clamp(-50.0, 50.0)
        g2_t2 = torch.nan_to_num(g2_t2, nan=0.0, posinf=1e4, neginf=-1e4).clamp(-50.0, 50.0)
        return g2_t1, g2_t2, x

    def forward(self, sparse_x, dense_x):
        # Deep PLE + LayerNorm under fp16 autocast can overflow; run forward in fp32 on CUDA.
        _ac = torch.amp.autocast('cuda', enabled=False) if sparse_x.is_cuda else nullcontext()
        with _ac:
            g2_t1, g2_t2, _x = self._ple_trunk(sparse_x, dense_x)
            p_ctr = torch.sigmoid(self.tower_ctr(g2_t1).squeeze(1)).clamp(1e-7, 1 - 1e-7)
            p_cvr = torch.sigmoid(self.tower_cvr(g2_t2).squeeze(1)).clamp(1e-7, 1 - 1e-7)
            p_ctcvr = (p_ctr * p_cvr).clamp(1e-7, 1 - 1e-7)
            return p_ctr, p_cvr, p_ctcvr


class ESMM_PLE_Wide(ESMM_PLE):
    """R5: ESMM_PLE + a Wide (linear) interaction term on the same embeddings, added to both CTR and
    CVR logits — the lever that lifted SharedBottom 0.578 -> 0.64. Tests whether PLE's gated shared/
    task experts PLUS the wide term beat the classic Wide&Deep (CTCVR 0.6508). Subclasses ESMM_PLE
    (reuses its experts/gates/towers); only adds the two wide heads and the logit addition."""

    def __init__(self, field_cardinalities, num_dense, embed_dim=18, **kwargs):
        super().__init__(field_cardinalities, num_dense, embed_dim=embed_dim, **kwargs)
        input_dim = self.num_fields * self.embed_dim + num_dense
        self.wide_ctr = nn.Linear(input_dim, 1)
        self.wide_cvr = nn.Linear(input_dim, 1)
        _init_linear(self.wide_ctr)
        _init_linear(self.wide_cvr)

    def forward(self, sparse_x, dense_x):
        # Same fp32 guard as ESMM_PLE (deep gated stack overflows under fp16 autocast).
        _ac = torch.amp.autocast('cuda', enabled=False) if sparse_x.is_cuda else nullcontext()
        with _ac:
            g2_t1, g2_t2, x = self._ple_trunk(sparse_x, dense_x)
            logit_ctr = self.tower_ctr(g2_t1).squeeze(1) + self.wide_ctr(x).squeeze(1)
            logit_cvr = self.tower_cvr(g2_t2).squeeze(1) + self.wide_cvr(x).squeeze(1)
            p_ctr = torch.sigmoid(logit_ctr).clamp(1e-7, 1 - 1e-7)
            p_cvr = torch.sigmoid(logit_cvr).clamp(1e-7, 1 - 1e-7)
            p_ctcvr = (p_ctr * p_cvr).clamp(1e-7, 1 - 1e-7)
            return p_ctr, p_cvr, p_ctcvr


class _ESMMCrossNet(nn.Module):
    """DCN-V2 cross network: x_{l+1} = x0 * (W_l x_l + b_l) + x_l, for L layers (explicit
    multiplicative feature crosses, full d x d weight per layer)."""

    def __init__(self, d, num_layers=3):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(d, d) for _ in range(num_layers)])
        for m in self.layers:
            _init_linear(m)

    def forward(self, x0):
        x = x0
        for lin in self.layers:
            x = x0 * lin(x) + x
        return x


class ESMM_PLE_Cross(ESMM_PLE):
    """R7: ESMM_PLE + a DCN-V2 cross network (instead of the wide linear term) on the same
    embeddings, projected into both CTR and CVR logits. Tests whether explicit multiplicative
    crosses beat the plain linear wide term (PLE_Wide = CTCVR 0.6728)."""

    def __init__(self, field_cardinalities, num_dense, embed_dim=18, num_cross_layers=3, **kwargs):
        super().__init__(field_cardinalities, num_dense, embed_dim=embed_dim, **kwargs)
        input_dim = self.num_fields * self.embed_dim + num_dense
        self.cross = _ESMMCrossNet(input_dim, num_cross_layers)
        self.cross_ctr = nn.Linear(input_dim, 1)
        self.cross_cvr = nn.Linear(input_dim, 1)
        _init_linear(self.cross_ctr)
        _init_linear(self.cross_cvr)

    def forward(self, sparse_x, dense_x):
        _ac = torch.amp.autocast('cuda', enabled=False) if sparse_x.is_cuda else nullcontext()
        with _ac:
            g2_t1, g2_t2, x = self._ple_trunk(sparse_x, dense_x)
            xc = self.cross(x)
            logit_ctr = self.tower_ctr(g2_t1).squeeze(1) + self.cross_ctr(xc).squeeze(1)
            logit_cvr = self.tower_cvr(g2_t2).squeeze(1) + self.cross_cvr(xc).squeeze(1)
            p_ctr = torch.sigmoid(logit_ctr).clamp(1e-7, 1 - 1e-7)
            p_cvr = torch.sigmoid(logit_cvr).clamp(1e-7, 1 - 1e-7)
            p_ctcvr = (p_ctr * p_cvr).clamp(1e-7, 1 - 1e-7)
            return p_ctr, p_cvr, p_ctcvr


class ESMM_PLE_WideCross(ESMM_PLE_Wide):
    """R7: ESMM_PLE + BOTH the wide linear term (inherited from ESMM_PLE_Wide) AND a DCN-V2 cross
    network, added to the CTR/CVR logits. Tests whether wide (linear) and cross (multiplicative)
    interactions are complementary on top of the best MTL head."""

    def __init__(self, field_cardinalities, num_dense, embed_dim=18, num_cross_layers=3, **kwargs):
        super().__init__(field_cardinalities, num_dense, embed_dim=embed_dim, **kwargs)
        input_dim = self.num_fields * self.embed_dim + num_dense
        self.cross = _ESMMCrossNet(input_dim, num_cross_layers)
        self.cross_ctr = nn.Linear(input_dim, 1)
        self.cross_cvr = nn.Linear(input_dim, 1)
        _init_linear(self.cross_ctr)
        _init_linear(self.cross_cvr)

    def forward(self, sparse_x, dense_x):
        _ac = torch.amp.autocast('cuda', enabled=False) if sparse_x.is_cuda else nullcontext()
        with _ac:
            g2_t1, g2_t2, x = self._ple_trunk(sparse_x, dense_x)
            xc = self.cross(x)
            logit_ctr = self.tower_ctr(g2_t1).squeeze(1) + self.wide_ctr(x).squeeze(1) + self.cross_ctr(xc).squeeze(1)
            logit_cvr = self.tower_cvr(g2_t2).squeeze(1) + self.wide_cvr(x).squeeze(1) + self.cross_cvr(xc).squeeze(1)
            p_ctr = torch.sigmoid(logit_ctr).clamp(1e-7, 1 - 1e-7)
            p_cvr = torch.sigmoid(logit_cvr).clamp(1e-7, 1 - 1e-7)
            p_ctcvr = (p_ctr * p_cvr).clamp(1e-7, 1 - 1e-7)
            return p_ctr, p_cvr, p_ctcvr


# --------------- AdaOrder / multi-stack cross-network classes (Round 20260609) ---------------


class _ESMMCrossNetExposed(nn.Module):
    """DCN-V2 cross network variant that returns ALL per-layer outputs.

    Recurrence: x_{l+1} = x0 * W_l(x_l) + x_l  (identical to _ESMMCrossNet).
    forward(x0) returns [x_1, ..., x_K] — one entry per cross layer.
    Used by ESMM_PLE_AdaOrderCross to gate over depth.
    """

    def __init__(self, d, num_layers=4):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(d, d) for _ in range(num_layers)])
        for m in self.layers:
            _init_linear(m)

    def forward(self, x0):
        """Returns list of per-layer outputs [x_1, ..., x_K]."""
        x = x0
        outputs = []
        for lin in self.layers:
            x = x0 * lin(x) + x
            outputs.append(x)
        return outputs


class ESMM_PLE_AdaOrderCross(ESMM_PLE_Wide):
    """AdaOrder: ESMM_PLE_WideCross with per-task adaptive gating over cross-layer depth.

    One SHARED _ESMMCrossNetExposed stack; each task {ctr, cvr} has a gate α_t over the K depth
    outputs; the task's cross contribution is sum_k α_{t,k} * x_k, folded into the task logit
    exactly as ESMM_PLE_WideCross folds its cross output.

    gate_mode options:
      'task'           (default) α_t = softmax(θ_t), θ_t a learned K-dim parameter per task.
      'shared'         single learned θ used by both tasks (ablation A1).
      'frozen_uniform' fixed α = 1/K, no learning (A2).
      'order_dropout'  training: sample α ~ softmax(N(0,1) noise); eval: uniform 1/K (A8).
      'instance'       α_t = softmax(MLP_t(x0)) with MLP d→gate_hidden→K (A5).

    gate_init ∈ {'uniform','shallow','deep'}:
      'uniform'  θ initialised to zeros (softmax → uniform).
      'shallow'  θ[0] += 2.0  (bias toward first / shallowest cross layer).
      'deep'     θ[-1] += 2.0 (bias toward last / deepest cross layer).

    AMP: subclasses ESMM_PLE_Wide (itself subclassing ESMM_PLE), so train_esmm_parquet_rowgroups
    already disables AMP via isinstance(model, ESMM_PLE).
    """

    def __init__(
        self,
        field_cardinalities,
        num_dense,
        embed_dim=18,
        num_cross_layers=4,
        gate_mode='task',
        gate_init='uniform',
        gate_hidden=16,
        **kwargs,
    ):
        super().__init__(field_cardinalities, num_dense, embed_dim=embed_dim, **kwargs)
        input_dim = self.num_fields * self.embed_dim + num_dense
        self.num_cross_layers = int(num_cross_layers)
        self.gate_mode = gate_mode
        self.gate_init = gate_init
        self.gate_hidden = gate_hidden

        self.cross = _ESMMCrossNetExposed(input_dim, self.num_cross_layers)
        self.cross_ctr = nn.Linear(input_dim, 1)
        self.cross_cvr = nn.Linear(input_dim, 1)
        _init_linear(self.cross_ctr)
        _init_linear(self.cross_cvr)

        K = self.num_cross_layers
        if gate_mode == 'task':
            self.gate_theta_ctr = nn.Parameter(torch.zeros(K))
            self.gate_theta_cvr = nn.Parameter(torch.zeros(K))
            self._apply_gate_init(self.gate_theta_ctr)
            self._apply_gate_init(self.gate_theta_cvr)
        elif gate_mode == 'shared':
            self.gate_theta = nn.Parameter(torch.zeros(K))
            self._apply_gate_init(self.gate_theta)
        elif gate_mode in ('frozen_uniform', 'order_dropout'):
            pass  # no learnable parameters for gate
        elif gate_mode == 'instance':
            self.gate_mlp_ctr = nn.Sequential(
                nn.Linear(input_dim, gate_hidden),
                nn.ReLU(),
                nn.Linear(gate_hidden, K),
            )
            self.gate_mlp_cvr = nn.Sequential(
                nn.Linear(input_dim, gate_hidden),
                nn.ReLU(),
                nn.Linear(gate_hidden, K),
            )
            for mlp in (self.gate_mlp_ctr, self.gate_mlp_cvr):
                for m in mlp.modules():
                    if isinstance(m, nn.Linear):
                        _init_linear(m)
        else:
            raise ValueError(f'Unknown gate_mode: {gate_mode!r}. '
                             f'Valid: task, shared, frozen_uniform, order_dropout, instance.')

    def _apply_gate_init(self, theta):
        """Apply gate_init bias to a learned gate parameter in-place."""
        with torch.no_grad():
            if self.gate_init == 'uniform':
                pass  # zeros → uniform softmax
            elif self.gate_init == 'shallow':
                theta[0] += 2.0
            elif self.gate_init == 'deep':
                theta[-1] += 2.0
            else:
                raise ValueError(f'Unknown gate_init: {self.gate_init!r}. Valid: uniform, shallow, deep.')

    def _get_alpha(self, task, x0):
        """Return alpha weights (K,) for the given task; x0 is (B, d) for instance mode."""
        K = self.num_cross_layers
        mode = self.gate_mode
        if mode == 'task':
            theta = self.gate_theta_ctr if task == 'ctr' else self.gate_theta_cvr
            return torch.softmax(theta, dim=0)  # (K,)
        elif mode == 'shared':
            return torch.softmax(self.gate_theta, dim=0)  # (K,)
        elif mode == 'frozen_uniform':
            return torch.full((K,), 1.0 / K, device=x0.device, dtype=x0.dtype)
        elif mode == 'order_dropout':
            if self.training:
                noise = torch.randn(K, device=x0.device, dtype=x0.dtype)
                return torch.softmax(noise, dim=0)
            else:
                return torch.full((K,), 1.0 / K, device=x0.device, dtype=x0.dtype)
        elif mode == 'instance':
            mlp = self.gate_mlp_ctr if task == 'ctr' else self.gate_mlp_cvr
            return torch.softmax(mlp(x0), dim=-1)  # (B, K)
        raise RuntimeError(f'Unreachable gate_mode: {mode!r}')

    def _weighted_cross(self, layer_outputs, alpha):
        """Combine K layer outputs with weight alpha.

        alpha shape: (K,) for non-instance modes, (B, K) for instance mode.
        Returns (B, d).
        """
        stacked = torch.stack(layer_outputs, dim=1)  # (B, K, d)
        if alpha.dim() == 1:
            # Broadcast (K,) → (1, K, 1)
            return (stacked * alpha.view(1, -1, 1)).sum(dim=1)
        else:
            # instance: alpha is (B, K)
            return (stacked * alpha.unsqueeze(-1)).sum(dim=1)

    def forward(self, sparse_x, dense_x):
        _ac = torch.amp.autocast('cuda', enabled=False) if sparse_x.is_cuda else nullcontext()
        with _ac:
            g2_t1, g2_t2, x = self._ple_trunk(sparse_x, dense_x)
            layer_outputs = self.cross(x)  # list of K tensors (B, d)
            alpha_ctr = self._get_alpha('ctr', x)
            alpha_cvr = self._get_alpha('cvr', x)
            xc_ctr = self._weighted_cross(layer_outputs, alpha_ctr)
            xc_cvr = self._weighted_cross(layer_outputs, alpha_cvr)
            logit_ctr = (self.tower_ctr(g2_t1).squeeze(1)
                         + self.wide_ctr(x).squeeze(1)
                         + self.cross_ctr(xc_ctr).squeeze(1))
            logit_cvr = (self.tower_cvr(g2_t2).squeeze(1)
                         + self.wide_cvr(x).squeeze(1)
                         + self.cross_cvr(xc_cvr).squeeze(1))
            p_ctr = torch.sigmoid(logit_ctr).clamp(1e-7, 1 - 1e-7)
            p_cvr = torch.sigmoid(logit_cvr).clamp(1e-7, 1 - 1e-7)
            p_ctcvr = (p_ctr * p_cvr).clamp(1e-7, 1 - 1e-7)
            return p_ctr, p_cvr, p_ctcvr

    def get_gate_weights(self):
        """Return softmaxed gate weights per task.

        Returns dict{'ctr': np.ndarray(K), 'cvr': np.ndarray(K)}.
        For 'instance' and 'order_dropout'/'frozen_uniform' modes, returns the appropriate
        value: None for instance (input-dependent), uniform vector for frozen/dropout.
        """
        K = self.num_cross_layers
        mode = self.gate_mode
        if mode == 'instance':
            return {'ctr': None, 'cvr': None}
        elif mode in ('frozen_uniform', 'order_dropout'):
            uniform = np.full(K, 1.0 / K, dtype=np.float64)
            return {'ctr': uniform, 'cvr': uniform}
        elif mode == 'task':
            with torch.no_grad():
                ctr = torch.softmax(self.gate_theta_ctr, dim=0).cpu().numpy()
                cvr = torch.softmax(self.gate_theta_cvr, dim=0).cpu().numpy()
            return {'ctr': ctr, 'cvr': cvr}
        elif mode == 'shared':
            with torch.no_grad():
                w = torch.softmax(self.gate_theta, dim=0).cpu().numpy()
            return {'ctr': w, 'cvr': w}
        raise RuntimeError(f'Unreachable gate_mode: {mode!r}')


class ESMM_PLE_TaskCross(ESMM_PLE_Wide):
    """DTN-class control: per-task SEPARATE plain _ESMMCrossNet stacks (no gating).

    Each task has its own K-layer cross network, each folded into its task's logit as in
    ESMM_PLE_WideCross. Wide term inherited from ESMM_PLE_Wide for apples-to-apples comparison.

    AMP disabled via isinstance(model, ESMM_PLE) check in train_esmm_parquet_rowgroups.
    """

    def __init__(self, field_cardinalities, num_dense, embed_dim=18, num_cross_layers=4, **kwargs):
        super().__init__(field_cardinalities, num_dense, embed_dim=embed_dim, **kwargs)
        input_dim = self.num_fields * self.embed_dim + num_dense
        self.cross_ctr_net = _ESMMCrossNet(input_dim, num_cross_layers)
        self.cross_cvr_net = _ESMMCrossNet(input_dim, num_cross_layers)
        self.cross_ctr = nn.Linear(input_dim, 1)
        self.cross_cvr = nn.Linear(input_dim, 1)
        _init_linear(self.cross_ctr)
        _init_linear(self.cross_cvr)

    def forward(self, sparse_x, dense_x):
        _ac = torch.amp.autocast('cuda', enabled=False) if sparse_x.is_cuda else nullcontext()
        with _ac:
            g2_t1, g2_t2, x = self._ple_trunk(sparse_x, dense_x)
            xc_ctr = self.cross_ctr_net(x)
            xc_cvr = self.cross_cvr_net(x)
            logit_ctr = (self.tower_ctr(g2_t1).squeeze(1)
                         + self.wide_ctr(x).squeeze(1)
                         + self.cross_ctr(xc_ctr).squeeze(1))
            logit_cvr = (self.tower_cvr(g2_t2).squeeze(1)
                         + self.wide_cvr(x).squeeze(1)
                         + self.cross_cvr(xc_cvr).squeeze(1))
            p_ctr = torch.sigmoid(logit_ctr).clamp(1e-7, 1 - 1e-7)
            p_cvr = torch.sigmoid(logit_cvr).clamp(1e-7, 1 - 1e-7)
            p_ctcvr = (p_ctr * p_cvr).clamp(1e-7, 1 - 1e-7)
            return p_ctr, p_cvr, p_ctcvr


class ESMM_PLE_EPNetGate(ESMM_PLE_Wide):
    """EPNet-style gating control: per-task small MLP gates the input before a SHARED cross stack.

    Each task has a gating MLP d→gate_hidden→d with 2*sigmoid output that scales x0 per task.
    Each task's gated x0_t is fed into the shared plain _ESMMCrossNet; the resulting cross output
    is folded into the task logit as in ESMM_PLE_WideCross. Wide term kept.

    AMP disabled via isinstance(model, ESMM_PLE) check in train_esmm_parquet_rowgroups.
    """

    def __init__(
        self,
        field_cardinalities,
        num_dense,
        embed_dim=18,
        num_cross_layers=3,
        gate_hidden=64,
        **kwargs,
    ):
        super().__init__(field_cardinalities, num_dense, embed_dim=embed_dim, **kwargs)
        input_dim = self.num_fields * self.embed_dim + num_dense
        self.cross = _ESMMCrossNet(input_dim, num_cross_layers)
        self.cross_ctr = nn.Linear(input_dim, 1)
        self.cross_cvr = nn.Linear(input_dim, 1)
        _init_linear(self.cross_ctr)
        _init_linear(self.cross_cvr)
        # EPNet-style per-task gate: d → gate_hidden → d, activation 2*sigmoid (output in (0,2))
        self.epnet_gate_ctr = nn.Sequential(
            nn.Linear(input_dim, gate_hidden),
            nn.ReLU(),
            nn.Linear(gate_hidden, input_dim),
        )
        self.epnet_gate_cvr = nn.Sequential(
            nn.Linear(input_dim, gate_hidden),
            nn.ReLU(),
            nn.Linear(gate_hidden, input_dim),
        )
        for gate in (self.epnet_gate_ctr, self.epnet_gate_cvr):
            for m in gate.modules():
                if isinstance(m, nn.Linear):
                    _init_linear(m)

    def forward(self, sparse_x, dense_x):
        _ac = torch.amp.autocast('cuda', enabled=False) if sparse_x.is_cuda else nullcontext()
        with _ac:
            g2_t1, g2_t2, x = self._ple_trunk(sparse_x, dense_x)
            x0_ctr = x * (2.0 * torch.sigmoid(self.epnet_gate_ctr(x)))
            x0_cvr = x * (2.0 * torch.sigmoid(self.epnet_gate_cvr(x)))
            xc_ctr = self.cross(x0_ctr)
            xc_cvr = self.cross(x0_cvr)
            logit_ctr = (self.tower_ctr(g2_t1).squeeze(1)
                         + self.wide_ctr(x).squeeze(1)
                         + self.cross_ctr(xc_ctr).squeeze(1))
            logit_cvr = (self.tower_cvr(g2_t2).squeeze(1)
                         + self.wide_cvr(x).squeeze(1)
                         + self.cross_cvr(xc_cvr).squeeze(1))
            p_ctr = torch.sigmoid(logit_ctr).clamp(1e-7, 1 - 1e-7)
            p_cvr = torch.sigmoid(logit_cvr).clamp(1e-7, 1 - 1e-7)
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
    track_grad_snr=False,
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

    track_grad_snr: if True, install a GradSNRTracker on the model (requires model to
    have a 'cross.layers' attribute, e.g. ESMM_PLE_WideCross_NDM), accumulate() after
    each backward, compute_snr() at epoch end (attached to train_meta as 'grad_snr'),
    then reset(). Zero overhead when False (default).
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
    if hasattr(model, 'compute_egean_loss'):
        if use_amp:
            print(
                '[train_esmm_parquet_rowgroups] ESMM_EGEAN: forcing use_amp=False '
                '(custom loss path; AMP disabled).'
            )
        use_amp = False
    if hasattr(model, 'compute_dcmt_loss'):
        if use_amp:
            print(
                '[train_esmm_parquet_rowgroups] ESMM_DCMT: forcing use_amp=False '
                '(custom loss path; AMP disabled).'
            )
        use_amp = False
    # ESCM²-DR flag: when the model exposes compute_escm2_loss(), route through that
    # path. The 4th column in row-group tensors becomes y_purchase (not y_ctcvr).
    # EGEAN and DCMT share the same routing (custom loss uses y_purchase as 4th col).
    _use_escm2_dr = (
        hasattr(model, 'compute_escm2_loss')
        or hasattr(model, 'compute_egean_loss')
        or hasattr(model, 'compute_dcmt_loss')
    )
    _use_egean = hasattr(model, 'compute_egean_loss')
    _use_dcmt  = hasattr(model, 'compute_dcmt_loss')
    # Optional GradSNR tracker — installed after model.to(device) so named_parameters() is stable.
    _snr_tracker = None
    if track_grad_snr:
        _snr_tracker = GradSNRTracker(model, layer_prefix='cross.layers')
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
                if _use_escm2_dr:
                    return sp, dn, y_click, y_purchase
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
        if _use_escm2_dr:
            del sub
            return sp, dn, y_click, y_purchase
        y_ctcvr = y_click * y_purchase
        del y_purchase
        del sub
        return sp, dn, y_click, y_ctcvr

    if compiled_active and not _use_escm2_dr and not _use_egean:
        # torch.compile warmup: skip for ESCM²-DR / EGEAN (custom loss path; AMP disabled anyway)
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
            if _use_egean:
                # EGEAN path: ycc_b carries y_purchase (not y_ctcvr); AMP disabled.
                model(sp_b, dn_b)
                loss = model.compute_egean_loss(yc_b, ycc_b, global_step=opt_steps)
                loss.backward()
                optimizer.step()
            elif _use_dcmt:
                # DCMT path: ycc_b carries y_purchase (not y_ctcvr); AMP disabled.
                model(sp_b, dn_b)
                loss = model.compute_dcmt_loss(yc_b, ycc_b, global_step=opt_steps)
                loss.backward()
                optimizer.step()
            elif _use_escm2_dr:
                # ESCM²-DR path: ycc_b carries y_purchase (not y_ctcvr); AMP disabled.
                model(sp_b, dn_b)
                loss = model.compute_escm2_loss(yc_b, ycc_b, global_step=opt_steps)
                loss.backward()
                optimizer.step()
            elif use_amp_cuda:
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
            if _snr_tracker is not None:
                _snr_tracker.accumulate()
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
        if _snr_tracker is not None:
            _epoch_snr = _snr_tracker.compute_snr()
            _snr_tracker.reset()
            # Attach per-epoch SNR to losses_out as a parallel list; also stash on tracker.
            if not hasattr(_snr_tracker, '_epoch_snr_history'):
                _snr_tracker._epoch_snr_history = []
            _snr_tracker._epoch_snr_history.append(_epoch_snr)
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
    if _snr_tracker is not None:
        # Attach the per-epoch SNR history; also expose last-epoch SNR at top level.
        train_meta['grad_snr'] = getattr(_snr_tracker, '_epoch_snr_history', [])
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


# --------------- Parameter count helper (Round 20260609) ---------------


def count_parameters(model):
    """Return total number of trainable parameters in a nn.Module."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# --------------- Calibration + significance utilities (Round 20260609) ---------------


def evaluate_ece(probs, labels, n_bins=20):
    """Expected calibration error with equal-width bins on probability.

    Parameters
    ----------
    probs  : array-like of float, predicted probabilities in [0, 1].
    labels : array-like of int/float, binary ground-truth labels.
    n_bins : int, number of equal-width probability bins (default 20).

    Returns
    -------
    float : scalar ECE.
    """
    p = np.clip(np.asarray(probs, dtype=np.float64).ravel(), 1e-9, 1 - 1e-9)
    y = np.asarray(labels, dtype=np.float64).ravel()
    n = len(p)
    if n == 0:
        return float('nan')
    edges = np.linspace(0.0, 1.0, int(n_bins) + 1)
    ece = 0.0
    for i in range(int(n_bins)):
        lo, hi = edges[i], edges[i + 1]
        m = (p >= lo) & (p <= hi) if i == int(n_bins) - 1 else (p >= lo) & (p < hi)
        cnt = int(m.sum())
        if cnt == 0:
            continue
        ece += (cnt / n) * abs(p[m].mean() - y[m].mean())
    return float(ece)


def user_grouped_bootstrap_auc_diff(
    labels,
    preds_a,
    preds_b,
    group_ids,
    n_boot=2000,
    seed=0,
    max_groups=None,
):
    """Bootstrap AUC difference (A − B) with group (user) resampling.

    Resamples GROUPS with replacement; AUC computed with weighted Mann–Whitney statistics
    (O(N log N) per bootstrap, no materialisation of resampled arrays).

    Parameters
    ----------
    labels    : array-like (N,), binary labels.
    preds_a   : array-like (N,), predictions for model A.
    preds_b   : array-like (N,), predictions for model B.
    group_ids : array-like (N,), group/user identifier per sample.
    n_boot    : int, number of bootstrap replicates (default 2000).
    seed      : int, random seed (default 0).
    max_groups: int or None. If set, randomly sub-sample this many groups before
                bootstrapping (useful when group count is very large; sub-sampling
                is done once, before the bootstrap loop). If None, all groups used.

    Returns
    -------
    dict with keys: delta (float), ci_low (float), ci_high (float), p_value (float).
      delta   = AUC_A − AUC_B computed on the ORIGINAL (unresampled) sample.
      ci_low/high = 2.5th/97.5th percentile of the bootstrap delta distribution.
      p_value = two-sided bootstrap test of delta=0:
                2 * min(frac(boot<=0), frac(boot>=0)), clamped to [1/n_boot, 1].
    """
    rng = np.random.RandomState(seed)
    labels = np.asarray(labels, dtype=np.float64).ravel()
    preds_a = np.asarray(preds_a, dtype=np.float64).ravel()
    preds_b = np.asarray(preds_b, dtype=np.float64).ravel()
    group_ids = np.asarray(group_ids).ravel()

    unique_groups, group_inv = np.unique(group_ids, return_inverse=True)
    G = len(unique_groups)

    if max_groups is not None and G > max_groups:
        chosen = rng.choice(G, size=int(max_groups), replace=False)
        mask = np.isin(group_inv, chosen)
        labels = labels[mask]
        preds_a = preds_a[mask]
        preds_b = preds_b[mask]
        group_ids = group_ids[mask]
        unique_groups, group_inv = np.unique(group_ids, return_inverse=True)
        G = len(unique_groups)

    # -----------------------------------------------------------------------
    # Vectorized weighted AUC (fully NumPy, no Python loops over tie-blocks)
    # -----------------------------------------------------------------------
    def _weighted_auc(y, p, w):
        """Tie-correct weighted AUC via score-group blocks (fully vectorized).

        For each score value s, all pos/neg pairs within the same block contribute 0.5
        (tie-correction), while pairs strictly ordered contribute 0 or 1.

        U = sum_blocks [ pos_w_in_block * (neg_w_below + 0.5 * neg_w_in_block) ]
        AUC = U / (total_pos_w * total_neg_w)
        """
        w = np.asarray(w, dtype=np.float64)
        order = np.argsort(p, kind='stable')    # ascending by score; O(N log N)
        y_s = y[order]
        w_s = w[order]
        p_s = p[order]

        pos_w = np.dot(y_s, w_s)
        neg_w = np.dot(1.0 - y_s, w_s)
        if pos_w == 0.0 or neg_w == 0.0:
            return float('nan')

        return _weighted_auc_presorted(y_s, p_s, w_s, pos_w, neg_w)

    def _weighted_auc_presorted(y_s, p_s, w_s, pos_w, neg_w):
        """Vectorized Mann-Whitney AUC on arrays already sorted by ascending score.

        Parameters
        ----------
        y_s, p_s, w_s : 1-D float64 arrays, sorted by p_s ascending.
        pos_w, neg_w  : pre-computed total positive/negative weight sums.

        Returns
        -------
        float AUC in [0, 1], or nan if pos_w==0 or neg_w==0.
        """
        # --- identify tie-group boundaries via score uniqueness ---
        # block_id[i] = index of the unique score value at position i
        _, block_id = np.unique(p_s, return_inverse=True)
        B = int(block_id[-1]) + 1                   # number of distinct score values

        # reduceat indices: first occurrence of each block
        starts = np.searchsorted(block_id, np.arange(B))

        # per-block positive and negative weight sums via add.reduceat
        pw_s = y_s * w_s
        nw_s = (1.0 - y_s) * w_s
        block_pos = np.add.reduceat(pw_s, starts)   # shape (B,)
        block_neg = np.add.reduceat(nw_s, starts)   # shape (B,)

        # neg_w_below[k] = sum of block_neg[0..k-1]  (exclusive prefix sum)
        neg_below = np.empty(B, dtype=np.float64)
        neg_below[0] = 0.0
        np.cumsum(block_neg[:-1], out=neg_below[1:])

        # U = sum_k block_pos[k] * (neg_below[k] + 0.5 * block_neg[k])
        u = np.dot(block_pos, neg_below + 0.5 * block_neg)
        return float(u / (pos_w * neg_w))

    # -----------------------------------------------------------------------
    # Pre-sort ONCE outside the bootstrap loop — O(N log N) paid once.
    # Per iteration we only need O(N) fancy-index to permute weights.
    # -----------------------------------------------------------------------
    order_a = np.argsort(preds_a, kind='stable')
    order_b = np.argsort(preds_b, kind='stable')

    y_sa = labels[order_a]
    y_sb = labels[order_b]
    p_sa = preds_a[order_a]
    p_sb = preds_b[order_b]

    # Pre-compute block structures (score-unique decomposition) for A and B
    _, block_id_a = np.unique(p_sa, return_inverse=True)
    Ba = int(block_id_a[-1]) + 1
    starts_a = np.searchsorted(block_id_a, np.arange(Ba))

    _, block_id_b = np.unique(p_sb, return_inverse=True)
    Bb = int(block_id_b[-1]) + 1
    starts_b = np.searchsorted(block_id_b, np.arange(Bb))

    def _auc_fast(y_s, starts, B, w_s):
        """AUC from pre-sorted/pre-blocked arrays plus a freshly-permuted weight vector."""
        pw_s = y_s * w_s
        nw_s = (1.0 - y_s) * w_s
        pos_w = pw_s.sum()
        neg_w = nw_s.sum()
        if pos_w == 0.0 or neg_w == 0.0:
            return float('nan')
        block_pos = np.add.reduceat(pw_s, starts)
        block_neg = np.add.reduceat(nw_s, starts)
        neg_below = np.empty(B, dtype=np.float64)
        neg_below[0] = 0.0
        np.cumsum(block_neg[:-1], out=neg_below[1:])
        u = np.dot(block_pos, neg_below + 0.5 * block_neg)
        return float(u / (pos_w * neg_w))

    # delta_obs: AUC difference on the ORIGINAL (unresampled) sample
    uniform_w = np.ones(len(labels), dtype=np.float64)
    delta_obs = _weighted_auc(labels, preds_a, uniform_w) - _weighted_auc(labels, preds_b, uniform_w)

    boot_deltas = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        boot_g = rng.randint(0, G, size=G)           # resample groups with replacement
        boot_counts = np.bincount(boot_g, minlength=G)   # (G,) counts
        sample_w = boot_counts[group_inv].astype(np.float64)  # (N,)
        if sample_w.sum() == 0.0:
            boot_deltas[b] = 0.0
            continue
        # Permute weights into the pre-sorted orders — O(N) each, no re-sort
        auc_a = _auc_fast(y_sa, starts_a, Ba, sample_w[order_a])
        auc_b = _auc_fast(y_sb, starts_b, Bb, sample_w[order_b])
        boot_deltas[b] = auc_a - auc_b

    valid = boot_deltas[np.isfinite(boot_deltas)]
    if len(valid) == 0:
        return {'delta': float('nan'), 'ci_low': float('nan'), 'ci_high': float('nan'), 'p_value': float('nan')}
    delta = float(delta_obs) if np.isfinite(delta_obs) else float('nan')
    ci_low = float(np.percentile(valid, 2.5))
    ci_high = float(np.percentile(valid, 97.5))
    # two-sided bootstrap test of delta=0:
    # p = 2 * min(frac(boot<=0), frac(boot>=0)), clamped to [1/n_boot, 1]
    frac_le = float(np.mean(valid <= 0))
    frac_ge = float(np.mean(valid >= 0))
    p_value = min(1.0, max(1.0 / n_boot, 2.0 * min(frac_le, frac_ge)))
    return {'delta': delta, 'ci_low': ci_low, 'ci_high': ci_high, 'p_value': p_value}


# ---------------------------------------------------------------------------
# Cycle-2 NegDisc-CVR models (NDM / ChorusCVR-style, GateObserver, GradSNR)
# ---------------------------------------------------------------------------


class _CTunCVRTower(nn.Module):
    """Auxiliary tower predicting click ∧ ¬convert (CTunCVR) from a representation.

    Identical MLP architecture to _ESMM_PLETower: d_model → d_model//2 → 1.
    Operated in entire-space (all samples see a gradient from the auxiliary task).
    """

    def __init__(self, d_model: int) -> None:
        super().__init__()
        hidden = max(1, d_model // 2)
        self.fc1 = nn.Linear(d_model, hidden)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(hidden, 1)
        _init_linear(self.fc1)
        _init_linear(self.fc2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x))).squeeze(1)


class _UnCVRTower(nn.Module):
    """Auxiliary tower predicting P(unconverted | click) — the unCVR factor in ChorusCVR.

    Identical architecture to _CTunCVRTower: d_model → d_model//2 → 1.
    The CTunCVR score is formed OUTSIDE this tower as y_ctuncvr = p_ctr * y_uncvr (Eq. 6).
    """

    def __init__(self, d_model: int) -> None:
        super().__init__()
        hidden = max(1, d_model // 2)
        self.fc1 = nn.Linear(d_model, hidden)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(hidden, 1)
        _init_linear(self.fc1)
        _init_linear(self.fc2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x))).squeeze(1)


class ESMM_PLE_WideCross_NDM(ESMM_PLE_WideCross):
    """Champion backbone (PLE + Wide + Cross) augmented with NDM (ChorusCVR Eq. 5–10).

    NDM adds:
    - An explicit unCVR tower predicting P(unconverted | click) → y_uncvr.
    - y_ctuncvr = p_ctr * y_uncvr  (two-stage product, Eq. 6; entire-space).
    - L_ctuncvr: BCE(y_ctuncvr, click*(1−purchase)) entire-space hard labels (Eq. 5+7).
    - L_align_IPW: four-term symmetric IPW alignment (Eq. 10):
        CVR  ← soft target 1−sg(y_uncvr), weighted by p_ctr (click) and 1−p_ctr (unclick).
        unCVR ← soft target 1−sg(p_cvr),  weighted by p_ctr (click) and 1−p_ctr (unclick).
      All four terms run over ALL samples; no masking.

    Total loss = L_ctr + L_ctcvr + ndm_weight × (L_ctuncvr + L_align_IPW)

    ndm_mode:
      'ndm'   — full NDM (default).
      'hard'  — identical parameters, NDM loss terms zeroed out (control arm).
                Both aux towers (uncvr_tower + legacy ctuncvr_tower slot) still built;
                param-matched to ndm mode.
      'smooth'— no alignment; unclicked CVR labels replaced by constant smooth_mass.
                Models N3 label-smoothing control.

    with_observer (bool): attach a GateObserverHead over the cross stack (M1/H3).
    track_grad_snr (bool): install GradSNRTracker on the cross layers (N4).

    Constructor signature: (field_cardinalities, num_dense, embed_dim=18, **kw)
    where kw may include num_cross_layers, d_model, expert_hidden, etc.
    Extra NDM kwargs: ndm_weight, ndm_mode, smooth_mass, with_observer, track_grad_snr.

    AMP: subclasses ESMM_PLE (via ESMM_PLE_WideCross), so train_esmm_parquet_rowgroups
    already disables AMP via isinstance(model, ESMM_PLE).
    """

    def __init__(
        self,
        field_cardinalities,
        num_dense,
        embed_dim: int = DEFAULT_EMBED_DIM,
        num_cross_layers: int = 3,
        ndm_weight: float = 1.0,
        ndm_mode: str = 'ndm',
        smooth_mass: float = 0.0,
        with_observer: bool = False,
        track_grad_snr: bool = False,
        ndm_impl: str = 'chorus_v2',
        soft_temp: float = 1.0,
        **kw,
    ):
        super().__init__(
            field_cardinalities, num_dense, embed_dim=embed_dim,
            num_cross_layers=num_cross_layers, **kw,
        )
        if ndm_mode not in ('ndm', 'hard', 'smooth'):
            raise ValueError(f'ndm_mode must be ndm/hard/smooth, got {ndm_mode!r}')
        self.ndm_impl = ndm_impl
        self.ndm_weight = float(ndm_weight)
        self.ndm_mode = ndm_mode
        self.smooth_mass = float(smooth_mass)
        # soft_temp: temperature scaling for soft alignment targets in 'ndm' mode.
        # Sharpens (temp<1) or softens (temp>1) the binary soft labels before BCE.
        # For binary p ∈ (0,1): p_sharp = p^(1/T) / (p^(1/T) + (1-p)^(1/T)).
        # temp=1.0 → no change (identity). Exposed as a kwarg; default=1.0 for
        # backward compatibility. See Round 11 sensitivity grid.
        self.soft_temp = float(soft_temp)

        # d_model from parent (ESMM_PLE sets it via PLE level; tower output is d_model)
        d_model = int(self.tower_cvr.fc1.in_features)

        # Auxiliary towers: always built (param-matched in hard mode, loss disabled).
        # uncvr_tower predicts P(unconverted | click); y_ctuncvr formed as p_ctr * y_uncvr.
        # ctuncvr_tower kept for param-parity with earlier checkpoints (unused in loss).
        self.uncvr_tower = _UnCVRTower(d_model)
        self.ctuncvr_tower = _CTunCVRTower(d_model)

        # Replace the plain cross stack (from ESMM_PLE_WideCross parent) with the exposed
        # variant so per-layer outputs are available for the observer and training loop.
        # Numerical equivalence: _ESMMCrossNetExposed uses the identical recurrence
        #   x_{l+1} = x0 * W_l(x_l) + x_l  (same as _ESMMCrossNet), so main-task logits
        # are unchanged — only the return value differs (list vs final tensor).
        input_dim = self.num_fields * self.embed_dim + num_dense
        self.cross = _ESMMCrossNetExposed(input_dim, num_cross_layers)
        # _last_cross_layer_outputs holds the most recent list of per-layer tensors (detached)
        # for the training loop / observer; set in forward().
        self._last_cross_layer_outputs: list = []

        # Optional GateObserverHead (M1)
        self._with_observer = bool(with_observer)
        if with_observer:
            self.gate_observer = GateObserverHead(
                input_dim=input_dim,
                num_cross_layers=num_cross_layers,
            )

        # Optional GradSNRTracker (N4)
        self._track_grad_snr = bool(track_grad_snr)
        if track_grad_snr:
            self.grad_snr_tracker = GradSNRTracker(self, layer_prefix='cross.layers')

    def forward(self, sparse_x, dense_x):
        """Returns (p_ctr, p_cvr, p_ctcvr).  NDM loss computed in compute_ndm_loss().

        Side effects:
          self._last_y_uncvr    — sigmoid output of uncvr_tower, shape (B,) (with grad)
          self._last_y_ctuncvr  — p_ctr * y_uncvr two-stage product (Eq. 6), shape (B,)
          self._last_logit_ctuncvr — legacy alias: raw logit from ctuncvr_tower (B,)
          self._last_cross_layer_outputs — list of K detached (B, d) tensors, one per
              cross layer, for the observer training loop (detached so no gradient leaks
              into the backbone through the observer path).
          self._last_logit_ctr / self._last_logit_cvr — detached backbone logits for
              the observer forward pass.
        """
        eps = 1e-7
        _ac = torch.amp.autocast('cuda', enabled=False) if sparse_x.is_cuda else nullcontext()
        with _ac:
            g2_t1, g2_t2, x = self._ple_trunk(sparse_x, dense_x)
            # _ESMMCrossNetExposed returns a list [x_1, ..., x_K]; final element is x_K
            layer_outputs = self.cross(x)   # list of K tensors (B, d)
            xc = layer_outputs[-1]          # x_K — same as _ESMMCrossNet final output
            logit_ctr = (self.tower_ctr(g2_t1).squeeze(1)
                         + self.wide_ctr(x).squeeze(1)
                         + self.cross_ctr(xc).squeeze(1))
            logit_cvr = (self.tower_cvr(g2_t2).squeeze(1)
                         + self.wide_cvr(x).squeeze(1)
                         + self.cross_cvr(xc).squeeze(1))
            p_ctr = torch.sigmoid(logit_ctr).clamp(eps, 1 - eps)
            p_cvr = torch.sigmoid(logit_cvr).clamp(eps, 1 - eps)
            p_ctcvr = (p_ctr * p_cvr).clamp(eps, 1 - eps)
            # unCVR auxiliary tower: P(unconverted | click) — Eq. 6 factorisation
            y_uncvr = torch.sigmoid(self.uncvr_tower(g2_t2)).clamp(eps, 1 - eps)
            # Two-stage CTunCVR product: y_ctuncvr = p_ctr * y_uncvr (Eq. 6)
            y_ctuncvr = (p_ctr * y_uncvr).clamp(eps, 1 - eps)
            self._last_y_uncvr = y_uncvr
            self._last_y_ctuncvr = y_ctuncvr
            # Legacy slot (ctuncvr_tower kept for param-parity; not used in loss)
            self._last_logit_ctuncvr = self.ctuncvr_tower(g2_t2)
            # Store detached per-layer cross outputs + backbone logits for observer use.
            self._last_cross_layer_outputs = [t.detach() for t in layer_outputs]
            self._last_logit_ctr = logit_ctr.detach()
            self._last_logit_cvr = logit_cvr.detach()
            return p_ctr, p_cvr, p_ctcvr

    def compute_ndm_loss(
        self,
        p_ctr: torch.Tensor,
        p_cvr: torch.Tensor,
        p_ctcvr: torch.Tensor,
        y_click: torch.Tensor,
        y_purchase: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the full NDM loss given model outputs and labels.

        Returns a scalar loss tensor.  In 'hard' mode the NDM terms are zeroed
        (backbone loss only); in 'smooth' mode a constant label-smoothing replaces
        the alignment losses on unclicked CVR labels.

        Implements ChorusCVR Eq. 5–10:
          L_ctuncvr: BCE(y_ctuncvr, click*(1−purchase)) entire-space hard labels.
          L_align_IPW: four-term symmetric IPW alignment (Eq. 10), all samples.

        Caller is responsible for the standard ESMM terms; this function returns the
        COMPLETE loss = L_ctr + L_ctcvr + ndm_weight * (L_ctuncvr + L_align_IPW).
        """
        eps = 1e-7
        yc = y_click.float()
        yp = y_purchase.float()
        y_ctcvr_label = (yc * yp).clamp(0.0, 1.0)

        pc = p_ctr.clamp(eps, 1 - eps)
        pcc = p_ctcvr.clamp(eps, 1 - eps)

        # Standard ESMM loss
        l_ctr = F.binary_cross_entropy(pc, yc)
        l_ctcvr = F.binary_cross_entropy(pcc, y_ctcvr_label)
        backbone_loss = l_ctr + l_ctcvr

        if self.ndm_mode == 'hard':
            # NDM terms disabled; aux towers constructed but loss-zeroed
            return backbone_loss

        # L_ctuncvr (Eq. 5+7): BCE over entire exposure space with hard label
        y_ctuncvr_hard = (yc * (1.0 - yp)).clamp(0.0, 1.0)
        l_ctuncvr = F.binary_cross_entropy(self._last_y_ctuncvr, y_ctuncvr_hard)

        pv = p_cvr.clamp(eps, 1 - eps)

        if self.ndm_mode == 'smooth':
            # N3 label-smoothing control: constant mass for unclicked CVR labels.
            # Clicked rows keep their hard purchase labels.
            smooth = float(self.smooth_mass)
            unclicked = (yc < 0.5)
            if unclicked.any():
                y_cvr_target = yp.clone()
                y_cvr_target[unclicked] = smooth
                l_soft_cvr = F.binary_cross_entropy(pv, y_cvr_target)
            else:
                # All rows clicked: use hard purchase labels (not 0.0).
                l_soft_cvr = F.binary_cross_entropy(pv, yp)
            return backbone_loss + self.ndm_weight * (l_ctuncvr + l_soft_cvr)

        # ndm_mode == 'ndm': ChorusCVR Eq. 10 — four-term symmetric IPW alignment
        # IPW weights (detached: no gradient through the weighting factors)
        w_click = p_ctr.detach()            # IPW for click space
        w_unclick = 1.0 - w_click          # IPW for unclick space

        # Normalisation denominators (at least 1 to avoid div-by-zero on edge batches)
        len_O = max(1, int((yc > 0.5).sum().item()))
        len_N = max(1, int((yc < 0.5).sum().item()))

        # Soft targets with stop-gradient on the cross-tower side (Eq. 10)
        _soft_raw_cvr   = (1.0 - self._last_y_uncvr.detach()).clamp(eps, 1.0 - eps)
        _soft_raw_uncvr = (1.0 - pv.detach()).clamp(eps, 1.0 - eps)
        # soft_temp sharpening: p_sharp = p^(1/T) / (p^(1/T) + (1-p)^(1/T))
        # temp=1.0 → identity; temp<1 → sharper (more confident); temp>1 → softer.
        if self.soft_temp != 1.0:
            _t = 1.0 / max(self.soft_temp, 1e-3)
            _p = _soft_raw_cvr
            soft_for_cvr = _p.pow(_t) / (_p.pow(_t) + (1.0 - _p).pow(_t)).clamp(min=eps)
            _p = _soft_raw_uncvr
            soft_for_uncvr = _p.pow(_t) / (_p.pow(_t) + (1.0 - _p).pow(_t)).clamp(min=eps)
        else:
            soft_for_cvr   = _soft_raw_cvr
            soft_for_uncvr = _soft_raw_uncvr

        y_uncvr = self._last_y_uncvr.clamp(eps, 1 - eps)

        # align1: CVR ← 1−sg(y_uncvr), click-weighted
        l_align = (F.binary_cross_entropy(pv, soft_for_cvr, reduction='none') * w_click).sum() / len_O
        # align3: CVR ← 1−sg(y_uncvr), unclick-weighted
        l_align = l_align + (F.binary_cross_entropy(pv, soft_for_cvr, reduction='none') * w_unclick).sum() / len_N
        # align2: unCVR ← 1−sg(p_cvr), click-weighted
        l_align = l_align + (F.binary_cross_entropy(y_uncvr, soft_for_uncvr, reduction='none') * w_click).sum() / len_O
        # align4: unCVR ← 1−sg(p_cvr), unclick-weighted
        l_align = l_align + (F.binary_cross_entropy(y_uncvr, soft_for_uncvr, reduction='none') * w_unclick).sum() / len_N

        return backbone_loss + self.ndm_weight * (l_ctuncvr + l_align)

    def observer_loss(
        self,
        y_click: torch.Tensor,
        y_purchase: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the observer's BCE loss from the most recent forward() call.

        Inputs are DETACHED (stored in self._last_cross_layer_outputs / _last_logit_ctr/cvr)
        so gradients flow only through the GateObserverHead's own parameters — the backbone
        is completely unaffected.

        y_click, y_purchase: labels for the current batch (same as passed to compute_ndm_loss).
        y_ctcvr = y_click * y_purchase.

        Returns a scalar tensor.  Caller adds it to the total loss with an appropriate weight.
        Raises RuntimeError if with_observer=False.
        """
        if not self._with_observer:
            raise RuntimeError('observer_loss() called but with_observer=False')
        y_ctcvr = (y_click.float() * y_purchase.float()).clamp(0.0, 1.0)
        return self.gate_observer(
            self._last_cross_layer_outputs,
            self._last_logit_ctr,
            self._last_logit_cvr,
            y_click,
            y_ctcvr,
        )

    def mean_ctuncvr_output(
        self,
        sparse_x: torch.Tensor,
        dense_x: torch.Tensor,
    ) -> float:
        """Forward pass returning mean y_ctuncvr (for measuring smooth_mass in N3)."""
        self.eval()
        with torch.no_grad():
            self(sparse_x, dense_x)
        return float(self._last_y_ctuncvr.mean().item())

    def get_gate_weights(self) -> dict:
        """Delegate to GateObserverHead if attached, else return None."""
        if self._with_observer:
            return self.gate_observer.get_gate_weights()
        return {'ctr': None, 'cvr': None}


class ESMM_NDM(ESMMModel_Wide):
    """Plain two-tower ESMM_Wide backbone with the same NDM machinery (ChorusCVR Eq. 5–10).

    Ceiling-control arm: confirms NDM effect holds on the simpler backbone (not
    just the champion); also provides a clean upper-bound reference.

    Same ndm_mode semantics as ESMM_PLE_WideCross_NDM.
    """

    def __init__(
        self,
        field_cardinalities,
        num_dense,
        embed_dim: int = DEFAULT_EMBED_DIM,
        ndm_weight: float = 1.0,
        ndm_mode: str = 'ndm',
        smooth_mass: float = 0.0,
        ndm_impl: str = 'chorus_v2',
        soft_temp: float = 1.0,
        **kw,
    ):
        super().__init__(field_cardinalities, num_dense, embed_dim=embed_dim, **kw)
        if ndm_mode not in ('ndm', 'hard', 'smooth'):
            raise ValueError(f'ndm_mode must be ndm/hard/smooth, got {ndm_mode!r}')
        self.ndm_impl = ndm_impl
        self.ndm_weight = float(ndm_weight)
        self.ndm_mode = ndm_mode
        self.smooth_mass = float(smooth_mass)
        # soft_temp: temperature for soft alignment targets. Same semantics as in
        # ESMM_PLE_WideCross_NDM. See Round 11 sensitivity grid.
        self.soft_temp = float(soft_temp)

        # Auxiliary towers: operate on the embed+dense concat (shared repr design).
        # uncvr_tower predicts P(unconverted | click); ctuncvr_tower kept for param-parity.
        input_dim = self.num_fields * self.embed_dim + num_dense
        hidden = max(1, input_dim // 4)
        self.uncvr_tower = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )
        for m in self.uncvr_tower.modules():
            if isinstance(m, nn.Linear):
                _init_linear(m)
        self.ctuncvr_tower = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )
        for m in self.ctuncvr_tower.modules():
            if isinstance(m, nn.Linear):
                _init_linear(m)
        nn.init.normal_(self.unified_emb.weight, 0.0, 0.01)

    def forward(self, sparse_x, dense_x):
        eps = 1e-7
        idx = sparse_x.long() + self.field_offsets
        e = self.unified_emb(idx)
        x = torch.cat([e.flatten(1), dense_x], dim=1)
        logit_ctr = self.ctr_tower(x).squeeze(1) + self.wide_ctr(x).squeeze(1)
        logit_cvr = self.cvr_tower(x).squeeze(1) + self.wide_cvr(x).squeeze(1)
        p_ctr = torch.sigmoid(logit_ctr).clamp(eps, 1 - eps)
        p_cvr = torch.sigmoid(logit_cvr).clamp(eps, 1 - eps)
        p_ctcvr = (p_ctr * p_cvr).clamp(eps, 1 - eps)
        self._last_x = x
        # unCVR tower: P(unconverted | click)
        y_uncvr = torch.sigmoid(self.uncvr_tower(x).squeeze(1)).clamp(eps, 1 - eps)
        # Two-stage CTunCVR product: p_ctr * y_uncvr (Eq. 6)
        y_ctuncvr = (p_ctr * y_uncvr).clamp(eps, 1 - eps)
        self._last_y_uncvr = y_uncvr
        self._last_y_ctuncvr = y_ctuncvr
        # Legacy slot
        self._last_logit_ctuncvr = self.ctuncvr_tower(x).squeeze(1)
        return p_ctr, p_cvr, p_ctcvr

    def compute_ndm_loss(
        self,
        p_ctr: torch.Tensor,
        p_cvr: torch.Tensor,
        p_ctcvr: torch.Tensor,
        y_click: torch.Tensor,
        y_purchase: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the full NDM loss (ChorusCVR Eq. 5–10).

        Returns backbone + ndm_weight * (L_ctuncvr + L_align_IPW) in 'ndm' mode.
        Returns backbone only in 'hard' mode.
        Returns backbone + ndm_weight * (L_ctuncvr + L_smooth_cvr) in 'smooth' mode.
        """
        eps = 1e-7
        yc = y_click.float()
        yp = y_purchase.float()
        y_ctcvr_label = (yc * yp).clamp(0.0, 1.0)
        pc = p_ctr.clamp(eps, 1 - eps)
        pcc = p_ctcvr.clamp(eps, 1 - eps)
        l_ctr = F.binary_cross_entropy(pc, yc)
        l_ctcvr = F.binary_cross_entropy(pcc, y_ctcvr_label)
        backbone_loss = l_ctr + l_ctcvr

        if self.ndm_mode == 'hard':
            return backbone_loss

        # L_ctuncvr: BCE over entire exposure space with hard label (Eq. 5+7)
        y_ctuncvr_hard = (yc * (1.0 - yp)).clamp(0.0, 1.0)
        l_ctuncvr = F.binary_cross_entropy(self._last_y_ctuncvr, y_ctuncvr_hard)

        pv = p_cvr.clamp(eps, 1 - eps)

        if self.ndm_mode == 'smooth':
            # N3 label-smoothing control: constant mass for unclicked CVR labels.
            smooth = float(self.smooth_mass)
            unclicked = (yc < 0.5)
            if unclicked.any():
                y_cvr_target = yp.clone()
                y_cvr_target[unclicked] = smooth
                l_soft_cvr = F.binary_cross_entropy(pv, y_cvr_target)
            else:
                # All rows clicked: use hard purchase labels (not 0.0).
                l_soft_cvr = F.binary_cross_entropy(pv, yp)
            return backbone_loss + self.ndm_weight * (l_ctuncvr + l_soft_cvr)

        # ndm_mode == 'ndm': ChorusCVR Eq. 10 — four-term symmetric IPW alignment
        w_click = p_ctr.detach()
        w_unclick = 1.0 - w_click

        len_O = max(1, int((yc > 0.5).sum().item()))
        len_N = max(1, int((yc < 0.5).sum().item()))

        _soft_raw_cvr   = (1.0 - self._last_y_uncvr.detach()).clamp(eps, 1.0 - eps)
        _soft_raw_uncvr = (1.0 - pv.detach()).clamp(eps, 1.0 - eps)
        # soft_temp sharpening: same formula as ESMM_PLE_WideCross_NDM
        if self.soft_temp != 1.0:
            _t = 1.0 / max(self.soft_temp, 1e-3)
            _p = _soft_raw_cvr
            soft_for_cvr = _p.pow(_t) / (_p.pow(_t) + (1.0 - _p).pow(_t)).clamp(min=eps)
            _p = _soft_raw_uncvr
            soft_for_uncvr = _p.pow(_t) / (_p.pow(_t) + (1.0 - _p).pow(_t)).clamp(min=eps)
        else:
            soft_for_cvr   = _soft_raw_cvr
            soft_for_uncvr = _soft_raw_uncvr

        y_uncvr = self._last_y_uncvr.clamp(eps, 1 - eps)

        # Four alignment terms — all samples (no masking)
        l_align = (F.binary_cross_entropy(pv, soft_for_cvr, reduction='none') * w_click).sum() / len_O
        l_align = l_align + (F.binary_cross_entropy(pv, soft_for_cvr, reduction='none') * w_unclick).sum() / len_N
        l_align = l_align + (F.binary_cross_entropy(y_uncvr, soft_for_uncvr, reduction='none') * w_click).sum() / len_O
        l_align = l_align + (F.binary_cross_entropy(y_uncvr, soft_for_uncvr, reduction='none') * w_unclick).sum() / len_N

        return backbone_loss + self.ndm_weight * (l_ctuncvr + l_align)


class GateObserverHead(nn.Module):
    """AdaOrder-style per-task softmax gate observer trained on DETACHED cross-stack inputs.

    The observer's gate parameters see gradients from task losses only w.r.t. the gate
    parameters themselves (stop-gradient: backbone is unaffected).  This allows profiling
    the cross-depth preference without perturbing training dynamics (M1/H3).

    The head maintains two K-dim parameters (one per task); they optimise a soft cross-
    entropy-like loss from detached cross outputs, computed externally via
    observer_loss(layer_outputs_detached, logit_ctr_detached, logit_cvr_detached, labels).

    Attachable: model.gate_observer = GateObserverHead(input_dim, num_cross_layers)
    expose via with_observer=True flag in ESMM_PLE_WideCross_NDM.
    """

    def __init__(self, input_dim: int, num_cross_layers: int) -> None:
        super().__init__()
        self.K = int(num_cross_layers)
        # Per-task learned log-weights over K depths
        self.theta_ctr = nn.Parameter(torch.zeros(self.K))
        self.theta_cvr = nn.Parameter(torch.zeros(self.K))
        # Projection heads — created in __init__ so the optimizer registers them at
        # construction time (lazy creation in forward would leave them unregistered).
        self._proj_ctr = nn.Linear(int(input_dim), 1, bias=False)
        self._proj_cvr = nn.Linear(int(input_dim), 1, bias=False)
        _init_linear(self._proj_ctr)
        _init_linear(self._proj_cvr)

    def get_gate_weights(self) -> dict:
        with torch.no_grad():
            return {
                'ctr': torch.softmax(self.theta_ctr, dim=0).cpu().numpy(),
                'cvr': torch.softmax(self.theta_cvr, dim=0).cpu().numpy(),
            }

    def forward(
        self,
        layer_outputs_detached: list,  # list of K tensors (B, d), already .detach()ed
        logit_ctr_detached: torch.Tensor,   # (B,) backbone CTR logit, detached
        logit_cvr_detached: torch.Tensor,   # (B,) backbone CVR logit, detached
        y_click: torch.Tensor,
        y_ctcvr: torch.Tensor,
    ) -> torch.Tensor:
        """Return observer loss (does NOT affect backbone parameters)."""
        K = self.K
        alpha_ctr = torch.softmax(self.theta_ctr, dim=0)  # (K,)
        alpha_cvr = torch.softmax(self.theta_cvr, dim=0)  # (K,)
        stacked = torch.stack(layer_outputs_detached, dim=1)  # (B, K, d)
        # Weighted cross output per task
        xc_ctr = (stacked * alpha_ctr.view(1, K, 1)).sum(dim=1)  # (B, d)
        xc_cvr = (stacked * alpha_cvr.view(1, K, 1)).sum(dim=1)  # (B, d)
        # The observer only optimises its own affine projection on top
        # (no separate head needed — use dot product with the last layer weight direction)
        # Simple approach: compute a scalar contribution from the weighted output and
        # optimise it to minimise task loss from the observer's perspective.
        # We score each task's weighted cross output by its cosine alignment with the
        # backbone's detached logit direction.  Observer loss = task CE with observer preds.
        eps = 1e-7
        # Observer CTR score = backbone_logit + small linear combo of observer weighted cross
        obs_ctr = torch.sigmoid(
            logit_ctr_detached + self._proj_ctr(xc_ctr).squeeze(1)
        ).clamp(eps, 1 - eps)
        obs_cvr_logit = logit_cvr_detached + self._proj_cvr(xc_cvr).squeeze(1)
        obs_ctcvr = (obs_ctr * torch.sigmoid(obs_cvr_logit)).clamp(eps, 1 - eps)
        loss_obs = (
            F.binary_cross_entropy(obs_ctr, y_click.float())
            + F.binary_cross_entropy(obs_ctcvr, y_ctcvr.float())
        )
        return loss_obs


class GradSNRTracker:
    """Gradient signal-to-noise ratio tracker for named cross-layer weights.

    Accumulates per-minibatch gradient mean/std across an epoch; computes
    SNR_k = |mean_k| / (std_k + eps) per tracked layer after the epoch.

    Usage:
        tracker = GradSNRTracker(model, layer_prefix='cross.layers')
        # at the end of each batch (after loss.backward()):
        tracker.accumulate()
        # at the end of the epoch:
        snr_dict = tracker.compute_snr()   # {layer_name: snr_value}
        tracker.reset()

    Install as model.grad_snr_tracker; training loop calls these three methods
    only when track_grad_snr=True (zero overhead otherwise — tracker not created).
    """

    def __init__(self, model: nn.Module, layer_prefix: str = 'cross.layers') -> None:
        self.model = model
        self.layer_prefix = layer_prefix
        self._grad_means: dict[str, list] = {}
        self._grad_stds: dict[str, list] = {}
        self._tracked_names: list[str] = []
        for name, param in model.named_parameters():
            if layer_prefix in name and param.requires_grad:
                self._tracked_names.append(name)
                self._grad_means[name] = []
                self._grad_stds[name] = []

    def accumulate(self) -> None:
        """Record the mean and std of the current gradient for each tracked parameter.

        Uses population std (correction=0) so single-element tensors return 0.0
        instead of nan (Bessel correction is undefined for n=1).
        """
        for name in self._tracked_names:
            param = dict(self.model.named_parameters())[name]
            if param.grad is not None:
                g = param.grad.detach().float().view(-1)
                self._grad_means[name].append(g.mean().item())
                std_val = g.std(correction=0).item() if g.numel() > 0 else 0.0
                self._grad_stds[name].append(std_val)

    def compute_snr(self) -> dict[str, float]:
        """Return SNR_k = |mean_k| / (std_k + eps) averaged over the epoch."""
        eps = 1e-8
        result = {}
        for name in self._tracked_names:
            means = self._grad_means[name]
            stds = self._grad_stds[name]
            if not means:
                result[name] = float('nan')
                continue
            mean_of_means = float(np.mean(np.abs(means)))
            mean_of_stds = float(np.mean(stds))
            result[name] = mean_of_means / (mean_of_stds + eps)
        return result

    def reset(self) -> None:
        """Clear accumulated statistics (call at the start of each epoch)."""
        for name in self._tracked_names:
            self._grad_means[name] = []
            self._grad_stds[name] = []


# ---------------------------------------------------------------------------
# ESCM²-DR (Wang et al., SIGIR 2022, arXiv 2204.05125v2)
# ---------------------------------------------------------------------------
# Plain two-tower backbone (mirrors ESMMModel_Wide structure:
#   unified embedding N(0, 0.01), wide terms, CTR + CVR towers)
# + an IMPUTATION tower (δ̂, same shape as CVR tower, no sigmoid).
#
# forward()  → (p_ctr, p_cvr, p_ctcvr)  — side-effect: self._last_delta_hat
# compute_escm2_loss() → full ESCM²-DR objective per memo Eq. 22–27.
#
# Differences vs. paper defaults (see escm2dr-formula-verification.md §7):
#   D1: embed_dim=18 (paper uses 5)
#   D10: propensity floored at ips_clip_floor (default 0.1, matching paper Appendix A)


class _ImpTower(nn.Module):
    """Imputation tower: predicts δ̂ (BCE error estimate), no sigmoid, same MLP shape as CVR tower."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        hidden = max(1, d_model // 2)
        self.fc1 = nn.Linear(d_model, hidden)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(hidden, 1)
        _init_linear(self.fc1)
        _init_linear(self.fc2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x))).squeeze(1)


class ESMM_ESCM2DR(ESMMModel_Wide):
    """ESCM²-DR baseline: plain two-tower ESMM_Wide + imputation tower.

    Paper: Wang et al., SIGIR 2022, "ESCM²" (arXiv 2204.05125v2, Eq. 22–27).
    Spec:  escm2dr-formula-verification.md (code-confirmed against PaddleRec).

    Constructor args
    ----------------
    field_cardinalities, num_dense, embed_dim : same as ESMMModel_Wide.
    lambda_c     : weight on R_DR (Eq. 27, default 1.0; paper uses 0.1 on Ali-CCP).
    lambda_g     : weight on L_CTCVR (Eq. 27, default 1.0).
    ips_clip_floor : propensity floor for IPS computation (paper Appendix A: 0.1).
    dr_warmup_steps: DR terms zeroed while global_step < dr_warmup_steps (default 0).

    forward() returns (p_ctr, p_cvr, p_ctcvr) and stores self._last_delta_hat.
    compute_escm2_loss(y_click, y_purchase, global_step) returns the full ESCM²-DR loss.
    """

    def __init__(
        self,
        field_cardinalities,
        num_dense,
        embed_dim: int = DEFAULT_EMBED_DIM,
        lambda_c: float = 1.0,
        lambda_g: float = 1.0,
        ips_clip_floor: float = 0.1,
        dr_warmup_steps: int = 0,
        **kw,
    ):
        super().__init__(field_cardinalities, num_dense, embed_dim=embed_dim, **kw)
        self.lambda_c = float(lambda_c)
        self.lambda_g = float(lambda_g)
        self.ips_clip_floor = float(ips_clip_floor)
        self.dr_warmup_steps = int(dr_warmup_steps)

        # Imputation tower: same representation as CVR tower (shared embed+dense concat).
        # Uses an MLP over the same input_dim as the CTR/CVR towers.
        input_dim = self.num_fields * self.embed_dim + num_dense
        self.imp_tower = _ImpTower(input_dim)
        # _last_delta_hat: set by forward(), consumed by compute_escm2_loss()
        self._last_delta_hat: torch.Tensor = torch.zeros(1)

    def compute_escm2_loss(
        self,
        y_click: torch.Tensor,
        y_purchase: torch.Tensor,
        global_step: int = 0,
    ) -> torch.Tensor:
        """Full ESCM²-DR multi-task loss (Eq. 27 = L_CTR + λ_c·R_DR + λ_g·L_CTCVR).

        Must be called AFTER forward() for the same batch (consumes self._last_delta_hat).

        R_DR = R_DR^err + R_DR^imp  (Eq. 24)
          R_DR^err = mean(δ̂ + ê·(o/ô))        (Eq. 22; imputation over all rows)
          R_DR^imp = mean(ê²·(o/ô))            (Eq. 23; correction click-gated via IPS)

        where ê = BCE(r, r̂) − δ̂  (error residual, computable only for clicked rows,
        but o=0 for unclicked makes the IPS term zero automatically).

        Propensity (ô):
          - detached from CTR tower (Trick 2: paper Appendix A §gradient-truncation)
          - floored at ips_clip_floor (default 0.1; paper Appendix A §propensity-clipping)
          - IPS = o/ô → max value is 1/ips_clip_floor = 10 at default

        DR terms zeroed while global_step < dr_warmup_steps (alternating-training warm-up).

        Arguments
        ---------
        y_click    : (B,) binary click indicator o ∈ {0, 1}
        y_purchase : (B,) binary purchase indicator r ∈ {0, 1}  (defined over 𝒟)
        global_step: current optimizer step (used for DR warm-up gate)
        """
        # Reads _last_p_ctr/_last_p_cvr/_last_p_ctcvr/_last_delta_hat set by forward().
        # Pattern: call model(sp, dn) first, then model.compute_escm2_loss(yc, yp, step).
        p_ctr = self._last_p_ctr
        p_cvr = self._last_p_cvr
        p_ctcvr = self._last_p_ctcvr

        eps = 1e-7
        yc = y_click.float()
        yp = y_purchase.float()
        y_ctcvr_label = (yc * yp).clamp(0.0, 1.0)

        # L_CTR (Eq. 25): BCE(p_ctr, o) over entire space 𝒟
        pc = p_ctr.clamp(eps, 1 - eps)
        l_ctr = F.binary_cross_entropy(pc, yc)

        # L_CTCVR (Eq. 26): BCE(p_ctr * p_cvr, o * r) over entire space 𝒟
        pcc = p_ctcvr.clamp(eps, 1 - eps)
        l_ctcvr = F.binary_cross_entropy(pcc, y_ctcvr_label)

        if global_step < self.dr_warmup_steps:
            # DR warm-up: return backbone only (standard ESMM loss)
            return l_ctr + self.lambda_g * l_ctcvr

        # --- DR CVR regularizer R_DR (Eq. 22-24) ---
        pv = p_cvr.clamp(eps, 1 - eps)

        # δ = BCE(r, r̂) per-sample (gradients to CVR tower)
        delta = F.binary_cross_entropy(pv, yp, reduction='none')

        # δ̂ = imputation tower output (raw, no sigmoid; gradients to IMP tower)
        delta_hat = self._last_delta_hat  # (B,)

        # ê = δ − δ̂  (error residual)
        e_hat = delta - delta_hat

        # Propensity: detach ô from CTR tower (Trick 2); floor at ips_clip_floor (Trick 3)
        propensity = p_ctr.detach().clamp(min=self.ips_clip_floor)

        # IPS = o / ô (Trick 1: unclicked rows have o=0, so IPS=0 → correction term vanishes)
        # Max IPS = 1 / ips_clip_floor (e.g. 10 at default floor=0.1)
        ips = (yc / propensity)  # (B,); unclicked → 0.0; ips already bounded by floor

        # R_DR^err (Eq. 22) per-sample: δ̂ + ê·(o/ô)
        # Imputation term δ̂ runs over ALL rows (E_{𝒟}); correction ê·IPS click-gated via ips
        r_dr_err = delta_hat + e_hat * ips

        # R_DR^imp (Eq. 23) per-sample: ê²·(o/ô) — click-gated via ips
        r_dr_imp = e_hat.pow(2) * ips

        # R_DR (Eq. 24): mean over entire space 𝒟
        l_cvr = (r_dr_err + r_dr_imp).mean()

        # Full ESCM²-DR objective (Eq. 27)
        return l_ctr + self.lambda_c * l_cvr + self.lambda_g * l_ctcvr

    def forward(self, sparse_x, dense_x):
        """Two-tower ESMM + imputation head.

        Returns (p_ctr, p_cvr, p_ctcvr) — same contract as ESMMModel_Wide.
        Side-effects: self._last_delta_hat, self._last_p_ctr, _last_p_cvr, _last_p_ctcvr.
        """
        idx = sparse_x.long() + self.field_offsets
        e = self.unified_emb(idx)
        x = torch.cat([e.flatten(1), dense_x], dim=1)
        logit_ctr = self.ctr_tower(x).squeeze(1) + self.wide_ctr(x).squeeze(1)
        logit_cvr = self.cvr_tower(x).squeeze(1) + self.wide_cvr(x).squeeze(1)
        eps = 1e-7
        p_ctr = torch.sigmoid(logit_ctr).clamp(eps, 1 - eps)
        p_cvr = torch.sigmoid(logit_cvr).clamp(eps, 1 - eps)
        p_ctcvr = (p_ctr * p_cvr).clamp(eps, 1 - eps)
        # Imputation tower: δ̂ ∈ ℝ (predicts BCE loss magnitude; no sigmoid)
        self._last_delta_hat = self.imp_tower(x)
        # Stash for compute_escm2_loss (avoids re-forward)
        self._last_p_ctr = p_ctr
        self._last_p_cvr = p_cvr
        self._last_p_ctcvr = p_ctcvr
        return p_ctr, p_cvr, p_ctcvr


# ---------------------------------------------------------------------------
# EGEAN (Zhang et al., WWW 2025, arXiv 2412.06852)
# ---------------------------------------------------------------------------
# Exposure-Guided Embedding Alignment Network for CVR estimation.
# Adds over ESMMModel_Wide:
#   - LoRA adapters on shared embedding (separate per CTR/CVR task, rank r)
#   - EPNet: task-personalized embedding gate with TWO stop-gradient points
#   - PPNet: per-layer gate in CTR/CVR towers (also consuming detached EPNet output)
#   - Exposure MLP head (in-batch negative BCE; all rows in batch = positives)
#   - Imputation tower (_ImpTower, same as ESCM²-DR) predicts CVR loss error
#   - PVDR estimator (Eq. 12): ratio-of-batch-sums (numerator / denominator)
#   - MMD² metric loss (Eq. 6): RBF kernel, median bandwidth heuristic
#
# forward()  → (p_ctr, p_cvr, p_ctcvr)  — side-effects: self._last_*
# compute_egean_loss(y_click, y_purchase, global_step) → full EGEAN objective.
#
# Resolved ambiguities (see egean-formula-verification.md §10):
#   A1: L̂ = MSE(imp_output, BCE(r, r̂)) over click-space 𝒪
#   A2: all loss weights = 1.0 (paper unspecified); exposed as kwargs
#   A3: RBF kernel, median-heuristic bandwidth
#   A4: lambda_pvdr fixed hyperparameter, default 1.0 (StableDR collapse)
#   A5: lora_rank=8 default
#   A6: single-phase training by default (pretrain_steps=0)
#
# Differences vs. paper (see egean-formula-verification.md §8):
#   D1: embed_dim=18 (paper: 5); D2: Wide backbone; D3: 360→200→80 towers;
#   D4: batch 4096; D5: 5 epochs; D7: weight_decay=0 default (paper 1e-3);
#   D8: lora_rank=8 default; D13: N(0,0.01) unified embedding; D15: task index emb.


def _mmd_rbf(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """MMD² with RBF kernel and median-heuristic bandwidth.

    x, y: (B, d) tensors. Returns scalar >= 0.
    Bandwidth σ² = median pairwise distance² over pooled sample (AMBIGUITY A3).
    """
    # Pooled pairwise squared distances for bandwidth estimation
    xy = torch.cat([x, y], dim=0)                          # (2B, d)
    dists_sq = torch.cdist(xy, xy, p=2).pow(2)             # (2B, 2B)
    # Median of upper-triangle (no self-distances)
    n = xy.size(0)
    upper_mask = torch.ones(n, n, dtype=torch.bool, device=x.device).triu(diagonal=1)
    bw = dists_sq[upper_mask].median().clamp(min=1e-6)     # σ²

    def _rbf_gram(a, b):
        d2 = torch.cdist(a, b, p=2).pow(2)
        return torch.exp(-d2 / (2.0 * bw))

    kxx = _rbf_gram(x, x).mean()
    kyy = _rbf_gram(y, y).mean()
    kxy = _rbf_gram(x, y).mean()
    return (kxx + kyy - 2.0 * kxy).clamp(min=0.0)


class _EPNet(nn.Module):
    """EPNet gate unit U_ep: (task_emb ⊕ shared_emb) → δ_task ∈ ℝ^emb_dim (Eq. 2).

    task_dim: dimensionality of task embedding (same as embed_dim here; one per task).
    emb_dim:  dimensionality of shared / LoRA embedding.
    """

    def __init__(self, task_dim: int, emb_dim: int) -> None:
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(task_dim + emb_dim, emb_dim),
            nn.LeakyReLU(0.2),
            nn.Sigmoid(),          # output ∈ (0,1) per element → scaling vector δ_task
        )
        for m in self.gate.modules():
            if isinstance(m, nn.Linear):
                _init_linear(m)

    def forward(self, task_emb: torch.Tensor, shared_emb_detached: torch.Tensor) -> torch.Tensor:
        """Returns δ_task (B, emb_dim). shared_emb MUST already be detached by caller."""
        return self.gate(torch.cat([task_emb, shared_emb_detached], dim=-1))


class _PPNetLayer(nn.Module):
    """Single PPNet-gated MLP layer: H^{l+1} = f( (δ_task ⊗ H^l) · W + b ) (Eq. 4-5).

    Uses δ_task from EPNet output (detached before entering this module; caller's job).
    delta_task may have a different dimension than h (flat_emb_dim vs layer in_dim),
    so a per-layer gate projection aligns them: gate_proj: delta_dim → in_dim.
    in_dim → out_dim, activation: LeakyReLU(0.2) to match paper.
    """

    def __init__(self, in_dim: int, out_dim: int, delta_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.act = nn.LeakyReLU(0.2)
        # Project delta_task to in_dim so element-wise product aligns
        self.gate_proj = nn.Linear(delta_dim, in_dim, bias=False)
        _init_linear(self.linear)
        nn.init.xavier_uniform_(self.gate_proj.weight)  # A8: xavier_uniform for gate_proj (was ones_)

    def forward(self, h: torch.Tensor, delta_task: torch.Tensor) -> torch.Tensor:
        """h: (B, in_dim), delta_task: (B, delta_dim) — already detached. Returns (B, out_dim)."""
        gate = self.gate_proj(delta_task)      # (B, in_dim)
        return self.act(self.linear(h * gate))


class ESMM_EGEAN(ESMMModel_Wide):
    """EGEAN: Exposure-Guided Embedding Alignment Network (Zhang et al., WWW 2025).

    Paper: arXiv 2412.06852. Spec: egean-formula-verification.md.
    Backbone: ESMMModel_Wide (unified embedding N(0,0.01), Wide head, 360→200→80 towers).

    Architecture additions vs. backbone
    ------------------------------------
    - LoRA adapters on shared embedding per task (rank lora_rank, default 8; A5)
    - EPNet per task: task_emb ⊕ ∅(shared_emb) → δ_task (stop-grad on shared; Trick 2a)
    - PPNet-gated tower layers: δ_task ⊗ H^l at each DNN layer (detach O_ep; Trick 2b)
    - Exposure MLP head → in-batch negative BCE (all rows = positives)
    - Imputation tower (same as ESCM²-DR _ImpTower) → ê estimate
    - PVDR ratio-of-batch-sums estimator (Eq. 12; λ fixed; A4)
    - MMD² metric loss over CVR embedding vs shared embedding (Eq. 6; RBF; A3)

    Constructor kwargs
    ------------------
    lora_rank        : LoRA adapter rank r (default 8; A5).
    lambda_pvdr      : λ in PVDR denominator (default 1.0 = StableDR collapse; A4).
    mmd_weight       : weight on L_metric (default 1.0; A2).
    pretrain_steps   : steps before LoRA/EPNet/PPNet/exposure are active (default 0; A6).
    ctr_weight       : weight on L_CTR (default 1.0; A2).
    pvdr_weight      : weight on L_PVDR (default 1.0; A2).
    imp_weight       : weight on L̂ (default 1.0; A2).
    exp_weight       : weight on L_exp (default 1.0; A2).
    ips_clip_floor   : propensity floor for PVDR (default 0.01).

    forward() returns (p_ctr, p_cvr, p_ctcvr) — same contract as ESMMModel_Wide.
    Side-effects: self._last_{delta_hat, p_ctr, p_cvr, p_ctcvr, O_ep_cvr, E_shared_flat}.
    compute_egean_loss(y_click, y_purchase, global_step) → EGEAN multi-task loss.
    """

    def __init__(
        self,
        field_cardinalities,
        num_dense: int,
        embed_dim: int = DEFAULT_EMBED_DIM,
        lora_rank: int = 8,
        lambda_pvdr: float = 1.0,
        mmd_weight: float = 1.0,
        pretrain_steps: int = 0,
        ctr_weight: float = 1.0,
        pvdr_weight: float = 1.0,
        imp_weight: float = 1.0,
        exp_weight: float = 1.0,
        ips_clip_floor: float = 0.01,
        **kw,
    ):
        super().__init__(field_cardinalities, num_dense, embed_dim=embed_dim, **kw)

        self.lora_rank = int(lora_rank)
        self.lambda_pvdr = float(lambda_pvdr)
        self.mmd_weight = float(mmd_weight)
        self.pretrain_steps = int(pretrain_steps)
        self.ctr_weight = float(ctr_weight)
        self.pvdr_weight = float(pvdr_weight)
        self.imp_weight = float(imp_weight)
        self.exp_weight = float(exp_weight)
        self.ips_clip_floor = float(ips_clip_floor)

        n_fields = self.num_fields
        flat_emb_dim = n_fields * embed_dim
        input_dim = flat_emb_dim + num_dense

        # Task embeddings (one-hot style; CTR=index 0, CVR=index 1; D15)
        self.task_emb = nn.Embedding(2, embed_dim)
        nn.init.normal_(self.task_emb.weight, 0.0, 0.01)

        # LoRA adapters per task: ΔW = B·A  (A: emb→rank, B: rank→emb)
        # Applied per-sample as: E_task_lora = E_shared + (E_shared @ A^T) @ B^T
        # We store flat (B, flat_emb_dim) LoRA; apply via linear projections on flat emb.
        r = self.lora_rank
        self.lora_ctr_A = nn.Linear(flat_emb_dim, r, bias=False)
        self.lora_ctr_B = nn.Linear(r, flat_emb_dim, bias=False)
        self.lora_cvr_A = nn.Linear(flat_emb_dim, r, bias=False)
        self.lora_cvr_B = nn.Linear(r, flat_emb_dim, bias=False)
        # Init LoRA: A ~ N(0, 0.01), B = 0 (so ΔW = 0 at init)
        for lora_A in (self.lora_ctr_A, self.lora_cvr_A):
            nn.init.normal_(lora_A.weight, 0.0, 0.01)
        for lora_B in (self.lora_ctr_B, self.lora_cvr_B):
            nn.init.zeros_(lora_B.weight)

        # EPNet gates: (task_emb ⊕ ∅(shared_emb)) → δ_task ∈ ℝ^flat_emb_dim
        self.epnet_ctr = _EPNet(embed_dim, flat_emb_dim)
        self.epnet_cvr = _EPNet(embed_dim, flat_emb_dim)

        # PPNet-gated towers replace plain ctr_tower / cvr_tower.
        # Tower dims: input_dim → 360 → 200 → 80 → 1.
        # Each hidden layer is a _PPNetLayer (gated by δ_task, dim=flat_emb_dim).
        # The final projection (80→1) is a plain linear (no gating needed).
        # delta_dim = flat_emb_dim (O_ep_task is the gate vector, shape flat_emb_dim)
        tower_dims = (360, 200, 80)
        self.ppnet_ctr_layers = nn.ModuleList()
        self.ppnet_cvr_layers = nn.ModuleList()
        prev = input_dim
        for h_dim in tower_dims:
            self.ppnet_ctr_layers.append(_PPNetLayer(prev, h_dim, delta_dim=flat_emb_dim))
            self.ppnet_cvr_layers.append(_PPNetLayer(prev, h_dim, delta_dim=flat_emb_dim))
            prev = h_dim
        self.ppnet_ctr_head = nn.Linear(prev, 1)
        self.ppnet_cvr_head = nn.Linear(prev, 1)
        _init_linear(self.ppnet_ctr_head)
        _init_linear(self.ppnet_cvr_head)

        # Exposure MLP: flat_emb_dim + num_dense → 1 (Eq. 1)
        self.exposure_mlp = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.LeakyReLU(0.2),
            nn.Linear(input_dim // 2, 1),
        )
        for m in self.exposure_mlp.modules():
            if isinstance(m, nn.Linear):
                _init_linear(m)

        # Imputation tower (same _ImpTower as ESCM²-DR)
        self.imp_tower = _ImpTower(input_dim)

        # Stash tensors from forward() for compute_egean_loss()
        self._last_delta_hat: torch.Tensor = torch.zeros(1)
        self._last_p_exp: torch.Tensor = torch.zeros(1)
        self._last_p_exp_neg: torch.Tensor = torch.zeros(1)   # A7: in-batch negative exposure probs
        self._last_O_ep_cvr: torch.Tensor = torch.zeros(1)
        self._last_E_shared_flat: torch.Tensor = torch.zeros(1)

    def _ppnet_forward(
        self,
        x0: torch.Tensor,
        layers: nn.ModuleList,
        head: nn.Linear,
        delta_task_detached: torch.Tensor,
    ) -> torch.Tensor:
        """Run PPNet-gated tower: x0 → (gated layers) → head → logit.

        delta_task_detached: (B, flat_emb_dim) — MUST be detached by caller (Trick 2b).
        Each _PPNetLayer has an internal gate_proj that projects delta to layer in_dim.
        Returns (B,) logit.
        """
        h = x0
        for layer in layers:
            h = layer(h, delta_task_detached)
        return head(h).squeeze(1)

    def forward(self, sparse_x, dense_x):
        """EGEAN forward pass.

        Returns (p_ctr, p_cvr, p_ctcvr).
        Side-effects:
          self._last_p_ctr, _last_p_cvr, _last_p_ctcvr
          self._last_delta_hat  (B,)  imputation estimate
          self._last_p_exp      (B,)  exposure probability
          self._last_O_ep_cvr   (B, flat_emb_dim)  CVR personalized emb (for MMD)
          self._last_E_shared_flat (B, flat_emb_dim)  shared emb (for MMD)
        """
        eps = 1e-7

        # Shared embedding lookup
        idx = sparse_x.long() + self.field_offsets
        E = self.unified_emb(idx)                          # (B, n_fields, embed_dim)
        E_flat = E.flatten(1)                              # (B, flat_emb_dim)
        x_raw = torch.cat([E_flat, dense_x], dim=1)       # (B, input_dim)

        # LoRA-adapted embeddings per task
        E_ctr_lora = E_flat + self.lora_ctr_B(self.lora_ctr_A(E_flat))   # (B, flat_emb_dim)
        E_cvr_lora = E_flat + self.lora_cvr_B(self.lora_cvr_A(E_flat))

        # Exposure task (runs over all rows; Eq. 1, §2.1.1)
        # Positives: real batch rows (all exposed items).
        # In-batch negatives (A7): shuffle x_raw features so the MLP sees random
        # feature combinations that were NOT the real exposure context.
        self._last_p_exp = torch.sigmoid(
            self.exposure_mlp(x_raw).squeeze(1)
        ).clamp(eps, 1 - eps)
        # Shuffle rows to form negatives — each shuffled row is a random (user, item) pair
        # drawn from the same batch distribution, serving as a proxy non-exposed sample.
        _neg_perm = torch.randperm(x_raw.size(0), device=x_raw.device)
        x_raw_shuffled = x_raw[_neg_perm]
        self._last_p_exp_neg = torch.sigmoid(
            self.exposure_mlp(x_raw_shuffled).squeeze(1)
        ).clamp(eps, 1 - eps)

        # Task embeddings (same for all rows in batch)
        B = E_flat.size(0)
        task_idx_ctr = torch.zeros(B, dtype=torch.long, device=E_flat.device)
        task_idx_cvr = torch.ones(B, dtype=torch.long, device=E_flat.device)
        t_ctr = self.task_emb(task_idx_ctr)               # (B, embed_dim)
        t_cvr = self.task_emb(task_idx_cvr)

        # EPNet: δ_task = U_ep(task_emb ⊕ ∅(E_lora))  (Trick 2a: stop-grad on shared emb)
        delta_ctr = self.epnet_ctr(t_ctr, E_ctr_lora.detach())    # (B, flat_emb_dim)
        delta_cvr = self.epnet_cvr(t_cvr, E_cvr_lora.detach())

        # Eq. 3: O_ep = δ_task ⊗ E_lora (personalized embedding)
        O_ep_ctr = delta_ctr * E_ctr_lora                 # (B, flat_emb_dim)
        O_ep_cvr = delta_cvr * E_cvr_lora

        # PPNet forward with detached O_ep as gate (Trick 2b: stop-grad on EPNet output)
        x_ctr = torch.cat([O_ep_ctr, dense_x], dim=1)     # (B, input_dim)
        x_cvr = torch.cat([O_ep_cvr, dense_x], dim=1)
        delta_ctr_det = O_ep_ctr.detach()                 # ∅(O_ep) — gate for PPNet
        delta_cvr_det = O_ep_cvr.detach()

        logit_ctr = self._ppnet_forward(
            x_ctr, self.ppnet_ctr_layers, self.ppnet_ctr_head, delta_ctr_det
        ) + self.wide_ctr(x_raw).squeeze(1)
        logit_cvr = self._ppnet_forward(
            x_cvr, self.ppnet_cvr_layers, self.ppnet_cvr_head, delta_cvr_det
        ) + self.wide_cvr(x_raw).squeeze(1)

        p_ctr = torch.sigmoid(logit_ctr).clamp(eps, 1 - eps)
        p_cvr = torch.sigmoid(logit_cvr).clamp(eps, 1 - eps)
        p_ctcvr = (p_ctr * p_cvr).clamp(eps, 1 - eps)

        # Imputation tower
        self._last_delta_hat = self.imp_tower(x_raw)       # (B,)

        # Stash for compute_egean_loss
        self._last_p_ctr = p_ctr
        self._last_p_cvr = p_cvr
        self._last_p_ctcvr = p_ctcvr
        self._last_O_ep_cvr = O_ep_cvr                     # for MMD²
        self._last_E_shared_flat = E_flat                  # for MMD²

        return p_ctr, p_cvr, p_ctcvr

    def compute_egean_loss(
        self,
        y_click: torch.Tensor,
        y_purchase: torch.Tensor,
        global_step: int = 0,
    ) -> torch.Tensor:
        """Full EGEAN multi-task loss.

        Must be called AFTER forward() for the same batch.

        L_total = ctr_weight * L_CTR
                + pvdr_weight * L_PVDR   (Eq. 12, ratio-of-batch-sums)
                + imp_weight  * L̂       (A1: MSE over click space)
                + mmd_weight  * L_metric (Eq. 6, MMD² RBF)
                + exp_weight  * L_exp    (Eq. 1, in-batch neg BCE)

        Arguments
        ---------
        y_click    : (B,) binary click indicator o ∈ {0, 1}
        y_purchase : (B,) binary purchase indicator r ∈ {0, 1}  (over 𝒟)
        global_step: current optimizer step (unused for EGEAN; kept for API parity)
        """
        eps = 1e-7
        p_ctr = self._last_p_ctr
        p_cvr = self._last_p_cvr
        yc = y_click.float()
        yp = y_purchase.float()

        # L_CTR: BCE(p_ctr, o) over 𝒟
        l_ctr = F.binary_cross_entropy(p_ctr.clamp(eps, 1 - eps), yc)

        # L_exp: exposure BCE with proper in-batch sampled negatives (Eq. 1, §2.1.1).
        # All rows in the batch are positives (label=1, they were exposed/clicked).
        # In-batch negatives: shuffled versions of the same batch rows — exposure MLP
        # applied to shuffled features should predict LOW probability (label=0).
        # Resolution A7: positives = batch rows (label=1);
        #                negatives = shuffle-resampled batch rows (label=0).
        # This forces the exposure MLP to distinguish real exposed items from random
        # combinations, rather than the degenerate all-ones case which provides zero
        # gradient signal on the exposure head.
        p_exp = self._last_p_exp.clamp(eps, 1 - eps)
        # Stash shuffled-features exposure prob set by forward() when negatives enabled.
        # If _last_p_exp_neg size matches p_exp (i.e. forward() was called with a real
        # batch), use the proper positive + negative BCE.
        # If size doesn't match (e.g. test bypasses forward() or init placeholder),
        # fall back to positives-only (legacy) to avoid size mismatch errors.
        _exp_neg = getattr(self, '_last_p_exp_neg', None)
        if _exp_neg is not None and _exp_neg.shape == p_exp.shape:
            p_exp_neg = _exp_neg.clamp(eps, 1 - eps)
            # Positive term: BCE(p_exp, 1) per row
            l_exp_pos = F.binary_cross_entropy(p_exp, torch.ones_like(p_exp))
            # Negative term: BCE(p_exp_neg, 0) per row
            l_exp_neg = F.binary_cross_entropy(p_exp_neg, torch.zeros_like(p_exp_neg))
            l_exp = (l_exp_pos + l_exp_neg) * 0.5
        else:
            # Legacy fallback: positives only (original degenerate form); applies when
            # _last_p_exp_neg is the init placeholder (shape [1]) or forward() was bypassed.
            l_exp = F.binary_cross_entropy(p_exp, torch.ones_like(p_exp))

        # L̂: imputation loss = MSE(ê, δ(r, r̂)) over click space 𝒪 (A1)
        # δ(r, r̂) = BCE(p_cvr, yp) per sample — gradients flow into both CVR tower
        # (through delta_per_sample) AND into imp_tower (through delta_hat).
        # This is correct: the CVR tower is updated so its BCE loss is well-calibrated
        # (reducing the imputation error); the imp_tower tracks the CVR tower's loss.
        pv = p_cvr.clamp(eps, 1 - eps)
        delta_per_sample = F.binary_cross_entropy(pv, yp, reduction='none')  # (B,) — NOT detached
        delta_hat = self._last_delta_hat                                       # (B,)
        # MSE only over clicked rows (yc=1); mean over full batch (including zeros) stable
        l_imp = (yc * (delta_per_sample - delta_hat).pow(2)).mean()

        # L_PVDR (Eq. 12): ratio-of-batch-sums estimator
        # Propensity = detached p_ctr (CTR tower not updated via PVDR path)
        propensity = p_ctr.detach().clamp(min=self.ips_clip_floor)
        ips = yc / propensity                                                  # o / p̂ (B,)
        numerator = (yc * delta_hat / propensity).sum()                        # Σ o·ê/p̂
        denominator = (
            self.lambda_pvdr * float(yc.size(0))
            + (1.0 - self.lambda_pvdr) * ips.sum()
        )
        l_pvdr = numerator / denominator.clamp(min=1e-8)

        # L_metric: MMD²(CVR embedding, shared embedding) over click space 𝒪 (Eq. 6, §2.1.3).
        # A2-fix: Eq. 6 is defined over click space 𝒪 (paper §2.1.3: "within the click space").
        # Mask both embedding sets to clicked rows (yc>0.5) before _mmd_rbf.
        click_mask = yc > 0.5
        if click_mask.sum() >= 2:
            O_ep_cvr_click = self._last_O_ep_cvr[click_mask]
            E_shared_click = self._last_E_shared_flat[click_mask].detach()
            l_metric = _mmd_rbf(O_ep_cvr_click, E_shared_click)
        else:
            # Edge case: fewer than 2 clicked rows — skip MMD to avoid degenerate kernel.
            l_metric = self._last_O_ep_cvr.new_zeros(1).squeeze()

        return (
            self.ctr_weight  * l_ctr
            + self.pvdr_weight * l_pvdr
            + self.imp_weight  * l_imp
            + self.mmd_weight  * l_metric
            + self.exp_weight  * l_exp
        )


# ---------------------------------------------------------------------------
# DCMT (Zhu et al., ICDE 2023, arXiv:2302.06141)
# ---------------------------------------------------------------------------
# Direct Entire-Space Causal Multi-Task Framework for Post-Click CVR Estimation.
# Adds over ESMMModel_Wide:
#   - Counterfactual CVR tower (_CfCVRTower): shares deep params θ_d with factual CVR;
#     factual-specific and counterfactual-specific shallow heads on top.
#   - DCMT main loss: factual SNIPS-IPW (click space 𝒪) + counterfactual SNIPS-IPW (N*).
#   - Soft constraint L_cf = λ1 * |1 − (r̂ + r̂*)| over 𝒟.
#   - L2 regularization λ2 * ‖θ‖²_F.
#
# forward()  → (p_ctr, p_cvr, p_ctcvr)  — side-effects: self._last_p_cvr_cf
# compute_dcmt_loss(y_click, y_purchase, global_step) → full DCMT objective.
#
# Resolved ambiguities (see dcmt-formula-verification.md §6):
#   A1: propensity floor=1e-6 (SNIPS naturally caps variance)
#   A2: detach ô for propensity denominators (consistent with ESCM²-DR)
#   A3: SNIPS normalization per batch (Eq. 13)
#   A4: L1 absolute value for soft constraint (matches Eq. 9 notation)
#   A5: counterfactual tower r̂* discarded at inference; forward() returns factual only
#   A6: no warmup schedule by default
#
# Differences vs. paper (see dcmt-formula-verification.md §8):
#   D1: embed_dim=18 (paper: 32); D2: tower 360→200→80 (paper: 320-200-80)
#   D4: twin CVR tower (new); D9: λ2=0.0001 (paper); D10: propensity floor 1e-6


class ESMM_DCMT(ESMMModel_Wide):
    """DCMT: Direct Entire-Space Causal Multi-Task (Zhu et al., ICDE 2023).

    Paper: arXiv:2302.06141. Spec: dcmt-formula-verification.md.
    Backbone: ESMMModel_Wide (unified embedding, Wide head, 360→200→80 towers).

    Architecture additions vs. backbone
    ------------------------------------
    - Counterfactual CVR tower: separate head that shares the CVR MLP body (deep params θ_d)
      with the factual CVR tower. Additional counterfactual-specific head on top.
    - DCMT loss: factual SNIPS over 𝒪 + counterfactual SNIPS over N* + soft constraint.
    - L2 weight decay λ2 on all parameters.
    - Propensity detached from CTR tower (Ambiguity A2; consistent with ESCM²-DR Trick 2).

    Constructor kwargs
    ------------------
    lambda1   : soft constraint weight |1 - (r̂ + r̂*)| (paper best: 0.001; Fig 8c).
    lambda2   : L2 weight decay (paper: 0.0001).
    wcvr      : weight on E_DCMT (paper: 1.0, fixed — not tuned).
    wctcvr    : weight on E_CTCVR (paper: 1.0, fixed — not tuned).
    eps_prop  : propensity floor for SNIPS (default 1e-6; Ambiguity A1).

    forward() returns (p_ctr, p_cvr, p_ctcvr) — same contract as ESMMModel_Wide.
    Side-effect: self._last_p_cvr_cf (counterfactual CVR, with sigmoid; discarded at inference).
    compute_dcmt_loss(y_click, y_purchase, global_step) → DCMT multi-task loss.
    """

    def __init__(
        self,
        field_cardinalities,
        num_dense: int,
        embed_dim: int = DEFAULT_EMBED_DIM,
        lambda1: float = 0.001,
        lambda2: float = 0.0001,
        wcvr: float = 1.0,
        wctcvr: float = 1.0,
        eps_prop: float = 1e-6,
        **kw,
    ):
        super().__init__(field_cardinalities, num_dense, embed_dim=embed_dim, **kw)
        self.lambda1 = float(lambda1)
        self.lambda2 = float(lambda2)
        self.wcvr = float(wcvr)
        self.wctcvr = float(wctcvr)
        self.eps_prop = float(eps_prop)

        # Counterfactual CVR head: same input_dim as factual CVR tower.
        # Shares the deep backbone of the factual CVR tower (θ_d = cvr_tower params);
        # adds a separate shallow head on top (θ_cf^d = cvr_tower_cf_head).
        # Architecture: cvr_tower produces a 80-dim intermediate; the CF head maps 80→1.
        # Both heads share the 360→200→80 body (cvr_tower) — θ_d in paper notation.
        # Wide term: separate counterfactual wide linear (θ_cf^w).
        input_dim = self.num_fields * self.embed_dim + num_dense
        # Counterfactual shallow head (80→1; operates on cvr_tower's penultimate output)
        # cvr_tower.[-1] is the 80→1 head. We extract the 80-dim intermediate from the
        # shared body and pass it to both the factual head and the CF head.
        # To keep this lightweight, we re-use cvr_tower fully for the factual path and
        # add a separate _CfCVRHead (80→1) + wide_cvr_cf (linear) for the CF path.
        # The shared body (360→200→80 DNN) is identical by weight sharing.
        cvr_hidden_out = 80   # final hidden dim before the 80→1 output head

        # Counterfactual-specific components
        self.cvr_cf_head = nn.Linear(cvr_hidden_out, 1)
        self.wide_cvr_cf = nn.Linear(input_dim, 1)
        _init_linear(self.cvr_cf_head)
        _init_linear(self.wide_cvr_cf)

        # Stash for compute_dcmt_loss
        self._last_p_cvr_cf: torch.Tensor = torch.zeros(1)
        self._last_h_cvr: torch.Tensor = torch.zeros(1)   # penultimate CVR body output

    def forward(self, sparse_x, dense_x):
        """DCMT forward pass.

        Returns (p_ctr, p_cvr, p_ctcvr).
        Side-effects:
          self._last_p_ctr, _last_p_cvr, _last_p_ctcvr
          self._last_p_cvr_cf  (B,)  counterfactual CVR prediction r̂* ∈ (0,1)
          self._last_h_cvr     (B, 80) penultimate CVR body output (for CF head)
        """
        eps = 1e-7
        idx = sparse_x.long() + self.field_offsets
        e = self.unified_emb(idx)
        x = torch.cat([e.flatten(1), dense_x], dim=1)   # (B, input_dim)

        # CTR tower (standard)
        logit_ctr = self.ctr_tower(x).squeeze(1) + self.wide_ctr(x).squeeze(1)
        p_ctr = torch.sigmoid(logit_ctr).clamp(eps, 1 - eps)

        # CVR factual tower — extract penultimate output (80-dim) for CF head.
        # cvr_tower is a Sequential: [Linear(in, 360), ReLU, ..., Linear(200, 80), ReLU, Linear(80, 1)]
        # We run all layers EXCEPT the final 80→1 linear to get the shared body output.
        h_cvr = x
        for layer in self.cvr_tower[:-1]:    # all layers except final head
            h_cvr = layer(h_cvr)
        # h_cvr is now the 80-dim penultimate representation (shared θ_d)
        logit_cvr = self.cvr_tower[-1](h_cvr).squeeze(1) + self.wide_cvr(x).squeeze(1)
        p_cvr = torch.sigmoid(logit_cvr).clamp(eps, 1 - eps)

        # Counterfactual CVR head: uses same h_cvr (shared θ_d), separate head θ_cf^d
        logit_cvr_cf = self.cvr_cf_head(h_cvr).squeeze(1) + self.wide_cvr_cf(x).squeeze(1)
        p_cvr_cf = torch.sigmoid(logit_cvr_cf).clamp(eps, 1 - eps)

        p_ctcvr = (p_ctr * p_cvr).clamp(eps, 1 - eps)

        # Stash for compute_dcmt_loss
        self._last_p_ctr = p_ctr
        self._last_p_cvr = p_cvr
        self._last_p_ctcvr = p_ctcvr
        self._last_p_cvr_cf = p_cvr_cf
        self._last_h_cvr = h_cvr

        return p_ctr, p_cvr, p_ctcvr

    def compute_dcmt_loss(
        self,
        y_click: torch.Tensor,
        y_purchase: torch.Tensor,
        global_step: int = 0,
    ) -> torch.Tensor:
        """Full DCMT multi-task loss (Eq. 14 = L_CTR + wcvr*E_DCMT + wctcvr*L_CTCVR + l2).

        Must be called AFTER forward() for the same batch.

        E_DCMT = E_DCMT_main + L_cf
          E_DCMT_main = (SNIPS_factual + SNIPS_cf) / B  (Eq. 8 + 13)
          L_cf = lambda1 * |1 − (r̂ + r̂*)| over 𝒟     (Eq. 9)

        SNIPS_factual: Σ_{𝒪} e(r, r̂)/ô / Σ_{𝒪} (1/ô)   — per batch (Eq. 13)
        SNIPS_cf:      Σ_{N*} e(r*, r̂*)/(1−ô) / Σ_{N*} (1/(1−ô))

        where r* = 1 − r (counterfactual label flip; Trick 1).

        Arguments
        ---------
        y_click    : (B,) binary click indicator o ∈ {0, 1}
        y_purchase : (B,) binary purchase indicator r ∈ {0, 1}  (over 𝒟)
        global_step: current optimizer step (unused; kept for API parity)
        """
        eps = 1e-7
        p_ctr    = self._last_p_ctr
        p_cvr    = self._last_p_cvr
        p_ctcvr  = self._last_p_ctcvr
        p_cvr_cf = self._last_p_cvr_cf

        yc = y_click.float()
        yp = y_purchase.float()
        B  = float(yc.size(0))

        # --- Standard backbone losses (Eq. 15) ---
        l_ctr   = F.binary_cross_entropy(p_ctr.clamp(eps, 1 - eps), yc)
        l_ctcvr = F.binary_cross_entropy(
            p_ctcvr.clamp(eps, 1 - eps),
            (yc * yp).clamp(0.0, 1.0),
        )

        # --- Detach propensity from CTR tower (Ambiguity A2) ---
        ô = p_ctr.detach().clamp(self.eps_prop, 1.0 - self.eps_prop)

        # Counterfactual labels: r* = 1 − r (Trick 1)
        yp_cf = 1.0 - yp

        # --- Per-sample BCE losses ---
        e_f  = F.binary_cross_entropy(p_cvr.clamp(eps, 1 - eps),    yp,    reduction='none')
        e_cf = F.binary_cross_entropy(p_cvr_cf.clamp(eps, 1 - eps), yp_cf, reduction='none')

        # --- SNIPS factual: Σ_{𝒪} e_f/ô / Σ_{𝒪} 1/ô  (Eq. 13) ---
        click_mask   = yc > 0.5
        nonclick_mask = ~click_mask
        inv_o = 1.0 / ô                        # (B,) = 1/ô per row

        if click_mask.any():
            snips_num_f  = (e_f  * yc  * inv_o).sum()
            snips_den_f  = inv_o[click_mask].sum().clamp(min=eps)
            snips_f      = snips_num_f / snips_den_f
        else:
            snips_f = p_cvr.new_zeros(1).squeeze()

        # --- SNIPS counterfactual: Σ_{N*} e_cf/(1−ô) / Σ_{N*} 1/(1−ô) ---
        inv_1mo = 1.0 / (1.0 - ô)             # (B,) = 1/(1−ô) per row
        if nonclick_mask.any():
            snips_num_cf = (e_cf * (1.0 - yc) * inv_1mo).sum()
            snips_den_cf = inv_1mo[nonclick_mask].sum().clamp(min=eps)
            snips_cf     = snips_num_cf / snips_den_cf
        else:
            snips_cf = p_cvr_cf.new_zeros(1).squeeze()

        # --- E_DCMT_main (Eq. 8 + 13): SNIPS terms are already self-normalised means ---
        e_dcmt_main = snips_f + snips_cf

        # --- Soft constraint L_cf (Eq. 9): L1 |1 − (r̂ + r̂*)| over 𝒟 (A4) ---
        l_cf = self.lambda1 * (1.0 - (p_cvr + p_cvr_cf)).abs().mean()

        e_dcmt = e_dcmt_main + l_cf

        # --- L2 regularization (Eq. 14) ---
        l2 = sum(p.pow(2).sum() for p in self.parameters()) * self.lambda2

        # --- Full objective (Eq. 14) ---
        return l_ctr + self.wcvr * e_dcmt + self.wctcvr * l_ctcvr + l2