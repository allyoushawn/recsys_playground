"""Ali-CCP multi-task-learning (MTL) model architectures.

Cloned from `experiments/20260404_ali_cpp_esmm/esmm_ali_ccp_impl.py` (lines 178-733
of that file, as of the 20260609 AdaOrder round). This is an additive clone for a
public blog article's code links — it intentionally covers only the "classic" MTL
family used in that article:

  ESMMModel, ESMMModel_Wide                                  — two-tower ESMM (+ wide)
  ESMM_SharedBottom, ESMM_SharedBottomWide                   — shared-bottom MTL (+ wide)
  ESMM_MMoE, ESMM_MMoE_Wide                                  — MMoE (+ wide)
  ESMM_PLE, ESMM_PLE_Wide, ESMM_PLE_Cross, ESMM_PLE_WideCross — faithful PLE (+ wide/cross)

plus their shared helper classes (_ESMMExpertMLP, _ESMMGate, _ESMM_PLELevel,
_ESMM_PLETower, _ESMMCrossNet) and the _init_linear init helper.

OUT OF SCOPE (intentionally NOT included): the NDM/ESCM2/EGEAN/DCMT/AdaOrderCross/
TaskCross/EPNetGate family (ESMM_PLE_AdaOrderCross onward in the source file) — a
separate, later study not part of this article.
"""
import torch
import torch.nn as nn
from contextlib import nullcontext


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
