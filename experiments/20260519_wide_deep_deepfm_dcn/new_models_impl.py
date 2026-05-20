"""
Classic recsys ranking models — AliCCP ESMM-compatible implementations.

All models output (p_ctr, p_cvr, p_ctcvr) to plug into the existing
train_esmm_parquet_rowgroups / evaluate_esmm_multitask_streaming_parquet pipeline.

Models:
  WideAndDeepModel  — Cheng et al., Google 2016
  DeepFMModel       — Guo et al., HuaWei 2017
  DCNv2Model        — Wang et al., Google 2021 (parallel structure)
"""

import torch
import torch.nn as nn

_CLAMP_EPS = 1e-7


def _make_mlp(in_dim, hidden_dims, out_dim, dropout=0.0):
    layers = []
    prev = in_dim
    for h in hidden_dims:
        layers += [nn.Linear(prev, h), nn.ReLU()]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        prev = h
    layers.append(nn.Linear(prev, out_dim))
    return nn.Sequential(*layers)


def _kaiming_init(module):
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)


def _build_offsets(field_cardinalities):
    offsets = [0]
    for c in field_cardinalities[:-1]:
        offsets.append(offsets[-1] + c)
    return torch.tensor(offsets, dtype=torch.long)


# ---------------------------------------------------------------------------
# Wide & Deep  (Cheng et al., 2016)
# ---------------------------------------------------------------------------
class WideAndDeepModel(nn.Module):
    """
    Wide & Deep Learning for Recommender Systems (Google, KDD 2016).

    Architecture
    ============
    Wide part (memorization): linear layer on the full feature vector.
    Deep part (generalization): MLP on the same feature vector.

    Both wide and deep components have separate CTR and CVR heads.
    p_ctcvr = p_ctr * p_cvr  (ESMM entire-space constraint).

    Key insight: wide component memorizes co-occurrence patterns;
    deep component generalises to unseen feature combinations.
    """

    def __init__(self, field_cardinalities, num_dense, embed_dim=18,
                 deep_dims=(360, 200, 80), dropout=0.0, **kwargs):
        super().__init__()
        total_vocab = sum(field_cardinalities)
        self.register_buffer('_offsets', _build_offsets(field_cardinalities))

        self.embedding = nn.Embedding(total_vocab, embed_dim, padding_idx=0)
        in_dim = len(field_cardinalities) * embed_dim + num_dense

        # Wide: separate linear for CTR and CVR
        self.wide_ctr = nn.Linear(in_dim, 1)
        self.wide_cvr = nn.Linear(in_dim, 1)

        # Deep: shared backbone, separate output heads
        self.deep_backbone = _make_mlp(in_dim, deep_dims[:-1], deep_dims[-1], dropout)
        self.deep_ctr = nn.Linear(deep_dims[-1], 1)
        self.deep_cvr = nn.Linear(deep_dims[-1], 1)

        _kaiming_init(self)
        nn.init.normal_(self.embedding.weight, 0, 0.01)

    def _embed(self, sparse_x, dense_x):
        ids = sparse_x.long() + self._offsets.unsqueeze(0)
        emb = self.embedding(ids).flatten(1)           # [B, F*E]
        return torch.cat([emb, dense_x], dim=1)        # [B, F*E+D]

    def forward(self, sparse_x, dense_x):
        x = self._embed(sparse_x, dense_x)
        deep_h = self.deep_backbone(x)
        logit_ctr = self.wide_ctr(x).squeeze(1) + self.deep_ctr(deep_h).squeeze(1)
        logit_cvr = self.wide_cvr(x).squeeze(1) + self.deep_cvr(deep_h).squeeze(1)
        p_ctr = torch.sigmoid(logit_ctr).clamp(_CLAMP_EPS, 1 - _CLAMP_EPS)
        p_cvr = torch.sigmoid(logit_cvr).clamp(_CLAMP_EPS, 1 - _CLAMP_EPS)
        return p_ctr, p_cvr, (p_ctr * p_cvr).clamp(_CLAMP_EPS, 1 - _CLAMP_EPS)


# ---------------------------------------------------------------------------
# DeepFM  (Guo et al., 2017)
# ---------------------------------------------------------------------------
class DeepFMModel(nn.Module):
    """
    DeepFM: A Factorization-Machine based Neural Network for CTR Prediction
    (HuaWei, IJCAI 2017).

    Architecture
    ============
    FM component: captures pairwise feature interactions implicitly via
        second-order FM: 0.5 * (||Σ v_i||^2 - Σ ||v_i||^2)
    DNN component: captures high-order interactions via MLP.
    CTR logit = first-order bias + FM scalar + DNN scalar.

    CVR uses a lightweight DNN without FM (clicked-space signal is sparser,
    so FM interactions are noisier; simpler DNN generalises better).

    Key insight: eliminates manual feature engineering — FM automatically
    learns all pairwise interactions from raw sparse IDs.
    """

    def __init__(self, field_cardinalities, num_dense, embed_dim=18,
                 dnn_dims=(360, 200, 80), dropout=0.0, **kwargs):
        super().__init__()
        total_vocab = sum(field_cardinalities)
        self.register_buffer('_offsets', _build_offsets(field_cardinalities))

        # FM / DNN shared embeddings
        self.embedding = nn.Embedding(total_vocab, embed_dim, padding_idx=0)
        # First-order (per-token scalar bias)
        self.first_order = nn.Embedding(total_vocab, 1, padding_idx=0)
        self.global_bias = nn.Parameter(torch.zeros(1))

        in_dim = len(field_cardinalities) * embed_dim + num_dense

        # CTR: FM + DNN head
        self.dnn_ctr = _make_mlp(in_dim, dnn_dims[:-1], dnn_dims[-1], dropout)
        self.ctr_head = nn.Linear(dnn_dims[-1], 1)

        # CVR: separate DNN (no FM — cleaner signal on clicked subspace)
        self.dnn_cvr = _make_mlp(in_dim, dnn_dims[:-1], dnn_dims[-1], dropout)
        self.cvr_head = nn.Linear(dnn_dims[-1], 1)

        _kaiming_init(self)
        nn.init.normal_(self.embedding.weight, 0, 0.01)
        nn.init.zeros_(self.first_order.weight)

    def _fm_second_order(self, emb):
        # emb: [B, F, E]
        sum_of_sq = emb.sum(dim=1).pow(2)       # [B, E]
        sq_of_sum = emb.pow(2).sum(dim=1)        # [B, E]
        return 0.5 * (sum_of_sq - sq_of_sum).sum(dim=1)  # [B]

    def forward(self, sparse_x, dense_x):
        ids = sparse_x.long() + self._offsets.unsqueeze(0)
        emb = self.embedding(ids)                          # [B, F, E]
        x_dnn = torch.cat([emb.flatten(1), dense_x], 1)   # [B, F*E+D]

        # CTR: first-order + FM + DNN
        first = self.first_order(ids).squeeze(-1).sum(1) + self.global_bias  # [B]
        fm = self._fm_second_order(emb)                                        # [B]
        dnn_h = self.dnn_ctr(x_dnn)
        logit_ctr = first + fm + self.ctr_head(dnn_h).squeeze(1)

        # CVR: DNN only
        cvr_h = self.dnn_cvr(x_dnn)
        logit_cvr = self.cvr_head(cvr_h).squeeze(1)

        p_ctr = torch.sigmoid(logit_ctr).clamp(_CLAMP_EPS, 1 - _CLAMP_EPS)
        p_cvr = torch.sigmoid(logit_cvr).clamp(_CLAMP_EPS, 1 - _CLAMP_EPS)
        return p_ctr, p_cvr, (p_ctr * p_cvr).clamp(_CLAMP_EPS, 1 - _CLAMP_EPS)


# ---------------------------------------------------------------------------
# DCN V2  (Wang et al., 2021)
# ---------------------------------------------------------------------------
class _CrossLayer(nn.Module):
    """x_{l+1} = x_0 ⊙ (W_l * x_l + b_l) + x_l"""
    def __init__(self, d):
        super().__init__()
        self.linear = nn.Linear(d, d)

    def forward(self, x0, x):
        return x0 * self.linear(x) + x


class DCNv2Model(nn.Module):
    """
    DCN V2: Improved Deep & Cross Network (Google, WWW 2021).

    Architecture (Parallel structure)
    ==================================
    Cross network: L cross layers, each learning x_{l+1} = x0 ⊙ (Wx_l + b) + x_l.
        Uses a full d×d weight matrix (vs DCN v1's vector), enabling bidirectional
        feature interaction: W_ij and W_ji are independent parameters.
    Deep network: standard MLP on the same input.
    Parallel: both networks process x_0 simultaneously; outputs are concatenated
        and fed to the prediction head (lower latency than stacked).

    CTR head: linear(concat(cross_out, deep_out)).
    CVR head: linear(deep_out) only — cross features are noisier for CVR.

    Key insight: automatically learns explicit high-order feature crosses
    without manual feature engineering, at lower cost than attention.
    """

    def __init__(self, field_cardinalities, num_dense, embed_dim=18,
                 num_cross_layers=3, deep_dims=(360, 200, 80), dropout=0.0, **kwargs):
        super().__init__()
        total_vocab = sum(field_cardinalities)
        self.register_buffer('_offsets', _build_offsets(field_cardinalities))

        self.embedding = nn.Embedding(total_vocab, embed_dim, padding_idx=0)
        d = len(field_cardinalities) * embed_dim + num_dense

        # Cross network
        self.cross = nn.ModuleList([_CrossLayer(d) for _ in range(num_cross_layers)])

        # Deep network
        self.deep = _make_mlp(d, deep_dims[:-1], deep_dims[-1], dropout)

        # CTR: cross_out (d) || deep_out (deep_dims[-1]) → 1
        self.ctr_head = nn.Linear(d + deep_dims[-1], 1)
        # CVR: deep_out only
        self.cvr_head = nn.Linear(deep_dims[-1], 1)

        _kaiming_init(self)
        nn.init.normal_(self.embedding.weight, 0, 0.01)

    def forward(self, sparse_x, dense_x):
        ids = sparse_x.long() + self._offsets.unsqueeze(0)
        emb = self.embedding(ids).flatten(1)               # [B, F*E]
        x0 = torch.cat([emb, dense_x], dim=1)             # [B, d]

        # Cross network (parallel to deep)
        xc = x0
        for layer in self.cross:
            xc = layer(x0, xc)

        # Deep network
        xd = self.deep(x0)

        # Heads
        logit_ctr = self.ctr_head(torch.cat([xc, xd], dim=1)).squeeze(1)
        logit_cvr = self.cvr_head(xd).squeeze(1)

        p_ctr = torch.sigmoid(logit_ctr).clamp(_CLAMP_EPS, 1 - _CLAMP_EPS)
        p_cvr = torch.sigmoid(logit_cvr).clamp(_CLAMP_EPS, 1 - _CLAMP_EPS)
        return p_ctr, p_cvr, (p_ctr * p_cvr).clamp(_CLAMP_EPS, 1 - _CLAMP_EPS)
