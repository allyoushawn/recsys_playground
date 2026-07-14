"""Wukong — stacked factorization-machine blocks for scalable feature interaction (Meta 2024).

Architecture: stacked layers, each consisting of a Factorization Machine Block (FMB) that
captures pairwise token interactions via low-rank projection, and a Linear Compress Block
(LCB) that carries lower-order/linear signal via a bottleneck projection. FMB + LCB outputs
are concatenated, projected, layer-normed + residual, and reshaped back to token form for
the next layer.

Simplification note (vs. the full Wukong paper):
    The paper uses full-rank FM interaction matrices and more complex compression schemes.
    Here, FMB uses the standard FM identity ((sum^2 - sum_of_squares)/2) on low-rank
    projections of the tokens, yielding a vector of interaction features. LCB flattens
    and compresses the tokens to mlp_compress dims then expands back. The stacking,
    FMB+LCB concatenation, and projection-back structure are preserved faithfully.

Tokenization:
    history interest and target item embeddings are each split into n_field_tokens sub-tokens
    by reshaping embed_dim -> (n_field_tokens, embed_dim // n_field_tokens), then projecting
    to token_dim. Yields T = 2 * n_field_tokens tokens.

Reference: "Wukong: Towards a Scaling Law for Large-Scale Recommendation" (Meta 2024).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..baselines import _SeqRankModel
from ..din import _mlp


class _FMBlock(nn.Module):
    """Factorization Machine Block: pairwise token interactions via low-rank FM.

    Projects tokens to fm_rank, then applies the standard FM identity
    (sum^2 - sum_of_squares) / 2 across tokens to produce an interaction vector.
    """

    def __init__(self, token_dim: int, n_tokens: int, fm_rank: int) -> None:
        super().__init__()
        # Project each token to fm_rank.
        self.proj = nn.Linear(token_dim, fm_rank, bias=False)
        # Map interaction result (fm_rank) to token_dim for residual.
        self.out_proj = nn.Linear(fm_rank, token_dim, bias=True)
        self.n_tokens = n_tokens
        self.fm_rank = fm_rank

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, token_dim]
        v = self.proj(x)  # [B, T, fm_rank]
        sum_v = v.sum(dim=1)  # [B, fm_rank]
        sum_v_sq = sum_v ** 2  # [B, fm_rank]
        sq_sum = (v ** 2).sum(dim=1)  # [B, fm_rank]
        interaction = 0.5 * (sum_v_sq - sq_sum)  # [B, fm_rank]
        return self.out_proj(interaction)  # [B, token_dim]


class _LCBlock(nn.Module):
    """Linear Compress Block: flattens tokens, compresses to mlp_compress dims, back out.

    Carries lower-order / linear signal alongside the FMB's pairwise interactions.
    """

    def __init__(self, token_dim: int, n_tokens: int, mlp_compress: int) -> None:
        super().__init__()
        flat_dim = token_dim * n_tokens
        self.compress = nn.Linear(flat_dim, mlp_compress, bias=True)
        self.expand = nn.Linear(mlp_compress, token_dim, bias=True)
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, token_dim]
        B = x.shape[0]
        flat = x.reshape(B, -1)  # [B, T * token_dim]
        return self.expand(self.act(self.compress(flat)))  # [B, token_dim]


class _WukongLayer(nn.Module):
    """One Wukong layer: FMB + LCB -> concat -> project -> LayerNorm + residual -> reshape."""

    def __init__(self, token_dim: int, n_tokens: int, fm_rank: int, mlp_compress: int, dropout: float) -> None:
        super().__init__()
        self.fmb = _FMBlock(token_dim, n_tokens, fm_rank)
        self.lcb = _LCBlock(token_dim, n_tokens, mlp_compress)
        # FMB and LCB each output [B, token_dim]; concat -> [B, 2*token_dim].
        # Project back to [B, T, token_dim] via a token_dim -> token_dim projection
        # that will be broadcast/replicated across T for the residual form.
        self.merge = nn.Linear(2 * token_dim, token_dim, bias=True)
        self.norm = nn.LayerNorm(token_dim)
        self.dropout = nn.Dropout(dropout)
        self.n_tokens = n_tokens

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, token_dim]
        fmb_out = self.fmb(x)  # [B, token_dim]  — pooled interaction signal
        lcb_out = self.lcb(x)  # [B, token_dim]  — linear compression signal
        combined = torch.cat([fmb_out, lcb_out], dim=-1)  # [B, 2*token_dim]
        delta = self.merge(combined)  # [B, token_dim]
        # Broadcast delta across the token axis and add as residual.
        delta = self.dropout(delta).unsqueeze(1)  # [B, 1, token_dim]
        out = self.norm(x + delta)  # [B, T, token_dim]
        return out


class Wukong(_SeqRankModel):
    """Wukong: stacked FM-block ranking model for scalable feature interaction.

    Each layer applies a Factorization Machine Block (explicit pairwise interaction
    via low-rank FM) and a Linear Compress Block (bottleneck linear signal), concatenates
    their outputs, projects back to token representations, and applies LayerNorm + residual.

    Simplification vs. paper: FMB uses the FM identity on low-rank projections rather than
    full-rank outer products; residual is broadcast (pooled delta added to all tokens)
    rather than per-token reconstruction. Core FMB+LCB+stack identity is preserved.

    Args:
        num_items:      Item vocabulary size.
        embed_dim:      Item embedding dimension. Must be divisible by n_field_tokens.
        n_field_tokens: Number of sub-tokens per field (interest and target).
        token_dim:      Common token dimension for Wukong layers.
        n_layers:       Number of Wukong layers.
        fm_rank:        Rank for the low-rank FM projection in FMB.
        mlp_compress:   Bottleneck dimension for LCB.
        dropout:        Dropout probability.
    """

    def __init__(
        self,
        num_items: int,
        embed_dim: int = 32,
        n_field_tokens: int = 4,
        token_dim: int = 32,
        n_layers: int = 2,
        fm_rank: int = 8,
        mlp_compress: int = 16,
        dropout: float = 0.0,
        **kwargs,
    ) -> None:
        super().__init__(num_items, embed_dim)
        if embed_dim % n_field_tokens != 0:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by n_field_tokens ({n_field_tokens})"
            )
        self.n_field_tokens = n_field_tokens
        self.token_dim = token_dim
        sub_dim = embed_dim // n_field_tokens
        n_tokens = 2 * n_field_tokens

        # Project each sub-token from sub_dim to token_dim.
        self.sub_proj = nn.Linear(sub_dim, token_dim, bias=True)

        self.layers = nn.ModuleList([
            _WukongLayer(token_dim, n_tokens, fm_rank, mlp_compress, dropout)
            for _ in range(n_layers)
        ])

        # Head: flatten final tokens -> scalar logit.
        self.head = _mlp(token_dim * n_tokens, (token_dim,), 1)

    def _to_tokens(self, field_vec: torch.Tensor) -> torch.Tensor:
        """Split [B, embed_dim] -> [B, n_field_tokens, token_dim]."""
        B = field_vec.shape[0]
        sub_dim = self.embed_dim // self.n_field_tokens
        sub = field_vec.view(B, self.n_field_tokens, sub_dim)
        return self.sub_proj(sub)

    def forward(
        self,
        hist_ids: torch.Tensor,
        hist_mask: torch.Tensor,
        target_ids: torch.Tensor,
    ) -> torch.Tensor:
        """forward(hist_ids [B,L], hist_mask [B,L], target_ids [B]) -> logits [B]."""
        interest = self._pool(hist_ids, hist_mask)  # [B, E]
        target = self.item_emb(target_ids)  # [B, E]

        # Build token set: [B, 2*n_field_tokens, token_dim]
        interest_tok = self._to_tokens(interest)  # [B, nft, token_dim]
        target_tok = self._to_tokens(target)      # [B, nft, token_dim]
        x = torch.cat([interest_tok, target_tok], dim=1)  # [B, T, token_dim]

        for layer in self.layers:
            x = layer(x)

        # Flatten final tokens -> logit.
        B = x.shape[0]
        flat = x.reshape(B, -1)  # [B, T * token_dim]
        return self.head(flat).squeeze(-1)  # [B]
