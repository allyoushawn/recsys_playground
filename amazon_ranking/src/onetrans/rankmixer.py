"""RankMixer — attention-free token-mixing ranking model (ByteDance 2024).

Architecture: stacked blocks of (a) multi-head token-mixing MLP and (b) per-token
feed-forward network, with no self-attention. Designed for explicit feature
interaction via learned token-mixing across a small set of field-derived tokens.

Tokenization:
    history interest and target item embeddings are each split into ``n_field_tokens``
    sub-tokens by reshaping embed_dim -> (n_field_tokens, embed_dim // n_field_tokens),
    then projecting each sub-token to ``token_dim``. This yields T = 2 * n_field_tokens
    tokens for the mixer blocks.

Reference: "RankMixer: Scaling Ads CTR Prediction Model via Token Mixing" (ByteDance 2024).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..baselines import _SeqRankModel
from ..din import _mlp


class _TokenMixingHead(nn.Module):
    """Multi-head token-mixing: mixes information across tokens within each head.

    For each head, the token axis (T) is mixed via a learned linear projection
    (transpose + Linear(T->T) per head), then heads are merged.
    """

    def __init__(self, token_dim: int, n_tokens: int, n_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        assert token_dim % n_heads == 0, f"token_dim {token_dim} must be divisible by n_heads {n_heads}"
        self.n_heads = n_heads
        self.head_dim = token_dim // n_heads
        # Per-head token-mixing: a linear that operates on the token axis (T -> T).
        # Implemented as a single Linear(T -> T * n_heads) applied after reshaping.
        self.token_mix = nn.Linear(n_tokens, n_tokens, bias=True)
        self.out_proj = nn.Linear(token_dim, token_dim, bias=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, token_dim]
        B, T, D = x.shape
        h = self.n_heads
        hd = self.head_dim
        # Reshape to [B, T, n_heads, head_dim] then transpose to [B, n_heads, head_dim, T]
        x_h = x.view(B, T, h, hd).permute(0, 2, 3, 1)  # [B, h, hd, T]
        # Mix across token axis (T -> T) per head: apply Linear to last dim (T)
        mixed = self.token_mix(x_h)  # [B, h, hd, T]
        # Back to [B, T, token_dim]
        mixed = mixed.permute(0, 3, 1, 2).contiguous().view(B, T, D)  # [B, T, D]
        return self.dropout(self.out_proj(mixed))


class _RankMixerBlock(nn.Module):
    """One RankMixer block: pre-norm token-mixing + pre-norm per-token FFN, both with residual."""

    def __init__(self, token_dim: int, n_tokens: int, n_heads: int, ff_mult: int, dropout: float) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(token_dim)
        self.token_mix = _TokenMixingHead(token_dim, n_tokens, n_heads, dropout)
        self.norm2 = nn.LayerNorm(token_dim)
        ff_dim = token_dim * ff_mult
        self.ffn = nn.Sequential(
            nn.Linear(token_dim, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, token_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, token_dim]
        x = x + self.token_mix(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class RankMixer(_SeqRankModel):
    """RankMixer: attention-free feature-interaction ranking model.

    Splits interest and target field vectors into n_field_tokens sub-tokens each,
    projects to token_dim, and processes them through n_blocks of (token-mixing MLP
    + per-token FFN). No self-attention.

    Args:
        num_items:      Item vocabulary size.
        embed_dim:      Item embedding dimension. Must be divisible by n_field_tokens.
        n_field_tokens: Number of sub-tokens per field (interest and target).
        token_dim:      Common token dimension for mixer blocks.
        n_blocks:       Number of RankMixer blocks.
        n_heads:        Number of heads in the token-mixing module.
        ff_mult:        FFN hidden dim multiplier (hidden = token_dim * ff_mult).
        dropout:        Dropout probability.
    """

    def __init__(
        self,
        num_items: int,
        embed_dim: int = 32,
        n_field_tokens: int = 4,
        token_dim: int = 32,
        n_blocks: int = 2,
        n_heads: int = 4,
        ff_mult: int = 2,
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

        self.blocks = nn.ModuleList([
            _RankMixerBlock(token_dim, n_tokens, n_heads, ff_mult, dropout)
            for _ in range(n_blocks)
        ])

        # Head: mean-pool tokens -> scalar logit.
        self.head = _mlp(token_dim, (token_dim,), 1)

    def _to_tokens(self, field_vec: torch.Tensor) -> torch.Tensor:
        """Split [B, embed_dim] -> [B, n_field_tokens, token_dim]."""
        B = field_vec.shape[0]
        sub_dim = self.embed_dim // self.n_field_tokens
        # [B, n_field_tokens, sub_dim]
        sub = field_vec.view(B, self.n_field_tokens, sub_dim)
        # [B, n_field_tokens, token_dim]
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

        for block in self.blocks:
            x = block(x)

        # Mean-pool tokens -> logit.
        pooled = x.mean(dim=1)  # [B, token_dim]
        return self.head(pooled).squeeze(-1)  # [B]
