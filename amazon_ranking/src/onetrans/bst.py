"""Behaviour Sequence Transformer (BST, Alibaba 2019) adapted for (history, target) CTR scoring.

The key adaptation vs. the original paper: the target item is appended as the
*last* token in the input sequence so that Transformer self-attention can model
interactions between every history item and the target in a single forward pass.
A learned positional embedding is added to each position. The encoder is fully
bi-directional (non-causal), which is correct for offline CTR scoring where the
target identity is known. The target-position hidden state (optionally concatenated
with a masked mean of the history hidden states) is then fed into an MLP head to
produce a scalar logit.
"""

from __future__ import annotations

import math
from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn

from amazon_ranking.src.baselines import _SeqRankModel
from amazon_ranking.src.din import _mlp


class _TransformerBlock(nn.Module):
    """One pre-norm Transformer encoder block (MHA + position-wise FFN).

    Pre-norm (LayerNorm before the sub-layer) is more stable than post-norm for
    shallow networks and matches popular open-source BST re-implementations.
    """

    def __init__(
        self,
        embed_dim: int,
        n_heads: int,
        ff_dim: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim,
            n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Linear(ff_dim, embed_dim),
        )
        self.drop = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """x: [B, T, E].  key_padding_mask: [B, T] bool (True = ignore)."""
        # Self-attention sub-layer (pre-norm).
        residual = x
        x = self.norm1(x)
        x, _ = self.attn(x, x, x, key_padding_mask=key_padding_mask, need_weights=False)
        x = self.drop(x) + residual
        # FFN sub-layer (pre-norm).
        residual = x
        x = self.norm2(x)
        x = self.drop(self.ff(x)) + residual
        return x


class BST(_SeqRankModel):
    """Behaviour Sequence Transformer for (history, target) binary CTR scoring.

    Inherits the shared item embedding table and mask-aware mean-pool helper from
    :class:`~amazon_ranking.src.baselines._SeqRankModel`.

    Forward signature (same as DIN / all baselines)::

        forward(hist_ids [B, L], hist_mask [B, L], target_ids [B]) -> logits [B]

    Parameters
    ----------
    num_items:
        Vocabulary size.  Pad index is ``num_items`` (inherited).
    embed_dim:
        Item embedding dimension (= Transformer hidden size).
    n_layers:
        Number of stacked Transformer encoder blocks.
    n_heads:
        Number of attention heads.  Must divide ``embed_dim``.
    ff_dim:
        Feed-forward inner dimension.  Defaults to ``4 * embed_dim`` when
        ``None``, matching the original Transformer paper.
    dropout:
        Dropout probability applied inside MHA and FFN.
    max_len:
        Maximum history length to support.  Positional embedding table has
        ``max_len + 1`` rows (the extra row is for the appended target token).
        Sequences longer than ``max_len`` are silently truncated from the left.
    mlp_hidden:
        Hidden layer widths for the scoring MLP head.
    **kwargs:
        Extra keyword arguments are accepted and silently ignored to allow the
        shared ``build_model`` registry to forward unknown kwargs.
    """

    def __init__(
        self,
        num_items: int,
        embed_dim: int = 32,
        n_layers: int = 2,
        n_heads: int = 2,
        ff_dim: Optional[int] = None,
        dropout: float = 0.0,
        max_len: int = 50,
        mlp_hidden: Sequence[int] = (80, 40),
        **kwargs,
    ) -> None:
        super().__init__(num_items, embed_dim)
        ff_dim = ff_dim if ff_dim is not None else 4 * embed_dim
        self.max_len = int(max_len)
        # Positional embeddings: max_len history positions + 1 target position.
        self.pos_emb = nn.Embedding(max_len + 1, embed_dim)
        self.blocks = nn.ModuleList(
            [
                _TransformerBlock(embed_dim, n_heads, ff_dim, dropout)
                for _ in range(n_layers)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim)
        # MLP head: target hidden state [E] + mean of history hidden states [E].
        self.head = _mlp(embed_dim * 2, mlp_hidden, 1)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_sequence(
        self,
        hist_ids: torch.Tensor,
        hist_mask: torch.Tensor,
        target_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build token + position sequences with the target appended.

        Positional embeddings are assigned **right-aligned** (most-recent history
        item → position ``max_len - 1``, target → position ``max_len``), so that
        extra left-padding does not shift the positions of real items.  This makes
        the logit invariant to the amount of right-padding.

        Returns
        -------
        token_emb: [B, T, E]  (T = actual_len + 1, ≤ max_len + 1)
        seq_mask:  [B, T] bool True = position should be *ignored* (for MHA key_padding_mask)
        tgt_pos:   int  index of the target token in the T dimension (always T-1)
        """
        B, L = hist_ids.shape
        # Note: caller (forward) has already truncated to max_len before this call.

        # Item embeddings: history [B, L, E] + target [B, 1, E].
        hist_emb = self.item_emb(hist_ids)  # [B, L, E]
        tgt_emb = self.item_emb(target_ids).unsqueeze(1)  # [B, 1, E]

        # Mask-aligned positional indices: the *last* valid history slot always
        # receives position (max_len - 1), the second-to-last gets (max_len - 2),
        # and so on.  Padded slots receive position 0 (a dummy; they are ignored by
        # the attention key-padding mask and zeroed out by the embedding
        # padding_idx).  The target always receives position max_len.
        #
        # This scheme makes the positional encoding of real items invariant to how
        # many pad slots surround them, so the logit is stable across varying L.
        #
        # For each row, count valid items from the right: the j-th valid item from
        # the end maps to position (max_len - 1 - (num_valid - 1 - rank)).
        # Equivalently: for each slot i (0-indexed left-to-right),
        #   pos[i] = max_len - (num_valid - cumulative_valid_at_i)  [1-indexed count from right]
        # We compute this without a Python loop using a cumulative sum trick.
        cum_valid = hist_mask.cumsum(dim=1)  # [B, L]; cumulative count of valid slots
        num_valid = hist_mask.sum(dim=1, keepdim=True)  # [B, 1]
        # rank_from_right[b, i] = 0 for the last valid slot, 1 for the second-to-last, …
        # For padding slots this can be negative; we clamp to 0 before indexing.
        rank_from_right = num_valid - cum_valid  # [B, L]; ≥0 for valid, >0 for padding
        hist_pos_idx = (self.max_len - 1 - rank_from_right).long().clamp(min=0)  # [B, L]
        # Replace padded slots with 0 so they all use the same dummy position.
        hist_pos_idx = hist_pos_idx * hist_mask.long()

        tgt_pos_val = torch.full((B, 1), self.max_len, dtype=torch.long, device=hist_ids.device)

        # Item embeddings + per-token positional embeddings.
        pos_hist = self.pos_emb(hist_pos_idx)   # [B, L, E]
        pos_tgt = self.pos_emb(tgt_pos_val)     # [B, 1, E]

        # Concatenate history + target along the sequence axis.
        tokens = torch.cat([hist_emb + pos_hist, tgt_emb + pos_tgt], dim=1)  # [B, L+1, E]

        # Key-padding mask for MHA: True = position is PADDING (should be ignored).
        # History: 1-hist_mask (0=real → keep, 1=padding → ignore).
        # Target: always valid (False = keep).
        pad_mask_hist = (hist_mask == 0)  # [B, L] bool
        pad_mask_tgt = torch.zeros(B, 1, dtype=torch.bool, device=hist_ids.device)
        seq_mask = torch.cat([pad_mask_hist, pad_mask_tgt], dim=1)  # [B, L+1]

        tgt_pos = L  # index of the target token in the T=L+1 dimension
        return tokens, seq_mask, tgt_pos

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        hist_ids: torch.Tensor,
        hist_mask: torch.Tensor,
        target_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Compute ranking logits for a batch of (history, target) pairs.

        Parameters
        ----------
        hist_ids:   [B, L] long  — padded history item indices
        hist_mask:  [B, L] float — 1 for real positions, 0 for padding
        target_ids: [B] long     — target item indices

        Returns
        -------
        logits: [B] float  (raw, no sigmoid)
        """
        # Truncate to max_len before building the sequence so that both the
        # Transformer input and the pooling mask use the same (possibly shorter) L.
        if hist_ids.shape[1] > self.max_len:
            hist_ids = hist_ids[:, -self.max_len :]
            hist_mask = hist_mask[:, -self.max_len :]

        tokens, seq_mask, tgt_pos = self._build_sequence(hist_ids, hist_mask, target_ids)

        # Run Transformer encoder blocks.
        x = tokens
        for block in self.blocks:
            x = block(x, key_padding_mask=seq_mask)
        x = self.norm(x)  # [B, T, E]

        # Target representation: final hidden state at the target position.
        tgt_repr = x[:, tgt_pos, :]  # [B, E]

        # History representation: masked mean of valid history positions.
        hist_out = x[:, :tgt_pos, :]  # [B, L, E]
        w = hist_mask.unsqueeze(-1).to(hist_out.dtype)  # [B, L, 1]
        cnt = w.sum(dim=1).clamp(min=1.0)  # [B, 1]
        hist_repr = (hist_out * w).sum(dim=1) / cnt  # [B, E]

        # MLP head → scalar logit.
        feats = torch.cat([tgt_repr, hist_repr], dim=-1)  # [B, 2E]
        return self.head(feats).squeeze(-1)  # [B]
