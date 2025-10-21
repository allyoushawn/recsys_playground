"""Decoding constraints for valid Semantic ID generation.

Provides level-based masking and trie-based constraints to ensure generated SIDs
are valid and exist in the catalog.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import torch


# Level token ranges (after tokenization)
# L1: <sid_0> to <sid_255>
# L2: <sid_256> to <sid_511>
# L3: <sid_512> to <sid_767>
# L4: <sid_768> to <sid_1023>
LEVEL_RANGES = {
    1: range(0, 256),
    2: range(256, 512),
    3: range(512, 768),
    4: range(768, 1024),
}


def get_sid_token_ids(tokenizer, level: int | None = None) -> list[int]:
    """Get token IDs for SID tokens.

    Args:
        tokenizer: Tokenizer instance
        level: If provided, return only tokens for that level (1-4)

    Returns:
        List of token IDs
    """
    if level is None:
        # All SID tokens
        token_strs = [f"<sid_{i}>" for i in range(1024)]
    else:
        # Specific level
        level_range = LEVEL_RANGES[level]
        token_strs = [f"<sid_{i}>" for i in level_range]

    token_ids = tokenizer.convert_tokens_to_ids(token_strs)

    # Filter out None values (tokens not in vocabulary)
    valid_token_ids = [tid for tid in token_ids if tid is not None]

    if len(valid_token_ids) != len(token_ids):
        missing_count = len(token_ids) - len(valid_token_ids)
        print(f"Warning: {missing_count} SID tokens not found in tokenizer vocabulary")

    return valid_token_ids


def mask_logits_by_level(
    logits: torch.Tensor,
    level: int,
    tokenizer,
    mask_value: float = -float("inf"),
) -> torch.Tensor:
    """Mask logits to only allow tokens from a specific level.

    Args:
        logits: Logits tensor of shape [batch_size, vocab_size] or [vocab_size]
        level: Level number (1-4)
        tokenizer: Tokenizer instance
        mask_value: Value to set for masked positions

    Returns:
        Masked logits tensor
    """
    # Get valid token IDs for this level
    valid_token_ids = get_sid_token_ids(tokenizer, level=level)

    # Create mask
    mask = torch.full_like(logits, mask_value)
    mask[..., valid_token_ids] = 0  # Allow these tokens

    return logits + mask


class TrieConstraint:
    """Trie-based constraint for valid SID continuations."""

    def __init__(self, trie_path: str | Path):
        """Load trie from pickle file.

        Args:
            trie_path: Path to sid_trie.pkl
        """
        with open(trie_path, "rb") as f:
            trie = pickle.load(f)

        self.valid_c2 = trie["valid_c2"]  # c1 -> list of valid c2
        self.valid_c3 = trie["valid_c3"]  # (c1, c2) -> list of valid c3
        self.valid_c4 = trie["valid_c4"]  # (c1, c2, c3) -> list of valid c4

    def get_valid_codes(self, level: int, prefix: tuple[int, ...]) -> list[int]:
        """Get valid codes for the next level given prefix.

        Args:
            level: Level to generate (2, 3, or 4)
            prefix: Tuple of previous codes (c1) or (c1, c2) or (c1, c2, c3)

        Returns:
            List of valid codes for this level
        """
        if level == 2:
            # prefix = (c1,)
            c1 = prefix[0]
            return self.valid_c2.get(c1, [])
        elif level == 3:
            # prefix = (c1, c2)
            return self.valid_c3.get(prefix, [])
        elif level == 4:
            # prefix = (c1, c2, c3)
            return self.valid_c4.get(prefix, [])
        else:
            raise ValueError(f"Level must be 2, 3, or 4, got {level}")

    def mask_logits_by_trie(
        self,
        logits: torch.Tensor,
        level: int,
        prefix: tuple[int, ...],
        tokenizer,
        mask_value: float = -float("inf"),
    ) -> torch.Tensor:
        """Mask logits to only allow valid continuations from trie.

        Args:
            logits: Logits tensor of shape [batch_size, vocab_size] or [vocab_size]
            level: Level to generate (2, 3, or 4)
            prefix: Tuple of previous codes
            tokenizer: Tokenizer instance
            mask_value: Value to set for masked positions

        Returns:
            Masked logits tensor
        """
        if level == 1:
            # No trie constraint for L1, use level mask only
            return logits

        # Get valid codes from trie
        valid_codes = self.get_valid_codes(level, prefix)

        if not valid_codes:
            # No valid continuations - this shouldn't happen in practice
            # but handle gracefully by allowing all tokens in the level
            return logits

        # Map codes to token IDs
        # L2: codes are in [0,255], map to <sid_256> to <sid_511>
        # L3: codes are in [0,255], map to <sid_512> to <sid_767>
        # L4: codes are in [0,255], map to <sid_768> to <sid_1023>
        level_offset = (level - 1) * 256
        token_strs = [f"<sid_{code + level_offset}>" for code in valid_codes]
        raw_token_ids = tokenizer.convert_tokens_to_ids(token_strs)

        # Filter out None values (tokens not in vocabulary)
        valid_token_ids = [tid for tid in raw_token_ids if tid is not None]

        if not valid_token_ids:
            # All tokens missing - something is wrong with tokenizer
            print(f"Error: No valid token IDs found for level {level}, prefix {prefix}")
            return logits  # Return unmasked logits as fallback

        # Create mask
        mask = torch.full_like(logits, mask_value)
        mask[..., valid_token_ids] = 0  # Allow these tokens

        return logits + mask


def apply_sid_constraints(
    logits: torch.Tensor,
    level: int,
    prefix: tuple[int, ...] | None,
    tokenizer,
    trie: TrieConstraint | None = None,
) -> torch.Tensor:
    """Apply both level and trie constraints to logits.

    Args:
        logits: Logits tensor
        level: Current level (1-4)
        prefix: Prefix codes (empty for level 1)
        tokenizer: Tokenizer instance
        trie: Optional trie constraint

    Returns:
        Masked logits
    """
    # First apply level mask
    masked_logits = mask_logits_by_level(logits, level, tokenizer)

    # Then apply trie mask if available and level > 1
    if trie is not None and level > 1 and prefix:
        masked_logits = trie.mask_logits_by_trie(
            masked_logits, level, prefix, tokenizer
        )

    return masked_logits


def decode_sid_tokens(token_ids: list[int], tokenizer) -> tuple[int, ...] | None:
    """Decode token IDs to SID codes.

    Args:
        token_ids: List of 4 token IDs
        tokenizer: Tokenizer instance

    Returns:
        Tuple of (c1, c2, c3, c4) where codes are in [0, 255] range,
        or None if invalid
    """
    if len(token_ids) != 4:
        return None

    tokens = tokenizer.convert_ids_to_tokens(token_ids)
    codes = []

    for i, token in enumerate(tokens):
        if not token.startswith("<sid_") or not token.endswith(">"):
            return None

        try:
            # Extract number from <sid_XXX>
            token_num = int(token[5:-1])

            # Map back to code in [0, 255]
            level_offset = i * 256
            code = token_num - level_offset

            if not (0 <= code < 256):
                return None

            codes.append(code)
        except (ValueError, IndexError):
            return None

    return tuple(codes)
