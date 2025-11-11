"""Build conversational dialogs from user histories for LLM fine-tuning.

Creates JSONL files with chat-style examples for next-item prediction using Semantic IDs.
Also builds a trie of valid SID continuations for constrained decoding.
"""

from __future__ import annotations

import argparse
import json
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Any

import jsonlines
import numpy as np
from tqdm import tqdm


SYSTEM_PROMPT = """You are a recommender that must reply ONLY with the next product's Semantic ID as 4 tokens in order: L1, L2, L3, L4.
Valid token ranges by level:
- L1: <sid_0>.. <sid_255>
- L2: <sid_256>.. <sid_511>
- L3: <sid_512>.. <sid_767>
- L4: <sid_768>.. <sid_1023>
Do not output anything else."""


def build_sid_trie(semantic_ids: np.ndarray) -> dict[str, Any]:
    """Build a trie of valid SID continuations.

    Args:
        semantic_ids: Array of shape [num_items, 4] with SID codes

    Returns:
        Dictionary with:
            - valid_c2: dict mapping c1 -> set of valid c2
            - valid_c3: dict mapping (c1, c2) -> set of valid c3
            - valid_c4: dict mapping (c1, c2, c3) -> set of valid c4
    """
    valid_c2 = defaultdict(set)
    valid_c3 = defaultdict(set)
    valid_c4 = defaultdict(set)

    for sid in semantic_ids:
        c1, c2, c3, c4 = sid
        valid_c2[c1].add(c2)
        valid_c3[(c1, c2)].add(c3)
        valid_c4[(c1, c2, c3)].add(c4)

    # Convert sets to sorted lists for determinism
    trie = {
        "valid_c2": {k: sorted(v) for k, v in valid_c2.items()},
        "valid_c3": {k: sorted(v) for k, v in valid_c3.items()},
        "valid_c4": {k: sorted(v) for k, v in valid_c4.items()},
    }
    return trie


def format_sid_tokens(sid: np.ndarray | list) -> str:
    """Format a 4-element SID as space-separated tokens.

    Maps codes to token ranges: L1:[0,255], L2:[256,511], L3:[512,767], L4:[768,1023]
    """
    c1, c2, c3, c4 = sid
    return f"<sid_{c1}> <sid_{c2 + 256}> <sid_{c3 + 512}> <sid_{c4 + 768}>"


def format_history(history_sids: list[np.ndarray]) -> str:
    """Format a list of SIDs as history text."""
    lines = ["History:"]
    for sid in history_sids:
        lines.append(format_sid_tokens(sid))
    lines.append("Recommend next:")
    return "\n".join(lines)


def format_history_compact(history_sids: list[np.ndarray]) -> str:
    """Format history as comma-separated SIDs (compact format like reference)."""
    sid_strs = [format_sid_tokens(sid) for sid in history_sids]
    return "User's last purchases: " + ", ".join(sid_strs) + ". Next:"


def create_dialogs(
    user_sequences: dict[int, list[int]],
    semantic_ids: np.ndarray,
    item_to_sid: dict[str, list[int]],
    history_lengths: list[int] = [2, 3, 5],
    min_seq_len: int = 3,
) -> list[dict]:
    """Create conversational dialogs from user sequences with multiple variations.

    Following reference implementation approach:
    - Generates examples starting from position 2 (not 8)
    - Creates multiple variations per split point (last_2, last_3, last_5)
    - Uses compact format for consistency

    Args:
        user_sequences: Dict mapping user_id -> list of item_ids (chronological)
        semantic_ids: Array of shape [num_items, 4]
        item_to_sid: Dict mapping item_id (str) -> [c1, c2, c3, c4]
        history_lengths: List of history window sizes to generate (default: [2, 3, 5])
        min_seq_len: Minimum sequence length to create examples (default: 3)

    Returns:
        List of dialog dicts with "messages" key
    """
    dialogs = []
    stats = {f"last_{h}": 0 for h in history_lengths}

    for user_id, item_seq in tqdm(user_sequences.items(), desc="Building dialogs"):
        if len(item_seq) < min_seq_len:
            continue

        # For each split point in the sequence (starting from position 2)
        for split_point in range(2, len(item_seq)):
            history = item_seq[:split_point]
            target_item = item_seq[split_point]

            # Get target SID
            try:
                target_sid = semantic_ids[target_item]
            except (IndexError, KeyError):
                continue  # Skip if target not in semantic_ids

            # Create multiple variations with different history lengths
            for hist_len in history_lengths:
                # Take last N items from history
                history_subset = history[-hist_len:] if len(history) >= hist_len else history

                # Get SIDs for history
                try:
                    history_sids = [semantic_ids[item_id] for item_id in history_subset]
                except (IndexError, KeyError):
                    continue  # Skip if any item not in semantic_ids

                # Format as dialog with compact format
                user_msg = format_history_compact(history_sids)
                assistant_msg = format_sid_tokens(target_sid)

                dialog = {
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_msg},
                        {"role": "assistant", "content": assistant_msg},
                    ],
                    "type": f"seq_last_{hist_len}",  # Track variation type
                }
                dialogs.append(dialog)
                stats[f"last_{hist_len}"] += 1

    # Print statistics
    print("\nDialog generation statistics:")
    for hist_type, count in stats.items():
        print(f"  {hist_type}: {count:,} examples")

    return dialogs


def main():
    parser = argparse.ArgumentParser(
        description="Build SID dialogs for LLM fine-tuning"
    )
    parser.add_argument(
        "--artifacts_dir",
        type=str,
        default="/content/artifacts",
        help="Path to artifacts directory",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="/content/artifacts/llm",
        help="Output directory for dialogs and trie",
    )
    parser.add_argument(
        "--history_lengths",
        type=str,
        default="2,3,5",
        help="Comma-separated list of history lengths to generate (e.g., '2,3,5')",
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.95,
        help="Fraction of dialogs for training",
    )

    args = parser.parse_args()

    # Parse history lengths
    history_lengths = [int(x.strip()) for x in args.history_lengths.split(",")]

    # Create output directory
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading artifacts from {args.artifacts_dir}...")
    artifacts_dir = Path(args.artifacts_dir)

    # Load semantic IDs
    semantic_ids = np.load(artifacts_dir / "semantic_ids.npy")
    print(f"Loaded {len(semantic_ids)} semantic IDs")

    # Load mappings
    with open(artifacts_dir / "item_to_sid.json") as f:
        item_to_sid = json.load(f)
    with open(artifacts_dir / "sid_to_items.json") as f:
        sid_to_items = json.load(f)

    # Build trie
    print("Building SID trie...")
    trie = build_sid_trie(semantic_ids)
    trie_path = out_dir / "sid_trie.pkl"
    with open(trie_path, "wb") as f:
        pickle.dump(trie, f)
    print(f"Saved trie to {trie_path}")
    print(f"  - valid_c2: {len(trie['valid_c2'])} L1 codes")
    print(f"  - valid_c3: {len(trie['valid_c3'])} (L1,L2) pairs")
    print(f"  - valid_c4: {len(trie['valid_c4'])} (L1,L2,L3) triples")

    # Load user sequences
    # Check for preprocessed sequences or derive from splits
    user_seqs_path = artifacts_dir / "user_sequences.json"
    if user_seqs_path.exists():
        with open(user_seqs_path) as f:
            user_sequences = json.load(f)
        # Convert string keys to int if needed
        user_sequences = {
            int(k) if isinstance(k, str) else k: v
            for k, v in user_sequences.items()
        }
    else:
        print("Warning: user_sequences.json not found, using dummy data")
        # Create dummy sequences for testing
        num_users = 100
        user_sequences = {}
        for u in range(num_users):
            seq_len = np.random.randint(10, 30)
            user_sequences[u] = np.random.randint(0, len(semantic_ids), size=seq_len).tolist()

    print(f"Loaded sequences for {len(user_sequences)} users")

    # Calculate expected examples
    total_items = sum(len(seq) for seq in user_sequences.values())
    avg_seq_len = total_items / len(user_sequences)
    print(f"Average sequence length: {avg_seq_len:.2f}")

    # Create dialogs with multiple variations
    print(f"\nCreating dialogs with history lengths: {history_lengths}")
    print("Following reference implementation approach:")
    print("  - Starting from position 2 (not 8)")
    print("  - Multiple variations per split point")
    print("  - Compact format")
    dialogs = create_dialogs(
        user_sequences,
        semantic_ids,
        item_to_sid,
        history_lengths=history_lengths,
    )
    print(f"\nCreated {len(dialogs)} total dialogs")

    # Split train/valid
    np.random.seed(42)
    np.random.shuffle(dialogs)
    split_idx = int(len(dialogs) * args.train_ratio)
    train_dialogs = dialogs[:split_idx]
    valid_dialogs = dialogs[split_idx:]

    print(f"Split: {len(train_dialogs)} train, {len(valid_dialogs)} valid")

    # Save JSONL files
    train_path = out_dir / "dialogs_train.jsonl"
    valid_path = out_dir / "dialogs_valid.jsonl"

    with jsonlines.open(train_path, mode="w") as writer:
        writer.write_all(train_dialogs)
    print(f"Saved training dialogs to {train_path}")

    with jsonlines.open(valid_path, mode="w") as writer:
        writer.write_all(valid_dialogs)
    print(f"Saved validation dialogs to {valid_path}")

    # Print example
    print("\n=== Example Dialog ===")
    example = train_dialogs[0]
    for msg in example["messages"]:
        print(f"{msg['role'].upper()}:")
        print(msg["content"])
        print()

    print("Done!")


if __name__ == "__main__":
    main()
