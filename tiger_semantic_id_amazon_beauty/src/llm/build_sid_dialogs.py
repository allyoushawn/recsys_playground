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


def create_sid_to_title_dialogs(
    semantic_ids: np.ndarray,
    item_metadata: dict[int, dict],
) -> list[dict]:
    """Create Type A dialogs: SID → Title.

    Args:
        semantic_ids: Array of shape [num_items, 4]
        item_metadata: Dict mapping item_id -> {'title': str, ...}

    Returns:
        List of dialog dicts
    """
    dialogs = []

    for item_id, sid in enumerate(tqdm(semantic_ids, desc="Building SID→Title dialogs")):
        # Skip items without metadata
        if item_id not in item_metadata or 'title' not in item_metadata[item_id]:
            continue

        title = item_metadata[item_id]['title']
        if not title or title.strip() == '':
            continue

        sid_str = format_sid_tokens(sid)

        dialog = {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"What product has Semantic ID: {sid_str}?"},
                {"role": "assistant", "content": title},
            ],
            "type": "sid_to_title",
        }
        dialogs.append(dialog)

    return dialogs


def create_title_to_sid_dialogs(
    semantic_ids: np.ndarray,
    item_metadata: dict[int, dict],
    augment: bool = True,
) -> list[dict]:
    """Create Type B dialogs: Title → SID.

    Args:
        semantic_ids: Array of shape [num_items, 4]
        item_metadata: Dict mapping item_id -> {'title': str, ...}
        augment: If True, create variations with different prompts

    Returns:
        List of dialog dicts
    """
    dialogs = []

    # Different prompt templates for augmentation
    templates = [
        "What is the Semantic ID for '{title}'?",
        "Find the SID for product: {title}",
        "Product: {title}. What's its SID?",
    ]

    for item_id, sid in enumerate(tqdm(semantic_ids, desc="Building Title→SID dialogs")):
        # Skip items without metadata
        if item_id not in item_metadata or 'title' not in item_metadata[item_id]:
            continue

        title = item_metadata[item_id]['title']
        if not title or title.strip() == '':
            continue

        sid_str = format_sid_tokens(sid)

        # Generate base example + augmented variations
        num_variations = len(templates) if augment else 1
        for i in range(num_variations):
            template = templates[i] if augment else templates[0]
            user_msg = template.format(title=title)

            dialog = {
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                    {"role": "assistant", "content": sid_str},
                ],
                "type": "title_to_sid",
            }
            dialogs.append(dialog)

    return dialogs


def create_semantic_understanding_dialogs(
    semantic_ids: np.ndarray,
    sample_size: int = 5000,
) -> list[dict]:
    """Create Type D dialogs: Semantic understanding questions.

    Generates questions about SID relationships and hierarchical structure.

    Args:
        semantic_ids: Array of shape [num_items, 4]
        sample_size: Number of question pairs to generate

    Returns:
        List of dialog dicts
    """
    dialogs = []
    rng = np.random.RandomState(42)

    num_items = len(semantic_ids)
    if num_items < 2:
        return dialogs

    for _ in tqdm(range(sample_size), desc="Building semantic understanding dialogs"):
        # Sample two random items
        idx1, idx2 = rng.choice(num_items, size=2, replace=False)
        sid1, sid2 = semantic_ids[idx1], semantic_ids[idx2]

        # Calculate shared levels
        c1_match = sid1[0] == sid2[0]
        c2_match = sid1[1] == sid2[1]
        c3_match = sid1[2] == sid2[2]

        sid1_str = format_sid_tokens(sid1)
        sid2_str = format_sid_tokens(sid2)

        # Question type 1: Do they share the same category?
        if c1_match and c2_match and c3_match:
            answer = "Yes, they share L1, L2, and L3 categories."
        elif c1_match and c2_match:
            answer = "Yes, they share L1 and L2 categories."
        elif c1_match:
            answer = "Yes, they share the L1 category."
        else:
            answer = "No, they belong to different categories."

        dialog = {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Do these products share the same category?\n{sid1_str}\n{sid2_str}"},
                {"role": "assistant", "content": answer},
            ],
            "type": "semantic_understanding",
        }
        dialogs.append(dialog)

    return dialogs


def build_cooccurrence_matrix(
    user_sequences: dict[int, list[int]],
    num_items: int,
    window_size: int = 5,
) -> dict[int, list[tuple[int, int]]]:
    """Build item co-occurrence matrix from user sequences.

    Args:
        user_sequences: Dict mapping user_id -> list of item_ids
        num_items: Total number of items
        window_size: Consider items within this window as co-occurring

    Returns:
        Dict mapping item_id -> [(co_item_id, count), ...] sorted by count
    """
    from collections import defaultdict

    cooccurrence = defaultdict(lambda: defaultdict(int))

    for user_id, item_seq in tqdm(user_sequences.items(), desc="Building co-occurrence matrix"):
        # For each item, count co-occurrences with nearby items
        for i, item in enumerate(item_seq):
            # Look at items within window
            start = max(0, i - window_size)
            end = min(len(item_seq), i + window_size + 1)

            for j in range(start, end):
                if i != j:
                    other_item = item_seq[j]
                    cooccurrence[item][other_item] += 1

    # Convert to sorted lists
    cooccurrence_sorted = {}
    for item, co_items in cooccurrence.items():
        # Sort by count descending
        sorted_items = sorted(co_items.items(), key=lambda x: x[1], reverse=True)
        cooccurrence_sorted[item] = sorted_items

    return cooccurrence_sorted


def create_copurchase_dialogs(
    semantic_ids: np.ndarray,
    user_sequences: dict[int, list[int]],
    top_k: int = 10,
    examples_per_item: int = 3,
) -> list[dict]:
    """Create Type E dialogs: Co-purchase patterns.

    Args:
        semantic_ids: Array of shape [num_items, 4]
        user_sequences: Dict mapping user_id -> list of item_ids
        top_k: Consider top K co-purchased items
        examples_per_item: Number of examples to generate per item

    Returns:
        List of dialog dicts
    """
    dialogs = []
    num_items = len(semantic_ids)

    # Build co-occurrence matrix
    cooccurrence = build_cooccurrence_matrix(user_sequences, num_items)

    # Generate dialogs
    for item_id in tqdm(range(num_items), desc="Building co-purchase dialogs"):
        if item_id not in cooccurrence:
            continue

        sid = semantic_ids[item_id]
        sid_str = format_sid_tokens(sid)

        # Get top K co-purchased items
        co_items = cooccurrence[item_id][:top_k]

        # Generate multiple examples (sample from top K)
        num_examples = min(examples_per_item, len(co_items))
        for i in range(num_examples):
            if i >= len(co_items):
                break

            co_item_id, count = co_items[i]
            co_sid = semantic_ids[co_item_id]
            co_sid_str = format_sid_tokens(co_sid)

            dialog = {
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"Users who bought {sid_str} also frequently bought:"},
                    {"role": "assistant", "content": co_sid_str},
                ],
                "type": "copurchase",
            }
            dialogs.append(dialog)

    return dialogs


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
    parser.add_argument(
        "--data_types",
        type=str,
        default="C",
        help="Comma-separated list of data types to generate (A,B,C,D,E). Default: C (next-item only)",
    )
    parser.add_argument(
        "--metadata_path",
        type=str,
        help="Path to item metadata JSON (required for types A,B). Expected format: {item_id: {'title': '...', ...}}",
    )
    parser.add_argument(
        "--semantic_sample_size",
        type=int,
        default=5000,
        help="Number of semantic understanding examples to generate (type D)",
    )
    parser.add_argument(
        "--copurchase_top_k",
        type=int,
        default=10,
        help="Top K co-purchased items to consider (type E)",
    )
    parser.add_argument(
        "--copurchase_examples_per_item",
        type=int,
        default=3,
        help="Number of co-purchase examples per item (type E)",
    )

    args = parser.parse_args()

    # Parse arguments
    history_lengths = [int(x.strip()) for x in args.history_lengths.split(",")]
    data_types = [x.strip().upper() for x in args.data_types.split(",")]

    # Validate data types
    valid_types = {"A", "B", "C", "D", "E"}
    for dtype in data_types:
        if dtype not in valid_types:
            raise ValueError(f"Invalid data type: {dtype}. Must be one of {valid_types}")

    print(f"Generating data types: {', '.join(data_types)}")

    # Create output directory
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nLoading artifacts from {args.artifacts_dir}...")
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
    print("\nBuilding SID trie...")
    trie = build_sid_trie(semantic_ids)
    trie_path = out_dir / "sid_trie.pkl"
    with open(trie_path, "wb") as f:
        pickle.dump(trie, f)
    print(f"Saved trie to {trie_path}")
    print(f"  - valid_c2: {len(trie['valid_c2'])} L1 codes")
    print(f"  - valid_c3: {len(trie['valid_c3'])} (L1,L2) pairs")
    print(f"  - valid_c4: {len(trie['valid_c4'])} (L1,L2,L3) triples")

    # Load metadata if needed for types A or B
    item_metadata = None
    if "A" in data_types or "B" in data_types:
        if args.metadata_path:
            print(f"\nLoading item metadata from {args.metadata_path}...")
            with open(args.metadata_path) as f:
                item_metadata = json.load(f)
            # Convert string keys to int if needed
            item_metadata = {
                int(k) if isinstance(k, str) else k: v
                for k, v in item_metadata.items()
            }
            print(f"Loaded metadata for {len(item_metadata)} items")
        else:
            raise ValueError("--metadata_path required for data types A or B")

    # Load user sequences if needed for types C or E
    user_sequences = None
    if "C" in data_types or "E" in data_types:
        user_seqs_path = artifacts_dir / "user_sequences.json"
        if user_seqs_path.exists():
            print(f"\nLoading user sequences from {user_seqs_path}...")
            with open(user_seqs_path) as f:
                user_sequences = json.load(f)
            # Convert string keys to int if needed
            user_sequences = {
                int(k) if isinstance(k, str) else k: v
                for k, v in user_sequences.items()
            }
            print(f"Loaded sequences for {len(user_sequences)} users")
            total_items = sum(len(seq) for seq in user_sequences.values())
            avg_seq_len = total_items / len(user_sequences)
            print(f"Average sequence length: {avg_seq_len:.2f}")
        else:
            print("Warning: user_sequences.json not found")
            if "C" in data_types or "E" in data_types:
                raise ValueError("user_sequences.json required for data types C or E")

    # Generate dialogs for each type
    all_dialogs = []
    type_stats = {}

    print("\n" + "="*60)
    print("GENERATING TRAINING DIALOGS")
    print("="*60)

    if "A" in data_types:
        print("\n[Type A] SID → Title")
        type_a_dialogs = create_sid_to_title_dialogs(semantic_ids, item_metadata)
        all_dialogs.extend(type_a_dialogs)
        type_stats["A_sid_to_title"] = len(type_a_dialogs)
        print(f"  Generated {len(type_a_dialogs):,} examples")

    if "B" in data_types:
        print("\n[Type B] Title → SID")
        type_b_dialogs = create_title_to_sid_dialogs(semantic_ids, item_metadata, augment=True)
        all_dialogs.extend(type_b_dialogs)
        type_stats["B_title_to_sid"] = len(type_b_dialogs)
        print(f"  Generated {len(type_b_dialogs):,} examples (with augmentation)")

    if "C" in data_types:
        print(f"\n[Type C] Next-Item Prediction (history lengths: {history_lengths})")
        print("  - Starting from position 2 (not 8)")
        print("  - Multiple variations per split point")
        print("  - Compact format")
        type_c_dialogs = create_dialogs(
            user_sequences,
            semantic_ids,
            item_to_sid,
            history_lengths=history_lengths,
        )
        all_dialogs.extend(type_c_dialogs)
        # Count by variation
        for hist_len in history_lengths:
            count = sum(1 for d in type_c_dialogs if d.get("type") == f"seq_last_{hist_len}")
            type_stats[f"C_next_item_last_{hist_len}"] = count
        print(f"  Generated {len(type_c_dialogs):,} examples")

    if "D" in data_types:
        print(f"\n[Type D] Semantic Understanding (sample_size={args.semantic_sample_size})")
        type_d_dialogs = create_semantic_understanding_dialogs(
            semantic_ids,
            sample_size=args.semantic_sample_size,
        )
        all_dialogs.extend(type_d_dialogs)
        type_stats["D_semantic_understanding"] = len(type_d_dialogs)
        print(f"  Generated {len(type_d_dialogs):,} examples")

    if "E" in data_types:
        print(f"\n[Type E] Co-Purchase Patterns (top_k={args.copurchase_top_k}, examples_per_item={args.copurchase_examples_per_item})")
        type_e_dialogs = create_copurchase_dialogs(
            semantic_ids,
            user_sequences,
            top_k=args.copurchase_top_k,
            examples_per_item=args.copurchase_examples_per_item,
        )
        all_dialogs.extend(type_e_dialogs)
        type_stats["E_copurchase"] = len(type_e_dialogs)
        print(f"  Generated {len(type_e_dialogs):,} examples")

    print("\n" + "="*60)
    print(f"TOTAL: {len(all_dialogs):,} examples across {len(data_types)} types")
    print("="*60)

    # Split train/valid
    print("\nShuffling and splitting...")
    np.random.seed(42)
    np.random.shuffle(all_dialogs)
    split_idx = int(len(all_dialogs) * args.train_ratio)
    train_dialogs = all_dialogs[:split_idx]
    valid_dialogs = all_dialogs[split_idx:]

    print(f"Split: {len(train_dialogs):,} train, {len(valid_dialogs):,} valid")

    # Print statistics by type
    print("\n" + "="*60)
    print("STATISTICS BY DATA TYPE")
    print("="*60)
    for type_name, count in sorted(type_stats.items()):
        pct = 100.0 * count / len(all_dialogs)
        print(f"  {type_name:30s}: {count:7,} ({pct:5.2f}%)")

    # Save JSONL files
    print("\nSaving JSONL files...")
    train_path = out_dir / "dialogs_train.jsonl"
    valid_path = out_dir / "dialogs_valid.jsonl"

    with jsonlines.open(train_path, mode="w") as writer:
        writer.write_all(train_dialogs)
    print(f"  Saved training dialogs to {train_path}")

    with jsonlines.open(valid_path, mode="w") as writer:
        writer.write_all(valid_dialogs)
    print(f"  Saved validation dialogs to {valid_path}")

    # Print example dialogs for each type
    print("\n" + "="*60)
    print("EXAMPLE DIALOGS")
    print("="*60)

    # Group dialogs by type for sampling
    dialogs_by_type = defaultdict(list)
    for dialog in train_dialogs[:1000]:  # Sample from first 1000 to get variety
        dtype = dialog.get("type", "unknown")
        dialogs_by_type[dtype].append(dialog)

    # Print one example per type
    for dtype in sorted(dialogs_by_type.keys()):
        if dialogs_by_type[dtype]:
            print(f"\n[{dtype}]")
            example = dialogs_by_type[dtype][0]
            for msg in example["messages"]:
                print(f"{msg['role'].upper()}: {msg['content'][:100]}{'...' if len(msg['content']) > 100 else ''}")

    print("\n" + "="*60)
    print("✓ Done!")
    print("="*60)


if __name__ == "__main__":
    main()
