"""Inference wrapper for generating Semantic IDs with constraints.

Generates exactly 4 SID tokens with level and trie masking to ensure validity.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .constraints import (
    TrieConstraint,
    apply_sid_constraints,
    decode_sid_tokens,
    get_sid_token_ids,
)


SYSTEM_PROMPT = """You are a recommender that must reply ONLY with the next product's Semantic ID as 4 tokens in order: L1, L2, L3, L4.
Valid token ranges by level:
- L1: <sid_0>.. <sid_255>
- L2: <sid_256>.. <sid_511>
- L3: <sid_512>.. <sid_767>
- L4: <sid_768>.. <sid_1023>
Do not output anything else."""


class SIDRecommender:
    """Recommender that generates Semantic IDs with constraints."""

    def __init__(
        self,
        model_path: str,
        trie_path: str | None = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """Initialize recommender.

        Args:
            model_path: Path to fine-tuned model
            trie_path: Optional path to sid_trie.pkl for trie constraints
            device: Device to use
        """
        print(f"Loading model from {model_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map="auto",
        )
        self.model.eval()
        self.device = device

        # Load trie if provided
        self.trie = None
        if trie_path:
            print(f"Loading trie from {trie_path}...")
            self.trie = TrieConstraint(trie_path)

        print("Model loaded!")

    def format_history(self, history_sids: list[tuple[int, ...]]) -> str:
        """Format history SIDs as text.

        Args:
            history_sids: List of (c1, c2, c3, c4) tuples

        Returns:
            Formatted history string
        """
        lines = ["History:"]
        for sid in history_sids:
            c1, c2, c3, c4 = sid
            # Map to token ranges
            tokens = f"<sid_{c1}> <sid_{c2 + 256}> <sid_{c3 + 512}> <sid_{c4 + 768}>"
            lines.append(tokens)
        lines.append("Recommend next:")
        return "\n".join(lines)

    def generate_sid(
        self,
        history_sids: list[tuple[int, ...]] | None = None,
        user_text: str | None = None,
    ) -> tuple[int, ...] | None:
        """Generate next SID given history.

        Args:
            history_sids: List of (c1, c2, c3, c4) tuples
            user_text: Optional natural language query (overrides history)

        Returns:
            Generated (c1, c2, c3, c4) or None if invalid
        """
        # Build user message
        if user_text:
            user_msg = user_text
        elif history_sids:
            user_msg = self.format_history(history_sids)
        else:
            raise ValueError("Must provide either history_sids or user_text")

        # Build messages
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ]

        # Apply chat template
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        # Generate 4 tokens with constraints
        generated_tokens = []
        prefix_codes = []

        with torch.no_grad():
            for level in range(1, 5):  # L1, L2, L3, L4
                # Forward pass
                outputs = self.model(**inputs)
                logits = outputs.logits[:, -1, :]  # [1, vocab_size]

                # Apply constraints
                masked_logits = apply_sid_constraints(
                    logits,
                    level=level,
                    prefix=tuple(prefix_codes) if prefix_codes else None,
                    tokenizer=self.tokenizer,
                    trie=self.trie,
                )

                # Greedy decode
                next_token_id = masked_logits.argmax(dim=-1).item()
                generated_tokens.append(next_token_id)

                # Decode to get code
                token_str = self.tokenizer.decode([next_token_id])
                if not token_str.startswith("<sid_"):
                    print(f"Warning: Invalid token at level {level}: {token_str}")
                    return None

                # Extract code
                code_num = int(token_str[5:-1])
                level_offset = (level - 1) * 256
                code = code_num - level_offset
                prefix_codes.append(code)

                # Update inputs for next token
                inputs["input_ids"] = torch.cat(
                    [inputs["input_ids"], torch.tensor([[next_token_id]]).to(self.device)],
                    dim=1,
                )
                inputs["attention_mask"] = torch.cat(
                    [inputs["attention_mask"], torch.ones((1, 1)).to(self.device)],
                    dim=1,
                )

        # Decode final SID
        sid = decode_sid_tokens(generated_tokens, self.tokenizer)
        return sid

    def recommend(
        self,
        history_sids: list[tuple[int, ...]],
        sid_to_items: dict[str, list[str]],
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """Generate recommendation and map to items.

        Args:
            history_sids: List of (c1, c2, c3, c4) tuples
            sid_to_items: Dict mapping "c1,c2,c3,c4" -> list of item IDs
            top_k: Number of items to return

        Returns:
            List of dicts with "sid" and "items" keys
        """
        # Generate SID
        generated_sid = self.generate_sid(history_sids=history_sids)

        if generated_sid is None:
            return []

        # Map to items
        sid_key = ",".join(map(str, generated_sid))
        items = sid_to_items.get(sid_key, [])

        return [
            {
                "sid": generated_sid,
                "items": items[:top_k],
            }
        ]


def main():
    parser = argparse.ArgumentParser(description="Generate SIDs for recommendations")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to fine-tuned model",
    )
    parser.add_argument(
        "--sid_trie",
        type=str,
        help="Path to sid_trie.pkl",
    )
    parser.add_argument(
        "--sid_to_items",
        type=str,
        required=True,
        help="Path to sid_to_items.json",
    )
    parser.add_argument(
        "--history_file",
        type=str,
        help="Path to JSON file with user histories",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Interactive mode",
    )

    args = parser.parse_args()

    # Load recommender
    recommender = SIDRecommender(
        model_path=args.model,
        trie_path=args.sid_trie,
    )

    # Load sid_to_items
    with open(args.sid_to_items) as f:
        sid_to_items = json.load(f)

    if args.interactive:
        print("\n=== Interactive Mode ===")
        print("Enter history SIDs as comma-separated tuples, e.g.:")
        print("  64,313,125,0;64,447,194,0;112,201,11,4")
        print("Type 'quit' to exit\n")

        while True:
            user_input = input("History: ").strip()
            if user_input.lower() == "quit":
                break

            try:
                # Parse history
                history_sids = []
                for sid_str in user_input.split(";"):
                    codes = tuple(map(int, sid_str.split(",")))
                    if len(codes) != 4:
                        print("Error: Each SID must have exactly 4 codes")
                        continue
                    history_sids.append(codes)

                # Generate
                results = recommender.recommend(
                    history_sids=history_sids,
                    sid_to_items=sid_to_items,
                    top_k=5,
                )

                # Display
                if results:
                    result = results[0]
                    print(f"\nGenerated SID: {result['sid']}")
                    print(f"Mapped to {len(result['items'])} items:")
                    for item_id in result['items']:
                        print(f"  - {item_id}")
                else:
                    print("No valid SID generated")

                print()

            except Exception as e:
                print(f"Error: {e}\n")

    elif args.history_file:
        # Batch mode
        with open(args.history_file) as f:
            histories = json.load(f)

        print(f"\nProcessing {len(histories)} histories...")

        for i, history_data in enumerate(histories[:10]):  # Process first 10
            history_sids = [tuple(sid) for sid in history_data["history"]]
            results = recommender.recommend(
                history_sids=history_sids,
                sid_to_items=sid_to_items,
                top_k=5,
            )

            print(f"\n[{i+1}] History length: {len(history_sids)}")
            if results:
                result = results[0]
                print(f"    Generated SID: {result['sid']}")
                print(f"    Items: {result['items'][:3]}")
            else:
                print("    No valid SID generated")

    else:
        print("Error: Must specify either --interactive or --history_file")


if __name__ == "__main__":
    main()
