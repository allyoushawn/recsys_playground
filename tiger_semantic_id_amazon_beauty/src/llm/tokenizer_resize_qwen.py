"""Resize Qwen tokenizer and model to add Semantic ID tokens.

Adds 1,027 new tokens:
- <SID_START>, <SID_END>, <REC>
- <sid_0> through <sid_1023> (1,024 tokens)

Initializes new embeddings and saves the extended model.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def add_sid_tokens(tokenizer: AutoTokenizer) -> list[str]:
    """Add SID tokens to tokenizer.

    Returns:
        List of new token strings
    """
    new_tokens = ["<SID_START>", "<SID_END>", "<REC>"]

    # Add <sid_0> through <sid_1023>
    for i in range(1024):
        new_tokens.append(f"<sid_{i}>")

    # Add tokens (they become special tokens)
    num_added = tokenizer.add_tokens(new_tokens, special_tokens=True)
    print(f"Added {num_added} new tokens")

    return new_tokens


def initialize_new_embeddings(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    original_vocab_size: int,
) -> None:
    """Initialize embeddings for new tokens.

    Uses mean of existing embeddings as initialization.
    """
    # Resize token embeddings
    model.resize_token_embeddings(len(tokenizer))

    # Get embedding layers
    input_embeddings = model.get_input_embeddings()
    output_embeddings = model.get_output_embeddings()

    # Compute mean of existing embeddings
    with torch.no_grad():
        # Input embeddings
        existing_input = input_embeddings.weight[:original_vocab_size]
        mean_input = existing_input.mean(dim=0, keepdim=True)
        # Initialize new rows with mean
        input_embeddings.weight[original_vocab_size:] = mean_input.repeat(
            len(tokenizer) - original_vocab_size, 1
        )

        # Output embeddings (LM head)
        if output_embeddings is not None:
            existing_output = output_embeddings.weight[:original_vocab_size]
            mean_output = existing_output.mean(dim=0, keepdim=True)
            output_embeddings.weight[original_vocab_size:] = mean_output.repeat(
                len(tokenizer) - original_vocab_size, 1
            )

    print(f"Initialized {len(tokenizer) - original_vocab_size} new embedding rows")


def main():
    parser = argparse.ArgumentParser(
        description="Resize Qwen tokenizer to add SID tokens"
    )
    parser.add_argument(
        "--base",
        type=str,
        default="Qwen/Qwen2.5-8B-Instruct",
        help="Base model name or path",
    )
    parser.add_argument(
        "--out",
        type=str,
        required=True,
        help="Output directory for extended model",
    )
    parser.add_argument(
        "--torch_dtype",
        type=str,
        default="bfloat16",
        choices=["float32", "float16", "bfloat16"],
        help="Model dtype",
    )

    args = parser.parse_args()

    # Map dtype string to torch dtype
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    torch_dtype = dtype_map[args.torch_dtype]

    print(f"Loading base model: {args.base}")
    print(f"Using dtype: {args.torch_dtype}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.base,
        trust_remote_code=True,
    )
    original_vocab_size = len(tokenizer)
    print(f"Original vocab size: {original_vocab_size}")

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        args.base,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
        device_map="auto",
    )
    print(f"Model loaded with {model.num_parameters():,} parameters")

    # Add new tokens
    print("\nAdding SID tokens...")
    new_tokens = add_sid_tokens(tokenizer)
    print(f"New vocab size: {len(tokenizer)}")
    print(f"Example new tokens: {new_tokens[:5]} ... {new_tokens[-5:]}")

    # Initialize embeddings
    print("\nInitializing new embeddings...")
    initialize_new_embeddings(model, tokenizer, original_vocab_size)

    # Save extended model and tokenizer
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nSaving to {out_dir}...")
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)

    print("\n=== Verification ===")
    print(f"Tokenizer vocab size: {len(tokenizer)}")
    print(f"Model embedding size: {model.get_input_embeddings().weight.shape[0]}")

    # Test tokenization of SID tokens
    test_tokens = ["<sid_0>", "<sid_512>", "<sid_1023>"]
    for token in test_tokens:
        token_id = tokenizer.convert_tokens_to_ids(token)
        print(f"  {token} -> ID {token_id}")

    print("\nDone!")


if __name__ == "__main__":
    main()
