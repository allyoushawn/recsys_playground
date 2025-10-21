"""Stage A: Fine-tune only embeddings to learn new SID tokens.

Freezes all model parameters except input and output embeddings.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)


def load_dialogs(path: str) -> list[dict]:
    """Load JSONL dialogs."""
    import jsonlines

    dialogs = []
    with jsonlines.open(path) as reader:
        for obj in reader:
            dialogs.append(obj)
    return dialogs


def preprocess_dialog(example: dict, tokenizer, max_length: int = 512) -> dict:
    """Convert dialog to tokenized format for causal LM.

    Args:
        example: Dict with "messages" key
        tokenizer: Tokenizer instance
        max_length: Maximum sequence length

    Returns:
        Dict with "input_ids", "attention_mask", "labels"
    """
    messages = example["messages"]

    # Apply chat template
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )

    # Tokenize with padding to max_length
    encoded = tokenizer(
        text,
        truncation=True,
        max_length=max_length,
        padding="max_length",  # Pad to max_length for consistent batch sizes
        return_tensors=None,  # Return lists, not tensors
    )

    # For causal LM, labels = input_ids
    # Set padding token labels to -100 so they're ignored in loss
    labels = encoded["input_ids"].copy()
    labels = [
        -100 if token_id == tokenizer.pad_token_id else token_id
        for token_id in labels
    ]
    encoded["labels"] = labels

    return encoded


def freeze_all_except_embeddings(model):
    """Freeze all parameters except input and output embeddings."""
    # Freeze all parameters
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze input embeddings
    for param in model.get_input_embeddings().parameters():
        param.requires_grad = True

    # Unfreeze output embeddings (LM head)
    output_embeddings = model.get_output_embeddings()
    if output_embeddings is not None:
        for param in output_embeddings.parameters():
            param.requires_grad = True

    # Count trainable params
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")


def main():
    parser = argparse.ArgumentParser(
        description="Stage A: Fine-tune embeddings only"
    )
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to dialogs_train.jsonl",
    )
    parser.add_argument(
        "--valid",
        type=str,
        required=True,
        help="Path to dialogs_valid.jsonl",
    )
    parser.add_argument(
        "--in_model",
        type=str,
        required=True,
        help="Input model path (base or resized)",
    )
    parser.add_argument(
        "--out_model",
        type=str,
        required=True,
        help="Output model path",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-4,
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--warmup_ratio",
        type=float,
        default=0.03,
    )
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=50,
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=500,
    )
    parser.add_argument(
        "--bf16",
        action="store_true",
        help="Use bfloat16",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Enable gradient checkpointing",
    )

    args = parser.parse_args()

    print("=== Stage A: Vocabulary Extension ===")
    print(f"Input model: {args.in_model}")
    print(f"Output model: {args.out_model}")

    # Load tokenizer and model
    print("\nLoading model...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.in_model,
        trust_remote_code=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.in_model,
        torch_dtype=torch.bfloat16 if args.bf16 else torch.float32,
        trust_remote_code=True,
        device_map="auto",
    )
    print(f"Loaded model with {model.num_parameters():,} parameters")

    # Freeze all except embeddings
    print("\nFreezing parameters...")
    freeze_all_except_embeddings(model)

    # Enable gradient checkpointing if requested
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        print("Gradient checkpointing enabled")

    # Load data
    print("\nLoading data...")
    train_dialogs = load_dialogs(args.data)
    valid_dialogs = load_dialogs(args.valid)
    print(f"Train: {len(train_dialogs)} dialogs")
    print(f"Valid: {len(valid_dialogs)} dialogs")

    # Convert to datasets
    train_dataset = Dataset.from_list(train_dialogs)
    valid_dataset = Dataset.from_list(valid_dialogs)

    # Preprocess
    print("\nPreprocessing...")
    max_length = 512
    train_dataset = train_dataset.map(
        lambda x: preprocess_dialog(x, tokenizer, max_length=max_length),
        remove_columns=train_dataset.column_names,
        desc="Preprocessing train",
    )
    valid_dataset = valid_dataset.map(
        lambda x: preprocess_dialog(x, tokenizer, max_length=max_length),
        remove_columns=valid_dataset.column_names,
        desc="Preprocessing valid",
    )

    # Data collator - no need for padding since we already padded
    # Use default data collator that just converts to tensors
    from transformers import default_data_collator
    data_collator = default_data_collator

    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.out_model,
        overwrite_output_dir=True,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_strategy="steps",
        eval_steps=args.save_steps,
        save_total_limit=2,
        bf16=args.bf16,
        dataloader_num_workers=2,
        remove_unused_columns=False,
        report_to="none",
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        data_collator=data_collator,
    )

    # Train
    print("\nStarting training...")
    trainer.train()

    # Save final model
    print(f"\nSaving final model to {args.out_model}...")
    trainer.save_model(args.out_model)
    tokenizer.save_pretrained(args.out_model)

    print("\nDone!")


if __name__ == "__main__":
    main()
