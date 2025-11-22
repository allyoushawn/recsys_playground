"""Stage B: LoRA fine-tuning with PEFT for memory-efficient training.

Trains only a small subset of parameters using LoRA adapters on the SID recommendation task.
Memory efficient - typically uses ~20-25GB VRAM instead of ~60GB for full fine-tuning.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from datasets import Dataset, disable_progress_bar
from peft import LoraConfig, get_peft_model, TaskType
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
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


def main():
    parser = argparse.ArgumentParser(
        description="Stage B: LoRA fine-tuning with PEFT"
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
        help="Input model path (from Stage A)",
    )
    parser.add_argument(
        "--out_model",
        type=str,
        required=True,
        help="Output LoRA adapter path",
    )
    parser.add_argument(
        "--sid_trie",
        type=str,
        help="Path to sid_trie.pkl (optional, for constrained eval)",
    )
    parser.add_argument(
        "--lora_r",
        type=int,
        default=16,
        help="LoRA rank (default: 16, higher = more parameters)",
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=32,
        help="LoRA alpha (default: 32, scaling factor)",
    )
    parser.add_argument(
        "--lora_dropout",
        type=float,
        default=0.05,
        help="LoRA dropout (default: 0.05)",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=4,
        help="Batch size per device (default: 4, can increase with LoRA)",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=8,
        help="Gradient accumulation steps (default: 8)",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Learning rate (default: 1e-4, higher than full fine-tuning)",
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=3,
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

    print("=== Stage B: LoRA Fine-tuning with PEFT ===")
    print(f"Input model: {args.in_model}")
    print(f"Output adapter: {args.out_model}")
    print(f"LoRA config: r={args.lora_r}, alpha={args.lora_alpha}, dropout={args.lora_dropout}")

    # Disable tqdm progress bars to reduce notebook log spam
    disable_progress_bar()

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
    print(f"Loaded base model with {model.num_parameters():,} parameters")

    # Configure LoRA
    # Target all attention and MLP layers for Qwen architecture
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        bias="none",
    )

    # Apply LoRA
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Enable gradient checkpointing if requested
    if args.gradient_checkpointing:
        model.enable_input_require_grads()
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

    # Data collator
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
        disable_tqdm=True,
        log_level="warning",  # Only show warnings and errors
        logging_first_step=False,  # Skip logging the first step
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

    # Save LoRA adapter
    print(f"\nSaving LoRA adapter to {args.out_model}...")
    model.save_pretrained(args.out_model)
    tokenizer.save_pretrained(args.out_model)

    print("\nDone! To use this model:")
    print(f"  1. Load base model from: {args.in_model}")
    print(f"  2. Load LoRA adapter from: {args.out_model}")
    print("  3. Use PeftModel.from_pretrained() to merge them")


if __name__ == "__main__":
    main()
