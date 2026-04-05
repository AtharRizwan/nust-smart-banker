"""
finetune/train.py
=================
Fine-tunes Qwen/Qwen2.5-3B-Instruct with QLoRA using Unsloth (preferred)
or HuggingFace PEFT + TRL (Windows fallback).

Usage (from project root, with .venv active):
    # Quick sanity-check (5 steps only):
    python finetune/train.py --max-steps 5

    # Full 3-epoch run:
    python finetune/train.py --epochs 3

    # Custom hyperparameters:
    python finetune/train.py --epochs 5 --lr 1e-4 --batch-size 4

Outputs:
    finetune/outputs/lora_adapter/   ← LoRA weights + tokenizer
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# ── Project root on sys.path ─────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# ── Paths ─────────────────────────────────────────────────────────────────────
TRAIN_FILE    = ROOT / "finetune" / "data" / "train.jsonl"
EVAL_FILE     = ROOT / "finetune" / "data" / "eval.jsonl"
OUTPUT_DIR    = ROOT / "finetune" / "outputs" / "lora_adapter"
CACHE_DIR     = ROOT / ".cache" / "huggingface"

# ── Model & QLoRA defaults ────────────────────────────────────────────────────
BASE_MODEL    = "Qwen/Qwen2.5-3B-Instruct"
LORA_R        = 16
LORA_ALPHA    = 32
LORA_DROPOUT  = 0.05
LORA_TARGETS  = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]


# ─────────────────────────────────────────────────────────────────────────────
# Training entry point
# ─────────────────────────────────────────────────────────────────────────────

def train(
    epochs: int = 3,
    lr: float = 2e-4,
    batch_size: int = 2,
    grad_accum: int = 4,
    max_seq_len: int = 1024,
    max_steps: int = -1,
    warmup_ratio: float = 0.05,
) -> None:
    # ── Pre-flight checks ────────────────────────────────────────────────────
    if not TRAIN_FILE.exists():
        print(f"[ERROR] Training data not found: {TRAIN_FILE}")
        print("  → Run `python finetune/build_dataset.py` first.")
        sys.exit(1)

    print("=" * 60)
    print("NUST Smart Banker — QLoRA Fine-tuning")
    print("=" * 60)
    print(f"  Base model     : {BASE_MODEL}")
    print(f"  Train file     : {TRAIN_FILE}")
    print(f"  Eval file      : {EVAL_FILE}")
    print(f"  Output dir     : {OUTPUT_DIR}")
    print(f"  LoRA r / alpha : {LORA_R} / {LORA_ALPHA}")
    print(f"  Epochs         : {epochs}")
    print(f"  LR             : {lr}")
    print(f"  Batch size     : {batch_size}  (grad_accum={grad_accum})")
    print(f"  Max seq len    : {max_seq_len}")
    if max_steps > 0:
        print(f"  Max steps      : {max_steps}  [DRY-RUN MODE]")
    print()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Set HF cache to E: drive (matches settings.py) ───────────────────────
    os.environ["HF_HOME"]           = str(CACHE_DIR)
    os.environ["HF_HUB_CACHE"]      = str(CACHE_DIR / "hub")
    os.environ["TRANSFORMERS_CACHE"] = str(CACHE_DIR / "transformers")

    # ── Load dataset ─────────────────────────────────────────────────────────
    from datasets import load_dataset  # type: ignore

    print("[1/4] Loading dataset …")
    data_files = {"train": str(TRAIN_FILE)}
    if EVAL_FILE.exists():
        data_files["validation"] = str(EVAL_FILE)

    dataset = load_dataset("json", data_files=data_files)
    print(f"       Train: {len(dataset['train'])} samples")
    if "validation" in dataset:
        print(f"       Eval : {len(dataset['validation'])} samples")

    # ── Try Unsloth first; fall back to PEFT/TRL ────────────────────────────
    try:
        _train_unsloth(dataset, epochs, lr, batch_size, grad_accum,
                       max_seq_len, max_steps, warmup_ratio)
    except ImportError as exc:
        print(f"\n  [INFO] Unsloth not available ({exc}).")
        print("  → Falling back to HuggingFace PEFT + TRL.\n")
        _train_peft(dataset, epochs, lr, batch_size, grad_accum,
                    max_seq_len, max_steps, warmup_ratio)


# ─────────────────────────────────────────────────────────────────────────────
# Unsloth training path
# ─────────────────────────────────────────────────────────────────────────────

def _train_unsloth(dataset, epochs, lr, batch_size, grad_accum,
                   max_seq_len, max_steps, warmup_ratio) -> None:
    from unsloth import FastLanguageModel  # type: ignore
    from trl import SFTTrainer, SFTConfig  # type: ignore

    print("[2/4] Loading model with Unsloth …")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL,
        max_seq_length=max_seq_len,
        dtype=None,           # auto-detect (bf16 on Ampere+, fp16 otherwise)
        load_in_4bit=True,
        cache_dir=str(CACHE_DIR),
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=LORA_TARGETS,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )

    _run_sft(model, tokenizer, dataset, epochs, lr, batch_size, grad_accum,
             max_seq_len, max_steps, warmup_ratio, backend="unsloth")


# ─────────────────────────────────────────────────────────────────────────────
# HuggingFace PEFT + TRL fallback (works on Windows)
# ─────────────────────────────────────────────────────────────────────────────

def _train_peft(dataset, epochs, lr, batch_size, grad_accum,
                max_seq_len, max_steps, warmup_ratio) -> None:
    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training  # type: ignore
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig  # type: ignore
    from trl import SFTTrainer, SFTConfig  # type: ignore

    print("[2/4] Loading model with PEFT (4-bit BitsAndBytes) …")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL,
        cache_dir=str(CACHE_DIR),
        trust_remote_code=True,
        padding_side="right",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        cache_dir=str(CACHE_DIR),
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=LORA_TARGETS,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    _run_sft(model, tokenizer, dataset, epochs, lr, batch_size, grad_accum,
             max_seq_len, max_steps, warmup_ratio, backend="peft")


# ─────────────────────────────────────────────────────────────────────────────
# Shared SFTTrainer run (used by both paths)
# ─────────────────────────────────────────────────────────────────────────────

def _run_sft(model, tokenizer, dataset, epochs, lr, batch_size, grad_accum,
             max_seq_len, max_steps, warmup_ratio, backend: str) -> None:
    from trl import SFTTrainer, SFTConfig  # type: ignore

    # In TRL 1.0.0, if the dataset has a "messages" column, SFTTrainer
    # natively applies the tokenizer's chat template. No manual mapping needed!
    train_data = dataset["train"]
    eval_data  = dataset.get("validation")

    report_to = "wandb" if os.getenv("WANDB_API_KEY") else "none"
    if report_to == "wandb":
        print("       W&B logging enabled.")

    # TRL 1.0.0 API uses `max_length` inside SFTConfig
    sft_config = SFTConfig(
        output_dir=str(OUTPUT_DIR),
        max_length=max_seq_len,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        warmup_ratio=warmup_ratio,
        learning_rate=lr,
        lr_scheduler_type="cosine",
        weight_decay=0.01,
        fp16=False,
        bf16=True,
        logging_steps=10,
        eval_strategy="epoch" if eval_data is not None else "no",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=(eval_data is not None),
        report_to=report_to,
        run_name="nust-smart-banker-qlora",
        max_steps=max_steps,          # -1 = run all epochs; >0 = dry run
        seed=42,
    )

    print(f"[4/4] Starting training ({backend}) …\n")
    
    # TRL 1.0.0 uses `processing_class` instead of `tokenizer`
    # We pass the pre-wrapped model (Unsloth or PEFT) directly.
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_data,
        eval_dataset=eval_data,
        processing_class=tokenizer,
    )

    trainer.train()

    # Save final adapter
    print(f"\n  Saving LoRA adapter to: {OUTPUT_DIR}")
    trainer.save_model(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))

    print("\n" + "=" * 60)
    print("  ✓  Training complete!")
    print(f"     Adapter saved to: {OUTPUT_DIR.resolve()}")
    print("\nNext step: python finetune/merge_and_export.py")
    print("=" * 60)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune NUST Smart Banker with QLoRA")
    parser.add_argument("--epochs",      type=int,   default=3,    help="Number of training epochs")
    parser.add_argument("--lr",          type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--batch-size",  type=int,   default=2,    help="Per-device batch size")
    parser.add_argument("--grad-accum",  type=int,   default=4,    help="Gradient accumulation steps")
    parser.add_argument("--max-seq-len", type=int,   default=1024, help="Maximum sequence length")
    parser.add_argument("--max-steps",   type=int,   default=-1,
                        help="Stop after this many steps (-1 = full training). Use 5 for a dry run.")
    parser.add_argument("--warmup-ratio", type=float, default=0.05, help="LR warmup ratio")
    args = parser.parse_args()

    train(
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        max_seq_len=args.max_seq_len,
        max_steps=args.max_steps,
        warmup_ratio=args.warmup_ratio,
    )
