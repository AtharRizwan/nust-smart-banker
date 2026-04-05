"""
finetune/merge_and_export.py
============================
Merges the trained LoRA adapters into the base Qwen2.5-3B-Instruct model and
saves the full merged model to finetune/outputs/merged_model/.

After merging, update your .env (or configs/settings.py) to point the app at
the local fine-tuned model:
    LLM_MODEL_PATH=e:/Farhan-LLM/nust-smart-banker/finetune/outputs/merged_model

Usage (from project root, with .venv active):
    python finetune/merge_and_export.py
    python finetune/merge_and_export.py --adapter-path finetune/outputs/lora_adapter
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# ── Paths ─────────────────────────────────────────────────────────────────────
DEFAULT_ADAPTER  = ROOT / "finetune" / "outputs" / "lora_adapter"
DEFAULT_MERGED   = ROOT / "finetune" / "outputs" / "merged_model"
CACHE_DIR        = ROOT / ".cache" / "huggingface"


def merge(adapter_path: Path, output_path: Path) -> None:
    print("=" * 60)
    print("NUST Smart Banker — LoRA Merge & Export")
    print("=" * 60)

    if not adapter_path.exists():
        print(f"\n[ERROR] Adapter not found at: {adapter_path}")
        print("  → Run `python finetune/train.py` first.")
        sys.exit(1)

    # Read base model name from adapter config
    import json

    adapter_config_path = adapter_path / "adapter_config.json"
    if adapter_config_path.exists():
        with open(adapter_config_path) as f:
            adapter_config = json.load(f)
        base_model = adapter_config.get("base_model_name_or_path", "Qwen/Qwen2.5-3B-Instruct")
    else:
        base_model = "Qwen/Qwen2.5-3B-Instruct"

    print(f"\n  Base model : {base_model}")
    print(f"  Adapter    : {adapter_path}")
    print(f"  Output     : {output_path}")

    try:
        # ── Unsloth merge path ────────────────────────────────────────────
        from unsloth import FastLanguageModel

        print("\n[1/3] Loading base model + adapter via Unsloth …")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=str(adapter_path),
            dtype=None,
            load_in_4bit=True,
            cache_dir=str(CACHE_DIR),
        )

        print("[2/3] Merging LoRA weights into base model …")
        model = FastLanguageModel.for_inference(model)

        print("[3/3] Saving merged model …")
        output_path.mkdir(parents=True, exist_ok=True)
        model.save_pretrained_merged(
            str(output_path),
            tokenizer,
            save_method="merged_16bit",
        )

    except ImportError:
        # ── HuggingFace PEFT fallback (Windows) ───────────────────────────
        print("\n  [INFO] Unsloth not found — using HuggingFace PEFT for merging.")
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        print("[1/3] Loading base model …")
        tokenizer = AutoTokenizer.from_pretrained(
            base_model, cache_dir=str(CACHE_DIR), trust_remote_code=True
        )
        base = AutoModelForCausalLM.from_pretrained(
            base_model,
            cache_dir=str(CACHE_DIR),
            torch_dtype="auto",
            device_map="auto",
            trust_remote_code=True,
        )

        print("[2/3] Loading and merging LoRA adapter …")
        merged = PeftModel.from_pretrained(base, str(adapter_path))
        merged = merged.merge_and_unload()

        print("[3/3] Saving merged model …")
        output_path.mkdir(parents=True, exist_ok=True)
        merged.save_pretrained(str(output_path))
        tokenizer.save_pretrained(str(output_path))

    print("\n" + "=" * 60)
    print("  ✓  Merged model saved!")
    print(f"     Path: {output_path.resolve()}")
    print("\nTo use the fine-tuned model, add this to your .env file:")
    print(f"  LLM_MODEL_PATH={output_path.resolve()}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge LoRA adapters into base model")
    parser.add_argument(
        "--adapter-path", type=Path, default=DEFAULT_ADAPTER,
        help=f"Path to the LoRA adapter directory (default: {DEFAULT_ADAPTER})"
    )
    parser.add_argument(
        "--output-path", type=Path, default=DEFAULT_MERGED,
        help=f"Where to save the merged model (default: {DEFAULT_MERGED})"
    )
    args = parser.parse_args()
    merge(adapter_path=args.adapter_path, output_path=args.output_path)
