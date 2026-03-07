"""
setup_models.py — Pre-download all models for NUST Smart Banker.

Run this ONCE before starting the application:
    python setup_models.py

What it downloads (all to the project's .cache/ directory on E: drive):
  1. BAAI/bge-m3          – embedding model (~570 MB, safetensors)
  2. Qwen/Qwen2.5-3B-Instruct – LLM tokeniser + weights (~6 GB fp16, or ~1.8 GB in 4-bit at runtime)
  3. en_core_web_lg       – spaCy NLP model for Presidio PII detection (~700 MB)

After this script completes the app will start without any network downloads.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

# ── 1. Apply cache env-vars BEFORE any HF/torch imports ──────────────────────
# Must happen before importing configs.settings so the paths exist first.
BASE_DIR = Path(__file__).resolve().parent
CACHE_DIR = BASE_DIR / ".cache"

import os  # noqa: E402 (must come after BASE_DIR is defined)

os.environ["HF_HOME"] = str(CACHE_DIR / "huggingface")
os.environ["HF_HUB_CACHE"] = str(CACHE_DIR / "huggingface" / "hub")
os.environ["TRANSFORMERS_CACHE"] = str(CACHE_DIR / "huggingface" / "transformers")
os.environ["SENTENCE_TRANSFORMERS_HOME"] = str(CACHE_DIR / "sentence_transformers")
os.environ["TORCH_HOME"] = str(CACHE_DIR / "torch")

# Create cache directories upfront
for sub in ["huggingface/hub", "sentence_transformers", "torch"]:
    (CACHE_DIR / sub).mkdir(parents=True, exist_ok=True)

# ── helpers ───────────────────────────────────────────────────────────────────

def step(msg: str) -> None:
    print(f"\n{'─' * 60}")
    print(f"  {msg}")
    print(f"{'─' * 60}")


def ok(msg: str) -> None:
    print(f"  ✓ {msg}")


def warn(msg: str) -> None:
    print(f"  ⚠  {msg}")


# ── 2. Import project settings (re-applies env-vars for safety) ───────────────
sys.path.insert(0, str(BASE_DIR))
from configs.settings import EMBEDDING_MODEL_NAME, LLM_MODEL_NAME  # noqa: E402


# ══════════════════════════════════════════════════════════════════════════════
# Step 1 — Embedding model (BAAI/bge-m3)
# ══════════════════════════════════════════════════════════════════════════════
step(f"Downloading embedding model: {EMBEDDING_MODEL_NAME}")

try:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    # Quick smoke-test to ensure weights loaded correctly
    _ = model.encode("test", normalize_embeddings=True)
    ok(f"Embedding model ready  →  {CACHE_DIR / 'sentence_transformers'}")
    del model
except Exception as exc:
    warn(f"Embedding model download failed: {exc}")
    sys.exit(1)


# ══════════════════════════════════════════════════════════════════════════════
# Step 2 — LLM tokeniser (Qwen2.5-3B-Instruct)
#          We download the tokeniser separately so any tokeniser-only errors
#          are caught early and because it's tiny (~2 MB).
# ══════════════════════════════════════════════════════════════════════════════
step(f"Downloading tokeniser: {LLM_MODEL_NAME}")

try:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        LLM_MODEL_NAME,
        trust_remote_code=True,
    )
    ok(f"Tokeniser ready  →  {CACHE_DIR / 'huggingface' / 'hub'}")
    del tokenizer
except Exception as exc:
    warn(f"Tokeniser download failed: {exc}")
    sys.exit(1)


# ══════════════════════════════════════════════════════════════════════════════
# Step 3 — LLM weights (Qwen2.5-3B-Instruct)
#          Downloads in fp16 (~6 GB).  At runtime bitsandbytes quantises to
#          4-bit on-the-fly; no separate quantised download exists on HF Hub.
# ══════════════════════════════════════════════════════════════════════════════
step(f"Downloading LLM weights: {LLM_MODEL_NAME}  (~6 GB, this will take a while…)")

try:
    import torch
    from transformers import AutoModelForCausalLM

    print(f"  torch version : {torch.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU           : {torch.cuda.get_device_name(0)}")
        print(f"  VRAM          : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Download with weights_only safetensors — no GPU needed for the download step
    model = AutoModelForCausalLM.from_pretrained(
        LLM_MODEL_NAME,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map="cpu",          # stay on CPU during download-only step
        low_cpu_mem_usage=True,    # don't materialise full tensor in RAM
    )
    ok(f"LLM weights ready  →  {CACHE_DIR / 'huggingface' / 'hub'}")
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
except Exception as exc:
    warn(f"LLM weight download failed: {exc}")
    sys.exit(1)


# ══════════════════════════════════════════════════════════════════════════════
# Step 4 — spaCy model (en_core_web_lg) for Presidio PII detection
# ══════════════════════════════════════════════════════════════════════════════
step("Downloading spaCy model: en_core_web_lg  (~700 MB)")

try:
    import spacy  # noqa: F401

    try:
        nlp = spacy.load("en_core_web_lg")
        ok("spaCy en_core_web_lg already installed.")
        del nlp
    except OSError:
        print("  Downloading en_core_web_lg via spaCy CLI…")
        result = subprocess.run(
            [sys.executable, "-m", "spacy", "download", "en_core_web_lg"],
            check=True,
            capture_output=False,
        )
        ok("spaCy en_core_web_lg installed.")
except ImportError:
    warn("spaCy not installed — PII anonymisation will use regex fallback only.")
except subprocess.CalledProcessError as exc:
    warn(f"spaCy download failed (return code {exc.returncode}).")


# ══════════════════════════════════════════════════════════════════════════════
# Done
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'═' * 60}")
print("  ✅  All models downloaded successfully!")
print(f"  Cache location: {CACHE_DIR}")
print()
print("  You can now start the application with:")
print("      streamlit run app.py")
print(f"{'═' * 60}\n")
