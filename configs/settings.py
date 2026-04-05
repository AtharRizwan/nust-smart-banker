"""
Central configuration for NUST Smart Banker.
All tuneable constants live here so every module imports from one place.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env from the project root (silently a no-op if the file is absent)
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

# ─── Hugging Face ─────────────────────────────────────────────────────────────
# Required to download gated models without running `huggingface-cli login`.
# Set HF_TOKEN in a .env file at the project root (see .env.example).
HF_TOKEN: str = os.getenv("HF_TOKEN", "")

# ─── Project Paths ────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent

# Set Hugging Face and Torch cache to the project directory to avoid C: drive space issues
os.environ["HF_HOME"] = str(BASE_DIR / ".cache" / "huggingface")
os.environ["HF_HUB_CACHE"] = str(BASE_DIR / ".cache" / "huggingface" / "hub")
os.environ["TRANSFORMERS_CACHE"] = str(
    BASE_DIR / ".cache" / "huggingface" / "transformers"
)
os.environ["SENTENCE_TRANSFORMERS_HOME"] = str(
    BASE_DIR / ".cache" / "sentence_transformers"
)
os.environ["TORCH_HOME"] = str(BASE_DIR / ".cache" / "torch")

DATA_DIR = BASE_DIR / "data"
QDRANT_DIR = BASE_DIR / "qdrant_data"
UPLOADED_DOCS_DIR = BASE_DIR / "uploaded_docs"


# ─── Qdrant ───────────────────────────────────────────────────────────────────
QDRANT_COLLECTION_NAME = "nust_bank_docs"
# Embedding dimension for BAAI/bge-m3 dense vector
EMBEDDING_DIM = 1024

# ─── Embedding Model ──────────────────────────────────────────────────────────
EMBEDDING_MODEL_NAME = "BAAI/bge-m3"

# ─── Language Model ───────────────────────────────────────────────────────────
LLM_MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
# Optional: override with a local fine-tuned model path (set LLM_MODEL_PATH in .env)
# e.g.  LLM_MODEL_PATH=e:/Farhan-LLM/nust-smart-banker/finetune/outputs/merged_model
# When set, this takes priority over LLM_MODEL_NAME.
LLM_MODEL_PATH: str = os.getenv("LLM_MODEL_PATH", "")
# Effective model identifier — used everywhere a model name/path is needed
LLM_EFFECTIVE_MODEL: str = LLM_MODEL_PATH if LLM_MODEL_PATH else LLM_MODEL_NAME
LLM_MAX_NEW_TOKENS = 512
LLM_TEMPERATURE = 0.2
LLM_REPETITION_PENALTY = 1.1
# Load in 4-bit to fit within 6 GB VRAM
LLM_LOAD_IN_4BIT = True

# ─── Chunking ─────────────────────────────────────────────────────────────────
CHUNK_SIZE = 1536
CHUNK_OVERLAP = 128

# ─── Retrieval ────────────────────────────────────────────────────────────────
# Number of chunks to retrieve per query
RETRIEVAL_TOP_K = 5
# Dense score threshold below which we consider the query out-of-domain
# (normalised cosine similarity; range 0-1)
RELEVANCE_THRESHOLD = 0.35

# ─── XLSX Parsing ─────────────────────────────────────────────────────────────
# Sheets in the product knowledge XLSX that should NOT be parsed for Q&A
XLSX_SKIP_SHEETS = {"Sheet1"}

# ─── Guardrails ───────────────────────────────────────────────────────────────
GUARDRAILS_CONFIG_DIR = BASE_DIR / "configs"

# ─── Presidio (PII anonymisation) ─────────────────────────────────────────────
# Entity types to detect and anonymise
PII_ENTITIES = [
    "PHONE_NUMBER",
    "EMAIL_ADDRESS",
    "IBAN_CODE",
    "CREDIT_CARD",
]
