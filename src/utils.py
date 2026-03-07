"""
Shared text-cleaning and chunking utilities used across the pipeline.
"""

from __future__ import annotations

import re
import unicodedata
from typing import List

from langchain_text_splitters import RecursiveCharacterTextSplitter

from configs.settings import CHUNK_SIZE, CHUNK_OVERLAP


# ─── Text Cleaning ────────────────────────────────────────────────────────────


def clean_text(text: str) -> str:
    """
    Normalise and clean raw text extracted from data sources:
      - Normalise unicode (NFKC)
      - Collapse runs of whitespace / newlines
      - Strip bullet unicode noise (·, •, \xa0, \t etc.)
      - Remove Excel formula artefacts (strings starting with =')
    """
    if not text or not isinstance(text, str):
        return ""

    # Normalise unicode (handles \xa0, fancy quotes, etc.)
    text = unicodedata.normalize("NFKC", text)

    # Drop Excel formula cell references like ='Rate Sheet July 1 2024'!D23
    text = re.sub(r"='[^']+'![A-Z]+\d+", "", text)

    # Remove bullet noise characters and leading dots
    text = re.sub(r"[·•◦▪▸‣⁃]", "-", text)

    # Collapse horizontal whitespace (tabs, multiple spaces)
    text = re.sub(r"[ \t]+", " ", text)

    # Collapse excess newlines (keep at most 2)
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Strip leading/trailing whitespace per line
    lines = [line.strip() for line in text.splitlines()]
    text = "\n".join(line for line in lines if line)

    return text.strip()


def format_qa_chunk(question: str, answer: str, source: str = "") -> str:
    """
    Format a Q&A pair as a single text chunk for embedding.
    Including the question in the chunk improves retrieval precision because
    the embedding of 'Q: ... A: ...' aligns well with user query embeddings.
    """
    q = clean_text(question)
    a = clean_text(answer)
    chunk = f"Q: {q}\nA: {a}"
    if source:
        chunk = f"[Source: {source}]\n{chunk}"
    return chunk


# ─── Chunking ─────────────────────────────────────────────────────────────────


def get_text_splitter() -> RecursiveCharacterTextSplitter:
    """Return the project-standard text splitter."""
    return RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )


def split_documents(docs: list) -> list:
    """
    Split a list of LangChain Document objects into smaller chunks.
    Metadata is preserved on every child chunk.
    """
    splitter = get_text_splitter()
    return splitter.split_documents(docs)


# ─── Misc ─────────────────────────────────────────────────────────────────────


def is_meaningful(text: str, min_chars: int = 20) -> bool:
    """Return True if the text is long enough to be worth indexing."""
    return len(clean_text(text)) >= min_chars
