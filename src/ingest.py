"""
Data ingestion pipeline for NUST Smart Banker.

Responsibilities:
  1. Load documents from JSON (FAQ) and XLSX (Product Knowledge) files.
  2. Anonymise PII using Microsoft Presidio before any text is stored.
  3. Chunk documents with RecursiveCharacterTextSplitter.
  4. Embed chunks with BAAI/bge-m3 via the shared Retriever.
  5. Upsert vectors + payloads into the Qdrant collection (persisted to disk).

Usage:
    python -m src.ingest               # ingest default data files
    python -m src.ingest --file path   # ingest a single new file (real-time update)
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import uuid
from pathlib import Path
from typing import List, Optional

import openpyxl
from langchain_core.documents import Document

from configs.settings import (
    FAQ_JSON_PATH,
    PII_ENTITIES,
    PRODUCT_XLSX_PATH,
    UPLOADED_DOCS_DIR,
    XLSX_SKIP_SHEETS,
)
from src.utils import clean_text, format_qa_chunk, is_meaningful, split_documents

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")


# ─── PII Anonymisation ────────────────────────────────────────────────────────


def _build_anonymizer():
    """
    Lazily build Presidio AnalyzerEngine + AnonymizerEngine.
    Returns (analyzer, anonymizer) or (None, None) if Presidio is unavailable.
    """
    try:
        from presidio_analyzer import AnalyzerEngine
        from presidio_anonymizer import AnonymizerEngine

        analyzer = AnalyzerEngine()
        anonymizer = AnonymizerEngine()
        logger.info("Presidio anonymizer initialised.")
        return analyzer, anonymizer
    except Exception as exc:
        logger.warning("Presidio unavailable (%s) – PII anonymisation skipped.", exc)
        return None, None


_analyzer, _anonymizer = _build_anonymizer()


def anonymize_text(text: str) -> str:
    """
    Detect and replace PII entities in *text* with <ENTITY_TYPE> placeholders.
    Falls back to a regex-based approach if Presidio is unavailable.
    """
    if not text:
        return text

    # ── Presidio path ────────────────────────────────────────────────────────
    if _analyzer and _anonymizer:
        try:
            results = _analyzer.analyze(
                text=text,
                entities=PII_ENTITIES,
                language="en",
            )
            if results:
                anonymized = _anonymizer.anonymize(text=text, analyzer_results=results)
                return anonymized.text
            return text
        except Exception as exc:
            logger.debug("Presidio anonymization error: %s", exc)

    # ── Regex fallback ───────────────────────────────────────────────────────
    # Email
    text = re.sub(r"[\w.+-]+@[\w-]+\.[a-zA-Z]{2,}", "<EMAIL>", text)
    # Pakistani mobile numbers (+92-XXX-XXXXXXX or 03XX-XXXXXXX)
    text = re.sub(r"(\+92[-\s]?|0)3\d{2}[-\s]?\d{7}", "<PHONE>", text)
    # Generic phone numbers
    text = re.sub(r"\b\d{3,4}[-.\s]\d{3,4}[-.\s]\d{4}\b", "<PHONE>", text)
    # IBAN (Pakistan: PK + 22 chars)
    text = re.sub(r"\bPK\d{2}[A-Z0-9]{20}\b", "<IBAN>", text)
    # CNIC (XXXXX-XXXXXXX-X)
    text = re.sub(r"\b\d{5}-\d{7}-\d\b", "<CNIC>", text)

    return text


# ─── JSON FAQ Loader ──────────────────────────────────────────────────────────


def load_json_faq(path: Path = FAQ_JSON_PATH) -> List[Document]:
    """
    Parse funds_transfer_app_features_faq.json and return a list of Documents.
    Each document corresponds to one Q&A pair; metadata carries the category.
    """
    logger.info("Loading FAQ JSON: %s", path)
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)

    documents: List[Document] = []
    categories = data.get("categories", [])

    for cat in categories:
        category_name = clean_text(cat.get("category", "General"))
        for item in cat.get("questions", []):
            question = clean_text(str(item.get("question", "")))
            answer = clean_text(str(item.get("answer", "")))

            if not is_meaningful(question) or not is_meaningful(answer):
                continue

            # Anonymise before storing
            question = anonymize_text(question)
            answer = anonymize_text(answer)

            content = format_qa_chunk(question, answer, source=category_name)
            documents.append(
                Document(
                    page_content=content,
                    metadata={
                        "source": "funds_transfer_faq",
                        "category": category_name,
                        "doc_type": "faq",
                    },
                )
            )

    logger.info("  → %d Q&A documents loaded from FAQ JSON.", len(documents))
    return documents


# ─── XLSX Product Knowledge Loader ───────────────────────────────────────────


def _extract_sheet_text(ws) -> List[tuple[str, str]]:
    """
    Extract Q&A pairs from a single XLSX worksheet.

    Strategy:
      - Scan every row; collect all non-None cell values by joining them.
      - A row whose joined text ends with '?' is treated as a *question*.
      - All subsequent non-empty rows until the next question are joined as
        the *answer* for that question.
      - Rows that are clearly navigation artefacts ('Main', sheet names) are
        dropped.
    """
    pairs: List[tuple[str, str]] = []
    current_q: Optional[str] = None
    answer_parts: List[str] = []

    # Words that indicate a navigation/header cell – not Q&A content
    nav_pattern = re.compile(r"^(main|rate sheet|sheet\d*)$", re.IGNORECASE)

    for row in ws.iter_rows(values_only=True):
        # Gather all non-None cell text in the row, skip formula refs and nav
        cell_texts = []
        for cell in row:
            if cell is None:
                continue
            val = clean_text(str(cell))
            if not val:
                continue
            if nav_pattern.match(val.strip()):
                continue
            # Skip Excel formula references
            if val.startswith("='"):
                continue
            cell_texts.append(val)

        if not cell_texts:
            continue

        row_text = " ".join(cell_texts).strip()

        if not is_meaningful(row_text, min_chars=10):
            continue

        # Decide if this row is a question or answer continuation
        is_question = row_text.rstrip().endswith("?") or (
            # Numbered questions like "1.What are the benefits..."
            re.match(r"^\d+[\.\)]", row_text) and len(row_text) < 200
        )

        if is_question:
            # Save the previous Q&A pair before starting a new question
            if current_q and answer_parts:
                pairs.append((current_q, "\n".join(answer_parts)))
            current_q = row_text
            answer_parts = []
        else:
            # Accumulate answer lines
            if current_q is not None:
                answer_parts.append(row_text)

    # Flush last pair
    if current_q and answer_parts:
        pairs.append((current_q, "\n".join(answer_parts)))

    return pairs


def load_xlsx_products(path: Path = PRODUCT_XLSX_PATH) -> List[Document]:
    """
    Parse NUST Bank-Product-Knowledge.xlsx.
    Each sheet (except skip-listed ones) represents one product.
    Returns a list of Documents, one per Q&A pair extracted.
    """
    logger.info("Loading Product XLSX: %s", path)
    wb = openpyxl.load_workbook(path, data_only=True)
    documents: List[Document] = []

    for sheet_name in wb.sheetnames:
        if sheet_name.strip() in XLSX_SKIP_SHEETS:
            logger.debug("  Skipping sheet: %s", sheet_name)
            continue

        ws = wb[sheet_name]
        pairs = _extract_sheet_text(ws)

        for question, answer in pairs:
            question = anonymize_text(question)
            answer = anonymize_text(answer)

            if not is_meaningful(answer):
                continue

            content = format_qa_chunk(question, answer, source=sheet_name.strip())
            documents.append(
                Document(
                    page_content=content,
                    metadata={
                        "source": "product_knowledge",
                        "product": sheet_name.strip(),
                        "doc_type": "product_faq",
                    },
                )
            )

        logger.info("  → Sheet %-25s : %d Q&A pairs", sheet_name, len(pairs))

    logger.info("  → %d total documents loaded from XLSX.", len(documents))
    return documents


# ─── Generic Plain-Text / JSON Loader (for real-time uploads) ─────────────────


def load_text_file(path: Path, source_label: str = "") -> List[Document]:
    """
    Load a plain .txt file and return it as a single Document.
    Used for real-time document uploads via the UI.
    """
    text = path.read_text(encoding="utf-8", errors="replace")
    text = clean_text(anonymize_text(text))
    if not is_meaningful(text, min_chars=50):
        return []
    label = source_label or path.stem
    return [
        Document(
            page_content=text,
            metadata={"source": label, "doc_type": "upload", "filename": path.name},
        )
    ]


def load_uploaded_json(path: Path, source_label: str = "") -> List[Document]:
    """
    Load an uploaded JSON file that follows the same FAQ schema as the main
    dataset (categories → questions), or falls back to loading it as plain text.
    """
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        if "categories" in data:
            # Reuse the FAQ loader with the new path
            return load_json_faq(path)
    except Exception:
        pass
    # Fallback: treat it as raw text
    return load_text_file(path, source_label)


# ─── Main Ingestion Entry Point ───────────────────────────────────────────────


def ingest_all() -> int:
    """
    Load all default data sources, chunk them, embed, and upsert to Qdrant.
    Returns the total number of chunks indexed.
    """
    # Import here to avoid circular dependency at module load time
    from src.retriever import get_retriever

    logger.info("=== Starting full data ingestion ===")

    docs: List[Document] = []
    docs.extend(load_json_faq())
    docs.extend(load_xlsx_products())

    if not docs:
        logger.error("No documents loaded — check data files.")
        return 0

    logger.info("Total raw documents: %d", len(docs))
    chunks = split_documents(docs)
    logger.info("Total chunks after splitting: %d", len(chunks))

    retriever = get_retriever()
    retriever.upsert_chunks(chunks)
    logger.info("=== Ingestion complete: %d chunks indexed. ===", len(chunks))
    return len(chunks)


def ingest_file(file_path: str | Path, source_label: str = "") -> int:
    """
    Ingest a single new file into the existing Qdrant collection.
    Supports .txt, .json, and .xlsx.  Used by the UI for real-time updates.
    Returns the number of newly indexed chunks.
    """
    from src.retriever import get_retriever

    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    logger.info("Ingesting new file: %s", path.name)

    ext = path.suffix.lower()
    if ext == ".json":
        docs = load_uploaded_json(path, source_label or path.stem)
    elif ext in {".xlsx", ".xls"}:
        docs = load_xlsx_products(path)
    else:
        docs = load_text_file(path, source_label or path.stem)

    if not docs:
        logger.warning("No meaningful content found in %s", path.name)
        return 0

    chunks = split_documents(docs)
    retriever = get_retriever()
    retriever.upsert_chunks(chunks)
    logger.info("Ingested %d chunks from %s.", len(chunks), path.name)
    return len(chunks)


# ─── CLI ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NUST Smart Banker – Ingest data")
    parser.add_argument(
        "--file",
        type=str,
        default=None,
        help="Path to a single file to ingest (optional; omit to ingest all defaults)",
    )
    parser.add_argument(
        "--label",
        type=str,
        default="",
        help="Human-readable source label for the uploaded file",
    )
    args = parser.parse_args()

    if args.file:
        n = ingest_file(args.file, args.label)
        print(f"Indexed {n} chunks from {args.file}.")
    else:
        n = ingest_all()
        print(f"Full ingestion complete: {n} chunks indexed.")
