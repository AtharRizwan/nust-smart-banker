"""
Embedding and retrieval layer for NUST Smart Banker.

Provides:
  - BankRetriever: wraps BAAI/bge-m3 embeddings + Qdrant persistent collection
  - Hybrid search: dense cosine similarity + BM25 keyword re-ranking via
    Reciprocal Rank Fusion (RRF)
  - upsert_chunks(): index new Document chunks at any time (real-time updates)
  - search(): return top-K most relevant chunks for a query string
"""

from __future__ import annotations

import logging
import uuid
from typing import List, Tuple

from langchain_core.documents import Document
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from rank_bm25 import BM25Okapi

# configs.settings MUST be imported before sentence_transformers so that
# SENTENCE_TRANSFORMERS_HOME / HF_HOME env-vars are set before the library loads.
from configs.settings import (
    EMBEDDING_DIM,
    EMBEDDING_MODEL_NAME,
    HF_TOKEN,
    QDRANT_COLLECTION_NAME,
    QDRANT_DIR,
    RETRIEVAL_TOP_K,
)
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

# ─── Singleton ────────────────────────────────────────────────────────────────
_retriever_instance: "BankRetriever | None" = None


def get_retriever() -> "BankRetriever":
    """Return (or lazily create) the global BankRetriever singleton."""
    global _retriever_instance
    if _retriever_instance is None:
        _retriever_instance = BankRetriever()
    return _retriever_instance


# ─── BankRetriever ────────────────────────────────────────────────────────────


class BankRetriever:
    """
    Manages BAAI/bge-m3 embeddings and a persistent Qdrant collection.

    Hybrid search workflow:
      1. Dense search  – cosine similarity between query embedding and stored
         document embeddings.  Returns top K*3 candidates.
      2. BM25 re-rank  – keyword-based scoring over the candidate texts.
      3. RRF fusion    – combines both ranked lists to produce a final top-K.
    """

    def __init__(self) -> None:
        import torch

        # token=None is treated as "no token" by sentence-transformers
        _hf_token: str | None = HF_TOKEN or None
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info("Loading embedding model on %s: %s", device, EMBEDDING_MODEL_NAME)
        # `token` kwarg was added in sentence-transformers 2.3; fall back to
        # the older `use_auth_token` parameter for backwards compatibility.
        try:
            self.embed_model = SentenceTransformer(
                EMBEDDING_MODEL_NAME, device=device, token=_hf_token
            )
        except TypeError:
            self.embed_model = SentenceTransformer(
                EMBEDDING_MODEL_NAME, device=device, use_auth_token=_hf_token
            )

        # Persist Qdrant to disk so vectors survive app restarts
        QDRANT_DIR.mkdir(parents=True, exist_ok=True)
        self.client = QdrantClient(path=str(QDRANT_DIR))
        logger.info("Qdrant client connected (path=%s).", QDRANT_DIR)

        self._ensure_collection()

        # BM25 index: built per-search over the dense-search candidate set
        # (~top_k * 3 docs).  No caching — rebuilding over ~30 candidates is
        # fast enough.  The fields below are kept for future optimisation but
        # are currently unused by the search path.
        self._bm25: BM25Okapi | None = None
        self._bm25_docs: List[Document] | None = None

    # ── Collection Management ─────────────────────────────────────────────────

    def _ensure_collection(self) -> None:
        """Create the Qdrant collection if it does not already exist."""
        existing = [c.name for c in self.client.get_collections().collections]
        if QDRANT_COLLECTION_NAME not in existing:
            self.client.create_collection(
                collection_name=QDRANT_COLLECTION_NAME,
                vectors_config=qmodels.VectorParams(
                    size=EMBEDDING_DIM,
                    distance=qmodels.Distance.COSINE,
                ),
            )
            logger.info("Created Qdrant collection '%s'.", QDRANT_COLLECTION_NAME)
        else:
            logger.info(
                "Qdrant collection '%s' already exists (%d vectors).",
                QDRANT_COLLECTION_NAME,
                self.count(),
            )

    def count(self) -> int:
        """Return the number of vectors currently in the collection."""
        try:
            info = self.client.get_collection(QDRANT_COLLECTION_NAME)
            # points_count is the non-deprecated field in recent Qdrant versions
            return info.points_count or info.vectors_count or 0
        except Exception:
            return 0

    # ── Embedding ─────────────────────────────────────────────────────────────

    def embed(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a list of strings using BGE-M3.
        BGE models perform best with a query instruction prefix for queries;
        for documents we embed as-is.
        """
        return self.embed_model.encode(
            texts,
            batch_size=32,
            show_progress_bar=False,
            normalize_embeddings=True,  # cosine similarity == dot product
        ).tolist()

    def embed_query(self, query: str) -> List[float]:
        """Embed a single query string with the BGE instruction prefix."""
        # BGE-M3 works best with this instruction prefix for retrieval queries
        prefixed = f"Represent this sentence for searching relevant passages: {query}"
        return self.embed_model.encode(
            prefixed,
            normalize_embeddings=True,
        ).tolist()

    # ── Upsert ────────────────────────────────────────────────────────────────

    def upsert_chunks(self, chunks: List[Document]) -> None:
        """
        Embed a list of Document chunks and upsert them into Qdrant.
        Duplicate content is not de-duplicated at this stage; callers should
        ensure they don't ingest the same file twice unless intended.
        """
        if not chunks:
            logger.warning("upsert_chunks called with empty chunk list.")
            return

        texts = [c.page_content for c in chunks]
        logger.info("Embedding %d chunks…", len(chunks))
        vectors = self.embed(texts)

        points = [
            qmodels.PointStruct(
                id=str(uuid.uuid4()),
                vector=vec,
                payload={
                    "page_content": chunk.page_content,
                    **chunk.metadata,
                },
            )
            for chunk, vec in zip(chunks, vectors)
        ]

        # Upsert in batches of 256 to keep memory usage bounded
        batch_size = 256
        for i in range(0, len(points), batch_size):
            self.client.upsert(
                collection_name=QDRANT_COLLECTION_NAME,
                points=points[i : i + batch_size],
            )

        logger.info(
            "Upserted %d vectors into '%s'.", len(points), QDRANT_COLLECTION_NAME
        )

        # Invalidate BM25 index so it rebuilds on next search
        self._bm25 = None
        self._bm25_docs = []

    # ── BM25 Index ────────────────────────────────────────────────────────────

    def _build_bm25(self, candidates: List[Document]) -> BM25Okapi:
        """Build a BM25 index over a candidate document list."""
        tokenized = [doc.page_content.lower().split() for doc in candidates]
        return BM25Okapi(tokenized)

    # ── Search ────────────────────────────────────────────────────────────────

    def search(
        self,
        query: str,
        top_k: int = RETRIEVAL_TOP_K,
    ) -> List[Tuple[Document, float]]:
        """
        Hybrid search: dense vector search + BM25 re-rank via RRF.

        Returns a list of (Document, score) tuples sorted by relevance,
        highest first.  Score is the RRF-fused normalised score in [0, 1].
        """
        # ── 1. Dense search (retrieve top_k * 3 candidates) ──────────────────
        query_vec = self.embed_query(query)
        dense_k = min(top_k * 3, max(self.count(), 1))

        # qdrant-client >= 1.7 replaced client.search() with client.query_points()
        response = self.client.query_points(
            collection_name=QDRANT_COLLECTION_NAME,
            query=query_vec,
            limit=dense_k,
            with_payload=True,
            with_vectors=False,
        )
        dense_results = response.points

        if not dense_results:
            return []

        # Convert to Documents
        candidate_docs: List[Document] = []
        dense_scores: List[float] = []
        for hit in dense_results:
            payload = hit.payload or {}
            content = payload.pop("page_content", "")
            candidate_docs.append(Document(page_content=content, metadata=payload))
            dense_scores.append(float(hit.score))

        # ── 2. BM25 re-rank over candidates ──────────────────────────────────
        bm25 = self._build_bm25(candidate_docs)
        query_tokens = query.lower().split()
        bm25_raw_scores = bm25.get_scores(query_tokens)

        # Normalise BM25 scores to [0, 1]
        bm25_max = max(bm25_raw_scores) if max(bm25_raw_scores) > 0 else 1.0
        bm25_norm = [s / bm25_max for s in bm25_raw_scores]

        # ── 3. Reciprocal Rank Fusion ─────────────────────────────────────────
        # Build rank lists (1-indexed)
        dense_rank = {
            i: rank + 1
            for rank, i in enumerate(
                sorted(
                    range(len(dense_scores)),
                    key=lambda x: dense_scores[x],
                    reverse=True,
                )
            )
        }
        bm25_rank = {
            i: rank + 1
            for rank, i in enumerate(
                sorted(range(len(bm25_norm)), key=lambda x: bm25_norm[x], reverse=True)
            )
        }

        RRF_K = 60  # standard constant
        rrf_scores = {
            i: 1 / (RRF_K + dense_rank[i]) + 1 / (RRF_K + bm25_rank[i])
            for i in range(len(candidate_docs))
        }

        # Sort by RRF score, take top_k
        ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

        results: List[Tuple[Document, float]] = []
        rrf_max = ranked[0][1] if ranked else 1.0
        for idx, rrf_score in ranked:
            normalised = rrf_score / rrf_max  # normalise to [0, 1] for threshold check
            results.append((candidate_docs[idx], normalised))

        return results
