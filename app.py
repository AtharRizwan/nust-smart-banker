"""
NUST Smart Banker – Streamlit Customer Interface

Tabs:
  1. Chat    – customer-facing chat with the NUST Bank AI assistant
  2. Admin   – upload new documents (JSON, XLSX, TXT) to update the knowledge base

Run:
    streamlit run app.py
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path

import streamlit as st

# ── Make project root importable regardless of working directory ──────────────
sys.path.insert(0, str(Path(__file__).resolve().parent))

from configs.settings import UPLOADED_DOCS_DIR, QDRANT_DIR

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# ─── Page Config ─────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="NUST Bank – Virtual Assistant",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────

st.markdown(
    """
<style>
    /* Bank-themed colour palette */
    :root {
        --nust-green:  #00704A;
        --nust-dark:   #004D33;
        --nust-light:  #E8F5EE;
        --nust-accent: #FFD700;
    }

    /* Header bar */
    .bank-header {
        background: linear-gradient(135deg, var(--nust-dark), var(--nust-green));
        padding: 1.2rem 2rem;
        border-radius: 12px;
        margin-bottom: 1rem;
        color: white;
        display: flex;
        align-items: center;
        gap: 1rem;
    }
    .bank-header h1 { margin: 0; font-size: 1.6rem; }
    .bank-header p  { margin: 0; opacity: 0.85; font-size: 0.9rem; }

    /* Chat bubbles */
    .user-bubble {
        background: var(--nust-light);
        border-left: 4px solid var(--nust-green);
        padding: 0.75rem 1rem;
        border-radius: 0 12px 12px 12px;
        margin: 0.4rem 0;
    }
    .assistant-bubble {
        background: white;
        border-left: 4px solid var(--nust-accent);
        padding: 0.75rem 1rem;
        border-radius: 0 12px 12px 12px;
        margin: 0.4rem 0;
        box-shadow: 0 1px 4px rgba(0,0,0,0.06);
    }

    /* Source pills */
    .source-pill {
        display: inline-block;
        background: var(--nust-light);
        color: var(--nust-dark);
        border: 1px solid var(--nust-green);
        border-radius: 20px;
        padding: 2px 10px;
        font-size: 0.75rem;
        margin: 2px;
    }

    /* Sidebar */
    .sidebar-stat {
        background: var(--nust-light);
        border-radius: 8px;
        padding: 0.6rem 1rem;
        margin: 0.4rem 0;
        font-size: 0.85rem;
    }

    /* Upload zone */
    .upload-info {
        background: #f8f9fa;
        border: 1px dashed #ccc;
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
        color: #666;
        font-size: 0.85rem;
    }
</style>
""",
    unsafe_allow_html=True,
)


# ─── Session State Initialisation ────────────────────────────────────────────


def _init_session() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "chain_ready" not in st.session_state:
        st.session_state.chain_ready = False
    if "ingested" not in st.session_state:
        st.session_state.ingested = False
    if "doc_count" not in st.session_state:
        st.session_state.doc_count = 0


_init_session()


# ─── Cached resource loaders ─────────────────────────────────────────────────


@st.cache_resource(show_spinner="Loading retriever and embedding model…")
def _load_retriever():
    from src.retriever import get_retriever

    return get_retriever()


@st.cache_resource(
    show_spinner="Loading Qwen2.5-3B language model (this may take a moment)…"
)
def _load_chain():
    from src.rag_chain import get_chain

    return get_chain()


def _ensure_ingested(retriever) -> None:
    """Run full ingestion if the Qdrant collection is empty."""
    if retriever.count() == 0 and not st.session_state.ingested:
        with st.spinner("Indexing knowledge base for the first time…"):
            from src.ingest import ingest_all

            n = ingest_all()
        st.session_state.ingested = True
        st.session_state.doc_count = n
        st.success(f"Knowledge base ready: {n} document chunks indexed.")
    else:
        st.session_state.ingested = True
        st.session_state.doc_count = retriever.count()


# ─── Sidebar ─────────────────────────────────────────────────────────────────


def render_sidebar(retriever) -> None:
    with st.sidebar:
        st.markdown("## 🏦 NUST Bank Assistant")
        st.markdown("---")

        # Status indicators
        doc_count = retriever.count()
        st.markdown(
            f'<div class="sidebar-stat">📚 <b>Indexed documents:</b> {doc_count}</div>',
            unsafe_allow_html=True,
        )

        model_status = (
            "✅ Ready" if st.session_state.chain_ready else "⏳ Loading on first query"
        )
        st.markdown(
            f'<div class="sidebar-stat">🤖 <b>LLM status:</b> {model_status}</div>',
            unsafe_allow_html=True,
        )

        st.markdown("---")
        st.markdown("### About")
        st.markdown(
            "This assistant is powered by **Qwen2.5-3B-Instruct** with a "
            "Retrieval-Augmented Generation (RAG) pipeline over NUST Bank's "
            "product knowledge base."
        )
        st.markdown("---")

        if st.button("🗑️ Clear chat history"):
            st.session_state.messages = []
            st.rerun()

        st.markdown("---")
        st.caption("NUST Smart Banker v1.0 · Prototype")


# ─── Chat Tab ─────────────────────────────────────────────────────────────────


def render_chat_tab(chain) -> None:
    # Header
    st.markdown(
        """
    <div class="bank-header">
        <div>
            <h1>🏦 NUST Bank Virtual Assistant</h1>
            <p>Ask me anything about our accounts, loans, transfers, and app features.</p>
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Render existing chat history
    for msg in st.session_state.messages:
        role = msg["role"]
        content = msg["content"]
        sources = msg.get("sources", [])

        with st.chat_message(role, avatar="👤" if role == "user" else "🏦"):
            st.markdown(content)
            if sources and role == "assistant":
                with st.expander("Sources", expanded=False):
                    for src in sources:
                        score_pct = int(src.get("score", 0) * 100)
                        st.markdown(
                            f'<span class="source-pill">{src["label"]} ({score_pct}%)</span>',
                            unsafe_allow_html=True,
                        )

    # Suggested starter questions (only when chat is empty)
    if not st.session_state.messages:
        st.markdown("#### Suggested questions")
        starter_cols = st.columns(2)
        starters = [
            "What accounts does NUST Bank offer?",
            "How do I change my funds transfer limit?",
            "Who can apply for NUST Personal Finance?",
            "What is the Little Champs Account?",
            "How do I reset my password in the mobile app?",
            "Does NUST Bank offer auto financing?",
        ]
        for i, q in enumerate(starters):
            col = starter_cols[i % 2]
            with col:
                if st.button(q, key=f"starter_{i}", use_container_width=True):
                    _process_query(q, chain)
                    st.rerun()

    # Chat input
    if prompt := st.chat_input("Ask a question about NUST Bank…"):
        _process_query(prompt, chain)
        st.rerun()


def _process_query(query: str, chain) -> None:
    """Append user message, run the chain, append assistant response."""
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": query})

    # Build chat history for context (exclude the message we just added)
    chat_history = [
        {"role": m["role"], "content": m["content"]}
        for m in st.session_state.messages[:-1]
        if m["role"] in ("user", "assistant")
    ]

    with st.spinner("Thinking…"):
        t0 = time.perf_counter()
        response = chain.answer(query, chat_history=chat_history)
        elapsed = time.perf_counter() - t0

    st.session_state.chain_ready = True

    answer_text = response.answer
    if response.is_blocked:
        answer_text = f"⚠️ {response.answer}"

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer_text,
            "sources": response.sources,
            "latency": round(elapsed, 2),
        }
    )


# ─── Admin / Upload Tab ───────────────────────────────────────────────────────


def render_admin_tab(retriever) -> None:
    st.markdown("## Admin – Update Knowledge Base")
    st.markdown(
        "Upload new bank documents here. They will be immediately indexed and "
        "available for customer queries — no restart required."
    )

    st.markdown("---")

    # Supported formats info
    st.markdown(
        """
    <div class="upload-info">
        <b>Supported file formats:</b><br>
        <code>.json</code> (FAQ schema: categories → questions)<br>
        <code>.xlsx</code> (Product knowledge sheet format)<br>
        <code>.txt</code>  (Plain text — policies, notices, FAQs)
    </div>
    """,
        unsafe_allow_html=True,
    )

    st.markdown("")

    uploaded_file = st.file_uploader(
        "Choose a document to upload",
        type=["json", "xlsx", "txt"],
        help="The document will be parsed, anonymised, chunked, and indexed automatically.",
    )

    source_label = st.text_input(
        "Source label (optional)",
        placeholder="e.g. 'New Credit Card FAQ' or 'Updated Transfer Policy'",
        help="A human-readable name for this document that will appear in source citations.",
    )

    col1, col2 = st.columns([1, 3])
    with col1:
        ingest_btn = st.button(
            "📥 Ingest Document", type="primary", use_container_width=True
        )
    with col2:
        if st.button("🔄 Refresh stats", use_container_width=True):
            st.session_state.doc_count = retriever.count()
            st.rerun()

    if ingest_btn and uploaded_file is not None:
        # Save to uploaded_docs folder
        UPLOADED_DOCS_DIR.mkdir(parents=True, exist_ok=True)
        save_path = UPLOADED_DOCS_DIR / uploaded_file.name

        with open(save_path, "wb") as fh:
            fh.write(uploaded_file.getbuffer())

        with st.spinner(f"Ingesting {uploaded_file.name}…"):
            try:
                from src.ingest import ingest_file

                n_chunks = ingest_file(save_path, source_label or uploaded_file.name)
                st.session_state.doc_count = retriever.count()

                if n_chunks > 0:
                    st.success(
                        f"Successfully indexed **{n_chunks} chunks** from "
                        f"**{uploaded_file.name}**. "
                        f"Total collection size: {retriever.count()} vectors."
                    )
                else:
                    st.warning(
                        f"No meaningful content could be extracted from {uploaded_file.name}. "
                        "Check the file format and content."
                    )
            except Exception as exc:
                st.error(f"Ingestion failed: {exc}")
                logger.exception("Ingestion error for %s", uploaded_file.name)

    elif ingest_btn and uploaded_file is None:
        st.warning("Please select a file to upload first.")

    # Show previously uploaded documents
    st.markdown("---")
    st.markdown("### Previously Uploaded Documents")
    uploaded_files = (
        sorted(UPLOADED_DOCS_DIR.glob("*")) if UPLOADED_DOCS_DIR.exists() else []
    )
    if uploaded_files:
        for f in uploaded_files:
            size_kb = round(f.stat().st_size / 1024, 1)
            st.markdown(f"- `{f.name}` ({size_kb} KB)")
    else:
        st.info("No documents have been uploaded yet.")


# ─── Architecture Diagram Tab ─────────────────────────────────────────────────


def render_architecture_tab() -> None:
    st.markdown("## System Architecture")
    arch_img = Path(__file__).parent / "architecture" / "architecture.png"
    if arch_img.exists():
        st.image(
            str(arch_img),
            caption="NUST Smart Banker – RAG System Architecture",
            use_container_width=True,
        )
    else:
        st.info(
            "Architecture diagram not yet generated. "
            "Run `python architecture/diagram.py` to create it."
        )

    st.markdown("""
    ### Component Overview

    | Component | Technology | Role |
    |---|---|---|
    | **User Interface** | Streamlit | Chat UI + Admin document upload |
    | **Input Guardrails** | Custom regex + NeMo Guardrails | Jailbreak & injection detection |
    | **RAG Chain** | LangChain | Orchestrates retrieval + generation |
    | **Embedding Model** | BAAI/bge-m3 | Dense vector representations |
    | **Vector Store** | Qdrant (disk-persistent) | Semantic similarity search |
    | **BM25 Re-ranker** | rank-bm25 | Keyword-based re-ranking |
    | **Language Model** | Qwen2.5-3B-Instruct (4-bit) | Answer generation |
    | **PII Anonymiser** | Microsoft Presidio | Remove personal data before indexing |
    | **Output Guardrails** | Custom rules | Sanitise LLM output |

    ### Query Flow

    ```
    User Query
        ↓ Input Guardrails (jailbreak / PII / harmful content check)
        ↓ BGE-M3 Query Embedding
        ↓ Qdrant Dense Search  ┐
        ↓ BM25 Keyword Search  ├─ Reciprocal Rank Fusion → Top-5 Chunks
        ↓                      ┘
        ↓ Relevance Threshold Check (out-of-domain guard)
        ↓ Prompt Assembly (System + Context + Query)
        ↓ Qwen2.5-3B-Instruct Generation
        ↓ Output Guardrails (PII / competitor / template-leak check)
        ↓ Response to User
    ```
    """)


# ─── Main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    # Load retriever (and trigger ingestion if needed)
    retriever = _load_retriever()
    _ensure_ingested(retriever)

    # Render sidebar
    render_sidebar(retriever)

    # Main content tabs
    tab_chat, tab_admin, tab_arch = st.tabs(
        ["💬 Chat", "📁 Admin / Upload", "🏗️ Architecture"]
    )

    with tab_chat:
        chain = _load_chain()
        render_chat_tab(chain)

    with tab_admin:
        render_admin_tab(retriever)

    with tab_arch:
        render_architecture_tab()


if __name__ == "__main__":
    main()
