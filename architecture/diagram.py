"""
Generate the NUST Smart Banker system architecture diagram.

Produces:
  architecture/architecture.png   (high-res PNG for README / reports)
  architecture/diagram.html       (interactive HTML version — open in browser)

Run:
    python architecture/diagram.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless backend — no display needed

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
from matplotlib.lines import Line2D

OUT_PNG  = Path(__file__).parent / "architecture.png"
OUT_HTML = Path(__file__).parent / "diagram.html"

# ── Palette ───────────────────────────────────────────────────────────────────
P = {
    "user":      "#1565C0",
    "ui":        "#00704A",
    "guard":     "#C62828",
    "rag":       "#6A1B9A",
    "bge":       "#1976D2",
    "qdrant":    "#00796B",
    "bm25":      "#00838F",
    "llm":       "#E65100",
    "data":      "#455A64",
    "arrow":     "#78909C",
    "arrow_ok":  "#00704A",
    "bg":        "#F8F9FC",
    "white":     "#FFFFFF",
}

# ── Layer background colours (very light) ─────────────────────────────────────
LAYERS = {
    "presentation": ("#EBF5F0", "#00704A"),
    "safety_in":    ("#FDECEA", "#C62828"),
    "orchestration":("#F3E8FF", "#6A1B9A"),
    "retrieval":    ("#E3F2FD", "#1565C0"),
    "datasources":  ("#ECEFF1", "#455A64"),
    "generation":   ("#FFF3E0", "#E65100"),
    "safety_out":   ("#FDECEA", "#C62828"),
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def layer_band(ax, y_top, height, face, label, label_color, xleft=0.02, xright=0.98):
    """Draw a horizontal band representing an architecture layer."""
    rect = FancyBboxPatch(
        (xleft, y_top - height), xright - xleft, height,
        boxstyle="round,pad=0.005",
        linewidth=0, facecolor=face, alpha=0.55, zorder=0,
    )
    ax.add_patch(rect)
    ax.text(
        xleft + 0.012, y_top - 0.008,
        label,
        ha="left", va="top",
        fontsize=7.5, fontweight="bold",
        color=label_color, alpha=0.8, zorder=1,
    )


def node(ax, cx, cy, w, h, title, subtitle="", color="#333", fontsize=8.5):
    """Draw a rounded-rectangle node with title and optional subtitle."""
    box = FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle="round,pad=0.012",
        linewidth=0, facecolor=color,
        zorder=3,
        clip_on=False,
    )
    # subtle shadow
    shadow = FancyBboxPatch(
        (cx - w / 2 + 0.003, cy - h / 2 - 0.004), w, h,
        boxstyle="round,pad=0.012",
        linewidth=0, facecolor="#000000", alpha=0.08,
        zorder=2, clip_on=False,
    )
    ax.add_patch(shadow)
    ax.add_patch(box)

    ty = cy + (0.012 if subtitle else 0)
    ax.text(cx, ty, title,
            ha="center", va="center",
            fontsize=fontsize, fontweight="bold",
            color="white", zorder=4, clip_on=False)
    if subtitle:
        ax.text(cx, cy - 0.016, subtitle,
                ha="center", va="center",
                fontsize=6.5, color=(1, 1, 1, 0.80),
                style="italic", zorder=4, clip_on=False,
                wrap=False)


def arrow(ax, x1, y1, x2, y2, label="", color="#78909C", labelside="right"):
    """Draw an annotated arrow between two points."""
    ax.annotate(
        "",
        xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(
            arrowstyle="-|>",
            color=color, lw=1.6,
            mutation_scale=12,
            connectionstyle="arc3,rad=0.0",
        ),
        zorder=5,
    )
    if label:
        mx = (x1 + x2) / 2
        my = (y1 + y2) / 2
        dx = 0.014 if labelside == "right" else -0.014
        ax.text(mx + dx, my, label,
                ha="left" if labelside == "right" else "right",
                va="center",
                fontsize=6.5, color=color,
                style="italic", zorder=6,
                bbox=dict(boxstyle="round,pad=0.02",
                          fc=P["bg"], ec="none", alpha=0.85))


# ── Main diagram ──────────────────────────────────────────────────────────────

def build_diagram():
    FIG_W, FIG_H = 18, 13
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, 1.0)
    ax.set_aspect("auto")
    ax.axis("off")
    fig.patch.set_facecolor(P["bg"])
    ax.set_facecolor(P["bg"])

    # ── Title ──────────────────────────────────────────────────────────────
    ax.text(0.50, 0.975,
            "NUST Smart Banker — RAG System Architecture",
            ha="center", va="top",
            fontsize=15, fontweight="bold", color="#0D1B2A")
    ax.text(0.50, 0.952,
            "Retrieval-Augmented Generation · Multi-layer Safety Guardrails · "
            "Qwen2.5-3B-Instruct · BAAI/bge-m3 · Qdrant",
            ha="center", va="top",
            fontsize=8.5, color="#6B7A8D")

    # ── Layer Y coords (top edge, height) ──────────────────────────────────
    #   Layers arranged top-to-bottom with equal gaps.
    #   Each (y_top, height) pair
    L = {
        "pres" :  (0.935, 0.100),
        "sin"  :  (0.810, 0.090),
        "orch" :  (0.690, 0.090),
        "retr" :  (0.560, 0.110),
        "data" :  (0.415, 0.110),
        "gen"  :  (0.265, 0.100),
        "sout" :  (0.130, 0.090),
    }

    # draw layer bands
    configs = [
        ("pres",  LAYERS["presentation"],  "① Presentation Layer"),
        ("sin",   LAYERS["safety_in"],     "② Safety Layer — Input Guardrails"),
        ("orch",  LAYERS["orchestration"], "③ Orchestration Layer"),
        ("retr",  LAYERS["retrieval"],     "④ Retrieval & Indexing Layer — Hybrid Search"),
        ("data",  LAYERS["datasources"],   "⑤ Knowledge Base — Data Sources"),
        ("gen",   LAYERS["generation"],    "⑥ Generation Layer"),
        ("sout",  LAYERS["safety_out"],    "⑦ Safety Layer — Output Guardrails"),
    ]
    for key, (face, lc), label in configs:
        yt, h = L[key]
        layer_band(ax, yt, h, face, label, lc)

    # ── Node centres (cx, cy) ─────────────────────────────────────────────
    NW, NH = 0.155, 0.058   # standard node width / height
    WW = 0.36               # wide node width

    # Presentation
    cy_pres  = L["pres"][0]  - L["pres"][1]  / 2
    cx_user  = 0.16
    cx_ui    = 0.54

    node(ax, cx_user, cy_pres, NW, NH, "User", "Customer", P["user"])
    node(ax, cx_ui,   cy_pres, WW, NH, "Streamlit UI",
         "Chat · Admin / Upload · Architecture tabs", P["ui"])

    # Input guardrails
    cy_sin = L["sin"][0] - L["sin"][1] / 2
    node(ax, 0.50, cy_sin, 0.62, NH,
         "Input Guardrails",
         "Jailbreak (20+ regex) · Hard-block keywords · PII detection · Length limit (2 000 chars)",
         P["guard"])

    # Orchestration
    cy_orch = L["orch"][0] - L["orch"][1] / 2
    node(ax, 0.50, cy_orch, 0.65, NH,
         "RAG Chain (LangChain)",
         "Retrieval → Relevance gate (threshold 0.30) → Prompt assembly → Generation",
         P["rag"])

    # Retrieval
    cy_retr = L["retr"][0] - L["retr"][1] / 2
    cx_bge    = 0.18
    cx_qdrant = 0.50
    cx_bm25   = 0.82
    node(ax, cx_bge,    cy_retr, 0.24, NH, "BGE-M3 Embeddings",
         "BAAI/bge-m3 · dim=1024 · CUDA", P["bge"])
    node(ax, cx_qdrant, cy_retr, 0.26, NH, "Qdrant Vector Store",
         "Disk-persistent · HNSW · nust_bank_docs", P["qdrant"])
    node(ax, cx_bm25,   cy_retr, 0.24, NH, "BM25 Re-ranker",
         "rank-bm25 · Okapi · RRF fusion", P["bm25"])

    # Data sources
    cy_data = L["data"][0] - L["data"][1] / 2
    cx_faq  = 0.20
    cx_xlsx = 0.50
    cx_upl  = 0.80
    node(ax, cx_faq,  cy_data, 0.26, NH, "FAQ JSON",
         "funds_transfer_faq.json · Q&A categories", P["data"])
    node(ax, cx_xlsx, cy_data, 0.30, NH, "Product XLSX",
         "34 product sheets · 200+ Q&A pairs", P["data"])
    node(ax, cx_upl,  cy_data, 0.26, NH, "Admin Uploads",
         ".txt / .json / .xlsx · Real-time ingest", P["data"])

    # Generation
    cy_gen = L["gen"][0] - L["gen"][1] / 2
    node(ax, 0.50, cy_gen, 0.60, NH,
         "Qwen2.5-3B-Instruct",
         "3.09B params · 4-bit NF4 (bitsandbytes) · ~2.0 GB VRAM · 32K context · Temp=0.2",
         P["llm"])

    # Output guardrails
    cy_sout = L["sout"][0] - L["sout"][1] / 2
    node(ax, 0.50, cy_sout, 0.62, NH,
         "Output Guardrails",
         "Template leak · Competitor redaction · PII strip · Length cap (3 000 chars) · NeMo Guardrails",
         P["guard"])

    # ── Arrows ────────────────────────────────────────────────────────────

    # User ↔ UI (horizontal, bidirectional)
    mid_y = cy_pres
    ax.annotate("", xy=(cx_ui - WW/2, mid_y + 0.010),
                xytext=(cx_user + NW/2, mid_y + 0.010),
                arrowprops=dict(arrowstyle="-|>", color=P["ui"], lw=1.6, mutation_scale=11), zorder=5)
    ax.text((cx_user + cx_ui)/2, mid_y + 0.022, "query",
            ha="center", fontsize=6.5, color=P["ui"], style="italic")

    ax.annotate("", xy=(cx_user + NW/2, mid_y - 0.010),
                xytext=(cx_ui - WW/2, mid_y - 0.010),
                arrowprops=dict(arrowstyle="-|>", color=P["arrow"], lw=1.6, mutation_scale=11), zorder=5)
    ax.text((cx_user + cx_ui)/2, mid_y - 0.024, "answer",
            ha="center", fontsize=6.5, color=P["arrow"], style="italic")

    # UI → Input guardrails (down)
    arrow(ax, cx_ui, cy_pres - NH/2, 0.50, cy_sin + NH/2,
          "user query", P["arrow"])

    # Input guardrails → RAG chain (down)
    arrow(ax, 0.50, cy_sin - NH/2, 0.50, cy_orch + NH/2,
          "safe query", P["arrow"])

    # RAG chain → BGE-M3 (diagonal left-down)
    arrow(ax, 0.38, cy_orch - NH/2, cx_bge + 0.04, cy_retr + NH/2,
          "embed query", P["arrow"], labelside="left")

    # RAG chain → BM25 (diagonal right-down)
    arrow(ax, 0.62, cy_orch - NH/2, cx_bm25 - 0.04, cy_retr + NH/2,
          "BM25 keyword", P["arrow"])

    # BGE-M3 → Qdrant (horizontal right)
    arrow(ax, cx_bge + NW/2 + 0.01, cy_retr, cx_qdrant - 0.13, cy_retr,
          "dense vector", P["bge"])

    # BM25 → Qdrant (horizontal left)
    arrow(ax, cx_bm25 - NW/2 - 0.01, cy_retr, cx_qdrant + 0.13, cy_retr,
          "BM25 score", P["bm25"], labelside="left")

    # Qdrant → RAG chain (up, labelled "top-K chunks")
    arrow(ax, cx_qdrant, cy_retr + NH/2, 0.50, cy_orch - NH/2,
          "top-5 chunks", P["qdrant"])

    # Data sources → Qdrant (up arrows)
    for cx_src in [cx_faq, cx_xlsx, cx_upl]:
        ax.annotate("", xy=(cx_qdrant, cy_retr - NH/2),
                    xytext=(cx_src, cy_data + NH/2),
                    arrowprops=dict(arrowstyle="-|>", color="#90A4AE", lw=1.2,
                                   mutation_scale=10,
                                   connectionstyle="arc3,rad=0.0"), zorder=5)
    ax.text(0.50, (cy_data + NH/2 + cy_retr - NH/2)/2,
            "ingest pipeline", ha="center", fontsize=6.5,
            color="#90A4AE", style="italic",
            bbox=dict(boxstyle="round,pad=0.02", fc=P["bg"], ec="none", alpha=0.85))

    # RAG chain → LLM (down)
    arrow(ax, 0.50, cy_orch - NH/2, 0.50, cy_gen + NH/2,
          "formatted prompt", P["llm"])

    # LLM → Output guardrails (down)
    arrow(ax, 0.50, cy_gen - NH/2, 0.50, cy_sout + NH/2,
          "raw response", P["guard"])

    # Output guardrails → UI (long upward curved arrow on left side)
    ax.annotate(
        "",
        xy=(0.10, cy_pres - NH/2),
        xytext=(0.10, cy_sout - NH/2),
        arrowprops=dict(
            arrowstyle="-|>",
            color=P["arrow_ok"], lw=2.0,
            mutation_scale=12,
            connectionstyle="arc3,rad=0.0",
        ),
        zorder=5,
    )
    ax.text(0.068, (cy_pres + cy_sout)/2, "sanitised\nanswer",
            ha="center", va="center", fontsize=6.5,
            color=P["arrow_ok"], style="italic", rotation=90,
            bbox=dict(boxstyle="round,pad=0.03", fc=P["bg"], ec="none", alpha=0.85))

    # ── RRF label in retrieval band ────────────────────────────────────────
    ax.text(cx_qdrant, cy_retr - NH/2 - 0.022, "⚡ Reciprocal Rank Fusion (RRF, k=60)",
            ha="center", va="top", fontsize=7, color=P["qdrant"], fontweight="bold")

    # ── Legend ─────────────────────────────────────────────────────────────
    items = [
        (P["user"],   "User"),
        (P["ui"],     "UI / Response"),
        (P["guard"],  "Guardrails"),
        (P["rag"],    "RAG Orchestration"),
        (P["bge"],    "Embeddings"),
        (P["qdrant"], "Vector Store"),
        (P["bm25"],   "BM25 Re-ranker"),
        (P["llm"],    "Language Model"),
        (P["data"],   "Data Sources"),
    ]
    handles = [mpatches.Patch(color=c, label=lbl) for c, lbl in items]
    ax.legend(
        handles=handles,
        loc="lower right",
        fontsize=7.5,
        framealpha=0.95,
        ncol=3,
        bbox_to_anchor=(0.98, 0.005),
        title="Component Types",
        title_fontsize=8,
    )

    # ── Save ───────────────────────────────────────────────────────────────
    plt.tight_layout(pad=0.4)
    fig.savefig(OUT_PNG, dpi=180, bbox_inches="tight", facecolor=P["bg"])
    plt.close(fig)
    print(f"✓ PNG  saved → {OUT_PNG}")
    print(f"✓ HTML saved → {OUT_HTML}  (open in browser for interactive view)")
    return OUT_PNG


if __name__ == "__main__":
    build_diagram()
