"""
Generate the NUST Smart Banker system architecture diagram.

Produces: architecture/architecture.png

Run:
    python architecture/diagram.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# ── Output path ──────────────────────────────────────────────────────────────
OUT_PATH = Path(__file__).parent / "architecture.png"

# ── Colour palette ───────────────────────────────────────────────────────────
C = {
    "user": "#2196F3",  # blue
    "ui": "#00704A",  # NUST green
    "guard": "#E53935",  # red (guardrails)
    "rag": "#7B1FA2",  # purple
    "retrieval": "#0288D1",  # teal-blue
    "llm": "#F57C00",  # orange
    "storage": "#388E3C",  # dark green
    "output": "#00704A",  # NUST green
    "arrow": "#455A64",  # dark grey
    "bg": "#FAFAFA",
    "box_text": "white",
}

# ── Helper: rounded rectangle with label ─────────────────────────────────────


def draw_box(ax, x, y, w, h, label, sublabel="", color="#333333", fontsize=9):
    box = FancyBboxPatch(
        (x - w / 2, y - h / 2),
        w,
        h,
        boxstyle="round,pad=0.015",
        linewidth=1.5,
        edgecolor="white",
        facecolor=color,
        zorder=3,
    )
    ax.add_patch(box)
    ax.text(
        x,
        y + (0.015 if sublabel else 0),
        label,
        ha="center",
        va="center",
        fontsize=fontsize,
        fontweight="bold",
        color=C["box_text"],
        zorder=4,
        wrap=True,
    )
    if sublabel:
        ax.text(
            x,
            y - 0.035,
            sublabel,
            ha="center",
            va="center",
            fontsize=7,
            color=(1, 1, 1, 0.85),
            zorder=4,
            style="italic",
        )


def draw_arrow(ax, x1, y1, x2, y2, label="", color="#455A64"):
    ax.annotate(
        "",
        xy=(x2, y2),
        xytext=(x1, y1),
        arrowprops=dict(
            arrowstyle="-|>",
            color=color,
            lw=1.8,
            mutation_scale=14,
        ),
        zorder=2,
    )
    if label:
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        ax.text(
            mx + 0.01,
            my,
            label,
            ha="left",
            va="center",
            fontsize=7,
            color=color,
            zorder=5,
        )


# ── Build figure ──────────────────────────────────────────────────────────────


def build_diagram():
    fig, ax = plt.subplots(figsize=(14, 9))
    ax.set_xlim(0, 1.4)
    ax.set_ylim(0, 1.0)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.patch.set_facecolor(C["bg"])

    # ── Title ────────────────────────────────────────────────────────────────
    ax.text(
        0.7,
        0.97,
        "NUST Smart Banker – RAG System Architecture",
        ha="center",
        va="top",
        fontsize=13,
        fontweight="bold",
        color="#1A1A2E",
    )

    # ── Layer backgrounds ─────────────────────────────────────────────────────
    def layer_bg(y_center, height, label, color):
        rect = FancyBboxPatch(
            (0.02, y_center - height / 2),
            1.36,
            height,
            boxstyle="round,pad=0.008",
            linewidth=0,
            facecolor=color,
            alpha=0.08,
            zorder=0,
        )
        ax.add_patch(rect)
        ax.text(
            0.04,
            y_center + height / 2 - 0.018,
            label,
            ha="left",
            va="top",
            fontsize=7,
            color=color,
            fontweight="bold",
            zorder=1,
        )

    layer_bg(0.845, 0.10, "Presentation Layer", C["ui"])
    layer_bg(0.69, 0.10, "Safety Layer", C["guard"])
    layer_bg(0.53, 0.10, "Orchestration Layer", C["rag"])
    layer_bg(0.355, 0.14, "Retrieval & Storage Layer", C["retrieval"])
    layer_bg(0.175, 0.14, "Generation Layer", C["llm"])

    # ── Boxes ─────────────────────────────────────────────────────────────────
    BW, BH = 0.20, 0.065

    # Presentation
    draw_box(ax, 0.22, 0.845, BW, BH, "User", "Customer", C["user"])
    draw_box(ax, 0.70, 0.845, 0.36, BH, "Streamlit UI", "Chat + Admin tabs", C["ui"])

    # Safety – Input
    draw_box(
        ax,
        0.70,
        0.69,
        0.44,
        BH,
        "Input Guardrails",
        "Jailbreak · PII · Harmful content · Length",
        C["guard"],
    )

    # Orchestration
    draw_box(
        ax,
        0.70,
        0.53,
        0.38,
        BH,
        "RAG Chain (LangChain)",
        "Retrieval → Prompt assembly → Generation",
        C["rag"],
    )

    # Retrieval & Storage
    draw_box(
        ax, 0.33, 0.355, 0.28, BH, "BGE-M3 Embeddings", "BAAI/bge-m3", C["retrieval"]
    )
    draw_box(
        ax,
        0.70,
        0.355,
        0.26,
        BH,
        "Qdrant Vector Store",
        "Disk-persistent",
        C["storage"],
    )
    draw_box(
        ax, 1.07, 0.355, 0.24, BH, "BM25 Re-ranker", "rank-bm25 · RRF", C["retrieval"]
    )

    # Data sources (below storage)
    draw_box(ax, 0.58, 0.215, 0.24, BH, "FAQ JSON", "15 Q&A pairs", "#546E7A")
    draw_box(
        ax, 0.87, 0.215, 0.24, BH, "Product XLSX", "34 sheets · 200+ Q&A", "#546E7A"
    )
    draw_box(ax, 1.16, 0.215, 0.20, BH, "Uploads", "Real-time updates", "#546E7A")

    # Generation
    draw_box(
        ax,
        0.55,
        0.175,
        0.38,
        BH,
        "Qwen2.5-3B-Instruct",
        "4-bit quantised · RTX 4050 (6 GB VRAM)",
        C["llm"],
    )

    # Safety – Output
    draw_box(
        ax,
        0.70,
        0.53 - 0.16,
        0.44,
        BH,
        "Output Guardrails",
        "PII strip · Competitor redact · Template leak",
        C["guard"],
    )

    # ── Arrows ────────────────────────────────────────────────────────────────
    A = C["arrow"]
    # User → UI
    draw_arrow(ax, 0.32, 0.845, 0.52, 0.845, "query", A)
    # UI → Input guardrails
    draw_arrow(ax, 0.70, 0.812, 0.70, 0.722, "", A)
    # Input guardrails → RAG chain
    draw_arrow(ax, 0.70, 0.657, 0.70, 0.563, "", A)
    # RAG chain → BGE-M3
    draw_arrow(ax, 0.57, 0.497, 0.38, 0.388, "embed query", A)
    # BGE-M3 → Qdrant
    draw_arrow(ax, 0.48, 0.355, 0.57, 0.355, "dense search", A)
    # RAG chain → BM25
    draw_arrow(ax, 0.83, 0.497, 1.02, 0.388, "BM25 search", A)
    # BM25 → Qdrant (RRF)
    draw_arrow(ax, 0.95, 0.355, 0.84, 0.355, "RRF fuse", A)
    # Qdrant → RAG chain (context)
    draw_arrow(ax, 0.70, 0.322, 0.70, 0.265, "top-5 chunks", A)
    # Context → Qwen
    draw_arrow(ax, 0.655, 0.248, 0.60, 0.208, "prompt", A)
    # Qwen → Output guardrails
    draw_arrow(ax, 0.62, 0.175, 0.70, 0.393 - 0.025, "response", A)
    # Output guardrails → UI
    draw_arrow(ax, 0.70, 0.40, 0.70, 0.815, "safe answer", C["ui"])
    # UI → User
    draw_arrow(ax, 0.52, 0.845, 0.32, 0.845, "answer", C["ui"])

    # Data sources → Qdrant
    draw_arrow(ax, 0.58, 0.248, 0.64, 0.322, "", "#78909C")
    draw_arrow(ax, 0.87, 0.248, 0.76, 0.322, "", "#78909C")
    draw_arrow(ax, 1.16, 0.248, 0.84, 0.322, "", "#78909C")

    # ── Legend ────────────────────────────────────────────────────────────────
    legend_items = [
        (C["user"], "User"),
        (C["ui"], "UI / Response"),
        (C["guard"], "Guardrails"),
        (C["rag"], "RAG Orchestration"),
        (C["retrieval"], "Retrieval"),
        (C["llm"], "Language Model"),
        (C["storage"], "Vector Store / Data"),
    ]
    handles = [mpatches.Patch(color=c, label=lbl) for c, lbl in legend_items]
    ax.legend(
        handles=handles,
        loc="lower left",
        fontsize=7.5,
        framealpha=0.9,
        ncol=4,
        bbox_to_anchor=(0.02, 0.01),
    )

    plt.tight_layout()
    fig.savefig(OUT_PATH, dpi=160, bbox_inches="tight", facecolor=C["bg"])
    print(f"Architecture diagram saved to {OUT_PATH}")
    return OUT_PATH


if __name__ == "__main__":
    build_diagram()
