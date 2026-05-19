"""Generate the three matplotlib PNGs that illustrate Phase 3 results.

Writes to docs/static/phase3/:
- layer_sweep.png         — ρ vs Gemma layer, concept-level seed[0]
- embed_variants.png      — ρ for three embed-text regimes at layer 13
- nw_alignment_schematic.png — pedagogical NW alignment of "woof" vs "bark"

Data source: the ρ table in semanticphonology.md §3 (copied here verbatim).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

OUT_DIR = Path(__file__).resolve().parents[1] / "docs" / "static" / "phase3"

CUTOVER_THRESHOLD = 0.15
STRONG_CLAIM_THRESHOLD = 0.20

LAYER_SWEEP = [
    (5, 0.0079),
    (9, 0.0197),
    (13, 0.0365),
    (17, 0.0154),
    (21, 0.0348),
    (25, 0.0174),
]

EMBED_VARIANTS = [
    ("63 concept\nseed[0]", 0.0365),
    ("1574 attribute\nslug + seed + attr", 0.0292),
    ("1574 attribute\nslug + attr only", 0.0178),
]


def plot_layer_sweep(out_path: Path) -> None:
    layers, rhos = zip(*LAYER_SWEEP, strict=True)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(
        [str(layer) for layer in layers],
        rhos,
        color="#4a6fa5",
        edgecolor="#22384f",
        zorder=3,
    )
    for rect, rho in zip(bars, rhos, strict=True):
        ax.text(
            rect.get_x() + rect.get_width() / 2,
            rect.get_height() + 0.004,
            f"{rho:.4f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax.axhline(
        CUTOVER_THRESHOLD,
        color="#c0392b",
        linestyle="--",
        linewidth=1.4,
        label=f"cutover threshold ρ ≥ {CUTOVER_THRESHOLD}",
        zorder=2,
    )
    ax.axhline(
        STRONG_CLAIM_THRESHOLD,
        color="#7d3c98",
        linestyle=":",
        linewidth=1.4,
        label=f"§5 strong-claim threshold ρ ≥ {STRONG_CLAIM_THRESHOLD}",
        zorder=2,
    )

    ax.set_ylim(0, max(STRONG_CLAIM_THRESHOLD, max(rhos)) * 1.4)
    ax.set_xlabel("Gemma 2 (2B) layer index")
    ax.set_ylabel("Spearman ρ")
    ax.set_title("Phase 3 layer sweep — 63 concept anchors, seed[0] embed")
    ax.grid(axis="y", alpha=0.3, zorder=0)
    ax.legend(loc="upper right", framealpha=0.95)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_embed_variants(out_path: Path) -> None:
    labels, rhos = zip(*EMBED_VARIANTS, strict=True)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(
        labels,
        rhos,
        color=["#4a6fa5", "#5b9bd5", "#9dc3e6"],
        edgecolor="#22384f",
        zorder=3,
    )
    for rect, rho in zip(bars, rhos, strict=True):
        ax.text(
            rect.get_x() + rect.get_width() / 2,
            rect.get_height() + 0.004,
            f"{rho:.4f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    ax.axhline(
        CUTOVER_THRESHOLD,
        color="#c0392b",
        linestyle="--",
        linewidth=1.4,
        label=f"cutover threshold ρ ≥ {CUTOVER_THRESHOLD}",
        zorder=2,
    )

    ax.set_ylim(0, CUTOVER_THRESHOLD * 1.4)
    ax.set_ylabel("Spearman ρ")
    ax.set_title("Embed-text regime at Gemma layer 13")
    ax.grid(axis="y", alpha=0.3, zorder=0)
    ax.legend(loc="upper right", framealpha=0.95)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_nw_alignment(out_path: Path) -> None:
    """Pedagogical schematic: woof /wʊf/ vs bark /bɑɹk/ NW alignment.

    Optimal alignment (illustrative, not from the actual NW kernel):
      w  ʊ  -  f
      b  ɑ  ɹ  k
    Three substitutions + one gap.
    """
    top = ["w", "ʊ", "—", "f"]
    bottom = ["b", "ɑ", "ɹ", "k"]
    ops = ["sub", "sub", "gap", "sub"]
    op_colors = {"sub": "#e67e22", "gap": "#7f8c8d", "match": "#27ae60"}

    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.set_xlim(0, len(top))
    ax.set_ylim(0, 4)
    ax.axis("off")

    cell_w = 0.9
    for i, (t, b, op) in enumerate(zip(top, bottom, ops, strict=True)):
        x = i + 0.5
        bg_color = op_colors[op]
        ax.add_patch(
            mpatches.Rectangle(
                (i + 0.05, 2.1),
                cell_w,
                0.9,
                facecolor=bg_color,
                alpha=0.18,
                edgecolor=bg_color,
            )
        )
        ax.add_patch(
            mpatches.Rectangle(
                (i + 0.05, 1.0),
                cell_w,
                0.9,
                facecolor=bg_color,
                alpha=0.18,
                edgecolor=bg_color,
            )
        )
        ax.text(x, 2.55, t, ha="center", va="center", fontsize=22, family="serif")
        ax.text(x, 1.45, b, ha="center", va="center", fontsize=22, family="serif")
        ax.text(x, 0.55, op, ha="center", va="center", fontsize=10, color=bg_color)

    ax.text(-0.15, 2.55, "woof", ha="right", va="center", fontsize=12, style="italic")
    ax.text(-0.15, 1.45, "bark", ha="right", va="center", fontsize=12, style="italic")
    ax.set_title(
        "Needleman-Wunsch alignment, illustrative — /wʊf/ vs /bɑɹk/",
        fontsize=12,
    )

    legend_handles = [
        mpatches.Patch(color=op_colors["sub"], alpha=0.6, label="substitute"),
        mpatches.Patch(color=op_colors["gap"], alpha=0.6, label="insert gap"),
        mpatches.Patch(color=op_colors["match"], alpha=0.6, label="match"),
    ]
    ax.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.05),
        ncol=3,
        frameon=False,
    )

    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plot_layer_sweep(OUT_DIR / "layer_sweep.png")
    plot_embed_variants(OUT_DIR / "embed_variants.png")
    plot_nw_alignment(OUT_DIR / "nw_alignment_schematic.png")
    print(f"wrote 3 PNGs to {OUT_DIR}")


if __name__ == "__main__":
    main()
