"""Generate matplotlib PNGs that illustrate the Phase 3 result.

Four plots, written to docs/static/phase3/:

- two_populations.png — both populations on the same axes. The 63
  iconic anchors live in a tiny cluster near the bottom (the model
  considers them mutually close). The 2000 substrate meaning-clusters
  live a long way up, in a different region of the model's geometry.
  This is the *why* of the failure: the anchors aren't anywhere near
  the meanings they were supposed to anchor.

- anchor_pairs_scatter.png — zoomed view of the iconic-anchor pairs
  only, with a few named callouts. There is *some* sound-meaning
  correlation among iconic words themselves.

- substrate_pairs_scatter.png — zoomed view of the 10,000 random
  substrate pairs. The cloud is round. This is the data behind the
  falsified headline.

- shape_overlay.png — both populations rescaled into a common
  [0,1] square and overlaid. Tests the intuition that the two clouds
  "share a shape." They share a sampling fan but only one has the
  diagonal climb that would mean sound and meaning track each other.

Reads:
- src/conlang/lab/results/anchor_pairs.parquet
- /media/menser/fauna/interlingua/data/processed/phase3_spearman_pairs.parquet
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

OUT_DIR = Path(__file__).resolve().parents[1] / "docs" / "static" / "phase3"
ANCHOR_PAIRS = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "conlang"
    / "lab"
    / "results"
    / "anchor_pairs.parquet"
)
SUBSTRATE_PAIRS = Path(
    "/media/menser/fauna/interlingua/data/processed/phase3_spearman_pairs.parquet"
)
SIGNED_PARQUET = Path("/media/menser/fauna/interlingua/data/processed/anchors-v1.parquet")

ANCHOR_COLOR = "#c0392b"
SUBSTRATE_COLOR = "#4a6fa5"

X_LABEL = "how different the two words sound"
Y_LABEL = "how different the model thinks they are"


def _signed_concepts() -> set[str]:
    return set(pd.read_parquet(SIGNED_PARQUET)["concept"])


def _load_anchor_signed() -> pd.DataFrame:
    df = pd.read_parquet(ANCHOR_PAIRS)
    signed = _signed_concepts()
    return df[df.a.isin(signed) & df.b.isin(signed)].copy()


def plot_two_populations(out_path: Path) -> None:
    anchor = _load_anchor_signed()
    sub = pd.read_parquet(SUBSTRATE_PAIRS)

    fig, ax = plt.subplots(figsize=(7.5, 5.5))

    ax.scatter(
        sub.phonological_distance,
        sub.cosine_distance,
        s=3,
        color=SUBSTRATE_COLOR,
        alpha=0.10,
        edgecolors="none",
        zorder=2,
    )
    ax.scatter(
        anchor.phon,
        anchor.cos,
        s=18,
        color=ANCHOR_COLOR,
        alpha=0.75,
        edgecolors="none",
        zorder=3,
    )

    # Direct annotations on the plot — no legend box that covers data.
    ax.annotate(
        "the 2000 meaning-clusters\nlive up here",
        xy=(11.5, 1.00),
        xytext=(11.5, 0.78),
        ha="right",
        fontsize=11,
        color=SUBSTRATE_COLOR,
        arrowprops=dict(arrowstyle="->", color=SUBSTRATE_COLOR, lw=1.2, alpha=0.7),
    )
    ax.annotate(
        "the 63 iconic anchors\nlive down here",
        xy=(11.5, 0.005),
        xytext=(11.5, 0.20),
        ha="right",
        fontsize=11,
        color=ANCHOR_COLOR,
        arrowprops=dict(arrowstyle="->", color=ANCHOR_COLOR, lw=1.2, alpha=0.7),
    )

    ax.set_xlabel(X_LABEL)
    ax.set_ylabel(Y_LABEL)
    ax.set_title("Two populations in the model's mind, on the same axes")
    ax.grid(alpha=0.25, zorder=0)

    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_anchor_pairs(out_path: Path) -> None:
    df = _load_anchor_signed()

    fig, ax = plt.subplots(figsize=(7.5, 5.5))

    ax.scatter(
        df.phon,
        df.cos,
        s=14,
        color=ANCHOR_COLOR,
        alpha=0.45,
        edgecolors="none",
        zorder=2,
    )

    callouts = [
        ("snake_hissing", "cat_hissing"),
        ("cow_mooing", "sheep_bleating"),
        ("bee_buzzing", "snake_hissing"),
        ("cat_meowing", "cat_purring"),
        ("thunder", "bell_ringing"),
    ]
    for a, b in callouts:
        row = df[((df.a == a) & (df.b == b)) | ((df.a == b) & (df.b == a))]
        if len(row) == 0:
            continue
        r = row.iloc[0]
        label = f"{r.seed_a} · {r.seed_b}"
        ax.scatter(r.phon, r.cos, s=70, color="#7b1d10", edgecolor="black", zorder=4)
        ax.annotate(
            label,
            (r.phon, r.cos),
            xytext=(8, 6),
            textcoords="offset points",
            fontsize=9,
            color="#3a0a04",
            zorder=5,
        )

    ax.set_xlabel(X_LABEL)
    ax.set_ylabel(Y_LABEL)
    ax.set_title("Just the iconic anchors (zoomed in) — some loose trend")
    ax.grid(alpha=0.25, zorder=0)
    ax.set_xlim(left=-0.3)
    ax.set_ylim(bottom=-0.001)

    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_substrate_pairs(out_path: Path) -> None:
    df = pd.read_parquet(SUBSTRATE_PAIRS)

    fig, ax = plt.subplots(figsize=(7.5, 5.5))

    ax.scatter(
        df.phonological_distance,
        df.cosine_distance,
        s=4,
        color=SUBSTRATE_COLOR,
        alpha=0.15,
        edgecolors="none",
        zorder=2,
    )

    ax.set_xlabel(X_LABEL)
    ax.set_ylabel(Y_LABEL)
    ax.set_title("Just the meaning-clusters (zoomed in) — no trend at all")
    ax.grid(alpha=0.25, zorder=0)

    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_shape_overlay(out_path: Path) -> None:
    """Rescale both populations to [0, 1] in both axes and overlay.

    Tests the visual intuition that the two clouds 'share a shape.'
    They share the sampling fan (more dots on the right than the left)
    but only the anchor cloud has the diagonal climb that Spearman
    measures.
    """
    from scipy.stats import spearmanr

    anchor = _load_anchor_signed()
    sub = pd.read_parquet(SUBSTRATE_PAIRS)

    def rescale(x):
        import numpy as np

        x = np.asarray(x, dtype=float)
        return (x - x.min()) / (x.max() - x.min())

    ax_r = rescale(anchor.phon)
    ay_r = rescale(anchor.cos)
    sx_r = rescale(sub.phonological_distance)
    sy_r = rescale(sub.cosine_distance)

    rho_a, _ = spearmanr(anchor.phon, anchor.cos)
    rho_s, _ = spearmanr(sub.phonological_distance, sub.cosine_distance)

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    ax.scatter(
        sx_r,
        sy_r,
        s=4,
        color=SUBSTRATE_COLOR,
        alpha=0.10,
        edgecolors="none",
        label=f"meaning-clusters  (ρ = {rho_s:.2f})",
        zorder=2,
    )
    ax.scatter(
        ax_r,
        ay_r,
        s=20,
        color=ANCHOR_COLOR,
        alpha=0.55,
        edgecolors="none",
        label=f"iconic anchors  (ρ = {rho_a:.2f})",
        zorder=3,
    )
    ax.set_xlabel("how different they sound  (rescaled to [0, 1])")
    ax.set_ylabel("how different the model thinks they are  (rescaled to [0, 1])")
    ax.set_title("Both clouds, rescaled into the same box")
    ax.grid(alpha=0.25, zorder=0)
    ax.legend(loc="lower right", framealpha=0.95)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plot_two_populations(OUT_DIR / "two_populations.png")
    plot_anchor_pairs(OUT_DIR / "anchor_pairs_scatter.png")
    plot_substrate_pairs(OUT_DIR / "substrate_pairs_scatter.png")
    plot_shape_overlay(OUT_DIR / "shape_overlay.png")
    for stale in ("layer_sweep.png", "embed_variants.png", "nw_alignment_schematic.png"):
        p = OUT_DIR / stale
        if p.exists():
            p.unlink()
    print(f"wrote 4 PNGs to {OUT_DIR}")


if __name__ == "__main__":
    main()
