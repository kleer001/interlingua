"""Bridge between Gemma Scope SAE features and Tegmark-style function vectors.

Per `prior-work.md` §1 and `spec.md` §4 Stage 3: each of the 12 Function-Vectors
relations gives a *direction* in the model's hidden-state space (the mean of all
subject→object difference vectors for that relation). SAE decoder vectors live
in the same hidden-state space, so cosine between an SAE decoder and a relation
centroid measures how much that feature encodes that transformation.

Two alignment views are computed:

- **Raw** — cosine in the full 2304-dim hidden-state space. Faithful to what
  the SAE decoder actually does, but includes the token-length distractor
  direction Tegmark identified.
- **LDA-projected** — cosine in an 8-dim discriminant subspace fit on the
  labeled diffs. The distractor is implicitly projected out. Cleaner separation
  but loses signal that doesn't help discriminate the 12 relations.

Both are useful: raw says "what does this feature emit?", LDA says "which of the
known transformations does it encode most distinctively?".
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_CRYSTAL_DIR = Path("/media/menser/fauna/interlingua/data/interim/crystals")


@dataclass
class CrystalBridge:
    """Result of aligning an SAE decoder matrix against the FV crystals."""

    relations: list[str]                # length R, ordered
    centroids: np.ndarray                # (R, D_model) raw mean diffs
    lda_scalings: np.ndarray             # (D_model, K) — K <= R - 1
    lda_centroids: np.ndarray            # (R, K) — centroids in LDA space
    alignment_raw: np.ndarray            # (N_features, R) cosine in raw space
    alignment_lda: np.ndarray            # (N_features, R) cosine in LDA space


def load_relation_diffs(crystal_dir: Path) -> dict[str, np.ndarray]:
    """Read every CSV in `crystal_dir` as a (n_pairs, d_model) matrix of
    subject→object hidden-state differences.

    Returns dict relation_name → array, keyed by csv stem (e.g. "english-french").
    """
    files = sorted(crystal_dir.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"no CSVs in {crystal_dir}")
    return {f.stem: pd.read_csv(f, header=None).to_numpy(dtype=np.float32) for f in files}


def relation_centroids(diffs: dict[str, np.ndarray]) -> tuple[list[str], np.ndarray]:
    """Per-relation mean difference vector. Returns (names, (R, D))."""
    names = list(diffs.keys())
    centroids = np.stack([diffs[n].mean(axis=0) for n in names])
    return names, centroids.astype(np.float32)


def fit_lda_basis(
    diffs: dict[str, np.ndarray],
    n_components: int = 8,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit Linear Discriminant Analysis on the labeled diffs and return:
    - scalings: (D, K) projection matrix (K <= n_components, K <= R-1)
    - mean: (D,) overall mean used for centering during projection

    The scalings columns span the K-dim subspace where the R relation classes
    are most separable. To project a vector v: v_lda = (v - mean) @ scalings.
    """
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

    X = np.concatenate(list(diffs.values()), axis=0)
    y = np.concatenate(
        [np.full(diff.shape[0], i, dtype=np.int64) for i, diff in enumerate(diffs.values())]
    )
    lda = LinearDiscriminantAnalysis(n_components=n_components)
    lda.fit(X, y)
    return lda.scalings_[:, :n_components].astype(np.float32), lda.xbar_.astype(np.float32)


def _cosine_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Cosine sim of every row of a against every row of b. Returns (len(a), len(b))."""
    a_n = a / np.clip(np.linalg.norm(a, axis=1, keepdims=True), 1e-12, None)
    b_n = b / np.clip(np.linalg.norm(b, axis=1, keepdims=True), 1e-12, None)
    return a_n @ b_n.T


def build_bridge(
    decoder: np.ndarray,
    diffs: dict[str, np.ndarray],
    n_lda_components: int = 8,
) -> CrystalBridge:
    """Compute the full alignment of an SAE decoder against the given relation diffs.

    `decoder` shape: (N_features, D_model).
    """
    names, centroids = relation_centroids(diffs)
    scalings, mean = fit_lda_basis(diffs, n_components=n_lda_components)

    # LDA-space versions
    decoder_lda = (decoder - mean) @ scalings
    centroids_lda = (centroids - mean) @ scalings

    return CrystalBridge(
        relations=names,
        centroids=centroids,
        lda_scalings=scalings,
        lda_centroids=centroids_lda,
        alignment_raw=_cosine_matrix(decoder, centroids),
        alignment_lda=_cosine_matrix(decoder_lda, centroids_lda),
    )


def coverage_stats(
    alignment: np.ndarray, thresholds: list[float]
) -> dict[float, dict[str, float | int]]:
    """For each threshold T: how many SAE features have at least one crystal
    with cosine ≥ T (positive direction)? Reports count and fraction.

    NOTE: a feature's "best" alignment can be high merely because all 12 crystal
    centroids share a common distractor direction (Tegmark §3). Use
    `distinctiveness_stats` for a better signal.
    """
    n_features = alignment.shape[0]
    best_per_feature = alignment.max(axis=1)
    out: dict[float, dict[str, float | int]] = {}
    for t in thresholds:
        n = int((best_per_feature >= t).sum())
        out[t] = {
            "n_features": n,
            "fraction": n / n_features,
        }
    return out


def distinctiveness_stats(
    alignment: np.ndarray,
    margin_thresholds: list[float],
) -> dict[float, dict[str, float | int]]:
    """For each margin T: how many features prefer their best crystal by ≥ T
    over their 2nd-best? Distinctive features are encoding a real transformation;
    non-distinctive features have similar alignment with many crystals (likely
    a distractor or a non-relation feature)."""
    n_features = alignment.shape[0]
    sorted_desc = np.sort(alignment, axis=1)[:, ::-1]
    margin = sorted_desc[:, 0] - sorted_desc[:, 1]
    out: dict[float, dict[str, float | int]] = {}
    for t in margin_thresholds:
        n = int((margin >= t).sum())
        out[t] = {"n_features": n, "fraction": n / n_features}
    return out


def distinctive_top_features_per_crystal(
    alignment: np.ndarray,
    relations: list[str],
    k: int = 10,
    min_margin: float = 0.0,
) -> dict[str, list[tuple[int, float, float]]]:
    """Like top_features_per_crystal, but ranks by *margin* (best - 2nd-best)
    rather than raw alignment, and only includes features whose best crystal
    matches the relation in question.

    Returns dict[relation] -> list of (feature_idx, alignment_score, margin)
    sorted by margin descending.
    """
    sorted_desc = np.sort(alignment, axis=1)[:, ::-1]
    margin = sorted_desc[:, 0] - sorted_desc[:, 1]
    best_idx = alignment.argmax(axis=1)

    out: dict[str, list[tuple[int, float, float]]] = {}
    for j, name in enumerate(relations):
        # features whose best crystal IS this one
        mask = (best_idx == j) & (margin >= min_margin)
        candidates = np.where(mask)[0]
        if candidates.size == 0:
            out[name] = []
            continue
        ordered = candidates[np.argsort(-margin[candidates])][:k]
        out[name] = [
            (int(i), float(alignment[i, j]), float(margin[i])) for i in ordered
        ]
    return out


def top_features_per_crystal(
    alignment: np.ndarray,
    relations: list[str],
    k: int = 10,
) -> dict[str, list[tuple[int, float]]]:
    """For each relation: the top-k SAE feature indices ranked by cosine."""
    out: dict[str, list[tuple[int, float]]] = {}
    for j, name in enumerate(relations):
        col = alignment[:, j]
        idx = np.argsort(-col)[:k]
        out[name] = [(int(i), float(col[i])) for i in idx]
    return out


def best_crystal_per_feature(
    alignment: np.ndarray,
    relations: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """For each SAE feature: the index of its best-matching crystal and the
    score. Returns (best_idx_per_feature, best_score_per_feature)."""
    best_idx = alignment.argmax(axis=1)
    best_score = alignment.max(axis=1)
    return best_idx, best_score
