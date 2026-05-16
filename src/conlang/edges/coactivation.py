"""Co-activation edges (Stage 3, second edge type).

Run a corpus through Gemma 2 2B, hook the residual stream at the same layer
the SAE was trained on, pass each token's residual through the SAE encoder to
get feature activations, threshold, and accumulate pairwise co-fire counts on
a chosen subset of features.

The Tegmark replication established that *raw activation* differences between
labeled pairs encode named transformations. Co-activation lives at the *feature*
level: two features that both fire on the same token are encoding co-occurring
concepts, which is the semantic-field signal spec §4 Stage 3 wants.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class CoactivationResult:
    cofire: np.ndarray            # (n_slice, n_slice) pair-cofire counts (symmetric, diagonal = single-feature fires)
    fires: np.ndarray             # (n_slice,) per-feature fire counts
    total_tokens: int             # number of real (non-pad) tokens processed
    activation_threshold: float
    layer_index: int


def compute_coactivation(
    model,
    tokenizer,
    sae,
    sentences: list[str],
    slice_feature_ids: list[int],
    layer_index: int,
    activation_threshold: float = 1.0,
    batch_size: int = 8,
    max_length: int = 128,
) -> CoactivationResult:
    """For each pair of slice features: count tokens where both fired.

    `layer_index` is the index into HuggingFace's `output_hidden_states`. For a
    Gemma 2 2B residual-stream SAE labeled `layer_12/...` (output of decoder
    layer 12), pass `layer_index=13`.
    """
    n_slice = len(slice_feature_ids)
    cofire = np.zeros((n_slice, n_slice), dtype=np.int64)
    fires = np.zeros(n_slice, dtype=np.int64)
    total_tokens = 0
    feat_idx = torch.tensor(slice_feature_ids, dtype=torch.long, device=model.device)

    for i in range(0, len(sentences), batch_size):
        batch = sentences[i : i + batch_size]
        enc = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        ).to(model.device)
        with torch.no_grad():
            out = model(**enc, output_hidden_states=True, use_cache=False)
        h = out.hidden_states[layer_index]  # (B, T, D)
        mask = enc.attention_mask.bool()
        # collect real tokens across batch
        h_real = h[mask]  # (n_real, D)
        # cast to SAE's dtype for the encoder
        sae_dtype = next(sae.parameters()).dtype
        h_real = h_real.to(sae_dtype)
        acts = sae.encode(h_real)  # (n_real, n_sae_features)
        slice_acts = acts.index_select(1, feat_idx)  # (n_real, n_slice)
        fired = (slice_acts > activation_threshold).float()  # (n_real, n_slice)
        cofire_batch = (fired.T @ fired).long().cpu().numpy()
        cofire += cofire_batch
        fires += fired.long().sum(dim=0).cpu().numpy()
        total_tokens += int(fired.shape[0])

    return CoactivationResult(
        cofire=cofire,
        fires=fires,
        total_tokens=total_tokens,
        activation_threshold=activation_threshold,
        layer_index=layer_index,
    )


def pmi_normalize(cofire: np.ndarray, fires: np.ndarray, total_tokens: int) -> np.ndarray:
    """Pointwise mutual information between feature pairs.

    PMI(i,j) = log( p(i,j) / (p(i) * p(j)) )
            = log( cofire[i,j] * total / (fires[i] * fires[j]) )

    Sign:
      > 0 → fire together more than chance (positive association)
      = 0 → independent
      < 0 → fire together less than chance

    Symmetric. Diagonal is uninformative — set to 0.
    """
    n = cofire.shape[0]
    denom = np.outer(fires, fires).astype(np.float64)
    numer = cofire.astype(np.float64) * total_tokens
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(denom > 0, numer / denom, 0.0)
        pmi = np.where(ratio > 0, np.log(np.clip(ratio, 1e-12, None)), 0.0)
    np.fill_diagonal(pmi, 0.0)
    # If either feature never fired, set PMI to 0 (undefined).
    never = fires == 0
    pmi[never, :] = 0.0
    pmi[:, never] = 0.0
    return pmi


def top_cofiring_pairs(
    pmi: np.ndarray,
    cofire: np.ndarray,
    k: int = 20,
    min_cofire_count: int = 3,
) -> list[tuple[int, int, float, int]]:
    """Top-k feature pairs by PMI, filtered to pairs that actually co-fired ≥ min_cofire_count times.

    Returns list of (slice_idx_a, slice_idx_b, pmi, cofire_count) sorted by PMI descending.
    """
    n = pmi.shape[0]
    iu, ju = np.triu_indices(n, k=1)
    p = pmi[iu, ju]
    c = cofire[iu, ju]
    mask = c >= min_cofire_count
    order = np.argsort(-p[mask])
    out: list[tuple[int, int, float, int]] = []
    iu_m, ju_m = iu[mask], ju[mask]
    p_m, c_m = p[mask], c[mask]
    for idx in order[:k]:
        out.append((int(iu_m[idx]), int(ju_m[idx]), float(p_m[idx]), int(c_m[idx])))
    return out
