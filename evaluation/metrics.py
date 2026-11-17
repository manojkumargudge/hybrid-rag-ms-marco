"""Ranking metrics (binary relevance) and paired bootstrap confidence intervals."""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np


def reciprocal_rank_at_k(ranking: Sequence[int], relevant: set[int], k: int) -> float:
    for rank, pid in enumerate(ranking[:k], start=1):
        if pid in relevant:
            return 1.0 / rank
    return 0.0


def recall_at_k(ranking: Sequence[int], relevant: set[int], k: int) -> float:
    if not relevant:
        raise ValueError("Query has no relevant passages.")
    return len(set(ranking[:k]) & relevant) / len(relevant)


def ndcg_at_k(ranking: Sequence[int], relevant: set[int], k: int) -> float:
    if not relevant:
        raise ValueError("Query has no relevant passages.")
    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, pid in enumerate(ranking[:k], start=1)
        if pid in relevant
    )
    ideal = sum(1.0 / math.log2(rank + 1) for rank in range(1, min(len(relevant), k) + 1))
    return dcg / ideal


METRICS = {
    "mrr@10": lambda r, rel: reciprocal_rank_at_k(r, rel, 10),
    "recall@10": lambda r, rel: recall_at_k(r, rel, 10),
    "recall@100": lambda r, rel: recall_at_k(r, rel, 100),
    "ndcg@10": lambda r, rel: ndcg_at_k(r, rel, 10),
}


def percentile(values: Sequence[float], q: float) -> float:
    """Linear-interpolation percentile (numpy's default), q in [0, 100]."""
    if not values:
        raise ValueError("No values.")
    ordered = sorted(values)
    position = (len(ordered) - 1) * q / 100
    low, high = math.floor(position), math.ceil(position)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def resample_indices(n: int, n_resamples: int = 10_000, seed: int = 42) -> np.ndarray:
    """Bootstrap query-index matrix (n_resamples x n). Reusing the same matrix
    across systems makes every comparison paired on identical resamples."""
    return np.random.default_rng(seed).integers(0, n, size=(n_resamples, n), dtype=np.int32)


def bootstrap_mean_ci(
    values: Sequence[float],
    n_resamples: int = 10_000,
    seed: int = 42,
    alpha: float = 0.05,
    indices: np.ndarray | None = None,
) -> tuple[float, float, float]:
    """Mean and percentile bootstrap (1 - alpha) CI over queries."""
    array = np.asarray(values, dtype=float)
    if indices is None:
        indices = resample_indices(len(array), n_resamples, seed)
    means = array[indices].mean(axis=1)
    low, high = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(array.mean()), float(low), float(high)


def paired_bootstrap(
    a: Sequence[float],
    b: Sequence[float],
    n_resamples: int = 10_000,
    seed: int = 42,
    alpha: float = 0.05,
    indices: np.ndarray | None = None,
) -> dict:
    """Paired bootstrap of mean(a - b) over the same queries.

    Returns the observed difference, its (1 - alpha) percentile CI, and the
    fraction of resamples with difference <= 0 (one-sided, for a > b).
    """
    if len(a) != len(b) or len(a) == 0:
        raise ValueError("Need two equal-length, non-empty per-query score lists.")
    diffs = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    if indices is None:
        indices = resample_indices(len(diffs), n_resamples, seed)
    resampled = diffs[indices].mean(axis=1)
    low, high = np.percentile(resampled, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {
        "mean_diff": float(diffs.mean()),
        "ci_low": float(low),
        "ci_high": float(high),
        "p_diff_le_0": float((resampled <= 0).mean()),
        "n_queries": len(diffs),
        "n_resamples": int(indices.shape[0]),
    }
