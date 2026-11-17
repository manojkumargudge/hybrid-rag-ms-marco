"""Metric unit tests on hand-computed toy rankings."""

import math

import pytest

from evaluation.metrics import (
    bootstrap_mean_ci,
    ndcg_at_k,
    paired_bootstrap,
    percentile,
    recall_at_k,
    reciprocal_rank_at_k,
)


def test_reciprocal_rank():
    assert reciprocal_rank_at_k([5, 7, 9], {9}, 10) == pytest.approx(1 / 3)
    assert reciprocal_rank_at_k([9, 7], {9, 7}, 10) == 1.0
    # Relevant at rank 11 is outside the cutoff.
    assert reciprocal_rank_at_k(list(range(10)) + [99], {99}, 10) == 0.0


def test_recall():
    assert recall_at_k([1, 2, 3], {2, 4}, 10) == 0.5
    assert recall_at_k([1, 2, 3], {3}, 2) == 0.0
    assert recall_at_k([], {1}, 100) == 0.0
    with pytest.raises(ValueError):
        recall_at_k([1], set(), 10)


def test_ndcg_hand_computed():
    # Relevant at ranks 2 and 3, two relevant total:
    # DCG = 1/log2(3) + 1/log2(4); IDCG = 1/log2(2) + 1/log2(3).
    expected = (1 / math.log2(3) + 0.5) / (1 + 1 / math.log2(3))
    assert ndcg_at_k([0, 1, 2], {1, 2}, 10) == pytest.approx(expected)
    assert ndcg_at_k([1, 2, 0], {1, 2}, 10) == pytest.approx(1.0)
    assert ndcg_at_k([0, 0, 0], {1}, 10) == 0.0


def test_ndcg_ideal_is_capped_at_k():
    # 3 relevant but k=1: ideal DCG is one hit at rank 1.
    assert ndcg_at_k([7, 8, 9], {7, 8, 9}, 1) == 1.0


def test_percentile_matches_linear_interpolation():
    values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    assert percentile(values, 50) == 5.5
    assert percentile(values, 95) == pytest.approx(9.55)
    assert percentile([42], 95) == 42


def test_bootstrap_mean_ci_brackets_mean_and_is_deterministic():
    values = [0.0, 1.0] * 50
    mean, low, high = bootstrap_mean_ci(values, n_resamples=2000)
    assert mean == 0.5
    assert low < 0.5 < high
    assert bootstrap_mean_ci(values, n_resamples=2000) == (mean, low, high)


def test_paired_bootstrap_detects_consistent_improvement():
    a = [1.0] * 30 + [0.5] * 30
    b = [0.5] * 60
    result = paired_bootstrap(a, b, n_resamples=2000)
    assert result["mean_diff"] == pytest.approx(0.25)
    assert result["ci_low"] > 0
    assert result["p_diff_le_0"] == 0.0


def test_paired_bootstrap_identical_systems():
    a = [0.2, 0.4, 0.9]
    result = paired_bootstrap(a, a, n_resamples=500)
    assert result["mean_diff"] == result["ci_low"] == result["ci_high"] == 0.0


def test_paired_bootstrap_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        paired_bootstrap([1.0], [1.0, 0.0])
