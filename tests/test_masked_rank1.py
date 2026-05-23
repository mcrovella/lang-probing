"""Verification of masked rank-1 ALS (filled by the overnight verify pass)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "experiments" / "perplexity_bleu_linear"))
from rank1_approximation import masked_rank1_als, masked_faithfulness  # noqa: E402


def make_low_rank_with_holes(shape, rank, hole_frac, seed):
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(shape[0], rank))
    B = rng.normal(size=(rank, shape[1]))
    M = A @ B
    mask = rng.random(size=shape) >= hole_frac
    for i in range(min(shape)):
        mask[i, i] = False
        M[i, i] = np.nan
    M = np.where(mask, M, np.nan)
    return M, mask


def test_masked_rank1_recovers_low_rank_with_holes():
    M, mask = make_low_rank_with_holes((8, 8), rank=1, hole_frac=0.2, seed=0)
    u, v = masked_rank1_als(M, mask, n_iter=300, seed=0)
    # A true rank-1 matrix is recovered almost perfectly on observed cells.
    assert masked_faithfulness(M, mask, u, v) > 0.999
    # The diagonal is excluded from the fit (never observed).
    assert not mask.diagonal().any()


def test_masked_faithfulness_ignores_unobserved_cells():
    M, mask = make_low_rank_with_holes((6, 6), rank=1, hole_frac=0.3, seed=1)
    u, v = masked_rank1_als(M, mask, n_iter=300, seed=1)
    f = masked_faithfulness(M, mask, u, v)
    # Corrupting an UNobserved cell must not change the observed-cell faithfulness.
    M2 = M.copy()
    M2[~mask] = 1e6
    assert masked_faithfulness(M2, mask, u, v) == pytest.approx(f)
