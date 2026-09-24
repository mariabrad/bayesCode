import matplotlib
matplotlib.use("Agg")  # no display, so plt.show() never blocks

import numpy as np
import pytest

from plotting import matched_errors, match_components


def test_match_components_recovers_permutation():
    rng = np.random.default_rng(0)
    K = 3
    C_small_true = rng.normal(size=(5, K))
    W_true = rng.dirichlet(np.ones(K), size=10).T

    perm = [2, 0, 1]
    C_small_est = C_small_true[:, perm]
    W_est = W_true[perm, :]

    C_small_m, W_m = match_components(C_small_true, C_small_est, W_true, W_est, K)
    np.testing.assert_allclose(C_small_m, C_small_true)
    np.testing.assert_allclose(W_m, W_true)


def test_matched_errors_zero_for_permuted_input():
    rng = np.random.default_rng(0)
    K = 3
    C_small_true = rng.normal(size=(5, K))
    W_true = rng.dirichlet(np.ones(K), size=10).T

    perm = [2, 0, 1]
    C_small_est = C_small_true[:, perm]
    W_est = W_true[perm, :]

    W_mae, C_mae = matched_errors(C_small_true, C_small_est, W_true, W_est)
    assert W_mae == pytest.approx(0.0, abs=1e-12)
    assert C_mae == pytest.approx(0.0, abs=1e-12)


def test_matched_errors_nonzero_for_unrelated_input():
    rng = np.random.default_rng(3)
    K = 3
    C_small_true = rng.normal(size=(5, K))
    W_true = rng.dirichlet(np.ones(K), size=10).T
    C_small_est = rng.normal(size=(5, K))
    W_est = rng.dirichlet(np.ones(K), size=10).T

    W_mae, C_mae = matched_errors(C_small_true, C_small_est, W_true, W_est)
    assert W_mae > 0.01
    assert C_mae > 0.01
