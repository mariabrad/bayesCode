import numpy as np
import pytest

from optimiser_alpha_gamma import (
    update_alpha,
    update_lambda,
    update_sigma2,
)


def test_update_sigma2_matches_mean_squared_residual():
    Y = np.array([[1.0, 2.0], [3.0, 5.0]])
    prediction = np.array([[0.5, 2.5], [2.0, 4.0]])
    residual = Y - prediction
    expected = np.mean(residual**2)
    np.testing.assert_allclose(
        update_sigma2(Y, prediction, np.eye(2), np.ones(2)), expected
    )


def test_update_lambda_matches_gaussian_precision_moment_update():
    C_small = np.array([[1.0, -2.0], [0.5, 1.5]])
    expected = (2 * 2) / np.sum(C_small**2)
    assert update_lambda(C_small, R=2, K=2) == pytest.approx(expected)


def test_update_lambda_applies_upper_cap():
    assert update_lambda(np.zeros((3, 2)), R=3, K=2) == 1e3


def test_update_alpha_returns_scalar_for_simplex_weights():
    rng = np.random.default_rng(0)
    weights = rng.dirichlet(2.5 * np.ones(3), size=500).T
    estimate = update_alpha(weights)
    assert np.isscalar(estimate)
    assert estimate > 0


@pytest.mark.parametrize("alpha_true", [0.5, 5.0])
def test_update_alpha_recovers_symmetric_dirichlet_concentration(alpha_true):
    rng = np.random.default_rng(11)
    weights = rng.dirichlet(alpha_true * np.ones(3), size=4000).T
    assert update_alpha(weights) == pytest.approx(alpha_true, rel=0.12)
