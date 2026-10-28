import numpy as np
import pytest

from create_grid import (
    apply_SVD_to_A,
    create_A_matrix,
    create_DT2_grid,
    create_gaussian_compartments,
    create_signal,
)
from optimiser_alpha_gamma import (
    _W_objective_and_gradient,
    _constrained_C_objective_and_gradient,
    update_C_constrained,
    update_W,
)
from optimiser_alpha_gamma import (
    define_initial_C_small_NNLS,
    define_initial_W_and_m,
    estimate_C,
    initialize_M0_monoexponential,
    run_constrained_optimisation,
)


def _finite_difference_gradient(objective, x, step=1e-6):
    gradient = np.empty_like(x)
    for i in range(x.size):
        plus = x.copy()
        minus = x.copy()
        plus[i] += step
        minus[i] -= step
        gradient[i] = (objective(plus) - objective(minus)) / (2 * step)
    return gradient


def test_constrained_c_analytic_gradient_matches_finite_differences():
    """Check likelihood plus every active C regularizer in logits space."""
    rng = np.random.default_rng(22)
    n_grid, K, R, n_meas, n_voxels = 6, 2, 3, 5, 4
    U, _ = np.linalg.qr(rng.normal(size=(n_meas, R)))
    s = np.array([1.0, 0.7, 0.2])
    V_mat, _ = np.linalg.qr(rng.normal(size=(n_grid, R)))
    logits = rng.normal(size=(n_grid, K)).ravel()
    Y = rng.normal(size=(n_meas, n_voxels))
    W = rng.dirichlet([1.0, 1.0], size=n_voxels).T
    M0 = rng.uniform(0.7, 1.3, size=n_voxels)
    args = (Y, U, s, V_mat, W, M0, 0.8, 0.3, 0.4, 0.2, 0.5, 2, 3)
    _, analytic = _constrained_C_objective_and_gradient(logits, *args)
    numerical = _finite_difference_gradient(
        lambda z: _constrained_C_objective_and_gradient(z, *args)[0], logits
    )
    np.testing.assert_allclose(analytic, numerical, rtol=2e-5, atol=2e-6)


def test_w_analytic_gradient_matches_finite_differences():
    rng = np.random.default_rng(23)
    logits = rng.normal(size=3)
    y = rng.normal(size=5)
    UC = rng.normal(size=(5, 3))
    args = (y, UC, 1.1, 0.7, 1.8, 1e-3, 0.4)
    _, analytic = _W_objective_and_gradient(logits, *args)
    numerical = _finite_difference_gradient(
        lambda z: _W_objective_and_gradient(z, *args)[0], logits
    )
    np.testing.assert_allclose(analytic, numerical, rtol=2e-5, atol=2e-6)


def test_weight_entropy_prior_promotes_dominant_simplex_weights():
    W0 = np.array([[0.60, 0.55], [0.40, 0.45]])
    W_est = update_W(
        np.zeros((2, 2)), np.zeros((2, 2)), sigma2=1.0, alpha=1.0,
        W=W0, m=np.ones(2), bound_eps=1e-3, weight_entropy=5.0,
    )
    assert np.all(W_est.max(axis=0) > W0.max(axis=0))
    np.testing.assert_allclose(W_est.sum(axis=0), 1.0, atol=1e-12)


def test_constrained_c_update_preserves_simplex_and_improves_fit():
    D_grid, T2_grid = create_DT2_grid(0.2, 2.0, 8, 20.0, 150.0, 8)
    A = create_A_matrix(D_grid, T2_grid, 0.0, 4.0, 5, 20.0, 180.0, 5)
    U, s, V_mat = apply_SVD_to_A(A, R=8)
    C_true = create_gaussian_compartments(
        D_grid, T2_grid, [0.4, 1.4], [0.1, 0.1], [90.0, 55.0], [10.0, 10.0]
    )
    W = np.array([[0.8, 0.2, 0.5], [0.2, 0.8, 0.5]])
    M0 = np.ones(3)
    Y = create_signal(A, C_true, W, M0, 0.0, np.random.default_rng(0))
    C0 = np.full_like(C_true, 1.0 / C_true.shape[0])
    B = (U * s[np.newaxis, :]) @ V_mat.T
    initial_error = np.sum((Y - B @ C0 @ W) ** 2)

    C_est = update_C_constrained(
        Y, U, s, V_mat, W, M0, C0, sigma2=1e-4, lam=0.0, maxiter=100
    )
    final_error = np.sum((Y - B @ C_est @ W) ** 2)
    assert (C_est >= 0).all()
    np.testing.assert_allclose(C_est.sum(axis=0), 1.0, atol=1e-12)
    assert final_error < initial_error


def test_constrained_c_smoothness_requires_grid_dimensions():
    with pytest.raises(ValueError):
        update_C_constrained(
            np.ones((2, 1)), np.ones((2, 1)), np.ones(1), np.ones((3, 1)),
            np.ones((1, 1)), np.ones(1), np.full((3, 1), 1 / 3),
            sigma2=1.0, smoothness=1.0,
        )


def test_entropy_sparsity_penalty_preserves_c_simplex():
    Y = np.ones((2, 1))
    U = np.ones((2, 1))
    s = np.ones(1)
    V_mat = np.array([[1.0], [0.5], [0.2]])
    C0 = np.full((3, 1), 1 / 3)
    C_est = update_C_constrained(
        Y, U, s, V_mat, np.ones((1, 1)), np.ones(1), C0,
        sigma2=1.0, sparsity=0.1, maxiter=20,
    )
    assert (C_est >= 0).all()
    np.testing.assert_allclose(C_est.sum(axis=0), 1.0, atol=1e-12)


def test_diversity_penalty_reduces_identical_component_overlap():
    Y = np.zeros((1, 1))
    U = np.ones((1, 1))
    s = np.ones(1)
    V_mat = np.ones((4, 1))
    C0 = np.tile(np.array([[0.7], [0.2], [0.05], [0.05]]), (1, 2))
    initial_overlap = np.dot(C0[:, 0], C0[:, 1])
    C_est = update_C_constrained(
        Y, U, s, V_mat, np.ones((2, 1)), np.ones(1), C0,
        sigma2=1.0, diversity=10.0, maxiter=100,
    )
    final_overlap = np.dot(C_est[:, 0], C_est[:, 1])
    assert final_overlap < initial_overlap
    np.testing.assert_allclose(C_est.sum(axis=0), 1.0, atol=1e-12)


def test_constrained_pipeline_recovers_finite_voxelwise_spectra_from_noisy_data():
    nD = nT2 = 8
    nB = nTE = 5
    K, n_voxels = 3, 12
    D_grid, T2_grid = create_DT2_grid(0.2, 2.0, nD, 20.0, 150.0, nT2)
    D_vals, T2_vals = np.linspace(0.2, 2.0, nD), np.linspace(20.0, 150.0, nT2)
    A = create_A_matrix(D_grid, T2_grid, 0.0, 4.0, nB, 20.0, 180.0, nTE)
    U, s, V_mat = apply_SVD_to_A(A, R=10)
    C_true = create_gaussian_compartments(
        D_grid, T2_grid, [0.4, 0.9, 1.6], [0.1] * K, [90.0, 55.0, 120.0], [10.0] * K
    )
    W_true = np.random.default_rng(1).dirichlet(2.0 * np.ones(K), size=n_voxels).T
    M0_true = np.random.default_rng(2).uniform(0.8, 1.2, n_voxels)
    sigma = 0.002
    Y = create_signal(A, C_true, W_true, M0_true, sigma, np.random.default_rng(3))
    b_vals, te_vals = np.linspace(0.0, 4.0, nB), np.linspace(20.0, 180.0, nTE)
    M0_init = initialize_M0_monoexponential(Y, b_vals, te_vals)
    C_small_init = define_initial_C_small_NNLS(
        Y, A, s, V_mat, range(n_voxels), K, D_vals, T2_vals,
        D_vals.min(), D_vals.max(), T2_vals.min(), T2_vals.max(), nD, nT2,
    )
    C_init = estimate_C(V_mat, s, C_small_init)
    W_init, _ = define_initial_W_and_m(K, n_voxels, np.random.default_rng(4))
    B = (U * s[np.newaxis, :]) @ V_mat.T
    initial_error = np.sum((Y - (B @ C_init @ W_init) * M0_init) ** 2)

    W_est, M0_est, C_est, _, _, _, _, fit_errs = run_constrained_optimisation(
        Y, U, s, V_mat, C_init, M0_init, K, n_iter=10, sigma2=sigma**2,
        alpha=1.0, lam=1e-8, W=W_init, update_alpha_flag=False,
        update_M0_flag=True, m0_prior=M0_init, m0_prior_sigma=0.15,
        c_maxiter=50, verbose_every=1000,
    )
    F_est = C_est @ W_est
    assert np.isfinite(F_est).all() and np.isfinite(M0_est).all()
    assert (C_est >= 0).all() and (W_est >= 0).all() and (M0_est > 0).all()
    np.testing.assert_allclose(C_est.sum(axis=0), 1.0, atol=1e-10)
    np.testing.assert_allclose(W_est.sum(axis=0), 1.0, atol=1e-10)
    assert fit_errs[-1] < initial_error
