"""Small, model-faithful tests for staged spectral recovery.

These tests deliberately use only 20 voxels so they remain quick while the
parameterization and alternating updates are being debugged.
"""

import numpy as np
import pytest

from create_grid import (
    apply_SVD_to_A,
    create_A_matrix,
    create_DT2_grid,
    create_M0_true,
    create_W_true,
    create_gaussian_compartments,
    create_signal,
)
from optimiser_alpha_gamma import (
    _lowest_b_data,
    define_initial_C_small,
    define_initial_W_and_m,
    initialize_M0_from_b0_T2,
    initialize_M0_monoexponential,
    run_optimisation,
    update_C_small,
    update_M0,
    update_W,
)


def _fixture(n_voxels=20):
    D_grid, T2_grid = create_DT2_grid(0.2, 2.0, 10, 20.0, 150.0, 10)
    A = create_A_matrix(D_grid, T2_grid, 0.0, 4.0, 6, 20.0, 180.0, 6)
    U, s, V_mat = apply_SVD_to_A(A, R=5)

    C_true = create_gaussian_compartments(
        D_grid,
        T2_grid,
        D_means=[0.4, 0.9, 1.6],
        D_stds=[0.08, 0.08, 0.08],
        T2_means=[90.0, 55.0, 120.0],
        T2_stds=[10.0, 10.0, 10.0],
    )
    W_true = create_W_true(3, n_voxels, 3.0, np.random.default_rng(10))
    M0_true = create_M0_true(0.8, 1.2, n_voxels, np.random.default_rng(11))
    Y = create_signal(
        A, C_true, W_true, M0_true, sigma_true=0.0, rng=np.random.default_rng(12)
    )
    C_small_true = np.diag(s) @ V_mat.T @ C_true
    return U, C_small_true, W_true, M0_true, Y


def test_fixed_components_recover_m0():
    U, C_small, W_true, M0_true, Y = _fixture()
    M0_est = update_M0(Y, U @ C_small, W_true)
    np.testing.assert_allclose(M0_est, M0_true, atol=1e-10)


def test_initialize_m0_from_b0_t2_fit():
    U, C_small, W_true, M0_true, Y = _fixture()
    b_vals = np.linspace(0.0, 4.0, 6)
    TE_vals = np.linspace(20.0, 180.0, 6)
    T2_vals = np.linspace(20.0, 150.0, 10)
    M0_init = initialize_M0_from_b0_T2(
        Y, b_vals, TE_vals, T2_vals
    )
    np.testing.assert_allclose(M0_init, M0_true, atol=0.03)


def test_initialize_m0_monoexponential_is_positive_and_reasonable():
    _, _, _, M0_true, Y = _fixture()
    TE_vals = np.linspace(20.0, 180.0, 6)
    b_vals = np.linspace(0.0, 4.0, 6)
    M0_init = initialize_M0_monoexponential(Y, b_vals, TE_vals)
    assert np.isfinite(M0_init).all()
    assert (M0_init > 0).all()
    np.testing.assert_allclose(M0_init, M0_true, atol=0.2)


def test_b0_selection_uses_zero_tolerance_not_lowest_available_b_value():
    Y = np.array([[1.0], [2.0], [3.0]])
    te, signals = _lowest_b_data(Y, [0.04, 0.06, 1.0], [10.0, 20.0, 10.0])
    np.testing.assert_allclose(te, [10.0])
    np.testing.assert_allclose(signals, [[1.0]])
    with pytest.raises(ValueError, match="b=0"):
        _lowest_b_data(Y, [0.06, 0.2, 1.0], [10.0, 20.0, 10.0])


def test_m0_initialization_is_invariant_to_measurement_order():
    _, _, _, M0_true, Y = _fixture()
    b_axis = np.linspace(0.0, 4.0, 6)
    te_axis = np.linspace(20.0, 180.0, 6)
    b_grid, te_grid = np.meshgrid(b_axis, te_axis, indexing="ij")
    permutation = np.random.default_rng(4).permutation(Y.shape[0])
    M0_init = initialize_M0_monoexponential(
        Y[permutation], b_grid.ravel()[permutation], te_grid.ravel()[permutation]
    )
    np.testing.assert_allclose(M0_init, M0_true, atol=0.2)


def test_m0_gaussian_prior_regularizes_closed_form_update():
    Y = np.array([[2.0], [2.0]])
    UC = np.array([[1.0], [1.0]])
    W = np.ones((1, 1))
    M0 = update_M0(Y, UC, W, prior_mean=np.array([1.0]), prior_sigma=0.5)
    np.testing.assert_allclose(M0, [8.0 / 6.0])


def test_fixed_c_and_m0_recover_w_on_simplex():
    U, C_small, W_true, M0_true, Y = _fixture()
    W_est = update_W(
        Y,
        U @ C_small,
        sigma2=1e-8,
        alpha=1.0,
        W=W_true.copy(),
        m=M0_true,
        bound_eps=1e-8,
    )
    np.testing.assert_allclose(W_est.sum(axis=0), 1.0, atol=1e-10)
    np.testing.assert_allclose(W_est, W_true, atol=1e-5)


def test_fixed_w_and_m0_recover_c_small():
    U, C_small_true, W_true, M0_true, Y = _fixture()
    C_small_est = update_C_small(
        Y,
        U,
        W_true,
        M0_true,
        lam=1e-10,
        sigma2=1.0,
        R=C_small_true.shape[0],
        K=C_small_true.shape[1],
    )
    np.testing.assert_allclose(C_small_est, C_small_true, atol=1e-5)


def test_joint_c_and_w_optimization_improves_reconstruction():
    U, C_small_true, W_true, M0_true, Y = _fixture()
    rng = np.random.default_rng(20)
    W0, _ = define_initial_W_and_m(3, W_true.shape[1], rng, alpha_init=5.0)
    C0 = define_initial_C_small(Y, U, W0, M0_true, K=3)
    initial_err = np.sum((Y - (U @ C0 @ W0) * M0_true) ** 2)

    _, _, _, _, _, _, losses, fit_errs = run_optimisation(
        Y,
        U,
        C0,
        M0_true,
        K=3,
        R=C_small_true.shape[0],
        eps=1e-10,
        n_iter=15,
        sigma2=1e-4,
        alpha=1.0,
        lam=1e-8,
        W=W0,
        update_alpha_flag=False,
        update_W_flag=True,
        update_M0_flag=False,
        update_C_flag=True,
        verbose_every=1000,
    )

    assert np.isfinite(losses).all()
    assert np.isfinite(fit_errs).all()
    assert fit_errs[-1] < initial_err


def test_joint_optimization_with_hyperparameter_updates_is_finite():
    U, C_small_true, W_true, M0_true, Y = _fixture()
    rng = np.random.default_rng(21)
    W0, _ = define_initial_W_and_m(3, W_true.shape[1], rng, alpha_init=5.0)
    C0 = define_initial_C_small(Y, U, W0, M0_true, K=3)

    result = run_optimisation(
        Y,
        U,
        C0,
        M0_true,
        K=3,
        R=C_small_true.shape[0],
        eps=1e-10,
        n_iter=5,
        sigma2=1e-3,
        alpha=2.0,
        lam=1.0,
        W=W0,
        update_alpha_flag=True,
        update_W_flag=True,
        update_M0_flag=False,
        update_C_flag=True,
        update_sigma2_flag=True,
        update_lambda_flag=True,
        verbose_every=1000,
    )
    W_est, _, _, alpha, sigma2, lam, losses, fit_errs = result
    assert np.isfinite(W_est).all()
    assert alpha > 0 and sigma2 > 0 and lam > 0
    assert np.isfinite(losses).all() and np.isfinite(fit_errs).all()


def test_joint_c_w_m0_optimization_remains_finite_and_improves_fit():
    U, C_small_true, W_true, M0_true, Y = _fixture()
    rng = np.random.default_rng(22)
    W0, M00 = define_initial_W_and_m(3, W_true.shape[1], rng, alpha_init=5.0)
    C0 = define_initial_C_small(Y, U, W0, M00, K=3)
    initial_err = np.sum((Y - (U @ C0 @ W0) * M00) ** 2)

    W_est, M0_est, _, _, _, _, _, fit_errs = run_optimisation(
        Y,
        U,
        C0,
        M00,
        K=3,
        R=C_small_true.shape[0],
        eps=1e-10,
        n_iter=15,
        sigma2=1e-4,
        alpha=1.0,
        lam=1e-8,
        W=W0,
        update_alpha_flag=False,
        update_W_flag=True,
        update_M0_flag=True,
        update_C_flag=True,
        verbose_every=1000,
    )
    assert np.isfinite(W_est).all() and np.isfinite(M0_est).all()
    assert (M0_est > 0).all()
    assert np.isfinite(fit_errs).all()
    assert fit_errs[-1] < initial_err


def test_joint_optimization_can_start_from_monoexponential_m0():
    U, C_small_true, W_true, M0_true, Y = _fixture()
    TE_vals = np.linspace(20.0, 180.0, 6)
    b_vals = np.linspace(0.0, 4.0, 6)
    M0_start = initialize_M0_monoexponential(Y, b_vals, TE_vals)
    rng = np.random.default_rng(23)
    W0, _ = define_initial_W_and_m(3, W_true.shape[1], rng, alpha_init=5.0)
    C0 = define_initial_C_small(Y, U, W0, M0_start, K=3)
    initial_err = np.sum((Y - (U @ C0 @ W0) * M0_start) ** 2)

    _, M0_est, _, _, _, _, _, fit_errs = run_optimisation(
        Y, U, C0, M0_start, K=3, R=C_small_true.shape[0], eps=1e-10,
        n_iter=15, sigma2=1e-4, alpha=1.0, lam=1e-8, W=W0,
        update_alpha_flag=False, update_W_flag=True, update_M0_flag=True,
        update_C_flag=True, verbose_every=1000,
    )
    assert np.isfinite(M0_est).all() and (M0_est > 0).all()
    assert np.isfinite(fit_errs).all()
    assert fit_errs[-1] < initial_err
