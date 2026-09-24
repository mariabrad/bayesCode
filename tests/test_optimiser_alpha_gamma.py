import numpy as np
import pytest
from optimiser_alpha_gamma import *
from create_grid import (
    create_DT2_grid, create_A_matrix, apply_SVD_to_A,
    create_gaussian_compartments, create_W_true, create_M0_true, create_signal,
)
from plotting import *

TRUE_D_MEANS = [0.4, 0.9, 1.6]


def _spectrum_setup():
    D_grid, T2_grid = create_DT2_grid(0.2, 2.0, 16, 20.0, 150.0, 16)
    A = create_A_matrix(D_grid, T2_grid, 0.0, 4.0, 8, 20.0, 180.0, 8)
    U, s, V_mat = apply_SVD_to_A(A, R=6)
    C_true = create_gaussian_compartments(D_grid, T2_grid, TRUE_D_MEANS, [0.08] * 3, [90, 55, 120], [10] * 3)

    rng = np.random.default_rng(0)
    W_true = create_W_true(3, 20, 3.0, rng)
    M0_true = create_M0_true(0.7, 1.3, 20, rng)
    Y = create_signal(A, C_true, W_true, M0_true, 0.002, rng)
    return Y, A, s, V_mat


def test_define_csmall_nnls_picks_compartments_close_to_true_D():
    Y, A, s, V_mat = _spectrum_setup()
    D_vals, T2_vals = np.linspace(0.2, 2.0, 16), np.linspace(20.0, 150.0, 16)

    _, _, _, centers, _, _ = define_initial_C_small_NNLS(
        Y, A, s, V_mat, voxel_idx=range(20), K=3, D_vals=D_vals, T2_vals=T2_vals,
        Dmin=0.2, Dmax=2.0, T2min=20.0, T2max=150.0, nD=16, nT2=16,
        return_diagnostics=True,
    )
    for D_c, _ in centers:
        closest_diff = min(abs(D_c - d) for d in TRUE_D_MEANS)
        assert closest_diff < 0.4


def test_define_initial_C_small_MADCO_picks_compartments_close_to_true_D():
    Y, A, s, V_mat = _spectrum_setup()
    D_vals, T2_vals = np.linspace(0.2, 2.0, 16), np.linspace(20.0, 150.0, 16)
    b_vals, TE_vals = np.linspace(0.0, 4.0, 8), np.linspace(20.0, 180.0, 8)

    _, _, _, centers, _, _ = define_initial_C_small_MADCO(
        Y, A, s, V_mat, voxel_idx=range(20), K=3, D_vals=D_vals, T2_vals=T2_vals,
        b_vals=b_vals, TE_vals=TE_vals, Dmin=0.2, Dmax=2.0, T2min=20.0, T2max=150.0,
        nB=8, nTE=8, nD=16, nT2=16, return_diagnostics=True,
    )
    for D_c, _ in centers:
        closest_diff = min(abs(D_c - d) for d in TRUE_D_MEANS)
        assert closest_diff < 0.4


def test_define_initial_W_and_m_is_valid_simplex():
    rng = np.random.default_rng(0)
    W, m = define_initial_W_and_m(K=3, n_voxels=10, rng=rng, alpha_init=5.0)
    assert W.shape == (3, 10)
    np.testing.assert_allclose(W.sum(axis=0), 1.0)
    assert m.shape == (10,)


def test_update_alpha_recovers_known_concentration():
    rng = np.random.default_rng(0)
    W = rng.dirichlet(3.0 * np.ones(3), size=500).T
    alpha_est = update_alpha(W)
    assert alpha_est == pytest.approx(3.0, abs=0.5)


def test_update_W_output_is_valid_simplex():
    rng = np.random.default_rng(0)
    K, R, n_voxels, N = 3, 4, 6, 8
    U = rng.normal(size=(N, R))
    C_small = rng.normal(size=(R, K))
    W = rng.dirichlet(np.ones(K), size=n_voxels).T
    M0 = np.ones(n_voxels)
    UC = U @ C_small
    Y = (UC @ W) * M0

    W_new = update_W(Y, UC, sigma2=1e-3, alpha=2.0, W=W, m=M0, bound_eps=1e-2)
    np.testing.assert_allclose(W_new.sum(axis=0), 1.0, atol=1e-6)


def test_update_M0_recovers_true_scale():
    rng = np.random.default_rng(0)
    K, R, n_voxels, N = 3, 4, 6, 8
    U = rng.normal(size=(N, R))
    C_small = rng.normal(size=(R, K))
    W = rng.dirichlet(np.ones(K), size=n_voxels).T
    M0_true = rng.uniform(0.7, 1.3, n_voxels)
    UC = U @ C_small
    Y = (UC @ W) * M0_true

    M0_est = update_M0(Y, UC, W)
    np.testing.assert_allclose(M0_est, M0_true, atol=1e-8)


def test_define_initial_C_small_matches_lstsq_solution():
    rng = np.random.default_rng(0)
    K, R, n_voxels, N = 3, 4, 6, 8
    U, _ = np.linalg.qr(rng.normal(size=(N, R)))  # orthonormal, like a real SVD U
    C_small_true = rng.normal(size=(R, K))
    W = rng.dirichlet(np.ones(K), size=n_voxels).T
    M0 = np.ones(n_voxels)
    Y = (U @ C_small_true) @ W  # noiseless, so lstsq should recover it exactly

    C_small_est = define_initial_C_small(Y, U, W, M0, K)
    np.testing.assert_allclose(C_small_est, C_small_true, atol=1e-8)




def test_update_C_small_recovers_true_value():
    np.random.seed(3)
    N, R, K, n_voxels = 6, 4, 2, 20
    U_test = np.linalg.qr(np.random.randn(N, R))[0]
    C_small_true = np.random.randn(R, K) * 0.5
    W = np.random.dirichlet(np.ones(K), size=n_voxels).T
    M0 = np.ones(n_voxels)
    Y = U_test @ C_small_true @ W

    C_small_est = update_C_small(Y, U_test, W, M0, lam=1e-6, sigma2=1.0, R=R, K=K)
    assert np.allclose(C_small_est, C_small_true, atol=1e-2)
    print("update_C_small OK, max err:", np.max(np.abs(C_small_est - C_small_true)))


def test_update_W_recovers_true_value():
    np.random.seed(3)
    N, R, K, n_voxels = 6, 4, 2, 20
    U_test = np.linalg.qr(np.random.randn(N, R))[0]
    C_small_true = np.random.randn(R, K) * 0.5
    W_true = np.random.dirichlet(np.ones(K), size=n_voxels).T
    M0 = np.ones(n_voxels)
    UC = U_test @ C_small_true
    Y = UC @ W_true

    W_est = update_W(Y, UC, sigma2=1.0, alpha=1.0, W=W_true.copy(), m=M0, bound_eps=1e-3)
    assert np.allclose(W_est, W_true, atol=1e-2)
    print("update_W OK, max err:", np.max(np.abs(W_est - W_true)))


def test_update_alpha_recovers_true_value():
    np.random.seed(3)
    K, alpha_true = 3, 4.0
    W = np.random.dirichlet(alpha_true * np.ones(K), size=1000).T

    alpha_est = update_alpha(W)
    assert np.isclose(alpha_est, alpha_true, atol=0.5)
    print("update_alpha OK, est:", alpha_est, "true:", alpha_true)


def test_update_M0_recovers_true_value():
    np.random.seed(3)
    N, R, K, n_voxels = 6, 4, 2, 20
    U_test = np.linalg.qr(np.random.randn(N, R))[0]
    C_small_true = np.random.randn(R, K) * 0.5
    W_true = np.random.dirichlet(np.ones(K), size=n_voxels).T
    M0_true = np.random.uniform(0.7, 1.3, n_voxels)
    UC = U_test @ C_small_true
    Y = UC @ W_true * M0_true

    M0_est = update_M0(Y, UC, W_true)
    assert np.allclose(M0_est, M0_true, atol=1e-2)
    print("update_M0 OK, max err:", np.max(np.abs(M0_est - M0_true)))


def test_run_optimisation_loss_decreases():
    rng = np.random.default_rng(0)
    K, R, n_voxels, N = 3, 4, 6, 8
    U, _ = np.linalg.qr(rng.normal(size=(N, R)))
    C_small_true = rng.normal(size=(R, K))
    W_true = rng.dirichlet(np.ones(K), size=n_voxels).T
    M0 = np.ones(n_voxels)
    Y = (U @ C_small_true) @ W_true * M0 + 0.01 * rng.standard_normal((N, n_voxels))

    W0, _ = define_initial_W_and_m(K, n_voxels, rng, alpha_init=5.0)
    C_small0 = define_initial_C_small(Y, U, W0, M0, K)

    *_, losses, fit_errs = run_optimisation(
        Y, U, C_small0, M0, K, R, eps=1e-10, n_iter=20,
        sigma2=1e-3, alpha=5.0, lam=1.0, W=W0,
        update_alpha_flag=False, update_W_flag=True,
        update_M0_flag=False, update_C_flag=True,
        verbose_every=100,
    )
    assert losses[-1] < losses[0]
    assert np.isfinite(losses).all()


# zero noise fixed-point tests 

def _zero_noise_setup(n_voxels=500):
    rng = np.random.default_rng(0)
    K, R, N = 3, 4, 8
    U, _ = np.linalg.qr(rng.normal(size=(N, R)))
    C_small_true = rng.normal(size=(R, K))
    W_true = rng.dirichlet(3.0 * np.ones(K), size=n_voxels).T
    M0_true = rng.uniform(0.7, 1.3, n_voxels)
    Y = (U @ C_small_true) @ W_true * M0_true
    return U, C_small_true, W_true, M0_true, Y, K, R


def test_W_is_fixed_point_at_zero_noise():
    U, C_small_true, W_true, M0_true, Y, K, R = _zero_noise_setup()
    W_est, *_ = run_optimisation(
        Y, U, C_small_true.copy(), M0_true, K, R, eps=1e-10, n_iter=5,
        sigma2=1e-10, alpha=3.0, lam=1e-8, W=W_true.copy(),
        update_alpha_flag=False, update_W_flag=True,
        update_M0_flag=False, update_C_flag=False, verbose_every=100,
    )
    np.testing.assert_allclose(W_est, W_true, atol=1e-6)


def test_C_small_is_fixed_point_at_zero_noise():
    U, C_small_true, W_true, M0_true, Y, K, R = _zero_noise_setup()
    _, _, C_small_est, *_ = run_optimisation(
        Y, U, C_small_true.copy(), M0_true, K, R, eps=1e-10, n_iter=5,
        sigma2=1e-10, alpha=3.0, lam=1e-8, W=W_true.copy(),
        update_alpha_flag=False, update_W_flag=False,
        update_M0_flag=False, update_C_flag=True, verbose_every=100,
    )
    np.testing.assert_allclose(C_small_est, C_small_true, atol=1e-8)


def test_alpha_is_recovered_from_true_W_at_zero_noise():
    U, C_small_true, W_true, M0_true, Y, K, R = _zero_noise_setup()
    _, _, _, alpha_est, *_ = run_optimisation(
        Y, U, C_small_true.copy(), M0_true, K, R, eps=1e-10, n_iter=1,
        sigma2=1e-10, alpha=3.0, lam=1e-8, W=W_true.copy(),
        update_alpha_flag=True, update_W_flag=False,
        update_M0_flag=False, update_C_flag=False, verbose_every=100,
    )
    assert alpha_est == pytest.approx(3.0, abs=0.2)


def test_M0_is_fixed_point_at_zero_noise():
    U, C_small_true, W_true, M0_true, Y, K, R = _zero_noise_setup()
    _, M0_est, *_ = run_optimisation(
        Y, U, C_small_true.copy(), M0_true, K, R, eps=1e-10, n_iter=5,
        sigma2=1e-10, alpha=3.0, lam=1e-8, W=W_true.copy(),
        update_alpha_flag=False, update_W_flag=False,
        update_M0_flag=True, update_C_flag=False, verbose_every=100,
    )
    np.testing.assert_allclose(M0_est, M0_true, atol=1e-8)


def test_sigma2_collapses_to_floor_at_zero_noise():
    U, C_small_true, W_true, M0_true, Y, K, R = _zero_noise_setup()
    _, _, _, _, sigma2_est, *_ = run_optimisation(
        Y, U, C_small_true.copy(), M0_true, K, R, eps=1e-10, n_iter=1,
        sigma2=1e-10, alpha=3.0, lam=1e-8, W=W_true.copy(),
        update_alpha_flag=False, update_W_flag=False,
        update_M0_flag=False, update_C_flag=False, update_sigma2_flag=True,
        verbose_every=100,
    )
    assert sigma2_est == pytest.approx(1e-6, abs=1e-9)  # SIGMA2_FLOOR


# NNLS/MADCO init: does optimisation reach the truth at zero noise? 

def test_nnls_init_converges_close_to_truth_at_zero_noise():
    D_grid, T2_grid = create_DT2_grid(0.2, 2.0, 12, 20.0, 150.0, 12)
    A = create_A_matrix(D_grid, T2_grid, 0.0, 4.0, 8, 20.0, 180.0, 8)
    U, s, V_mat = apply_SVD_to_A(A, R=5)
    C_true = create_gaussian_compartments(D_grid, T2_grid, TRUE_D_MEANS, [0.08] * 3, [90, 55, 120], [10] * 3)
    D_vals, T2_vals = np.linspace(0.2, 2.0, 12), np.linspace(20.0, 150.0, 12)

    rng = np.random.default_rng(0)
    W_true = create_W_true(3, 8, 3.0, rng)
    M0_true = create_M0_true(0.7, 1.3, 8, rng)
    Y = create_signal(A, C_true, W_true, M0_true, 0.0, rng)  # zero noise
    C_small_true = np.diag(s) @ V_mat.T @ C_true
    voxel_idx = np.arange(8)

    C_small_init = define_initial_C_small_NNLS(Y, A, s, V_mat, voxel_idx, K=3, D_vals=D_vals, T2_vals=T2_vals,
                                       Dmin=0.2, Dmax=2.0, T2min=20.0, T2max=150.0, nD=12, nT2=12)
    lam0 = define_initial_lambda(C_small_init, R=5, K=3)
    W0, _ = define_initial_W_and_m(3, 8, rng, alpha_init=5.0)

    W_est, _, C_small_est, *_ = run_optimisation(
        Y, U, C_small_init, M0_true, K=3, R=5, eps=1e-10, n_iter=300,
        sigma2=1e-10, alpha=5.0, lam=lam0, W=W0, bound_eps=1e-2,
        update_alpha_flag=True, update_W_flag=True,
        update_M0_flag=False, update_C_flag=True, verbose_every=1000,
    )
    W_mae, C_mae = matched_errors(C_small_true, C_small_est, W_true, W_est)
    # NNLS starts close enough to the true compartments so this should be a tight (not exact) recovery
    assert W_mae < 0.05
    assert C_mae < 0.05


def test_madco_init_stays_bounded_at_zero_noise():
    D_grid, T2_grid = create_DT2_grid(0.2, 2.0, 12, 20.0, 150.0, 12)
    A = create_A_matrix(D_grid, T2_grid, 0.0, 4.0, 8, 20.0, 180.0, 8)
    U, s, V_mat = apply_SVD_to_A(A, R=5)
    C_true = create_gaussian_compartments(D_grid, T2_grid, TRUE_D_MEANS, [0.08] * 3, [90, 55, 120], [10] * 3)
    D_vals, T2_vals = np.linspace(0.2, 2.0, 12), np.linspace(20.0, 150.0, 12)
    b_vals, TE_vals = np.linspace(0.0, 4.0, 8), np.linspace(20.0, 180.0, 8)

    rng = np.random.default_rng(0)
    W_true = create_W_true(3, 8, 3.0, rng)
    M0_true = create_M0_true(0.7, 1.3, 8, rng)
    Y = create_signal(A, C_true, W_true, M0_true, 0.0, rng)
    C_small_true = np.diag(s) @ V_mat.T @ C_true
    voxel_idx = np.arange(8)

    C_small_init = define_initial_C_small_MADCO(Y, A, s, V_mat, voxel_idx, K=3, D_vals=D_vals, T2_vals=T2_vals,
                                                  b_vals=b_vals, TE_vals=TE_vals, Dmin=0.2, Dmax=2.0,
                                                  T2min=20.0, T2max=150.0, nB=8, nTE=8, nD=12, nT2=12)
    lam0 = define_initial_lambda(C_small_init, R=5, K=3)
    W0, _ = define_initial_W_and_m(3, 8, rng, alpha_init=5.0)

    W_est, _, C_small_est, *_ = run_optimisation(
        Y, U, C_small_init, M0_true, K=3, R=5, eps=1e-10, n_iter=300,
        sigma2=1e-10, alpha=5.0, lam=lam0, W=W0, bound_eps=1e-2,
        update_alpha_flag=True, update_W_flag=True,
        update_M0_flag=False, update_C_flag=True, verbose_every=1000,
    )
    W_mae, C_mae = matched_errors(C_small_true, C_small_est, W_true, W_est)
    # MADCO's init lands in a worse basin of attraction than NNLS's so this documents that known limitation rather than asserting exact recovery
    assert np.isfinite(W_mae) and np.isfinite(C_mae)
    assert W_mae < 0.3
    assert C_mae < 0.3
