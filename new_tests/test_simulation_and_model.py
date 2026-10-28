import numpy as np
import pytest

from src.optimiser_alpha_gamma import apply_SVD_to_A
from src.signal_models import DiffusionT2Model
from src.simulation import create_W_true, create_M0_true, create_signal, gaussian_compartments


def acquisition(minB, maxB, nB, minTE, maxTE, nTE):
    """Every combination of nB b-values and nTE echo times, b-major (as the old create_A_matrix)."""
    b = np.linspace(minB, maxB, nB)
    TE = np.linspace(minTE, maxTE, nTE)
    return {"b": np.repeat(b, nTE), "TE": np.tile(TE, nB)}


def dt2_compartments(model, D_means, D_stds, T2_means, T2_stds):
    """Gaussian compartments on the D-T2 grid (as the old create_gaussian_compartments)."""
    centres = [{"D": d, "T2": t} for d, t in zip(D_means, T2_means)]
    widths = [{"D": d, "T2": t} for d, t in zip(D_stds, T2_stds)]
    return gaussian_compartments(model, centres, widths)


def test_model_grid_shapes():
    model = DiffusionT2Model(0.2, 2.0, 5, 20.0, 150.0, 6)
    D_grid, T2_grid = model.D, model.T2
    assert D_grid.shape == (30,)
    assert T2_grid.shape == (30,)


def test_model_A_shape():
    model = DiffusionT2Model(0.2, 2.0, 5, 20.0, 150.0, 6)
    D_grid, T2_grid = model.D, model.T2
    A = model.A(acquisition(0.0, 4.0, 3, 20.0, 180.0, 4))
    assert A.shape == (3 * 4, 5 * 6)


def test_model_A_values_are_bounded_signal_decay():
    # A models exp(-b*D) * exp(-TE/T2), so every entry should be in (0, 1]
    model = DiffusionT2Model(0.2, 2.0, 5, 20.0, 150.0, 6)
    D_grid, T2_grid = model.D, model.T2
    A = model.A(acquisition(0.0, 4.0, 3, 20.0, 180.0, 4))
    assert (A > 0).all()
    assert (A <= 1.0 + 1e-12).all()


def test_model_A_respects_arbitrary_measurement_order():
    model = DiffusionT2Model(0.2, 2.0, 5, 20.0, 150.0, 6)
    D_grid, T2_grid = model.D, model.T2
    b = np.array([2.0, 0.0, 4.0, 0.0])
    te = np.array([80.0, 180.0, 20.0, 20.0])
    A = model.A({"b": b, "TE": te})
    expected = np.exp(-np.outer(b, D_grid)) * np.exp(-np.outer(te, 1 / T2_grid))
    np.testing.assert_allclose(A, expected)


def test_model_A_rejects_unpaired_coordinates():
    model = DiffusionT2Model(0.2, 2.0, 5, 20.0, 150.0, 6)
    D_grid, T2_grid = model.D, model.T2
    with pytest.raises(ValueError):
        model.A({"b": [0.0, 1.0], "TE": [20.0]})


def test_apply_SVD_to_A_shapes():
    model = DiffusionT2Model(0.2, 2.0, 5, 20.0, 150.0, 6)
    D_grid, T2_grid = model.D, model.T2
    A = model.A(acquisition(0.0, 4.0, 3, 20.0, 180.0, 4))
    R = 5
    U, s, V_mat = apply_SVD_to_A(A, R)
    assert U.shape == (A.shape[0], R)
    assert s.shape == (R,)
    assert V_mat.shape == (A.shape[1], R)


def test_apply_SVD_to_A_full_rank_reconstructs_A():
    model = DiffusionT2Model(0.2, 2.0, 5, 20.0, 150.0, 6)
    D_grid, T2_grid = model.D, model.T2
    A = model.A(acquisition(0.0, 4.0, 3, 20.0, 180.0, 4))
    R = min(A.shape)
    U, s, V_mat = apply_SVD_to_A(A, R)
    A_recon = U @ np.diag(s) @ V_mat.T
    np.testing.assert_allclose(A_recon, A, atol=1e-10)


def test_apply_SVD_to_A_singular_values_are_sorted_descending():
    model = DiffusionT2Model(0.2, 2.0, 5, 20.0, 150.0, 6)
    D_grid, T2_grid = model.D, model.T2
    A = model.A(acquisition(0.0, 4.0, 3, 20.0, 180.0, 4))
    _, s, _ = apply_SVD_to_A(A, R=5)
    assert (np.diff(s) <= 0).all()


def test_gaussian_compartments_shape():
    model = DiffusionT2Model(0.2, 2.0, 5, 20.0, 150.0, 6)
    D_grid, T2_grid = model.D, model.T2
    C = dt2_compartments(model,
        D_means=[0.4, 0.9], D_stds=[0.1, 0.1],
        T2_means=[90, 55], T2_stds=[15, 10],
    )
    assert C.shape == (30, 2)


def test_gaussian_compartments_columns_sum_to_one():
    model = DiffusionT2Model(0.2, 2.0, 20, 20.0, 150.0, 20)
    D_grid, T2_grid = model.D, model.T2
    C = dt2_compartments(model,
        D_means=[0.4, 0.9, 1.6], D_stds=[0.10, 0.12, 0.16],
        T2_means=[90, 55, 120], T2_stds=[15, 10, 18],
    )
    np.testing.assert_allclose(C.sum(axis=0), 1.0, atol=1e-9)


def test_gaussian_compartments_nonnegative():
    model = DiffusionT2Model(0.2, 2.0, 20, 20.0, 150.0, 20)
    D_grid, T2_grid = model.D, model.T2
    C = dt2_compartments(model,
        D_means=[0.4, 0.9, 1.6], D_stds=[0.10, 0.12, 0.16],
        T2_means=[90, 55, 120], T2_stds=[15, 10, 18],
    )
    assert (C >= 0).all()


def test_create_W_true_shape():
    rng = np.random.default_rng(0)
    W = create_W_true(K=3, n_voxels=50, alpha_true=1.5, rng=rng)
    assert W.shape == (3, 50)


def test_create_W_true_is_valid_simplex():
    rng = np.random.default_rng(0)
    W = create_W_true(K=3, n_voxels=50, alpha_true=1.5, rng=rng)
    np.testing.assert_allclose(W.sum(axis=0), 1.0)
    assert (W >= 0).all()
    assert (W <= 1.0).all()


def test_create_W_true_reproducible_with_same_seed():
    W1 = create_W_true(3, 20, 1.5, np.random.default_rng(1))
    W2 = create_W_true(3, 20, 1.5, np.random.default_rng(1))
    np.testing.assert_array_equal(W1, W2)


def test_create_W_true_different_seeds_give_different_draws():
    W1 = create_W_true(3, 20, 1.5, np.random.default_rng(1))
    W2 = create_W_true(3, 20, 1.5, np.random.default_rng(2))
    assert not np.allclose(W1, W2)


def test_create_W_true_higher_alpha_is_more_uniform():
    # larger Dirichlet concentration -> weights cluster closer to 1/K -> lower variance
    rng_low = np.random.default_rng(0)
    rng_high = np.random.default_rng(0)
    W_low = create_W_true(K=3, n_voxels=2000, alpha_true=0.5, rng=rng_low)
    W_high = create_W_true(K=3, n_voxels=2000, alpha_true=20.0, rng=rng_high)
    assert W_high.var() < W_low.var()


def test_create_M0_true_shape_and_bounds():
    rng = np.random.default_rng(0)
    M0 = create_M0_true(0.7, 1.3, n_voxels=25, rng=rng)
    assert M0.shape == (25,)
    assert (M0 >= 0.7).all()
    assert (M0 <= 1.3).all()


def test_create_M0_true_reproducible_with_same_seed():
    M0_1 = create_M0_true(0.7, 1.3, 25, np.random.default_rng(3))
    M0_2 = create_M0_true(0.7, 1.3, 25, np.random.default_rng(3))
    np.testing.assert_array_equal(M0_1, M0_2)


def test_create_signal_shape():
    model = DiffusionT2Model(0.2, 2.0, 5, 20.0, 150.0, 6)
    D_grid, T2_grid = model.D, model.T2
    A = model.A(acquisition(0.0, 4.0, 3, 20.0, 180.0, 4))
    C = dt2_compartments(model, D_means=[0.4, 0.9], D_stds=[0.1, 0.1],
        T2_means=[90, 55], T2_stds=[15, 10],
    )
    rng = np.random.default_rng(0)
    W = create_W_true(2, 10, 1.5, rng)
    M0 = create_M0_true(0.7, 1.3, 10, rng)
    Y = create_signal(A, C, W, M0, sigma_true=0.002, rng=rng)
    assert Y.shape == (A.shape[0], 10)


def test_create_signal_zero_noise_matches_noiseless_forward_model():
    model = DiffusionT2Model(0.2, 2.0, 5, 20.0, 150.0, 6)
    D_grid, T2_grid = model.D, model.T2
    A = model.A(acquisition(0.0, 4.0, 3, 20.0, 180.0, 4))
    C = dt2_compartments(model, D_means=[0.4, 0.9], D_stds=[0.1, 0.1],
        T2_means=[90, 55], T2_stds=[15, 10],
    )
    rng = np.random.default_rng(0)
    W = create_W_true(2, 10, 1.5, rng)
    M0 = create_M0_true(0.7, 1.3, 10, rng)
    Y = create_signal(A, C, W, M0, sigma_true=0.0, rng=rng)
    expected = (A @ C @ W) * M0
    np.testing.assert_allclose(Y, expected)


def test_create_signal_reproducible_with_same_rng_state():
    model = DiffusionT2Model(0.2, 2.0, 5, 20.0, 150.0, 6)
    D_grid, T2_grid = model.D, model.T2
    A = model.A(acquisition(0.0, 4.0, 3, 20.0, 180.0, 4))
    C = dt2_compartments(model, D_means=[0.4, 0.9], D_stds=[0.1, 0.1],
        T2_means=[90, 55], T2_stds=[15, 10],
    )
    W = np.full((2, 10), 0.5)
    M0 = np.ones(10)

    Y1 = create_signal(A, C, W, M0, sigma_true=0.01, rng=np.random.default_rng(7))
    Y2 = create_signal(A, C, W, M0, sigma_true=0.01, rng=np.random.default_rng(7))
    np.testing.assert_array_equal(Y1, Y2)
