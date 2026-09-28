import numpy as np


def create_DT2_grid(minD, maxD, nD, minT2, maxT2, nT2):
    D_vals = np.linspace(minD, maxD, nD)
    T2_vals = np.linspace(minT2, maxT2, nT2)
    D_mesh, T2_mesh = np.meshgrid(D_vals, T2_vals, indexing="ij")
    D_grid = D_mesh.ravel()
    T2_grid = T2_mesh.ravel()
    return D_grid, T2_grid


def create_A_matrix(D_grid, T2_grid, minB, maxB, nB, minTE, maxTE, nTE):
    b_vals = np.linspace(minB, maxB, nB)
    TE_vals = np.linspace(minTE, maxTE, nTE)
    B_mesh, TE_mesh = np.meshgrid(b_vals, TE_vals, indexing="ij")
    return create_A_matrix_from_acquisition(D_grid, T2_grid, B_mesh.ravel(), TE_mesh.ravel())


def create_A_matrix_from_acquisition(D_grid, T2_grid, b_values, TE_values):
    """Build a forward matrix for arbitrary paired b/TE measurements."""
    b_values = np.asarray(b_values, dtype=float).ravel()
    TE_values = np.asarray(TE_values, dtype=float).ravel()
    if b_values.shape != TE_values.shape:
        raise ValueError("b_values and TE_values must have the same shape")
    A = np.exp(-np.outer(b_values, D_grid)) * np.exp(-np.outer(TE_values, 1 / T2_grid))
    return A


def apply_SVD_to_A(A, R):
    U_full, s_full, Vt_full = np.linalg.svd(A, full_matrices=False)
    U = U_full[:, :R]
    s = s_full[:R]
    V_mat = Vt_full[:R, :].T
    return U, s, V_mat


def create_gaussian_compartments(D_grid, T2_grid, D_means, D_stds, T2_means, T2_stds):
    K = len(D_means)
    C_true = np.zeros((len(D_grid), K))
    for k in range(K):
        C_true[:, k] = np.exp(-0.5 * ((D_grid - D_means[k]) / D_stds[k]) ** 2
                              - 0.5 * ((T2_grid - T2_means[k]) / T2_stds[k]) ** 2)
    C_true = C_true / (C_true.sum(axis=0, keepdims=True) + 1e-12)
    return C_true


def create_W_true(K, n_voxels, alpha_true, rng):
    return rng.dirichlet(alpha_true * np.ones(K), size=n_voxels).T


def create_M0_true(min_M0, max_M0, n_voxels, rng):
    return rng.uniform(min_M0, max_M0, n_voxels)


def create_signal(A, C_true, W_true, M0_true, sigma_true, rng):
    N = A.shape[0]
    n_voxels = W_true.shape[1]
    Y = (A @ C_true @ W_true) * M0_true + sigma_true * rng.standard_normal((N, n_voxels))
    return Y
