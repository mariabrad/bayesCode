import numpy as np


def gaussian_compartments(model, centres, widths):
    C = np.stack([np.exp(-0.5 * sum(((model.params[p] - c[p]) / w[p]) ** 2 for p in c))
                  for c, w in zip(centres, widths)], axis=1)
    return C / (C.sum(axis=0, keepdims=True) + 1e-12)


def create_W_true(K, n_voxels, alpha_true, rng):
    return rng.dirichlet(alpha_true * np.ones(K), size=n_voxels).T


def create_M0_true(min_M0, max_M0, n_voxels, rng):
    return rng.uniform(min_M0, max_M0, n_voxels)


def create_signal(A, C_true, W_true, M0_true, sigma_true, rng):
    N = A.shape[0]
    n_voxels = W_true.shape[1]
    Y = (A @ C_true @ W_true) * M0_true + sigma_true * rng.standard_normal((N, n_voxels))
    return Y
