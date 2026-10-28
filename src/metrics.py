"""Quantitative diagnostics for synthetic spectral-recovery experiments."""

import numpy as np
from scipy.optimize import linear_sum_assignment


def voxelwise_spectrum_mae(F_true, F_est):
    """Mean absolute error over all grid points and voxels."""
    F_true = np.asarray(F_true, dtype=float)
    F_est = np.asarray(F_est, dtype=float)
    if F_true.shape != F_est.shape:
        raise ValueError("F_true and F_est must have the same shape")
    return float(np.mean(np.abs(F_true - F_est)))


def m0_mae(M0_true, M0_est):
    """Mean absolute error of voxelwise signal scales."""
    M0_true = np.asarray(M0_true, dtype=float)
    M0_est = np.asarray(M0_est, dtype=float)
    if M0_true.shape != M0_est.shape:
        raise ValueError("M0_true and M0_est must have the same shape")
    return float(np.mean(np.abs(M0_true - M0_est)))


def dominant_peak_grid_error(F_true, F_est, nD, nT2):
    """Mean dominant-peak displacement, measured in spectral-grid cells."""
    F_true = np.asarray(F_true, dtype=float)
    F_est = np.asarray(F_est, dtype=float)
    if F_true.shape != F_est.shape or F_true.shape[0] != nD * nT2:
        raise ValueError("spectra must match and have nD * nT2 rows")
    errors = []
    for voxel in range(F_true.shape[1]):
        true_index = np.unravel_index(np.argmax(F_true[:, voxel]), (nD, nT2))
        est_index = np.unravel_index(np.argmax(F_est[:, voxel]), (nD, nT2))
        errors.append(np.linalg.norm(np.subtract(true_index, est_index)))
    return float(np.mean(errors))


def match_components(C_small_true, C_small_est, W_true, W_est, K):
    cost = np.zeros((K, K))
    for i in range(K):
        for j in range(K):
            cost[i, j] = np.sum((C_small_true[:, i] - C_small_est[:, j]) ** 2)
    true_idx, est_idx = linear_sum_assignment(cost)
    order = est_idx[np.argsort(true_idx)]
    return C_small_est[:, order], W_est[order, :]


def matched_errors(C_small_true, C_small_est, W_true, W_est):
    K = C_small_true.shape[1]
    C_small_m, W_m = match_components(C_small_true, C_small_est, W_true, W_est, K)
    W_mae = np.mean(np.abs(W_m - W_true))
    C_mae = np.mean(np.abs(C_small_m - C_small_true))
    return W_mae, C_mae


def order_components(C_true, C_est, W_est, K):
    # Cost[i, j] = dissimilarity between true component i and estimated component j.
    cost = np.array([
    [np.mean((C_true[:, i] - C_est[:, j]) ** 2) for j in range(K)]
    for i in range(K)
    ])

    true_indices, estimated_indices = linear_sum_assignment(cost)

    # Put estimated components into the ground-truth component order.
    order = estimated_indices[np.argsort(true_indices)]

    C_est = C_est[:, order]
    W_est = W_est[order, :]
    return C_est, W_est
