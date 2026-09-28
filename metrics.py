"""Quantitative diagnostics for synthetic spectral-recovery experiments."""

import numpy as np


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
