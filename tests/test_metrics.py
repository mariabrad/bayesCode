import numpy as np
import pytest

from metrics import dominant_peak_grid_error, m0_mae, voxelwise_spectrum_mae


def test_voxelwise_spectrum_mae_and_m0_mae():
    assert voxelwise_spectrum_mae(np.zeros((2, 2)), np.ones((2, 2))) == pytest.approx(1.0)
    assert m0_mae([1.0, 3.0], [2.0, 1.0]) == pytest.approx(1.5)


def test_dominant_peak_grid_error():
    F_true = np.zeros((9, 2))
    F_est = np.zeros((9, 2))
    F_true[0, 0] = F_est[0, 0] = 1.0
    F_true[8, 1] = 1.0
    F_est[5, 1] = 1.0  # (2, 2) versus (1, 2): one grid cell apart
    assert dominant_peak_grid_error(F_true, F_est, 3, 3) == pytest.approx(0.5)
