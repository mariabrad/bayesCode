import numpy as np
import pytest

from postprocessing import find_spectral_peaks, threshold_spectrum


def test_threshold_spectrum_removes_weak_mass_and_normalizes():
    f = np.array([0.01, 0.20, 0.02, 0.77])
    out = threshold_spectrum(f, relative_threshold=0.05)
    np.testing.assert_allclose(out, [0.0, 0.20 / 0.97, 0.0, 0.77 / 0.97])
    np.testing.assert_allclose(out.sum(), 1.0)


def test_threshold_spectrum_rejects_invalid_threshold():
    with pytest.raises(ValueError):
        threshold_spectrum(np.ones(4), relative_threshold=1.1)


def test_find_spectral_peaks_returns_sorted_hotspots():
    image = np.zeros((5, 6))
    image[1, 2] = 0.8
    image[3, 4] = 0.4
    image[0, 0] = 0.01
    peaks = find_spectral_peaks(image.ravel(), 5, 6, relative_threshold=0.1)
    assert [(p["D_index"], p["T2_index"]) for p in peaks] == [(1, 2), (3, 4)]
    assert peaks[0]["value"] > peaks[1]["value"]


def test_find_spectral_peaks_rejects_wrong_shape():
    with pytest.raises(ValueError):
        find_spectral_peaks(np.ones(5), 2, 3)


def test_find_spectral_peaks_detects_boundary_hotspots():
    image = np.zeros((5, 6))
    image[0, 0] = 1.0
    image[4, 5] = 0.8
    peaks = find_spectral_peaks(image.ravel(), 5, 6, relative_threshold=0.1)
    locations = {(p["D_index"], p["T2_index"]) for p in peaks}
    assert (0, 0) in locations
    assert (4, 5) in locations
