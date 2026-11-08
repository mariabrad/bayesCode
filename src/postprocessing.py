"""Clean up estimated spectra and find their peaks."""

import numpy as np
from scipy.ndimage import maximum_filter


def threshold_spectrum(f, relative_threshold=0.05):
    """Set values below relative_threshold * max to 0, then rescale so the spectrum sums to 1."""
    if not 0 <= relative_threshold < 1:
        raise ValueError("relative_threshold must be in [0, 1)")
    f = np.asarray(f, dtype=float).copy()
    f[f < relative_threshold * f.max()] = 0
    return f / f.sum()


def find_spectral_peaks(f, shape, relative_threshold=0.1):
    """Local maxima of a spectrum on a grid of the given shape (e.g. model.shape), strongest first.

    Peaks weaker than relative_threshold * max are ignored. Each peak is a dict
    with its grid position ("index", one entry per axis) and its value.
    """
    f = np.asarray(f, dtype=float)
    if f.size != np.prod(shape):
        raise ValueError(f"spectrum has {f.size} values, expected {np.prod(shape)}")
    image = f.reshape(shape)

    # a peak is at least as large as its 8 neighbours (outside the grid counts as -inf)
    is_peak = image == maximum_filter(image, size=3, mode="constant", cval=-np.inf)
    is_peak &= image > 0
    is_peak &= image >= relative_threshold * image.max()

    peaks = [{"index": tuple(int(i) for i in ij), "value": float(image[ij])}
             for ij in zip(*np.nonzero(is_peak))]
    return sorted(peaks, key=lambda p: p["value"], reverse=True)
