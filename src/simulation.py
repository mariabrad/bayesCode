import numpy as np
from .create_grid import create_W_true, create_M0_true, create_signal


def gaussian_compartments(model, centres, widths):
    C = np.stack([np.exp(-0.5 * sum(((model.params[p] - c[p]) / w[p]) ** 2 for p in c))
                  for c, w in zip(centres, widths)], axis=1)
    return C / (C.sum(axis=0, keepdims=True) + 1e-12)

