from abc import ABC, abstractmethod

import numpy as np

from optimiser_alpha_gamma import define_initial_C_NNLS, initialize_M0_monoexponential


class SignalModel(ABC):
    """Every signal model needs a grid (params, shape) and these three methods."""

    def make_grid(self, **axes):
        """Build the grid from named axes, e.g. make_grid(D=D_vals, T2=T2_vals)."""
        names = list(axes.keys())
        values = list(axes.values())

        # grid size along each axis, e.g. (40, 40)
        self.shape = tuple(len(v) for v in values)

        # every combination of axis values, flattened to one column per grid point
        meshes = np.meshgrid(*values, indexing="ij")
        self.params = {}
        for name, mesh in zip(names, meshes):
            self.params[name] = mesh.ravel()

    @abstractmethod
    def A(self, acq): ...

    @abstractmethod
    def initial_C(self, Y, A, K, **kwargs): ...

    @abstractmethod
    def initial_M0(self, Y, acq): ...



class DiffusionT2Model(SignalModel):
    def __init__(self, Dmin, Dmax, nD, T2min, T2max, nT2):
        self.nD, self.nT2 = nD, nT2
        self.D_vals = np.linspace(Dmin, Dmax, nD)
        self.T2_vals = np.linspace(T2min, T2max, nT2)
        self.make_grid(D=self.D_vals, T2=self.T2_vals)       # sets self.shape and self.params
        self.D, self.T2 = self.params["D"], self.params["T2"]

        self.extent = [T2min, T2max, Dmin, Dmax]          # for imshow: x = T2, y = D
        self.labels = ("T2 (ms)", "D (µm²/ms)")           # (x label, y label)


    def A(self, acq):
        return np.exp(-np.outer(acq["b"], self.D)) * np.exp(-np.outer(acq["TE"], 1.0 / self.T2))

    def initial_C(self, Y, A, K, **kwargs):
        return define_initial_C_NNLS(Y, A, range(Y.shape[1]), K, self.D_vals, self.T2_vals,
                                     self.D_vals.min(), self.D_vals.max(),
                                     self.T2_vals.min(), self.T2_vals.max(),
                                     self.nD, self.nT2, **kwargs)

    def initial_M0(self, Y, acq):
        return initialize_M0_monoexponential(Y, np.unique(acq["b"]), np.unique(acq["TE"]))
