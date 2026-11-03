from abc import ABC, abstractmethod

import numpy as np

from .optimiser_alpha_gamma import define_initial_C_NNLS, initialize_M0_monoexponential
from scipy.special import erf

class SignalModel(ABC):
    "Every signal model needs a grid (params, shape) and these three methods"

    def make_grid(self, **axes):
       
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
        b = np.asarray(acq["b"], dtype=float).ravel()
        TE = np.asarray(acq["TE"], dtype=float).ravel()
        if b.shape != TE.shape:
            raise ValueError("acq['b'] and acq['TE'] must have the same length")
        return np.exp(-np.outer(b, self.D)) * np.exp(-np.outer(TE, 1.0 / self.T2))

    def initial_C(self, Y, A, K, **kwargs):
        return define_initial_C_NNLS(Y, A, range(Y.shape[1]), K, self.D_vals, self.T2_vals,
                                     self.D_vals.min(), self.D_vals.max(),
                                     self.T2_vals.min(), self.T2_vals.max(),
                                     self.nD, self.nT2, **kwargs)

    def initial_M0(self, Y, acq):
        return initialize_M0_monoexponential(Y, np.unique(acq["b"]), np.unique(acq["TE"]))



class DTDModel(SignalModel):
 
    def __init__(self, Disomin, Disomax, nDiso, Ddelmin, Ddelmax, nDdel):
        self.nDiso, self.nDdel = nDiso, nDdel
        self.Diso_vals = np.linspace(Disomin, Disomax, nDiso)
        self.Ddel_vals = np.linspace(Ddelmin, Ddelmax, nDdel)
        self.make_grid(Diso=self.Diso_vals, Ddel=self.Ddel_vals)
        self.Diso, self.Ddel = self.params["Diso"], self.params["Ddel"]
 
        self.extent = [Ddelmin, Ddelmax, Disomin, Disomax]    # for imshow: x = Ddel, y = Diso
        self.labels = ("Ddel", "Diso (µm²/ms)")
 
    def A(self, acq):
        b = np.asarray(acq["b"], dtype=float)[:, None]
        beta = np.asarray(acq["beta"], dtype=float)[:, None]
        eps = 1e-7  
        
        alpha = 3 * b * self.Diso * beta * self.Ddel + eps  
        sqrt_alpha = np.emath.sqrt(alpha)
        return np.exp(-b * self.Diso) * np.exp(alpha / 3) * np.sqrt(np.pi) / 2 * np.real(erf(sqrt_alpha) / sqrt_alpha)
 
    def initial_C(self, Y, A, K, **kwargs):
        return define_initial_C_NNLS(Y, A, range(Y.shape[1]), K, self.Diso_vals, self.Ddel_vals,
                                     self.Diso_vals.min(), self.Diso_vals.max(),
                                     self.Ddel_vals.min(), self.Ddel_vals.max(),
                                     self.nDiso, self.nDdel, **kwargs)
 
    def initial_M0(self, Y, acq):
        b0 = np.asarray(acq["b"]) < 0.05                       #M0 - mean of Y over the b = 0 measurements

        return Y[b0].mean(axis=0)
 