from abc import ABC, abstractmethod

import numpy as np

from .optimiser_alpha_gamma import *
from scipy.special import erf

class SignalModel(ABC):
    "Every signal model needs a grid (params, shape) and these three methods"

    def make_grid(self, **axes):
       
        names = list(axes.keys())
        values = list(axes.values())
        self.axes = values

        # grid size along each axis, e.g. (40, 40)
        self.shape = tuple(len(v) for v in values)

        # every combination of axis values, flattened to one column per grid point
        meshes = np.meshgrid(*values, indexing="ij")
        self.params = {}
        for name, mesh in zip(names, meshes):
            self.params[name] = mesh.ravel()


    T2_default = 100.0   # ms and it is used by models without a T2 axis

    def t2_factor(self, acq):
        "exp(-TE/T2) for every measurement and grid point"
        TE = np.asarray(acq["TE"], dtype=float).ravel()[:, None]

        if "T2" in self.params:          # model has a T2 axis -> use its list
            T2 = self.params["T2"]
        else:                            # no T2 axis -> one default value for every grid point
            T2 = self.T2_default

        return np.exp(-TE / T2)


    def initial_constant(self, Y, acq):
        "Scale constant: M0 if there are several TEs, otherwise S0 (the b=0 signal at that one TE)"
        TE = acq.get("TE")
        if TE is not None and np.unique(TE).size > 1:
            return initialize_M0_from_b0(Y, self.A(acq), acq.get("b"))   # several TEs -> M0
        return initialize_S0_linear(Y, self, acq)                         # one TE -> S0


    @abstractmethod
    def A(self, acq): ...

    def initial_C(self, Y, A, K, **kwargs):
        "Default C: NNLS + k-means + one Gaussian per compartment, on this model's grid"
        return define_initial_C_grid(Y, A, range(Y.shape[1]), K, self.axes, **kwargs)

 



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
        return np.exp(-np.outer(b, self.D)) * self.t2_factor(acq)




class DTDModel(SignalModel):
 
    def __init__(self, Disomin, Disomax, nDiso, Ddelmin, Ddelmax, nDdel, T2_default=50.0):
        self.T2_default = T2_default          # ms, fixed T2 for every compartment (no T2 axis here)
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
        return np.exp(-b * self.Diso) * np.exp(alpha / 3) * np.sqrt(np.pi) / 2 * np.real(erf(sqrt_alpha) / sqrt_alpha) * self.t2_factor(acq)

    def initial_C(self, Y, A, K, **kwargs):
        "Same as the default, but NNLS only uses grid cells with Ddel > 0"
        return define_initial_C_grid(Y, A, range(Y.shape[1]), K, self.axes,
                                     allowed=self.Ddel > 0, **kwargs)


class DTDT2Model(DTDModel):
    "Diso x Ddel x T2: the DTD kernel times T2 decay. acq needs b, beta and TE"

    def __init__(self, Disomin, Disomax, nDiso, Ddelmin, Ddelmax, nDdel, T2min, T2max, nT2):
        self.Diso_vals = np.linspace(Disomin, Disomax, nDiso)
        self.Ddel_vals = np.linspace(Ddelmin, Ddelmax, nDdel)
        self.T2_vals = np.linspace(T2min, T2max, nT2)
        self.make_grid(Diso=self.Diso_vals, Ddel=self.Ddel_vals, T2=self.T2_vals)
        self.Diso, self.Ddel, self.T2 = self.params["Diso"], self.params["Ddel"], self.params["T2"]
        self.labels = ("Diso (µm²/ms)", "Ddel", "T2 (ms)")      # one label per axis, in grid order

    def A(self, acq):
        b = np.asarray(acq["b"], dtype=float)[:, None]
        beta = np.asarray(acq["beta"], dtype=float)[:, None]
        eps = 1e-7  
        
        alpha = 3 * b * self.Diso * beta * self.Ddel + eps  
        sqrt_alpha = np.emath.sqrt(alpha)
        return np.exp(-b * self.Diso) * np.exp(alpha / 3) * np.sqrt(np.pi) / 2 * np.real(erf(sqrt_alpha) / sqrt_alpha) * self.t2_factor(acq)

 
 