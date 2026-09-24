import warnings                              
import numpy as np                                    
from sklearn.linear_model import Ridge          
import dirichlet      
from scipy.special import gammaln, softmax      
from scipy.optimize import minimize   
from scipy.optimize import nnls
from sklearn.cluster import KMeans


warnings.filterwarnings(
    "ignore",                                        
    message="Values in x were outside bounds during a minimize step, clipping to bounds",  
    category=RuntimeWarning,                          
)

SIGMA2_FLOOR = 1e-6   # smallest sigma2 value allowed
BOUND_EPS = 1e-2      # smallest weight value allowed during optimization, keeps log(w) finite


def compute_loss(Y, UC, W, M0, alpha, lam, C_small, sigma2, w_floor=1e-10):
    resid = Y - (UC @ W) * M0
    fit_err = np.sum(resid**2)

    n_voxels = W.shape[1]
    K = W.shape[0]

    data_term = fit_err / (2 * sigma2)                       
    reg_term = (lam / 2) * np.sum(C_small**2)
    dir_term = (
        -(alpha - 1) * np.sum(np.log(np.maximum(W, w_floor)))
        + n_voxels * (K * gammaln(alpha) - gammaln(alpha * K))
    )
    return data_term + reg_term + dir_term, fit_err

def update_sigma2(Y, UC, W, M0):
    resid = Y - (UC @ W) * M0                          # same  definition as in compute_loss
    n_meas, n_voxels = Y.shape                         
    return np.sum(resid**2) / (n_meas * n_voxels)        # closed-form MLE mean squared residual over all entries


def update_lambda(C_small, R, K):
    new_lam = (R * K) / (np.sum(C_small**2) + 1e-12)     #  precision ~ (#params) / (sum of squares)
    return min(new_lam, 1e3)                              # cap lambda at 1e3 to avoid over-shrinking C_small to zero


def update_alpha(W):
    alpha_hat = dirichlet.mle(W.T)                        # fit Dirichlet concentration params via MLE, one per component
    return np.mean(alpha_hat)                          
                           

def update_W(Y, UC, sigma2, alpha, W, m, bound_eps=1e-2):
    n_voxels = W.shape[1]
    W_new = np.zeros_like(W)

    for v in range(n_voxels):
        yv = Y[:, v]

        def objective(z):
            w = softmax(z)
            w = np.maximum(w, bound_eps)
            w = w / w.sum()
            pred = m[v] * (UC @ w)
            resid = yv - pred
            data_term = np.sum(resid**2) / (2 * sigma2)

            K = w.shape[0]
            sym_dir_term = K * gammaln(alpha) - gammaln(K * alpha)
            dir_term = -(alpha - 1) * np.sum(np.log(w)) + sym_dir_term

            return data_term + dir_term

        w0 = np.clip(W[:, v], bound_eps, 1.0)
        w0 = w0 / w0.sum()
        z0 = np.log(w0 + 1e-12)

        res = minimize(objective, z0, method="L-BFGS-B")
        w_new = softmax(res.x)      # "ABNORMAL" stops happen at an already-converged point, so keep res.x

        w_new = np.maximum(w_new, bound_eps)
        w_new = w_new / w_new.sum()
        W_new[:, v] = w_new

    return W_new                                     


def update_M0(Y, UC, W):
    n_voxels = W.shape[1]                                          # number of voxels
    M0_new = np.zeros(n_voxels)                                     # allocate output array for updated per-voxel scales
    pred_unscaled = UC @ W                                          # unscaled prediction (before applying M0) for all voxels
    for v in range(n_voxels):                                     
        p = pred_unscaled[:, v]                                    
        num = np.dot(p, Y[:, v])                                  
        denom = np.dot(p, p) + 1e-12                                  
        M0_new[v] = num / denom                                       # closed-form least-squares scalar fit
    return M0_new                                                 


def update_C_small(Y, U, W, M0, lam, sigma2, R, K):
    Y_proj = U.T @ Y                      # project raw Y, no division
    X = (W * M0).T                        # fold M0 into the design matrix instead
    ridge_alpha = lam * max(sigma2, SIGMA2_FLOOR)
    C_small_new = np.zeros((R, K))
    for r in range(R):
        model = Ridge(alpha=ridge_alpha, fit_intercept=False)
        model.fit(X, Y_proj[r, :])
        C_small_new[r, :] = model.coef_
    return C_small_new                                             


def define_initial_C_small(Y, U, W, M0, K):
    Y_proj = U.T @ Y                          # (R, V) -  project raw Y, no division
    X = (W * M0[np.newaxis, :]).T             # (V, K) — add M0 into the  matrix
    C_small_init = np.linalg.lstsq(X, Y_proj.T, rcond=None)[0]   # (K, R)
    return C_small_init.T                     # (R, K)


def define_initial_C_small_NNLS(Y, A, s, V_mat, voxel_idx, K, D_vals, T2_vals,
                        Dmin, Dmax, T2min, T2max, nD, nT2, seed=0,
                        return_diagnostics=False):
    spectra = []
    for v in voxel_idx:
        f, _ = nnls(A, Y[:, v])
        spectra.append(f / f.sum() if f.sum() > 0 else f)
    tissue_spectrum = np.mean(spectra, axis=0)
    tissue_spectrum /= tissue_spectrum.sum()
    F_tissue = tissue_spectrum.reshape(nD, nT2)

    D_mesh, T2_mesh = np.meshgrid(D_vals, T2_vals, indexing='ij')
    grid_points = np.column_stack([
        (D_mesh.ravel() - Dmin) / (Dmax - Dmin),
        (T2_mesh.ravel() - T2min) / (T2max - T2min),
    ])
    kmeans = KMeans(n_clusters=K, n_init=10, random_state=seed)
    kmeans.fit(grid_points, sample_weight=tissue_spectrum)
    labels = kmeans.labels_.reshape(nD, nT2)

    D_step = D_vals[1] - D_vals[0]
    T2_step = T2_vals[1] - T2_vals[0]
    n_grid = nD * nT2
    C_new_gauss = np.zeros((n_grid, K))
    compartment_centers = []
    compartment_widths = []
    GD, GT2 = np.meshgrid(D_vals, T2_vals, indexing='ij')
    for i in range(K):
        idxD, idxT2 = np.nonzero(labels == i)
        w = F_tissue[idxD, idxT2]
        if w.sum() <= 0 or len(idxD) == 0:
            # fall back to a broad default (avoids divide-by-zero)
            D_c, T2_c = Dmin + (Dmax - Dmin) / 2, T2min + (T2max - T2min) / 2
            sD, sT2 = (Dmax - Dmin) / 4, (T2max - T2min) / 4
        else:
            D_c = np.sum(D_vals[idxD] * w) / w.sum()
            T2_c = np.sum(T2_vals[idxT2] * w) / w.sum()
            sD = np.sqrt(np.sum(w * (D_vals[idxD] - D_c) ** 2) / w.sum())
            sT2 = np.sqrt(np.sum(w * (T2_vals[idxT2] - T2_c) ** 2) / w.sum())
            sD = max(sD, D_step)      # floor to at least one grid step
            sT2 = max(sT2, T2_step)
        compartment_centers.append((D_c, T2_c))
        compartment_widths.append((sD, sT2))

        blob = np.exp(-0.5 * ((GD - D_c) / sD) ** 2) * np.exp(-0.5 * ((GT2 - T2_c) / sT2) ** 2)
        C_new_gauss[:, i] = blob.ravel() / blob.sum()

    C_small_init = np.diag(s) @ V_mat.T @ C_new_gauss

    if return_diagnostics:
        return C_small_init, tissue_spectrum, labels, compartment_centers, compartment_widths, C_new_gauss
    return C_small_init



def _data_driven_marginals(y, b_vals, TE_vals, D_vals, T2_vals, nB, nTE, n_avg=1):
    y_grid = y.reshape(nB, nTE)
    y_b0 = y_grid[:n_avg, :].mean(axis=0)
    y_te_min = y_grid[:, :n_avg].mean(axis=1)

    A_T2 = np.exp(-np.outer(TE_vals, 1.0 / T2_vals))
    m_T2, _ = nnls(A_T2, y_b0)
    m_T2 = m_T2 / m_T2.sum() if m_T2.sum() > 0 else m_T2

    A_D = np.exp(-np.outer(b_vals, D_vals))
    m_D, _ = nnls(A_D, y_te_min)
    m_D = m_D / m_D.sum() if m_D.sum() > 0 else m_D
    return m_D, m_T2


def _compute_marginals(f, D_vals, T2_vals):
    F = f.reshape(len(D_vals), len(T2_vals))
    m_D = F.sum(axis=1)
    m_T2 = F.sum(axis=0)
    m_D = m_D / m_D.sum() if m_D.sum() > 0 else m_D
    m_T2 = m_T2 / m_T2.sum() if m_T2.sum() > 0 else m_T2
    return m_D, m_T2


def _fit_MADCO_soft(A, S, D_vals, T2_vals, mt, reg_weight):
    f_start, _ = nnls(A, S)

    def objective(f):
        data_fit = np.sum((A @ f - S) ** 2)
        f_marginals = _compute_marginals(f, D_vals, T2_vals)
        marginal_error = sum(np.sum((e - t) ** 2) for e, t in zip(f_marginals, mt))
        return data_fit + reg_weight * marginal_error

    # only f >= 0; no sum-to-1 constraint, because the signal is scaled by M0
    bounds = [(0, None)] * A.shape[1]
    res = minimize(objective, f_start, bounds=bounds, method="L-BFGS-B")
    f = np.clip(res.x, 0, None)
    return f / f.sum() if f.sum() > 0 else f      # normalise afterwards, like the NNLS init


def define_initial_C_small_MADCO(Y, A, s, V_mat, voxel_idx, K, D_vals, T2_vals, b_vals, TE_vals,
                                  Dmin, Dmax, T2min, T2max, nB, nTE, nD, nT2,
                                  reg_weight=10.0, n_avg=1, seed=0, return_diagnostics=False):
    # per-voxel MADCO spectra: NNLS fit softly constrained to match data-driven marginals
    spectra = []
    for v in voxel_idx:
        y_v = Y[:, v]
        m_D, m_T2 = _data_driven_marginals(y_v, b_vals, TE_vals, D_vals, T2_vals, nB, nTE, n_avg=n_avg)
        spectra.append(_fit_MADCO_soft(A, y_v, D_vals, T2_vals, (m_D, m_T2), reg_weight))
    tissue_spectrum = np.mean(spectra, axis=0)
    tissue_spectrum /= tissue_spectrum.sum()
    F_tissue = tissue_spectrum.reshape(nD, nT2)

    # cluster the averaged spectrum into K compartments
    D_mesh, T2_mesh = np.meshgrid(D_vals, T2_vals, indexing='ij')
    grid_points = np.column_stack([
        (D_mesh.ravel() - Dmin) / (Dmax - Dmin),
        (T2_mesh.ravel() - T2min) / (T2max - T2min),
    ])
    kmeans = KMeans(n_clusters=K, n_init=10, random_state=seed)
    kmeans.fit(grid_points, sample_weight=tissue_spectrum)
    labels = kmeans.labels_.reshape(nD, nT2)

    D_step = D_vals[1] - D_vals[0]
    T2_step = T2_vals[1] - T2_vals[0]
    n_grid = nD * nT2
    C_new_gauss = np.zeros((n_grid, K))
    compartment_centers, compartment_widths = [], []
    GD, GT2 = np.meshgrid(D_vals, T2_vals, indexing='ij')
    for i in range(K):
        idxD, idxT2 = np.nonzero(labels == i)
        w = F_tissue[idxD, idxT2]
        if w.sum() <= 0 or len(idxD) == 0:
            # degenerate/empty cluster: fall back to a broad default (avoids divide-by-zero)
            D_c, T2_c = Dmin + (Dmax - Dmin) / 2, T2min + (T2max - T2min) / 2
            sD, sT2 = (Dmax - Dmin) / 4, (T2max - T2min) / 4
        else:
            D_c = np.sum(D_vals[idxD] * w) / w.sum()
            T2_c = np.sum(T2_vals[idxT2] * w) / w.sum()
            sD = np.sqrt(np.sum(w * (D_vals[idxD] - D_c) ** 2) / w.sum())
            sT2 = np.sqrt(np.sum(w * (T2_vals[idxT2] - T2_c) ** 2) / w.sum())
            sD = max(sD, D_step)      # floor to at least one grid step
            sT2 = max(sT2, T2_step)
        compartment_centers.append((D_c, T2_c))
        compartment_widths.append((sD, sT2))

        blob = np.exp(-0.5 * ((GD - D_c) / sD) ** 2) * np.exp(-0.5 * ((GT2 - T2_c) / sT2) ** 2)
        C_new_gauss[:, i] = blob.ravel() / blob.sum()

    C_small_init = np.diag(s) @ V_mat.T @ C_new_gauss

    if return_diagnostics:
        return C_small_init, tissue_spectrum, labels, compartment_centers, compartment_widths, C_new_gauss
    return C_small_init



def define_initial_W_and_m(K, n_voxels, rng, alpha_init=5.0):
    W = rng.dirichlet(alpha_init * np.ones(K), size=n_voxels).T
    m = np.ones(n_voxels)
    return W, m                                              


def define_initial_lambda(C_small_init, R, K):
    new_lam = (R * K) / (np.sum(C_small_init**2) + 1e-12)   # params / sum of squared initial coefficients
    return min(new_lam, 1e3)

def run_optimisation(
    Y, U, C_small, M0, K, R, eps, n_iter, sigma2, alpha, lam, W,
    bound_eps=1e-2,
    update_alpha_flag=True,
    update_W_flag=True,
    update_M0_flag=False,
    update_C_flag=False,
    update_sigma2_flag=False,
    update_lambda_flag=False,
    verbose_every=10,
):
    losses = []
    fit_errs = []

    for t in range(n_iter):
        UC = U @ C_small

        if update_W_flag:
            W = update_W(Y, UC, sigma2, alpha, W, M0, bound_eps)

        if update_M0_flag:
            M0 = update_M0(Y, UC, W)

        if update_C_flag:
            C_small = update_C_small(Y, U, W, M0, lam, sigma2, R, K)
            UC = U @ C_small

        if update_sigma2_flag:
            sigma2 = max(update_sigma2(Y, UC, W, M0), SIGMA2_FLOOR)

        if update_lambda_flag:
            lam = update_lambda(C_small, R, K)

        if update_alpha_flag:
            alpha = update_alpha(W)

        loss, fit_err = compute_loss(Y, UC, W, M0, alpha, lam, C_small, sigma2, eps)
        losses.append(loss)
        fit_errs.append(fit_err)

        if t % verbose_every == 0:
            print(f"iter {t}: loss={loss:.4f}, fit_err={fit_err:.6f}, "
                  f"sigma2={sigma2:.6f}, lam={lam:.4f}, alpha={alpha:.4f}")

    return W, M0, C_small, alpha, sigma2, lam, losses, fit_errs

    
def estimate_C(V_mat, s, C_small):
    C_est = np.maximum(V_mat @ np.diag(1.0 / (s + 1e-12)) @ C_small, 0)
    C_est = C_est / (C_est.sum(axis=0, keepdims=True) + 1e-12)
    return C_est