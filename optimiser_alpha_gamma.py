import warnings                              
import numpy as np                                    
from sklearn.linear_model import Ridge          
import dirichlet      
from scipy.special import gammaln, softmax      
from scipy.optimize import minimize, minimize_scalar
from scipy.optimize import nnls
from scipy.optimize import curve_fit
from sklearn.cluster import KMeans


warnings.filterwarnings(
    "ignore",                                        
    message="Values in x were outside bounds during a minimize step, clipping to bounds",  
    category=RuntimeWarning,                          
)

SIGMA2_FLOOR = 1e-6   # smallest sigma2 value allowed
BOUND_EPS = 1e-2      # smallest weight value allowed during optimization, keeps log(w) finite


def _measurement_vectors(b_values, TE_values, n_measurements):
    """Return per-measurement b/TE vectors, preserving arbitrary ordering.

    Inputs may be vectors of length ``n_measurements`` or grid axes. Grid axes
    are expanded using the same b-slowest, TE-fastest convention as
    ``create_A_matrix``; per-measurement vectors avoid any ordering assumption.
    """
    b_values = np.asarray(b_values, dtype=float).ravel()
    TE_values = np.asarray(TE_values, dtype=float).ravel()
    if b_values.size == n_measurements and TE_values.size == n_measurements:
        return b_values, TE_values
    if b_values.size * TE_values.size == n_measurements:
        b_grid, te_grid = np.meshgrid(b_values, TE_values, indexing="ij")
        return b_grid.ravel(), te_grid.ravel()
    raise ValueError("b_values and TE_values must be per-measurement vectors or complete grid axes")


def _mean_by_coordinate(coordinate, signals):
    """Average repeated measurements at identical acquisition coordinates."""
    values, inverse = np.unique(coordinate, return_inverse=True)
    mean_signals = np.empty((values.size, signals.shape[1]))
    for i in range(values.size):
        mean_signals[i] = signals[inverse == i].mean(axis=0)
    return values, mean_signals


def _lowest_b_data(Y, b_values, TE_values, b_tolerance=0.05):
    """Return measurements acquired at b=0, allowing a small b tolerance."""
    b, te = _measurement_vectors(b_values, TE_values, Y.shape[0])
    mask = np.isclose(b, 0.0, atol=b_tolerance, rtol=0.0)
    if not np.any(mask):
        raise ValueError(f"no b=0 measurements found within tolerance {b_tolerance}")
    return _mean_by_coordinate(te[mask], Y[mask])


def _solve_nnls(A, y):
    """Solve the nonnegative least-squares initializer robustly.

    The MRI forward matrix is deliberately ill-conditioned and typically has
    many more spectral-grid variables than measurements.  SciPy's default
    active-set budget (3 * n_variables) can therefore terminate before
    convergence.
    """
    maxiter = max(3 * A.shape[1], 20 * A.shape[1])
    return nnls(A, y, maxiter=maxiter)


def compute_loss(Y, UC, W, M0, alpha, lam, C_small, sigma2, w_floor=1e-10,
                 m0_prior=None, m0_prior_sigma=None, weight_entropy=0.0,
                 C=None,
                 c_smoothness=0.0, c_sparsity=0.0, c_diversity=0.0,
                 nD=None, nT2=None):
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
    m0_term = 0.0
    if m0_prior is not None or m0_prior_sigma is not None:
        if m0_prior is None or m0_prior_sigma is None or m0_prior_sigma <= 0:
            raise ValueError("m0_prior and a positive m0_prior_sigma must be supplied together")
        m0_term = np.sum((M0 - m0_prior) ** 2) / (2 * m0_prior_sigma**2)
    entropy_term = weight_entropy * (-np.sum(W * np.log(np.maximum(W, w_floor))))
    if (c_smoothness or c_sparsity or c_diversity) and C is None:
        raise ValueError("C must be supplied when full-grid C priors are active")
    c_term = 0.0
    if C is not None:
        c_term, _ = _c_grid_regularization(
            C, c_smoothness, c_sparsity, c_diversity, nD, nT2,
        )
    return data_term + reg_term + dir_term + m0_term + entropy_term + c_term, fit_err

def update_sigma2(Y, UC, W, M0):
    resid = Y - (UC @ W) * M0                          # same  definition as in compute_loss
    n_meas, n_voxels = Y.shape                         
    return np.sum(resid**2) / (n_meas * n_voxels)        # closed-form MLE mean squared residual over all entries


def update_lambda(C_small, R, K):
    new_lam = (R * K) / (np.sum(C_small**2) + 1e-12)     #  precision ~ (#params) / (sum of squares)
    return min(new_lam, 1e3)                              # cap lambda at 1e3 to avoid over-shrinking C_small to zero


def update_alpha(W, w_floor=1e-12):
    """Maximum-likelihood fit of a *symmetric* Dirichlet concentration.

    The previous implementation fitted a general Dirichlet and averaged its
    component-specific concentrations. That is not the MLE for the symmetric
    prior used by the model.
    """
    W = np.maximum(np.asarray(W, dtype=float), w_floor)
    W = W / W.sum(axis=0, keepdims=True)
    K, n_voxels = W.shape
    log_weights = np.sum(np.log(W))

    def negative_log_likelihood(log_alpha):
        alpha = np.exp(log_alpha)
        log_likelihood = (
            n_voxels * (gammaln(K * alpha) - K * gammaln(alpha))
            + (alpha - 1.0) * log_weights
        )
        return -log_likelihood

    result = minimize_scalar(
        negative_log_likelihood,
        bounds=(np.log(1e-3), np.log(1e3)),
        method="bounded",
    )
    return float(np.exp(result.x))
                           

def _bounded_simplex_from_logits(logits, bound_eps):
    """Map logits to simplex weights with a differentiable positive floor."""
    K = np.asarray(logits).size
    if not 0.0 <= bound_eps < 1.0 / K:
        raise ValueError("bound_eps must be non-negative and smaller than 1 / K")
    q = softmax(logits)
    return bound_eps + (1.0 - K * bound_eps) * q, q


def _W_objective_and_gradient(logits, y, UC, m0, sigma2, alpha, bound_eps,
                              weight_entropy=0.0):
    """Single-voxel conditional MAP objective and analytic logits gradient.

    ``weight_entropy`` is a sparsity-favouring prior on the simplex. A
    Dirichlet contribution is retained for backwards compatibility and can
    be disabled by setting ``alpha=1``.
    """
    w, q = _bounded_simplex_from_logits(logits, bound_eps)
    pred = m0 * (UC @ w)
    resid = y - pred
    data_term = np.sum(resid**2) / (2 * sigma2)
    K = w.shape[0]
    dir_term = -(alpha - 1) * np.sum(np.log(w)) + K * gammaln(alpha) - gammaln(K * alpha)
    grad_w = -m0 * (UC.T @ resid) / sigma2 - (alpha - 1.0) / w
    entropy = -np.sum(w * np.log(w + 1e-12))
    value = data_term + dir_term + weight_entropy * entropy
    grad_w += weight_entropy * (-(np.log(w + 1e-12) + 1.0))
    scale = 1.0 - K * bound_eps
    grad_z = scale * q * (grad_w - np.dot(grad_w, q))
    return value, grad_z


def update_W(Y, UC, sigma2, alpha, W, m, bound_eps=1e-2,
             return_diagnostics=False, weight_entropy=0.0):
    """Update simplex weights with an optional entropy MAP prior."""
    K = W.shape[0]
    n_voxels = W.shape[1]
    W_new = np.zeros_like(W)
    diagnostics = []

    for v in range(n_voxels):
        yv = Y[:, v]

        # Convert a valid bounded-simplex point to softmax coordinates.
        w0 = np.clip(W[:, v], bound_eps, 1.0)
        q0 = np.maximum((w0 - bound_eps) / (1.0 - K * bound_eps), 1e-12)
        q0 = q0 / q0.sum()
        z0 = np.log(q0)

        res = minimize(
            _W_objective_and_gradient, z0, jac=True, method="L-BFGS-B",
            args=(yv, UC, m[v], sigma2, alpha, bound_eps, weight_entropy),
        )
        w_new, _ = _bounded_simplex_from_logits(res.x, bound_eps)
        W_new[:, v] = w_new
        diagnostics.append({
            "success": bool(res.success), "status": int(res.status),
            "message": str(res.message), "fun": float(res.fun),
            "nit": int(res.nit), "grad_norm": float(np.linalg.norm(res.jac)),
        })

    if return_diagnostics:
        return W_new, diagnostics
    return W_new                                     


def initialize_W_from_C(Y, U, C_small, M0_init, bound_eps=1e-3):
    """Initialize voxel weights by simplex-constrained fitting with fixed C/M0.

    This solves, independently for each voxel,
    ``min_w ||Y_v - M0_v U C_small w||²`` subject to a positive simplex
    constraint.  ``alpha=1`` removes the Dirichlet term, and ``sigma2=1`` is
    immaterial because it only rescales this data-only objective.
    """
    C_small = np.asarray(C_small, dtype=float)
    K = C_small.shape[1]
    n_voxels = Y.shape[1]
    if np.asarray(M0_init).shape != (n_voxels,):
        raise ValueError("M0_init must have one value per voxel")
    W_uniform = np.full((K, n_voxels), 1.0 / K)
    return update_W(
        Y, U @ C_small, sigma2=1.0, alpha=1.0, W=W_uniform,
        m=M0_init, bound_eps=bound_eps,
    )


def update_M0(Y, UC, W, prior_mean=None, prior_sigma=None, sigma2=1.0):
    """Closed-form positive M0 MAP update under the model noise variance."""
    n_voxels = W.shape[1]                                          # number of voxels
    M0_new = np.zeros(n_voxels)                                     # allocate output array for updated per-voxel scales
    pred_unscaled = UC @ W                                          # unscaled prediction (before applying M0) for all voxels
    if prior_mean is not None or prior_sigma is not None:
        if prior_mean is None or prior_sigma is None or prior_sigma <= 0:
            raise ValueError("prior_mean and a positive prior_sigma must be supplied together")
        prior_mean = np.broadcast_to(np.asarray(prior_mean, dtype=float), (n_voxels,))
        if sigma2 <= 0:
            raise ValueError("sigma2 must be positive")
        # After multiplying the MAP normal equations by sigma2, the prior
        # precision is sigma2 / prior_sigma**2.
        prior_precision = sigma2 / prior_sigma**2
    else:
        prior_precision = 0.0
    for v in range(n_voxels):                                     
        p = pred_unscaled[:, v]                                    
        num = np.dot(p, Y[:, v])                                  
        denom = np.dot(p, p) + prior_precision + 1e-12
        if prior_precision:
            num += prior_precision * prior_mean[v]
        M0_new[v] = max(num / denom, 0.0)                             # positive scale parameter
    return M0_new                                                 


def initialize_M0_from_b0_T2(Y, b_values, TE_values, T2_vals, b_tolerance=0.05):
    """Estimate voxelwise M0 by NNLS fitting the b=0 multi-TE signal.

    For normalized spectra, the sum of fitted nonnegative T2 amplitudes
    estimates the voxel scale M0. Rows near b=0 are selected by their b
    values, not row position.
    """
    TE_b0, b0 = _lowest_b_data(Y, b_values, TE_values, b_tolerance)
    A_T2 = np.exp(-np.outer(TE_b0, 1.0 / T2_vals))
    M0 = np.empty(Y.shape[1])
    for voxel in range(Y.shape[1]):
        amplitudes, _ = nnls(A_T2, b0[:, voxel])
        M0[voxel] = amplitudes.sum()
    return M0


def initialize_M0_monoexponential(Y, b_values, TE_values, b_tolerance=0.05):
    """Estimate M0 from a mono-exponential fit to near-b=0 multi-TE data."""
    TE_b0, b0 = _lowest_b_data(Y, b_values, TE_values, b_tolerance)

    def model(te, m0, t2):
        return m0 * np.exp(-te / t2)

    M0 = np.empty(Y.shape[1])
    for voxel in range(Y.shape[1]):
        signal = np.maximum(b0[:, voxel], 1e-12)
        initial = [max(signal.max(), 1e-12), max(np.median(TE_b0), 1e-12)]
        fitted, _ = curve_fit(
            model,
            TE_b0,
            signal,
            p0=initial,
            bounds=([0.0, 1e-6], [np.inf, np.inf]),
            maxfev=2000,
        )
        M0[voxel] = fitted[0]
    return M0


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


def _softmax_columns(logits):
    """Map unconstrained grid logits to normalized nonnegative spectra."""
    return softmax(logits, axis=0)


def reduced_forward_matrix(U, s, V_mat):
    """Return U @ diag(s) @ V.T without forming an explicit diagonal matrix."""
    return (U * s[np.newaxis, :]) @ V_mat.T


def _grid_smoothness(C, nD, nT2):
    """Return first-difference grid penalty and its gradient for C."""
    grid = C.reshape(nD, nT2, C.shape[1])
    d_diff = grid[1:, :, :] - grid[:-1, :, :]
    t2_diff = grid[:, 1:, :] - grid[:, :-1, :]
    penalty = 0.5 * (np.sum(d_diff**2) + np.sum(t2_diff**2))
    gradient = np.zeros_like(grid)
    gradient[:-1, :, :] -= d_diff
    gradient[1:, :, :] += d_diff
    gradient[:, :-1, :] -= t2_diff
    gradient[:, 1:, :] += t2_diff
    return penalty, gradient.reshape(C.shape)


def _component_overlap(C):
    """Return squared pairwise-overlap penalty and its gradient."""
    overlap = C.T @ C
    np.fill_diagonal(overlap, 0.0)
    penalty = 0.5 * np.sum(overlap**2)
    gradient = 2.0 * C @ overlap
    return penalty, gradient


def _c_grid_regularization(C, smoothness=0.0, sparsity=0.0, diversity=0.0,
                           nD=None, nT2=None):
    """Evaluate full-grid C priors and their gradient in one shared place."""
    if smoothness and (nD is None or nT2 is None or nD * nT2 != C.shape[0]):
        raise ValueError("smoothness requires nD and nT2 matching the spectral grid")
    value = 0.0
    gradient = np.zeros_like(C)
    if smoothness:
        penalty, penalty_gradient = _grid_smoothness(C, nD, nT2)
        value += smoothness * penalty
        gradient += smoothness * penalty_gradient
    if sparsity:
        entropy = -np.sum(C * np.log(C + 1e-12))
        value += sparsity * entropy
        gradient += sparsity * (-(np.log(C + 1e-12) + 1.0))
    if diversity:
        penalty, penalty_gradient = _component_overlap(C)
        value += diversity * penalty
        gradient += diversity * penalty_gradient
    return value, gradient


def _constrained_C_objective_and_gradient(
    logits, Y, U, s, V_mat, W, M0, sigma2, lam=0.0,
    smoothness=0.0, sparsity=0.0, diversity=0.0, nD=None, nT2=None,
):
    """Objective/analytic gradient for full-grid C logits.

    Kept separate from the optimiser so tests can compare this derivative to
    finite differences.  This guards the joint C/W/M0 recovery against subtle
    calculus or array-orientation errors.
    """
    n_grid, K = V_mat.shape[0], W.shape[0]
    B = reduced_forward_matrix(U, s, V_mat)
    Q = s[:, np.newaxis] * V_mat.T
    C_current = _softmax_columns(np.asarray(logits).reshape(n_grid, K))
    prediction = (B @ C_current @ W) * M0
    residual = prediction - Y
    value = np.sum(residual**2) / (2 * sigma2)
    grad_C = B.T @ (residual * M0[np.newaxis, :]) @ W.T / sigma2
    if lam:
        C_small = Q @ C_current
        value += lam * np.sum(C_small**2) / 2
        grad_C += lam * (Q.T @ C_small)
    c_prior_value, c_prior_gradient = _c_grid_regularization(
        C_current, smoothness, sparsity, diversity, nD, nT2,
    )
    value += c_prior_value
    grad_C += c_prior_gradient
    grad_logits = C_current * (grad_C - np.sum(grad_C * C_current, axis=0, keepdims=True))
    return value, grad_logits.ravel()


def update_C_constrained(Y, U, s, V_mat, W, M0, C, sigma2, lam=0.0,
                         maxiter=100, smoothness=0.0, sparsity=0.0, diversity=0.0,
                         nD=None, nT2=None, return_diagnostics=False):
    """Update full-grid canonical spectra under simplex constraints.

    ``C`` has one normalized, nonnegative spectrum per component. The
    optimization operates on unconstrained logits but evaluates the exact
    reduced forward model U @ diag(s) @ V.T @ C.
    """
    n_grid, K = C.shape
    if smoothness and (nD is None or nT2 is None or nD * nT2 != n_grid):
        raise ValueError("smoothness requires nD and nT2 matching the spectral grid")
    C0 = np.clip(C, 1e-12, None)
    logits0 = np.log(C0).ravel()
    result = minimize(
        _constrained_C_objective_and_gradient,
        logits0,
        jac=True,
        method="L-BFGS-B",
        args=(Y, U, s, V_mat, W, M0, sigma2, lam, smoothness, sparsity,
              diversity, nD, nT2),
        options={"maxiter": maxiter},
    )
    C_new = _softmax_columns(result.x.reshape(n_grid, K))
    if return_diagnostics:
        return C_new, {
            "success": bool(result.success), "status": int(result.status),
            "message": str(result.message), "fun": float(result.fun),
            "nit": int(result.nit), "grad_norm": float(np.linalg.norm(result.jac)),
        }
    return C_new


def run_constrained_optimisation(
    Y, U, s, V_mat, C, M0, K, n_iter, sigma2, alpha, lam, W,
    bound_eps=1e-2, update_alpha_flag=True, update_W_flag=True,
    update_M0_flag=False, update_C_flag=True, update_sigma2_flag=False,
    update_lambda_flag=False, m0_prior=None, m0_prior_sigma=None,
    alpha_update_start=0, c_maxiter=100, verbose_every=10,
    c_smoothness=0.0, c_sparsity=0.0, c_diversity=0.0, nD=None, nT2=None,
    weight_entropy=0.0,
    return_diagnostics=False,
):
    """Alternating MAP optimization with canonical spectra constrained to simplices."""
    losses, fit_errs = [], []
    diagnostics = {"W": [], "C": []}
    for t in range(n_iter):
        C_small = (s[:, np.newaxis] * V_mat.T) @ C
        UC = U @ C_small
        if update_W_flag:
            if return_diagnostics:
                W, w_diagnostics = update_W(
                    Y, UC, sigma2, alpha, W, M0, bound_eps,
                    weight_entropy=weight_entropy,
                    return_diagnostics=True,
                )
                diagnostics["W"].append(w_diagnostics)
            else:
                W = update_W(
                    Y, UC, sigma2, alpha, W, M0, bound_eps,
                    weight_entropy=weight_entropy,
                )
        if update_M0_flag:
            M0 = update_M0(
                Y, UC, W, prior_mean=m0_prior, prior_sigma=m0_prior_sigma,
                sigma2=sigma2,
            )
        if update_C_flag:
            c_kwargs = dict(
                smoothness=c_smoothness, sparsity=c_sparsity, diversity=c_diversity,
                nD=nD, nT2=nT2, return_diagnostics=return_diagnostics,
            )
            c_result = update_C_constrained(
                Y, U, s, V_mat, W, M0, C, sigma2, lam, c_maxiter, **c_kwargs
            )
            if return_diagnostics:
                C, c_diagnostics = c_result
                diagnostics["C"].append(c_diagnostics)
            else:
                C = c_result
            C_small = (s[:, np.newaxis] * V_mat.T) @ C
            UC = U @ C_small
        if update_sigma2_flag:
            sigma2 = max(update_sigma2(Y, UC, W, M0), SIGMA2_FLOOR)
        if update_lambda_flag:
            lam = update_lambda(C_small, C_small.shape[0], K)
        if update_alpha_flag and t >= alpha_update_start:
            alpha = update_alpha(W)
        loss, fit_err = compute_loss(
            Y, UC, W, M0, alpha, lam, C_small, sigma2,
            m0_prior=m0_prior, m0_prior_sigma=m0_prior_sigma,
            weight_entropy=weight_entropy,
            C=C, c_smoothness=c_smoothness, c_sparsity=c_sparsity,
            c_diversity=c_diversity, nD=nD, nT2=nT2,
        )
        losses.append(loss)
        fit_errs.append(fit_err)
        if t % verbose_every == 0:
            print(f"iter {t}: loss={loss:.4f}, fit_err={fit_err:.6f}, sigma2={sigma2:.6f}")
    result = W, M0, C, alpha, sigma2, lam, losses, fit_errs
    return (*result, diagnostics) if return_diagnostics else result


def define_initial_C_small(Y, U, W, M0, K):
    Y_proj = U.T @ Y                          # (R, V) -  project raw Y, no division
    X = (W * M0[np.newaxis, :]).T             # (V, K) — add M0 into the  matrix
    C_small_init = np.linalg.lstsq(X, Y_proj.T, rcond=None)[0]   # (K, R)
    return C_small_init.T                     # (R, K)


def define_initial_C_NNLS(Y, A, voxel_idx, K, D_vals, T2_vals,
                          Dmin, Dmax, T2min, T2max, nD, nT2, seed=0,
                          return_diagnostics=False, min_fraction=0.0):
    """Build full-grid, physical canonical-spectrum initial values from NNLS.

    Voxelwise NNLS spectra are averaged, partitioned with weighted k-means,
    and represented by one normalized Gaussian D--T2 hotspot per component.
    This function intentionally has no SVD inputs: it returns ``C_init`` in
    the physical spectral grid, with shape ``(nD * nT2, K)``.

    Grid cells below ``min_fraction`` of the mean spectrum's maximum are set
    to zero before clustering (0 keeps everything).
    """
    spectra = []
    for v in voxel_idx:
        f, _ = _solve_nnls(A, Y[:, v])
        spectra.append(f / f.sum() if f.sum() > 0 else f)
    tissue_spectrum = np.mean(spectra, axis=0)
    tissue_spectrum /= tissue_spectrum.sum()
    if min_fraction > 0:
        tissue_spectrum[tissue_spectrum < min_fraction * tissue_spectrum.max()] = 0
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

    if return_diagnostics:
        return C_new_gauss, tissue_spectrum, labels, compartment_centers, compartment_widths
    return C_new_gauss


def define_initial_C_NNLS_mean(Y, A, voxel_idx, K, D_vals, T2_vals,
                          Dmin, Dmax, T2min, T2max, nD, nT2, seed=0,
                          return_diagnostics=False, min_fraction=0.0):

    # Average the signals over voxels first, then solve one NNLS on the average
    y_mean = Y[:, list(voxel_idx)].mean(axis=1)
    tissue_spectrum, _ = _solve_nnls(A, y_mean)
    tissue_spectrum /= tissue_spectrum.sum()
    if min_fraction > 0:
        # cells below min_fraction of the maximum are set to zero before clustering
        tissue_spectrum[tissue_spectrum < min_fraction * tissue_spectrum.max()] = 0
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

    if return_diagnostics:
        return C_new_gauss, tissue_spectrum, labels, compartment_centers, compartment_widths
    return C_new_gauss



def project_C_to_C_small(C, s, V_mat):
    """Project physical canonical spectra into retained SVD coordinates."""
    C = np.asarray(C, dtype=float)
    s = np.asarray(s, dtype=float)
    V_mat = np.asarray(V_mat, dtype=float)
    if V_mat.ndim != 2 or C.ndim != 2 or V_mat.shape[0] != C.shape[0]:
        raise ValueError("V_mat and C must be 2D with matching spectral-grid rows")
    if V_mat.shape[1] != s.size:
        raise ValueError("s must have one value per retained V_mat column")
    return (s[:, np.newaxis] * V_mat.T) @ C


def define_initial_C_small_NNLS(Y, A, s, V_mat, voxel_idx, K, D_vals, T2_vals,
                                Dmin, Dmax, T2min, T2max, nD, nT2, seed=0,
                                return_diagnostics=False):
    """Compatibility wrapper returning the SVD projection of NNLS-based C.

    Prefer :func:`define_initial_C_NNLS` followed explicitly by
    :func:`project_C_to_C_small` in new code.
    """
    result = define_initial_C_NNLS(
        Y, A, voxel_idx, K, D_vals, T2_vals, Dmin, Dmax, T2min, T2max,
        nD, nT2, seed=seed, return_diagnostics=return_diagnostics,
    )
    if return_diagnostics:
        C_init, tissue_spectrum, labels, centres, widths = result
        return (project_C_to_C_small(C_init, s, V_mat), tissue_spectrum,
                labels, centres, widths, C_init)
    return project_C_to_C_small(result, s, V_mat)



def _data_driven_marginals(y, b_vals, TE_vals, D_vals, T2_vals, nB=None, nTE=None,
                            n_avg=None, b_tolerance=1e-12, te_tolerance=1e-12):
    """Estimate 1D marginals from lowest-b and lowest-TE measurements by value."""
    y = np.asarray(y, dtype=float)
    b, te = _measurement_vectors(b_vals, TE_vals, y.size)
    b_mask = np.isclose(b, b.min(), atol=b_tolerance, rtol=0.0)
    te_b0, y_b0 = _mean_by_coordinate(te[b_mask], y[b_mask, None])

    te_mask = np.isclose(te, te.min(), atol=te_tolerance, rtol=0.0)
    b_te_min, y_te_min = _mean_by_coordinate(b[te_mask], y[te_mask, None])

    A_T2 = np.exp(-np.outer(te_b0, 1.0 / T2_vals))
    m_T2, _ = _solve_nnls(A_T2, y_b0[:, 0])
    m_T2 = m_T2 / m_T2.sum() if m_T2.sum() > 0 else m_T2

    A_D = np.exp(-np.outer(b_te_min, D_vals))
    m_D, _ = _solve_nnls(A_D, y_te_min[:, 0])
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
    f_start, _ = _solve_nnls(A, S)

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
    m0_prior=None,
    m0_prior_sigma=None,
    alpha_update_start=0,
    weight_entropy=0.0,
    verbose_every=10,
):
    losses = []
    fit_errs = []

    for t in range(n_iter):
        UC = U @ C_small

        if update_W_flag:
            W = update_W(
                Y, UC, sigma2, alpha, W, M0, bound_eps,
                weight_entropy=weight_entropy,
            )

        if update_M0_flag:
            M0 = update_M0(
                Y, UC, W, prior_mean=m0_prior, prior_sigma=m0_prior_sigma,
                sigma2=sigma2,
            )

        if update_C_flag:
            C_small = update_C_small(Y, U, W, M0, lam, sigma2, R, K)
            UC = U @ C_small

        if update_sigma2_flag:
            sigma2 = max(update_sigma2(Y, UC, W, M0), SIGMA2_FLOOR)

        if update_lambda_flag:
            lam = update_lambda(C_small, R, K)

        if update_alpha_flag and t >= alpha_update_start:
            alpha = update_alpha(W)

        loss, fit_err = compute_loss(
            Y, UC, W, M0, alpha, lam, C_small, sigma2, eps,
            m0_prior=m0_prior, m0_prior_sigma=m0_prior_sigma,
            weight_entropy=weight_entropy,
        )
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
