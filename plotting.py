import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linear_sum_assignment


def matched_errors(C_small_true, C_small_est, W_true, W_est):
    K = C_small_true.shape[1]
    C_small_m, W_m = match_components(C_small_true, C_small_est, W_true, W_est, K)
    W_mae = np.mean(np.abs(W_m - W_true))
    C_mae = np.mean(np.abs(C_small_m - C_small_true))
    return W_mae, C_mae


def match_components(C_small_true, C_small_est, W_true, W_est, K):
    cost = np.zeros((K, K))
    for i in range(K):
        for j in range(K):
            cost[i, j] = np.sum((C_small_true[:, i] - C_small_est[:, j]) ** 2)
    true_idx, est_idx = linear_sum_assignment(cost)
    order = est_idx[np.argsort(true_idx)]
    return C_small_est[:, order], W_est[order, :]


def plot_compartment_comparison_nomatch(C_true, C_est, K, nD, nT2, D_vals, T2_vals):
    fig, axes = plt.subplots(2, K, figsize=(4 * K, 8))
    for k in range(K):
        for row, C, label in zip(
            [0, 1],
            [C_true, C_est],
            ["True", "Estimated"],
        ):
            im = axes[row, k].imshow(
                C[:, k].reshape(nD, nT2),
                origin="lower",
                aspect="auto",
                extent=[T2_vals.min(), T2_vals.max(), D_vals.min(), D_vals.max()],
            )
            axes[row, k].set_title(f"{label} compartment {k+1}")
            axes[row, k].set_xlabel("T2")
            axes[row, k].set_ylabel("D")
            plt.colorbar(im, ax=axes[row, k])

    plt.tight_layout()
    plt.show()



def plot_compartment_comparison(C_true, C_est, K, nD, nT2, D_vals, T2_vals):
    true_order = np.argsort([np.argmax(C_true[:, k]) for k in range(K)])
    est_order = np.argsort([np.argmax(C_est[:, k]) for k in range(K)])

    C_true_matched = C_true[:, true_order]
    C_est_matched = C_est[:, est_order]

    fig, axes = plt.subplots(2, K, figsize=(4 * K, 8))
    for k in range(K):
        for row, C, label in zip(
            [0, 1],
            [C_true_matched, C_est_matched],
            ["True", "Estimated"],
        ):
            im = axes[row, k].imshow(
                C[:, k].reshape(nD, nT2),
                origin="lower",
                aspect="auto",
                extent=[T2_vals.min(), T2_vals.max(), D_vals.min(), D_vals.max()],
            )
            axes[row, k].set_title(f"{label} compartment {k+1}")
            axes[row, k].set_xlabel("T2")
            axes[row, k].set_ylabel("D")
            plt.colorbar(im, ax=axes[row, k])

    plt.tight_layout()
    plt.show()


def plot_voxel_grid_comparison(
    C_a, W_a, C_b, W_b, K, n_voxels, nD, nT2,
    n_show_per_block=20, label_a="True", label_b="Est",
):
    assert C_a.shape == C_b.shape == (nD * nT2, K), \
        f"C_a and C_b must be full-grid ({nD * nT2}, {K}), got {C_a.shape} and {C_b.shape}"
    assert W_a.shape == W_b.shape and W_a.shape[0] == K, \
        f"W_a and W_b must both be ({K}, n_voxels), got {W_a.shape} and {W_b.shape}"

    n_voxels = W_a.shape[1]                                  
    block_size = n_voxels // K
    n_show_per_block = min(n_show_per_block, block_size)     # can't show more voxels than we have
    voxel_idx = []
    for k in range(K):
        start_k = k * block_size
        end_k = (k + 1) * block_size if k < K - 1 else n_voxels
        voxel_idx.extend(np.linspace(start_k, end_k - 1, n_show_per_block, dtype=int))

    ncols = n_show_per_block
    nrows = K

    F_a = C_a @ W_a
    F_b = C_b @ W_b

    F_a_plot = F_a / (F_a.max(axis=0, keepdims=True) + 1e-12)
    F_b_plot = F_b / (F_b.max(axis=0, keepdims=True) + 1e-12)

    cell = 1.4
    fig, axes = plt.subplots(
        2 * nrows, ncols,
        figsize=(cell * ncols, cell * 2 * nrows)
    )
    axes = np.array(axes).reshape(2 * nrows, ncols)

    for i, v in enumerate(voxel_idx):
        r = i // ncols
        c = i % ncols

        a_map = F_a_plot[:, v].reshape(nD, nT2)
        b_map = F_b_plot[:, v].reshape(nD, nT2)

        a_map = a_map / (a_map.max() + 1e-12)
        b_map = b_map / (b_map.max() + 1e-12)

        ax_a = axes[2 * r, c]
        im = ax_a.imshow(a_map, origin="lower", aspect="equal",
                          vmin=0, vmax=1, cmap="viridis")
        ax_a.set_title(f"Voxel {v} - {label_a}", fontsize=7, pad=2)
        ax_a.axis("off")

        ax_b = axes[2 * r + 1, c]
        ax_b.imshow(b_map, origin="lower", aspect="equal",
                    vmin=0, vmax=1, cmap="viridis")
        ax_b.set_title(f"Voxel {v} - {label_b}", fontsize=7, pad=2)
        ax_b.axis("off")

    fig.text(0.5, 0.01, "T2", ha="center", va="bottom", fontsize=12)
    fig.text(0.01, 0.5, "D", ha="left", va="center", rotation="vertical", fontsize=12)

    plt.tight_layout(rect=[0.03, 0.03, 1, 0.96])
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.6)
    plt.show()

    return F_a_plot, F_b_plot


def plot_loss_curve(losses):
    plt.figure(figsize=(8, 6))
    plt.plot(losses)
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("Convergence of the Optimisation")
    plt.grid()
    plt.show()


def plot_init_diagnostics(
    tissue_spectrum, labels, compartment_centers, K, nD, nT2, D_vals, T2_vals,
    D_means=None, T2_means=None, compartment_widths=None, C_new_gauss=None,
    method_label="",
):
    """
    Diagnostic plots for a clustering-based C_small initializer (e.g.
    define_initial_C_small_NNLS / define_initial_C_small_MADCO), built from
    that function's return_diagnostics=True outputs. Shows: the averaged
    tissue spectrum, the K-means regions, the detected vs. true compartment
    centres, and (if compartment_widths/C_new_gauss are given) the fitted
    Gaussian compartments themselves.
    """
    suffix = f" ({method_label})" if method_label else ""

    # 1. averaged tissue spectrum
    plt.imshow(tissue_spectrum.reshape(nD, nT2).T, origin="lower", aspect="auto",
               extent=[D_vals[0], D_vals[-1], T2_vals[0], T2_vals[-1]])
    if D_means is not None and T2_means is not None:
        for d, t2 in zip(D_means, T2_means):
            plt.plot(d, t2, "r+", markersize=15, markeredgewidth=2)
    plt.colorbar()
    plt.title(f"Averaged tissue spectrum{suffix}")
    plt.show()

    # 2. K-means cluster regions
    plt.imshow(labels.T, origin="lower", aspect="auto",
               extent=[D_vals[0], D_vals[-1], T2_vals[0], T2_vals[-1]])
    plt.title(f"K-means regions (K={K}){suffix}")
    plt.show()

    # 3. spectrum with detected centres (red) vs true means (white)
    im = plt.imshow(tissue_spectrum.reshape(nD, nT2).T, origin="lower", aspect="auto",
                     extent=[D_vals[0], D_vals[-1], T2_vals[0], T2_vals[-1]])
    for d, t2 in compartment_centers:
        plt.plot(d, t2, "r+", markersize=15, markeredgewidth=2)
    if D_means is not None and T2_means is not None:
        for d, t2 in zip(D_means, T2_means):
            plt.plot(d, t2, "w+", markersize=15, markeredgewidth=2)
    plt.colorbar(im)
    plt.title(f"Detected centres (red) vs true (white){suffix}")
    plt.show()

    # 4. fitted Gaussian compartments themselves, if provided
    if compartment_widths is not None and C_new_gauss is not None:
        fig, axes = plt.subplots(1, K, figsize=(4 * K, 4))
        for i, ax in enumerate(axes):
            ax.imshow(C_new_gauss[:, i].reshape(nD, nT2).T, origin="lower", aspect="auto",
                      extent=[D_vals[0], D_vals[-1], T2_vals[0], T2_vals[-1]])
            ax.plot(*compartment_centers[i], "r+", markersize=15, markeredgewidth=2)
            sD_i, sT2_i = compartment_widths[i]
            ax.set_title(f"Compartment {i+1}: sD={sD_i:.3f}, sT2={sT2_i:.1f}")
        plt.tight_layout()
        plt.show()