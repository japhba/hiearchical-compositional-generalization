import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from network import _rescale_weight


def _kernel(X, cossim=False):
    """Dot-product or cosine similarity kernel."""
    K = X @ X.T
    if cossim:
        norms = jnp.sqrt(jnp.diag(K))
        K = K / (norms[:, None] * norms[None, :] + 1e-30)
    return K


def _imshow_centered(ax, K, title):
    """imshow with 0-centered colormap."""
    vmax = float(jnp.abs(K).max())
    im = ax.imshow(K, cmap="RdBu_r", aspect="equal", vmin=-vmax, vmax=vmax)
    ax.set(title=title, xlabel="Pattern p", ylabel="Pattern p'")
    return im


def _cross_kernel(X1, X2, cossim=False):
    """Cross-kernel between two sets of patterns."""
    K = X1 @ X2.T
    if cossim:
        n1 = jnp.sqrt(jnp.sum(X1 ** 2, axis=1))
        n2 = jnp.sqrt(jnp.sum(X2 ** 2, axis=1))
        K = K / (n1[:, None] * n2[None, :] + 1e-30)
    return K


def plot_data_kernels(X_train: np.ndarray, X_test: np.ndarray, cossim: bool = False):
    X_train, X_test = jnp.array(X_train), jnp.array(X_test)
    K_train = _kernel(X_train, cossim)
    K_test = _kernel(X_test, cossim)
    K_cross = _cross_kernel(X_train, X_test, cossim)
    label = "cossim" if cossim else "dot"
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, K, title in zip(axes, [K_train, K_test, K_cross],
                            [f"Train kernel ({label})", f"Test kernel ({label})", f"Train×Test kernel ({label})"]):
        im = _imshow_centered(ax, K, title)
        fig.colorbar(im, ax=ax, shrink=0.8)
    return fig


def plot_loss_curves(curves_shallow: dict, curves_deep: dict, depth: int):
    fig, ax = plt.subplots(figsize=(8, 4))
    s_sh, s_dp = curves_shallow["s"], curves_deep["s"]
    ax.plot(s_sh, curves_shallow["mse_train"], label="Shallow train", color="C0")
    ax.plot(s_dp, curves_deep["mse_train"], label=f"Deep (d={depth}) train", color="C1")
    if "mse_test" in curves_shallow:
        ax.plot(s_sh, curves_shallow["mse_test"], label="Shallow test", color="C0", ls="--")
    if "mse_test" in curves_deep:
        ax.plot(s_dp, curves_deep["mse_test"], label=f"Deep (d={depth}) test", color="C1", ls="--")
    ax.set(xlabel="s", ylabel="MSE", xscale="log", yscale="log")
    ax.legend()
    return fig


def _layer_hiddens(params, x, mup=False, ntp=True, activation="linear"):
    """Return list of hidden representations at each layer."""
    hiddens = []
    h = x
    for i, W in enumerate(params):
        is_last = i == len(params) - 1
        W_eff = _rescale_weight(W, downscale=ntp, mup=mup, is_last=is_last)
        h = h @ W_eff.T
        if activation != "linear" and not is_last:
            if activation == "relu":
                h = jax.nn.relu(h)
            elif activation == "tanh":
                h = jnp.tanh(h)
            else:
                raise ValueError(f"Unknown activation: {activation!r}")
        hiddens.append(h)
    return hiddens


def plot_shallow_kernels(params, X_tr, X_te, mup=False, ntp=True, cossim=False, activation="linear"):
    h_tr = _layer_hiddens(params, X_tr, mup=mup, ntp=ntp, activation=activation)[0]
    h_te = _layer_hiddens(params, X_te, mup=mup, ntp=ntp, activation=activation)[0]
    K_train = _kernel(h_tr, cossim)
    K_test = _kernel(h_te, cossim)
    K_cross = _cross_kernel(h_tr, h_te, cossim)
    label = "cossim" if cossim else "dot"
    fig, axes = plt.subplots(3, 1, figsize=(6, 16))
    for ax, K, title in zip(axes, [K_train, K_test, K_cross],
                            [f"Shallow — train ({label})", f"Shallow — test ({label})", f"Shallow — train×test ({label})"]):
        im = _imshow_centered(ax, K, title)
        fig.colorbar(im, ax=ax, shrink=0.8)
    fig.suptitle("Shallow network kernel", y=1.02)
    return fig


def plot_deep_kernels(params, X_tr, X_te, mup=False, ntp=True, cossim=False, activation="linear"):
    hs_tr = _layer_hiddens(params, X_tr, mup=mup, ntp=ntp, activation=activation)
    hs_te = _layer_hiddens(params, X_te, mup=mup, ntp=ntp, activation=activation)
    label = "cossim" if cossim else "dot"
    n_layers = len(hs_tr)
    fig, axes = plt.subplots(3, n_layers, figsize=(6 * n_layers, 14))
    if n_layers == 1:
        axes = axes[:, None]
    for col, (h_tr, h_te) in enumerate(zip(hs_tr, hs_te)):
        Ktr = _kernel(h_tr, cossim)
        Kte = _kernel(h_te, cossim)
        Kcross = _cross_kernel(h_tr, h_te, cossim)
        for row, (K, split) in enumerate(zip([Ktr, Kte, Kcross], ["train", "test", "train×test"])):
            ax = axes[row, col]
            im = _imshow_centered(ax, K, f"Layer {col + 1} — {split} ({label})")
            fig.colorbar(im, ax=ax, shrink=0.8)
    fig.suptitle("Deep network kernels per layer", y=1.02)
    return fig


def kernel_images(params, X_tr, X_te, mup=False, ntp=True, cossim=True, activation="linear"):
    """Compute kernel matrices at each layer and return as dict of wandb.Image."""
    import wandb
    hs_tr = _layer_hiddens(params, X_tr, mup=mup, ntp=ntp, activation=activation)
    hs_te = _layer_hiddens(params, X_te, mup=mup, ntp=ntp, activation=activation)
    label = "cossim" if cossim else "dot"
    images = {}
    for col, (h_tr, h_te) in enumerate(zip(hs_tr, hs_te)):
        for K, split in [(_kernel(h_tr, cossim), "train"), (_kernel(h_te, cossim), "test"),
                         (_cross_kernel(h_tr, h_te, cossim), "cross")]:
            fig, ax = plt.subplots(figsize=(4, 4))
            _imshow_centered(ax, K, f"L{col+1} {split} ({label})")
            fig.tight_layout()
            images[f"kernel/L{col+1}_{split}"] = wandb.Image(fig)
            plt.close(fig)
    return images


def plot_effective_svd(params_shallow, params_deep):
    def effective_weight(params):
        W = params[-1]
        for i in range(len(params) - 2, -1, -1):
            W = W @ params[i]
        return W

    W_shallow = effective_weight(params_shallow)
    W_deep = effective_weight(params_deep)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, W, label in zip(axes, [W_shallow, W_deep], ["Shallow", "Deep"]):
        U, S, Vt = jnp.linalg.svd(W, full_matrices=False)
        ax.bar(range(len(S)), S)
        ax.set(title=f"{label} — singular values of W_eff", xlabel="Index", ylabel="σ")
    return fig
