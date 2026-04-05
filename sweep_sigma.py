"""Sweep initialization variance (sigma) for hierarchy level resolution.

For each activation and sigma, trains via train.train() (with wandb) and measures
per-layer kernel alignment with the input kernel's eigenvectors — tracking how
each hierarchical component is preserved or discarded across depth.

Results (trained params + curves) are cached to disk so plotting changes
don't require retraining.
"""

import hashlib
import os
import pickle
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm

from task import generate_brownian_tree
from network import init_deep
from train import train
from plotting import _layer_hiddens, _kernel

# ---------- Config ----------
BRANCHING = (2, 2, 3, 4)  # depth-4 hierarchy, 48 leaves
D, N, DEPTH = 256, 256, 6
KAPPA0, DS, S_MAX = 1e-2, 3e-3, 10000.0
MUP, NTP = True, True
ACTIVATIONS = ["relu", "tanh"]
SHARED_DEPTH = 1
SIGMAS = np.geomspace(0.1, 10, 5)
N_EIGVECS = 12  # how many eigenvectors to show

# ---------- Cache ----------
CACHE_DIR = Path(os.environ.get("FAST_CACHE_DIR", "/var/tmp")) / "sweep_sigma_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _cache_key(phi, sigma):
    """Deterministic cache key from all training-relevant config."""
    cfg = (BRANCHING, D, N, DEPTH, KAPPA0, DS, S_MAX, MUP, NTP, SHARED_DEPTH, phi, f"{sigma:.6f}")
    h = hashlib.sha256(str(cfg).encode()).hexdigest()[:12]
    return f"{phi}_sigma{sigma:.4f}_{h}"


def _save_result(phi, sigma, params, curves, scores):
    key = _cache_key(phi, sigma)
    path = CACHE_DIR / f"{key}.pkl"
    data = {
        "params": [np.array(w) for w in params],
        "curves": {k: np.array(v) if hasattr(v, '__len__') else v for k, v in curves.items()},
        "scores": scores,
    }
    with open(path, "wb") as f:
        pickle.dump(data, f)


def _load_result(phi, sigma):
    key = _cache_key(phi, sigma)
    path = CACHE_DIR / f"{key}.pkl"
    if not path.exists():
        return None
    with open(path, "rb") as f:
        data = pickle.load(f)
    data["params"] = [jnp.array(w) for w in data["params"]]
    return data


# ---------- Data ----------
X_train, Y_train, paths_train = generate_brownian_tree(branching=BRANCHING, dim=D, seed_prefix=(0,), shared_depth=SHARED_DEPTH)
X_test, Y_test, paths_test = generate_brownian_tree(branching=BRANCHING, dim=D, seed_prefix=(1,), shared_depth=SHARED_DEPTH)
X_tr, Y_tr = jnp.array(X_train), jnp.array(Y_train)
X_te, Y_te = jnp.array(X_test), jnp.array(Y_test)

# ---------- Input kernel eigenbasis ----------
K_input = np.array(_kernel(X_tr, cossim=True))
eigenvalues, eigenvectors = np.linalg.eigh(K_input)
idx = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]


def eigvec_alignment(params, x, phi):
    """(n_layers, n_eigvecs) alignment: a_i(l) = v_i^T K_l v_i."""
    hiddens = _layer_hiddens(params, x, mup=MUP, ntp=NTP, phi=phi)
    scores = np.zeros((len(hiddens), N_EIGVECS))
    for li, h in enumerate(hiddens):
        K = np.array(_kernel(h, cossim=True))
        for i in range(N_EIGVECS):
            v = eigenvectors[:, i]
            scores[li, i] = v @ K @ v
    return scores


# ---------- Sweep (with caching) ----------
results = {}  # (phi, sigma) -> {"scores": ..., "curves": ..., "params": ...}

for phi in ACTIVATIONS:
    for sigma in tqdm(SIGMAS, desc=f"Sweep {phi}"):
        cached = _load_result(phi, sigma)
        if cached is not None:
            results[(phi, sigma)] = cached
            print(f"{phi} sigma={sigma:.3f} [cached]")
            continue

        params = init_deep(jax.random.key(42), D, BRANCHING[0], width=N, depth=DEPTH, sigma=sigma)
        params_trained, curves = train(
            params, X_tr, Y_tr, kappa0=KAPPA0, ds=DS, s_max=S_MAX, sigma=sigma,
            x_test=X_te, y_test=Y_te, mup=MUP, ntp=NTP, phi=phi, cossim=True,
            eigvec_basis=eigenvectors[:, :N_EIGVECS],
            run_name=f"sweep-{phi}-sigma={sigma:.3f}",
            log_every=100,
            config={"sweep": "sigma", "phi": phi, "branching": BRANCHING, "D": D, "N": N,
                    "depth": DEPTH, "ds": DS},
        )
        scores = eigvec_alignment(params_trained, X_tr, phi)
        results[(phi, sigma)] = {"params": params_trained, "scores": scores, "curves": curves}
        _save_result(phi, sigma, params_trained, curves, scores)
        print(f"{phi} sigma={sigma:.3f} eigvec alignment:\n{scores}")

# ---------- Plot: Eigenvector alignment heatmaps ----------
fig, axes = plt.subplots(len(ACTIVATIONS), len(SIGMAS), figsize=(3 * len(SIGMAS), 4 * len(ACTIVATIONS)), squeeze=False)
for row, phi in enumerate(ACTIVATIONS):
    for col, sigma in enumerate(SIGMAS):
        ax = axes[row, col]
        scores = results[(phi, sigma)]["scores"]
        im = ax.imshow(scores.T, aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1, interpolation="nearest")
        ax.set_xticks(range(DEPTH))
        ax.set_xticklabels(range(1, DEPTH + 1))
        ax.set_yticks(range(0, N_EIGVECS, 2))
        if row == len(ACTIVATIONS) - 1:
            ax.set_xlabel("Layer")
        if col == 0:
            ax.set_ylabel(f"{phi}\nEigvec index")
        ax.set_title(f"$\\sigma$={sigma:.2f}" if row == 0 else "")

fig.colorbar(im, ax=axes, shrink=0.6, label="$v_i^T K_l v_i$")
fig.suptitle(f"Eigenvector alignment — depth-{DEPTH}, $\\kappa_0$={KAPPA0}", y=1.02)
fig.tight_layout()
fig.savefig("sweep_sigma_hierarchy.png", dpi=150, bbox_inches="tight")
print("\nSaved sweep_sigma_hierarchy.png")

# ---------- Plot: Loss curves (train + test) ----------
sigma_colors = plt.cm.coolwarm(np.linspace(0, 1, len(SIGMAS)))

fig, axes = plt.subplots(2, len(ACTIVATIONS), figsize=(8 * len(ACTIVATIONS), 9), squeeze=False)
for col, phi in enumerate(ACTIVATIONS):
    ax_train, ax_test = axes[0, col], axes[1, col]
    for i, sigma in enumerate(SIGMAS):
        c = results[(phi, sigma)]["curves"]
        s = c["s"]
        mse_train = c["mse_train"]
        if np.any(np.isfinite(mse_train)):
            ax_train.plot(s, mse_train, color=sigma_colors[i], label=f"$\\sigma$={sigma:.2f}", alpha=0.8)
        if "mse_test" in c and np.any(np.isfinite(c["mse_test"])):
            ax_test.plot(s, c["mse_test"], color=sigma_colors[i], label=f"$\\sigma$={sigma:.2f}", alpha=0.8)
    ax_train.set(xlabel="s", ylabel="MSE (train)", xscale="log", yscale="log", title=f"{phi} — train")
    ax_test.set(xlabel="s", ylabel="MSE (test)", xscale="log", yscale="log", title=f"{phi} — test")
    ax_train.legend(fontsize=7, ncol=2)
    ax_test.legend(fontsize=7, ncol=2)

fig.suptitle(f"Training & test loss — depth-{DEPTH}, $\\kappa_0$={KAPPA0}, ds={DS}", y=1.02)
fig.tight_layout()
fig.savefig("sweep_sigma_loss.png", dpi=150, bbox_inches="tight")
print("Saved sweep_sigma_loss.png")

# ---------- Plot: Eigenvector alignment detail (panels=sigmas, lines=eigvecs) ----------
N_SHOW_EV = min(8, N_EIGVECS)
layers = np.arange(1, DEPTH + 1)
evec_colors = plt.cm.Spectral(np.linspace(0, 1, N_SHOW_EV))

# Filter to finite sigmas per activation
def finite_sigmas(phi):
    return [s for s in SIGMAS if np.isfinite(results[(phi, s)]["curves"]["mse_train"][-1])]

for phi in ACTIVATIONS:
    good_sigmas = finite_sigmas(phi)
    n_panels = len(good_sigmas)
    ncols = min(7, n_panels)
    nrows = (n_panels + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows), squeeze=False, sharey=True)

    for idx, sigma in enumerate(good_sigmas):
        ax = axes[idx // ncols, idx % ncols]
        align = results[(phi, sigma)]["scores"]
        for i in range(N_SHOW_EV):
            ax.plot(layers, align[:, i], "o-", color=evec_colors[i], markersize=4,
                    label=f"v{i} (λ={eigenvalues[i]:.1f})")
        ax.set_xlabel("Layer")
        ax.set_title(f"σ={sigma:.2f}")
        ax.axhline(0, color="gray", ls="--", lw=0.5)
        if idx % ncols == 0:
            ax.set_ylabel("$v_i^T K_l v_i$")

    # Hide unused panels
    for idx in range(n_panels, nrows * ncols):
        axes[idx // ncols, idx % ncols].set_visible(False)

    axes[0, -1].legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=7)
    fig.suptitle(f"Eigenvector alignment — {phi}, branching={BRANCHING}, depth-{DEPTH}", y=1.02)
    fig.tight_layout()
    fig.savefig(f"sweep_sigma_eigalign_{phi}.png", dpi=150, bbox_inches="tight")
    print(f"Saved sweep_sigma_eigalign_{phi}.png")

# Also save combined inset figure with v_i v_i^T matshows
fig_insets, axes_insets = plt.subplots(1, N_SHOW_EV, figsize=(2.5 * N_SHOW_EV, 2.5))
for i in range(N_SHOW_EV):
    ax = axes_insets[i]
    v = eigenvectors[:, i]
    vvT = np.outer(v, v)
    vmax = np.abs(vvT).max()
    ax.imshow(vvT, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="equal")
    ax.set_title(f"$v_{i} v_{i}^T$\nλ={eigenvalues[i]:.1f}", fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])
fig_insets.suptitle("Input kernel eigenvector outer products", y=1.02)
fig_insets.tight_layout()
fig_insets.savefig("sweep_sigma_eigvecs.png", dpi=150, bbox_inches="tight")
print("Saved sweep_sigma_eigvecs.png")
