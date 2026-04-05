"""Null-hypothesis visualization: kernel eigenvector alignment across layers.

Metric: for each layer's cosine-similarity kernel K_l, measure alignment with
each eigenvector of the input data kernel K_X:

    a_i(l) = v_i^T K_l v_i

where v_i are eigenvectors of K_X sorted by descending eigenvalue. Since the
input kernel's eigenvectors decompose by hierarchy level (coarsest = largest
eigenvalue), this tracks how each layer preserves or discards each hierarchical
component.

Constructs a deep linear network with weight matrices that progressively
coarse-grain the hierarchy, as a null hypothesis for what "successive
coarse-graining" looks like under this metric.
"""

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from task import generate_brownian_tree
from network import init_deep
from plotting import _kernel, _imshow_centered, _layer_hiddens

# ---------- Config ----------
BRANCHING = (2, 2, 3, 4)
N_LEVELS = len(BRANCHING)
N_LEAVES = int(np.prod(BRANCHING))  # 48
N_LAYERS = 6
DIM = 256
SHARED_DEPTH = 1

# ---------- Data ----------
X_train, Y_train, paths_train = generate_brownian_tree(
    branching=BRANCHING, dim=DIM, seed_prefix=(0,), shared_depth=SHARED_DEPTH,
)
X = np.array(X_train)  # (48, 256)
X_jnp = jnp.array(X)

# ---------- Input kernel eigenvectors ----------
K_input = np.array(_kernel(X_jnp, cossim=True))
eigenvalues, eigenvectors = np.linalg.eigh(K_input)
# Sort descending
idx = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]  # (48, 48), columns are eigenvectors


def eigenvector_alignment(hiddens, eigvecs=eigenvectors):
    """(n_layers, n_eigvecs) alignment: a_i(l) = v_i^T K_l v_i."""
    n_ev = eigvecs.shape[1]
    scores = np.zeros((len(hiddens), n_ev))
    for li, h in enumerate(hiddens):
        K = np.array(_kernel(h, cossim=True))
        for i in range(n_ev):
            v = eigvecs[:, i]
            scores[li, i] = v @ K @ v
    return scores


# ---------- Group averaging matrices ----------
def group_avg_matrix(paths, level):
    n = len(paths)
    groups = [p[:level + 1] for p in paths]
    A = np.zeros((n, n))
    for i in range(n):
        same = [j for j in range(n) if groups[j] == groups[i]]
        for j in same:
            A[i, j] = 1.0 / len(same)
    return A


# ---------- Construct coarse-graining network ----------
targets = [X]
H = X.copy()
for lv in reversed(range(N_LEVELS)):
    A = group_avg_matrix(paths_train, lv)
    H = A @ H
    targets.append(H.copy())
while len(targets) <= N_LAYERS:
    targets.append(targets[-1].copy())

params_constructed = []
for l in range(N_LAYERS):
    W_T, _, _, _ = np.linalg.lstsq(targets[l], targets[l + 1], rcond=None)
    params_constructed.append(jnp.array(W_T.T))

hiddens_constructed = _layer_hiddens(params_constructed, X_jnp, mup=False, ntp=False, phi=None)
for l, (h, t) in enumerate(zip(hiddens_constructed, targets[1:])):
    print(f"Layer {l+1}: reconstruction MSE = {float(jnp.mean((h - jnp.array(t)) ** 2)):.2e}")

# Control: random network
params_random = init_deep(jax.random.key(0), DIM, DIM, width=DIM, depth=N_LAYERS, sigma=1.0)
hiddens_random = _layer_hiddens(params_random, X_jnp, mup=False, ntp=False, phi=None)

# ---------- Compute alignment ----------
align_constructed = eigenvector_alignment(hiddens_constructed)
align_random = eigenvector_alignment(hiddens_random)

# ---------- Plot: alignment for top eigenvectors ----------
N_SHOW = min(12, N_LEAVES)
layers = np.arange(1, N_LAYERS + 1)
evec_colors = plt.cm.Spectral(np.linspace(0, 1, N_SHOW))

fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
for ax, align, title in zip(axes,
                             [align_constructed, align_random],
                             ["Constructed: progressive coarse-graining",
                              "Control: random init"]):
    for i in range(N_SHOW):
        ax.plot(layers, align[:, i], "o-", color=evec_colors[i],
                label=f"v{i} (λ={eigenvalues[i]:.1f})", alpha=0.8)
    ax.set_xlabel("Layer")
    ax.axhline(0, color="gray", ls="--", lw=0.5)
    ax.set_title(title)

axes[0].set_ylabel("$v_i^T K_l v_i$\n(eigenvector alignment)")
axes[-1].legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
fig.suptitle(f"Kernel eigenvector alignment across layers\nbranching={BRANCHING}, input kernel eigenbasis", y=1.05)
fig.tight_layout()
fig.savefig("_test_crosslayer_metric.png", dpi=150, bbox_inches="tight")
print("Saved _test_crosslayer_metric.png")

# ---------- Plot: kernels at each layer ----------
fig_k, axes_k = plt.subplots(1, N_LAYERS, figsize=(4 * N_LAYERS, 4))
for l, (ax, h) in enumerate(zip(axes_k, hiddens_constructed)):
    K = np.array(_kernel(h, cossim=True))
    n_fine_removed = min(l + 1, N_LEVELS)
    remaining = N_LEVELS - n_fine_removed
    if remaining > 0:
        _imshow_centered(ax, K, f"Layer {l+1}\n(levels 0–{remaining-1})")
    else:
        _imshow_centered(ax, K, f"Layer {l+1}\n(no structure)")
fig_k.suptitle(f"Constructed network kernels — branching={BRANCHING}, cosine similarity", y=1.02)
fig_k.tight_layout()
fig_k.savefig("_test_crosslayer_kernels.png", dpi=150, bbox_inches="tight")
print("Saved _test_crosslayer_kernels.png")

# ---------- Plot: eigenvector outer products as insets ----------
N_INSETS = min(8, N_SHOW)
fig_ev, axes_ev = plt.subplots(2, N_INSETS, figsize=(3 * N_INSETS, 7),
                                gridspec_kw={"height_ratios": [1, 2]})

# Top row: v_i v_i^T as matshow
for i in range(N_INSETS):
    ax = axes_ev[0, i]
    v = eigenvectors[:, i]
    vvT = np.outer(v, v)
    vmax = np.abs(vvT).max()
    ax.imshow(vvT, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="equal")
    ax.set_title(f"$v_{i} v_{i}^T$\nλ={eigenvalues[i]:.1f}", fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])

# Bottom row: alignment across layers for each eigenvector
for i in range(N_INSETS):
    ax = axes_ev[1, i]
    ax.plot(layers, align_constructed[:, i], "o-", color="C0", label="constructed")
    ax.plot(layers, align_random[:, i], "s--", color="C3", alpha=0.5, label="random")
    ax.set_xlabel("Layer")
    ax.set_ylim(-0.5, max(1.0, align_constructed[:, :N_INSETS].max() * 1.1))
    ax.axhline(0, color="gray", ls="--", lw=0.5)
    if i == 0:
        ax.set_ylabel("$v_i^T K_l v_i$")
        ax.legend(fontsize=7)

fig_ev.suptitle(f"Eigenvector structure & alignment — branching={BRANCHING}", y=1.02)
fig_ev.tight_layout()
fig_ev.savefig("_test_crosslayer_eigalign.png", dpi=150, bbox_inches="tight")
print("Saved _test_crosslayer_eigalign.png")
