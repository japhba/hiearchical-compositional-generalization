"""Sweep initialization variance (sigma) for a deep tanh network on the ultrametric task.

For each sigma, trains the network and measures per-layer hierarchy resolution:
at each layer, how well does the representation separate groups at each level
of the hierarchy (root / mid / leaf).

Metric: mean within-group cosine similarity minus mean between-group cosine similarity,
computed separately for each hierarchy level.
"""

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from task import generate_brownian_tree
from network import init_deep
from network_aux import get_eta, loss_fn
from plotting import _layer_hiddens, _kernel


# ---------- Config ----------
BRANCHING = (2, 3, 4)
D = 256
N = 256
DEPTH = 6
KAPPA0 = 1e-2
DS = 1e-1
S_MAX = 10000.0
MUP = True
NTP = True
ACTIVATION = "tanh"
SHARED_DEPTH = 1
LOG_EVERY = 100
SCAN_UNROLL = 10

SIGMAS = np.geomspace(0.1, 10, 7)

# ---------- Data ----------
X_train, Y_train, paths_train = generate_brownian_tree(
    branching=BRANCHING, dim=D, seed_prefix=(0,), shared_depth=SHARED_DEPTH,
)
X_test, Y_test, paths_test = generate_brownian_tree(
    branching=BRANCHING, dim=D, seed_prefix=(1,), shared_depth=SHARED_DEPTH,
)
X_tr, Y_tr = jnp.array(X_train), jnp.array(Y_train)
X_te, Y_te = jnp.array(X_test), jnp.array(Y_test)

n_levels = len(BRANCHING)  # 3 hierarchy levels
n_leaves = X_tr.shape[0]


def group_masks(paths, level):
    """Return (n, n) bool arrays for within-group and between-group at a given hierarchy level.

    Two leaves are 'within-group' at level l if they share the same ancestor path[:l+1].
    """
    n = len(paths)
    groups = [p[:level + 1] for p in paths]
    within = np.zeros((n, n), dtype=bool)
    between = np.zeros((n, n), dtype=bool)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if groups[i] == groups[j]:
                within[i, j] = True
            else:
                between[i, j] = True
    return within, between


# Precompute masks for each hierarchy level
masks = [group_masks(paths_train, l) for l in range(n_levels)]


def hierarchy_scores(params, x, paths):
    """Compute hierarchy resolution score per layer per level.

    Returns: (n_layers, n_levels) array of (within - between) cosine similarities.
    """
    hiddens = _layer_hiddens(params, x, mup=MUP, ntp=NTP, activation=ACTIVATION)
    scores = np.zeros((len(hiddens), n_levels))
    for li, h in enumerate(hiddens):
        K = np.array(_kernel(h, cossim=True))
        for lv in range(n_levels):
            within, between = masks[lv]
            w_mean = K[within].mean() if within.any() else 0.0
            b_mean = K[between].mean() if between.any() else 0.0
            scores[li, lv] = w_mean - b_mean
    return scores


# ---------- Minimal training loop (no wandb) ----------
def train_minimal(params, x, y, kappa0, sigma, x_test=None, y_test=None):
    """Lightweight training loop without wandb, returns trained params."""
    max_steps = int(S_MAX / DS)
    eta = get_eta(params, x, y, kappa0, mup=MUP, ntp=NTP, ds=DS)

    def scan_body(params, _):
        (loss, (mse, nll)), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            params, x, y, kappa0, mup=MUP, sigma=sigma, ntp=NTP, terms="all",
            activation=ACTIVATION,
        )
        params = [w - eta * g for w, g in zip(params, grads)]
        return params, mse

    @jax.jit
    def scan_chunk(params):
        return jax.lax.scan(scan_body, params, None, length=LOG_EVERY, unroll=SCAN_UNROLL)

    n_chunks = max_steps // LOG_EVERY
    ema = None
    ema_beta = 0.999
    for chunk_idx in range(n_chunks):
        params, mse_arr = scan_chunk(params)
        mse_val = float(mse_arr[-1])
        ema = mse_val if ema is None else ema_beta * ema + (1 - ema_beta) * mse_val

        if chunk_idx % 100 == 0:
            test_str = ""
            if x_test is not None:
                _, (mse_test, _) = loss_fn(params, x_test, y_test, kappa0,
                                           mup=MUP, sigma=sigma, ntp=NTP, terms="mse",
                                           activation=ACTIVATION)
                test_str = f", test_mse={float(mse_test):.2e}"
            print(f"  chunk {chunk_idx}/{n_chunks}: mse={mse_val:.2e}, ema={ema:.2e}{test_str}")

        # Simple early stopping
        if ema is not None and chunk_idx > 200 and ema < 1e-8:
            print(f"  converged at chunk {chunk_idx}")
            break

    return params


# ---------- Sweep ----------
all_scores = {}

for sigma in SIGMAS:
    print(f"\n{'='*60}")
    print(f"sigma = {sigma:.3f}")
    print(f"{'='*60}")

    key = jax.random.key(42)
    params = init_deep(key, D, BRANCHING[0], width=N, depth=DEPTH, sigma=sigma)

    params_trained = train_minimal(params, X_tr, Y_tr, KAPPA0, sigma,
                                   x_test=X_te, y_test=Y_te)

    scores = hierarchy_scores(params_trained, X_tr, paths_train)
    all_scores[sigma] = scores
    print(f"  Hierarchy scores (layers x levels):\n{scores}")


# ---------- Plot ----------
level_labels = [f"Level {l} ({BRANCHING[l]} branches)" for l in range(n_levels)]
colors = plt.cm.viridis(np.linspace(0, 1, n_levels))

n_sigmas = len(SIGMAS)
fig, axes = plt.subplots(1, n_sigmas, figsize=(4 * n_sigmas, 4), sharey=True)
if n_sigmas == 1:
    axes = [axes]

layers = np.arange(1, DEPTH + 1)

for ax, sigma in zip(axes, SIGMAS):
    scores = all_scores[sigma]
    for lv in range(n_levels):
        ax.plot(layers, scores[:, lv], 'o-', color=colors[lv], label=level_labels[lv])
    ax.set_xlabel("Layer")
    ax.set_title(f"$\\sigma$ = {sigma:.2f}")
    ax.axhline(0, color='gray', ls='--', lw=0.5)

axes[0].set_ylabel("Within - Between\n(cosine similarity)")
axes[-1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
fig.suptitle(f"Hierarchy resolution by layer — tanh depth-{DEPTH}, $\\kappa_0$={KAPPA0}", y=1.02)
fig.tight_layout()
fig.savefig("sweep_sigma_hierarchy.png", dpi=150, bbox_inches="tight")
plt.show()
print("\nSaved sweep_sigma_hierarchy.png")
