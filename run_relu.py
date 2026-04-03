"""Train a deep ReLU network on the ultrametric task and visualize per-layer kernels.

Tests the hypothesis that hierarchy is resolved progressively across layers:
early layers should resolve coarse (root-level) structure, deeper layers
resolve finer sub-branches.
"""

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from task import generate_brownian_tree
from network import init_deep
from train import train
from plotting import plot_deep_kernels, plot_data_kernels

# --- Config ---
BRANCHING = (2, 3, 4)  # 3-level hierarchy, 24 leaves
D = 256                 # observation dimension
N = 256                 # hidden width
DEPTH = 6               # enough layers to see progressive resolution
SIGMA = 0.8
KAPPA0 = 1e-1
DS = 1e-1
S_MAX = 10000.0
MUP = True
NTP = True
COSSIM = True
SHARED_DEPTH = 1
ACTIVATION = "tanh"

# --- Data ---
X_train, Y_train, paths_train = generate_brownian_tree(
    branching=BRANCHING, dim=D, seed_prefix=(0,), shared_depth=SHARED_DEPTH,
)
X_test, Y_test, paths_test = generate_brownian_tree(
    branching=BRANCHING, dim=D, seed_prefix=(1,), shared_depth=SHARED_DEPTH,
)
print(f"Train: X {X_train.shape}, Y {Y_train.shape}")
print(f"Test:  X {X_test.shape}, Y {Y_test.shape}")

X_tr, Y_tr = jnp.array(X_train), jnp.array(Y_train)
X_te, Y_te = jnp.array(X_test), jnp.array(Y_test)

# --- Init ---
key = jax.random.key(42)
d_out = BRANCHING[0]
params = init_deep(key, D, d_out, width=N, depth=DEPTH, sigma=SIGMA)
print(f"ReLU network: {[w.shape for w in params]}")

# --- Train ---
params_trained, curves = train(
    params, X_tr, Y_tr,
    kappa0=KAPPA0, ds=DS, s_max=S_MAX, sigma=SIGMA,
    x_test=X_te, y_test=Y_te,
    mup=MUP, ntp=NTP, cossim=COSSIM,
    activation=ACTIVATION,
    run_name=f"{ACTIVATION}-depth{DEPTH}",
    config={"arch": f"deep-{ACTIVATION}", "branching": BRANCHING, "D": D, "N": N, "depth": DEPTH, "sigma": SIGMA},
)

print(f"\nFinal train MSE: {curves['mse_train'][-1]:.6f}")
if "mse_test" in curves:
    print(f"Final test  MSE: {curves['mse_test'][-1]:.6f}")

# --- Visualize per-layer kernels ---
fig_data = plot_data_kernels(X_train, X_test, cossim=COSSIM)
fig_data.savefig(f"{ACTIVATION}_data_kernels.png", dpi=150, bbox_inches="tight")

fig_kernels = plot_deep_kernels(
    params_trained, X_tr, X_te, mup=MUP, ntp=NTP, cossim=COSSIM, activation=ACTIVATION,
)
fig_kernels.savefig(f"{ACTIVATION}_layer_kernels.png", dpi=150, bbox_inches="tight")

plt.show()
print(f"Saved {ACTIVATION}_data_kernels.png and {ACTIVATION}_layer_kernels.png")
