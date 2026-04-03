"""Demonstrate that erf activation + Gaussian weight init leads to NNGP kernel
fixpoints at 0 and c(sigma_w^2).

For a fully-connected DNN with erf activation and weights W_{ij} ~ N(0, sigma_w^2 / fan_in),
the diagonal kernel element q (expected squared pre-activation) obeys the recursion:

    q_{l+1} = sigma_w^2 * E_z[erf(z)^2],   z ~ N(0, q_l)
            = sigma_w^2 * (2/pi) * arcsin(2*q_l / (1 + 2*q_l))

Fixpoints satisfy q* = f(q*).  q* = 0 is always a fixpoint.

The derivative at q=0 is f'(0) = 4*sigma_w^2 / pi.  When f'(0) > 1, i.e.
sigma_w^2 > pi/4, the zero fixpoint is unstable and a non-trivial fixpoint c > 0 exists.
"""

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_platform_name", "cpu")


# ---------- analytic kernel map ----------
def kernel_map(q: float, sigma_w2: float) -> float:
    """One step of the NNGP diagonal-kernel recursion for erf activation."""
    return sigma_w2 * (2.0 / np.pi) * np.arcsin(2.0 * q / (1.0 + 2.0 * q))


def find_nontrivial_fixpoint(sigma_w2: float, tol: float = 1e-12) -> float:
    """Find the non-trivial fixpoint c by iterating from a large initial value."""
    q = 10.0
    for _ in range(10_000):
        q_new = kernel_map(q, sigma_w2)
        if abs(q_new - q) < tol:
            return q_new
        q = q_new
    return q


# ---------- empirical kernel from finite-width network ----------
def empirical_kernel_diagonal(
    key: jax.Array,
    x: jnp.ndarray,
    depth: int,
    width: int,
    sigma_w2: float,
) -> jnp.ndarray:
    """Compute diagonal of the empirical NNGP kernel at the last layer.

    Returns the mean (over neurons) squared pre-activation for each input.
    """
    h = x  # (n, d_in)
    sigma_w = jnp.sqrt(sigma_w2)
    for l in range(depth):
        key, subkey = jax.random.split(key)
        fan_in = h.shape[1]
        W = jax.random.normal(subkey, (width, fan_in)) * sigma_w / jnp.sqrt(fan_in)
        pre = h @ W.T  # pre-activation
        h = jax.lax.erf(pre)  # post-activation
    return jnp.mean(pre ** 2, axis=1)


def empirical_kernel_per_layer(
    key: jax.Array,
    x: jnp.ndarray,
    depth: int,
    width: int,
    sigma_w2: float,
) -> list[float]:
    """Return mean pre-activation variance (averaged over inputs and neurons) at each layer."""
    h = x  # (n, d_in)
    sigma_w = jnp.sqrt(sigma_w2)
    qs = []
    for l in range(depth):
        key, subkey = jax.random.split(key)
        fan_in = h.shape[1]
        W = jax.random.normal(subkey, (width, fan_in)) * sigma_w / jnp.sqrt(fan_in)
        pre = h @ W.T
        qs.append(float(jnp.mean(pre ** 2)))
        h = jax.lax.erf(pre)
    return qs


# ---------- tests ----------
def test_zero_is_fixpoint():
    """q=0 is always a fixpoint of the kernel map."""
    for sigma_w2 in [0.5, 1.0, 2.0, 5.0]:
        assert kernel_map(0.0, sigma_w2) == 0.0, f"q=0 not a fixpoint for sigma_w2={sigma_w2}"
    print("[PASS] q=0 is a fixpoint for all sigma_w^2")


def test_nontrivial_fixpoint_exists():
    """For sigma_w^2 > pi/2, a non-trivial fixpoint c > 0 exists."""
    critical = np.pi / 4  # ~0.7854
    print(f"  Critical variance: sigma_w^2 = pi/4 ≈ {critical:.4f}")
    print(f"  (f'(0) = 4*sigma_w^2/pi; unstable zero fixpoint when > 1)")

    # Below critical: iteration collapses to 0
    q = 0.5
    for _ in range(500):
        q = kernel_map(q, sigma_w2=0.5)
    assert q < 1e-10, "Expected collapse to 0 below critical variance"
    print(f"  sigma_w^2=0.5 < pi/4: q converged to {q:.2e} (→ 0 fixpoint)")

    # Above critical: converges to non-trivial fixpoint
    for sigma_w2 in [2.0, 3.0, 5.0]:
        c = find_nontrivial_fixpoint(sigma_w2)
        residual = abs(kernel_map(c, sigma_w2) - c)
        assert residual < 1e-10, f"Not a fixpoint: residual={residual}"
        assert c > 0.01, f"Fixpoint too small: c={c}"
        print(f"  sigma_w^2={sigma_w2}: non-trivial fixpoint c = {c:.6f}  (residual {residual:.2e})")

    print("[PASS] non-trivial fixpoint exists for sigma_w^2 > pi/4")


def test_kernel_recursion_converges():
    """Starting from arbitrary q0, the recursion converges to the predicted fixpoint."""
    sigma_w2 = 3.0
    c = find_nontrivial_fixpoint(sigma_w2)

    q_values = []
    q = 0.1  # arbitrary start
    for _ in range(200):
        q = kernel_map(q, sigma_w2)
        q_values.append(q)

    assert abs(q_values[-1] - c) < 1e-8, f"Did not converge: got {q_values[-1]}, expected {c}"
    print(f"  sigma_w^2=3.0, q0=0.1 → converged to {q_values[-1]:.6f} ≈ c={c:.6f}")
    print("[PASS] kernel recursion converges to predicted fixpoint")


def test_empirical_matches_theory():
    """Finite-width network kernel diagonal converges toward the theoretical fixpoint."""
    sigma_w2 = 3.0
    depth = 50  # deep enough to reach fixpoint
    width = 4096
    d_in = 20
    n_samples = 5

    c_theory = find_nontrivial_fixpoint(sigma_w2)

    key = jax.random.key(42)
    # Use unit-norm inputs so initial q ≈ 1
    key, subkey = jax.random.split(key)
    x = jax.random.normal(subkey, (n_samples, d_in))
    x = x / jnp.linalg.norm(x, axis=1, keepdims=True)

    key, subkey = jax.random.split(key)
    q_empirical = empirical_kernel_diagonal(subkey, x, depth, width, sigma_w2)
    q_mean = float(jnp.mean(q_empirical))

    # erf^2 is bounded by 1, so kernel diagonal is in [0, 1]
    # the fixpoint c should also be < 1 for erf
    print(f"  Theory fixpoint c = {c_theory:.6f}")
    print(f"  Empirical mean q  = {q_mean:.6f}  (width={width}, depth={depth})")
    # Allow 10% relative tolerance for finite-width effects
    assert abs(q_mean - c_theory) / c_theory < 0.10, (
        f"Empirical {q_mean:.4f} too far from theory {c_theory:.4f}"
    )
    print("[PASS] empirical kernel matches theoretical fixpoint")


def plot_fixpoints():
    """Three-panel figure showing the erf kernel fixpoint structure with empirical overlay."""
    import matplotlib.pyplot as plt

    # -- Run empirical networks for panels B and C --
    width = 4096
    depth_emp = 40
    d_in = 20
    n_samples = 10
    sigma_w2_list = [0.5, 2.0, 3.0, 5.0]
    color_map = {0.5: "C0", 2.0: "C1", 3.0: "C2", 5.0: "C4"}

    key = jax.random.key(0)
    key, subkey = jax.random.split(key)
    x = jax.random.normal(subkey, (n_samples, d_in))
    x = x / jnp.linalg.norm(x, axis=1, keepdims=True)  # unit norm → q0 ≈ 1

    empirical_traces = {}
    for sigma_w2 in sigma_w2_list:
        key, subkey = jax.random.split(key)
        empirical_traces[sigma_w2] = empirical_kernel_per_layer(
            subkey, x, depth_emp, width, sigma_w2
        )
    print(f"  Empirical networks computed (width={width}, depth={depth_emp})")

    # -- Also sweep sigma_w^2 for panel B empirical fixpoints --
    sw2_empirical = [1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    emp_fixpoints = []
    deep_for_fixpoint = 80  # deep enough to converge
    for sw2 in sw2_empirical:
        key, subkey = jax.random.split(key)
        qs = empirical_kernel_per_layer(subkey, x, deep_for_fixpoint, width, sw2)
        emp_fixpoints.append(qs[-1])  # last layer ≈ fixpoint
    print(f"  Empirical fixpoints computed for {len(sw2_empirical)} variance values")

    # -- Plot --
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # ---- Panel A: kernel map f(q) vs identity for several sigma_w^2 ----
    ax = axes[0]
    q = np.linspace(0, 5, 500)
    ax.plot(q, q, "k--", lw=1, label=r"$q$")
    for sigma_w2, color in [(0.5, "C0"), (np.pi / 4, "C3"), (2.0, "C1"), (3.0, "C2"), (5.0, "C4")]:
        fq = np.array([kernel_map(qi, sigma_w2) for qi in q])
        label = (r"$\sigma_w^2 = \pi/4$" if sigma_w2 == np.pi / 4
                 else rf"$\sigma_w^2 = {sigma_w2}$")
        ls = ":" if sigma_w2 == np.pi / 4 else "-"
        ax.plot(q, fq, color=color, ls=ls, lw=1.5, label=label)
        if sigma_w2 > np.pi / 4:
            c = find_nontrivial_fixpoint(sigma_w2)
            ax.plot(c, c, "o", color=color, ms=6, zorder=5)
    ax.plot(0, 0, "ko", ms=5, zorder=5)
    ax.set(xlabel=r"$q_\ell$", ylabel=r"$q_{\ell+1} = f(q_\ell)$",
           title="(A) Kernel map and fixpoints", xlim=(0, 5), ylim=(0, 5))
    ax.legend(fontsize=8, loc="upper left")

    # ---- Panel B: non-trivial fixpoint c vs sigma_w^2 (theory + empirical) ----
    ax = axes[1]
    sw2_vals = np.linspace(np.pi / 4 + 0.01, 8, 200)
    c_vals = [find_nontrivial_fixpoint(s) for s in sw2_vals]
    ax.plot(sw2_vals, c_vals, "C2", lw=2, label="Theory")
    ax.plot(sw2_empirical, emp_fixpoints, "kx", ms=8, mew=2, label=f"Empirical (w={width})")
    ax.axvline(np.pi / 4, color="C3", ls=":", lw=1, label=r"$\sigma_w^2 = \pi/4$ (critical)")
    ax.axhline(0, color="k", lw=0.5)
    ax.set(xlabel=r"$\sigma_w^2$", ylabel=r"Fixpoint $c(\sigma_w^2)$",
           title=r"(B) Non-trivial fixpoint $c$ vs $\sigma_w^2$")
    ax.legend(fontsize=8)

    # ---- Panel C: layer-wise convergence (theory lines + empirical markers) ----
    ax = axes[2]
    q0 = 1.0
    for sigma_w2 in sigma_w2_list:
        color = color_map[sigma_w2]
        # Theory trace
        qs = [q0]
        q_th = q0
        for _ in range(depth_emp):
            q_th = kernel_map(q_th, sigma_w2)
            qs.append(q_th)
        ax.plot(range(depth_emp + 1), qs, "-", color=color, lw=1.5,
                label=rf"$\sigma_w^2 = {sigma_w2}$ (theory)")
        # Empirical trace (layers 1..depth)
        emp_qs = empirical_traces[sigma_w2]
        ax.plot(range(1, depth_emp + 1), emp_qs, "x", color=color, ms=4, mew=1.2, alpha=0.7)
        # Fixpoint reference line
        if sigma_w2 > np.pi / 4:
            c = find_nontrivial_fixpoint(sigma_w2)
            ax.axhline(c, color=color, ls="--", lw=0.8, alpha=0.4)
    # Dummy for legend
    ax.plot([], [], "kx", ms=4, mew=1.2, label=f"Empirical (w={width})")
    ax.set(xlabel=r"Layer $\ell$", ylabel=r"$q_\ell$",
           title=r"(C) Kernel recursion ($q_0 = 1$)")
    ax.legend(fontsize=7, ncol=2)

    fig.tight_layout()
    fig.savefig("erf_kernel_fixpoints.png", dpi=150, bbox_inches="tight")
    print(f"Saved erf_kernel_fixpoints.png")
    return fig


if __name__ == "__main__":
    test_zero_is_fixpoint()
    print()
    test_nontrivial_fixpoint_exists()
    print()
    test_kernel_recursion_converges()
    print()
    test_empirical_matches_theory()
    print("\nAll tests passed.\n")

    plot_fixpoints()
