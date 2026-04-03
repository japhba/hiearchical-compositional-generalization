import jax
import jax.numpy as jnp
import numpy as np


def generate_brownian_tree(
    branching: tuple[int, ...] = (3, 3, 3),
    dim: int = 100,
    step_std: float | tuple[float, ...] = 1.0,
    seed_prefix: tuple[int, ...] = (),
    shared_depth: int = 0,
) -> tuple[np.ndarray, np.ndarray, list[tuple[int, ...]]]:
    """Generate hierarchical Brownian motion and return (X, Y, paths) dataset.

    Args:
        branching: number of children at each depth level
        dim: dimensionality of Brownian motion
        step_std: std of Brownian increment, either a single float (same at
            all levels) or a tuple of per-level stds matching len(branching)
        seed_prefix: prepended to path seeds at depths >= shared_depth.
            Use different prefixes for train/test to get different realizations.
        shared_depth: depths < shared_depth use seed=path (shared across
            prefixes), depths >= shared_depth use seed=seed_prefix+path.
            E.g. shared_depth=1 shares root-level increments across datasets.

    Returns:
        X: (num_leaves, dim) array of leaf observations
        Y: (num_leaves, branching[0]) one-hot root branch labels
        paths: list of full branch path tuples for each leaf
    """
    stds = (step_std,) * len(branching) if isinstance(step_std, (int, float)) else tuple(step_std)
    leaves = []
    labels = []
    paths = []

    def _make_seed(path: tuple[int, ...], depth: int):
        # Include depth in seed to prevent SeedSequence collisions across levels
        # (e.g. seed=(0,) and seed=(0,0,0) can be correlated due to trailing zeros)
        base = path if depth < shared_depth else seed_prefix + path
        return base + (depth,)

    def _recurse(parent_value: np.ndarray, path: tuple[int, ...], depth: int):
        if depth == len(branching):
            leaves.append(parent_value)
            labels.append(path[0])
            paths.append(path)
            return
        for b in range(branching[depth]):
            child_path = path + (b,)
            rng = np.random.default_rng(seed=_make_seed(child_path, depth))
            increment = rng.normal(scale=stds[depth], size=dim)
            _recurse(parent_value + increment, child_path, depth + 1)

    root = np.zeros(dim)
    for b in range(branching[0]):
        path = (b,)
        rng = np.random.default_rng(seed=_make_seed(path, 0))
        increment = rng.normal(scale=stds[0], size=dim)
        _recurse(root + increment, path, 1)

    X = np.stack(leaves)
    Y = np.eye(branching[0])[labels]
    return X, Y, paths


def make_teacher(
    branching: tuple[int, ...],
    dim: int,
    seed: int = 0,
) -> list[jnp.ndarray]:
    """Build a deep hierarchical block-diagonal teacher that reduces bottom-up.

    Architecture for branching=(2, 3, 4), dim=240:

        Layer 0: R^240 → R^24  block-diagonal (24 blocks of 10→1), tanh
        Layer 1: R^24  → R^6   block-diagonal (6 blocks of 4→1),   tanh
        Layer 2: R^6   → R^2   block-diagonal (2 blocks of 3→1),   tanh
        → output ∈ R^2

    depth = len(branching).  Each layer aggregates within groups at one
    tree level, reducing dimensionality until only branching[0] outputs
    remain.  This is a genuinely compositional function: leaf features
    are grouped, aggregated, and combined up the hierarchy.

    Returns:
        list of weight matrices [W_leaf, W_agg1, W_agg2, ...]
    """
    n_leaves = 1
    for b in branching:
        n_leaves *= b
    dims_per_leaf = dim // n_leaves

    key = jax.random.key(seed)
    params = []

    # Layer 0: leaf projection R^dim → R^n_leaves
    key, subkey = jax.random.split(key)
    leaf_keys = jax.random.split(subkey, n_leaves)
    blocks = [jax.random.normal(k, (1, dims_per_leaf)) / jnp.sqrt(dims_per_leaf) for k in leaf_keys]
    W0 = jax.scipy.linalg.block_diag(*blocks)
    if W0.shape[1] < dim:
        W0 = jnp.pad(W0, ((0, 0), (0, dim - W0.shape[1])))
    params.append(W0)

    # Aggregation layers: reduce bottom-up through the hierarchy
    n_units = n_leaves
    for level in reversed(branching[1:]):
        n_groups = n_units // level
        key, subkey = jax.random.split(key)
        block_keys = jax.random.split(subkey, n_groups)
        blocks = [jax.random.normal(bk, (1, level)) / jnp.sqrt(level) for bk in block_keys]
        W = jax.scipy.linalg.block_diag(*blocks)
        params.append(W)
        n_units = n_groups

    return params


def teacher_forward(
    teacher_params: list[jnp.ndarray],
    x: jnp.ndarray,
    beta: float = 1.0,
) -> jnp.ndarray:
    """Forward pass through the deep hierarchical teacher."""
    h = x
    for W in teacher_params:
        h = jnp.tanh(beta * (h @ W.T))
    return h


def generate_teacher_task(
    branching: tuple[int, ...],
    dim: int,
    seed: int = 0,
    beta: float = 1.0,
    step_std: float | tuple[float, ...] = 1.0,
    shared_depth: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[jnp.ndarray]]:
    """Generate train+test data: Brownian tree inputs + deep teacher labels.

    Inputs (one per leaf) come from a hierarchical Brownian tree, giving
    an ultrametric X sample kernel.  Labels are produced by a deep
    block-diagonal teacher network.

    Train and test share Brownian increments at depths < shared_depth,
    so they have the same coarse structure but differ at fine scales.

    Returns:
        X_train, Y_train, X_test, Y_test, teacher_params
    """
    teacher_params = make_teacher(branching, dim, seed=seed)

    X_train, _, _ = generate_brownian_tree(branching, dim, step_std=step_std, seed_prefix=(0,), shared_depth=shared_depth)
    X_test, _, _ = generate_brownian_tree(branching, dim, step_std=step_std, seed_prefix=(1,), shared_depth=shared_depth)

    Y_train = np.asarray(teacher_forward(teacher_params, jnp.array(X_train), beta=beta))
    Y_test = np.asarray(teacher_forward(teacher_params, jnp.array(X_test), beta=beta))

    return X_train, Y_train, X_test, Y_test, teacher_params


def get_data(
    task: str,
    **kwargs,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list | None]:
    """Unified data interface.

    Returns (X_train, Y_train, X_test, Y_test, teacher_params).
    teacher_params is None for non-teacher tasks.
    """
    teacher_params = None
    if task == "brownian":
        branching = kwargs.pop("branching", (3, 3, 3))
        dim = kwargs.pop("dim", 100)
        shared_depth = kwargs.pop("shared_depth", 1)
        kw = {k: v for k, v in kwargs.items() if k in ("step_std",)}
        X_train, Y_train, _ = generate_brownian_tree(branching=branching, dim=dim, seed_prefix=(0,), shared_depth=shared_depth, **kw)
        X_test, Y_test, _ = generate_brownian_tree(branching=branching, dim=dim, seed_prefix=(1,), shared_depth=shared_depth, **kw)
    elif task == "teacher":
        branching = kwargs.pop("branching", (3, 3, 3))
        dim = kwargs.pop("dim", 100)
        seed = kwargs.pop("seed", 0)
        beta = kwargs.pop("beta", 1.0)
        step_std = kwargs.pop("step_std", 1.0)
        shared_depth = kwargs.pop("shared_depth", 1)
        X_train, Y_train, X_test, Y_test, teacher_params = generate_teacher_task(branching=branching, dim=dim, seed=seed, beta=beta, step_std=step_std, shared_depth=shared_depth)
    else:
        raise ValueError(f"Unknown task {task!r}, expected 'brownian' or 'teacher'")

    print(f"Task={task}: Train X {X_train.shape}, Y {Y_train.shape} | Test X {X_test.shape}, Y {Y_test.shape}")
    return X_train, Y_train, X_test, Y_test, teacher_params
