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
