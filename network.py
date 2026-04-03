import jax
import jax.numpy as jnp


def _rescale_weight(w: jnp.ndarray, downscale: bool, mup: bool, is_last: bool) -> jnp.ndarray:
    """Rescale weight for NTP ↔ effective conversion.

    When downscale=True (NTP forward pass): multiply by fan_in^{-gamma/2}
    where gamma=2 for muP last layer with scalar output, gamma=1 otherwise.

    Mirrors recurrent_feature/tfl/network_aux.py:rescale_weight + get_scale.
    """
    if not downscale:
        return w
    fan_in = w.shape[1]
    gamma_scaling = 2 if (mup and is_last and w.shape[0] == 1) else 1
    return w * (fan_in ** gamma_scaling) ** -0.5


def init_shallow(key: jax.Array, d_in: int, d_out: int, sigma: float = 1.0) -> list[jnp.ndarray]:
    """Single linear layer W: (d_out, d_in).  NTP: entries ~ sigma."""
    W = jax.random.normal(key, (d_out, d_in)) * sigma
    return [W]


def init_deep(key: jax.Array, d_in: int, d_out: int, width: int = 1000, depth: int = 2, sigma: float = 1.0) -> list[jnp.ndarray]:
    """Chain of `depth` linear layers with hidden dim `width`.  NTP: entries ~ sigma."""
    keys = jax.random.split(key, depth)
    params = []
    for i in range(depth):
        fan_in = d_in if i == 0 else width
        fan_out = width if i < depth - 1 else d_out
        W = jax.random.normal(keys[i], (fan_out, fan_in)) * sigma
        params.append(W)
    return params


def forward(params: list[jnp.ndarray], x: jnp.ndarray, mup: bool = False, ntp: bool = True, activation: str = "linear") -> jnp.ndarray:
    """Forward pass with optional nonlinearity.

    Args:
        activation: 'linear' (no nonlinearity) or 'relu' (ReLU after each hidden layer, not the last).
    """
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
    return h
