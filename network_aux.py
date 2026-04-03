import jax.numpy as jnp

from network import _rescale_weight, forward


def _prior_var(w: jnp.ndarray, mup: bool, is_last: bool, sigma: float) -> float:
    """Prior variance of a single (unit-scale) weight entry.

    var = sigma^2 / fan_in^gamma  where gamma follows get_scale convention.
    """
    fan_in = w.shape[1]
    gamma_scaling = 2 if (mup and is_last and w.shape[0] == 1) else 1
    return sigma ** 2 / fan_in ** gamma_scaling


def get_eta(
    params: list[jnp.ndarray],
    x: jnp.ndarray,
    y: jnp.ndarray,
    kappa0: float,
    mup: bool = False,
    ntp: bool = True,
    ds: float = 1.0,
) -> float:
    """Physics-informed learning rate following NTP scaling.

    Ported from recurrent_feature/tfl/weights.py:get_initial_step_size.

        eta = kappa0 / N_muP  * N_ntp  * ds / (P * D_out)
    """
    width = max(w.shape[1] for w in params)
    P = x.shape[0]
    D_out = y.shape[-1]

    eta = kappa0 / (width if mup else 1)
    if ntp:
        eta *= width
    eta *= ds
    eta /= P * D_out
    return float(eta)


def logprob_prior(
    params: list[jnp.ndarray],
    mup: bool = False,
    ntp: bool = True,
    sigma: float = 1.0,
) -> jnp.ndarray:
    """MVN prior NLL (negative log-prior, sign-flipped).

    Ported from recurrent_feature/tfl/network_aux.py:logprob_prior.

        -log p(w) = 1/2 * sum_l  var_l^{-1} * ||rescale(w_l)||^2

    NTP=True  →  rescale(w) = w / sqrt(fan_in)  [or / fan_in for muP-last scalar]
    NTP=False →  rescale(w) = w   (weights already at unit scale)
    """
    nll = 0.0
    for i, w in enumerate(params):
        is_last = i == len(params) - 1
        var = _prior_var(w, mup, is_last, sigma)
        w_rescaled = _rescale_weight(w, downscale=ntp, mup=mup, is_last=is_last)
        nll = nll + jnp.sum(w_rescaled ** 2) / var
    return 0.5 * nll


def logprob_y(
    params: list[jnp.ndarray],
    x: jnp.ndarray,
    y: jnp.ndarray,
    kappa0: float,
    mup: bool = False,
    ntp: bool = True,
    activation: str = "linear",
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Data negative log-likelihood (sign-flipped logprob_y).

    Ported from recurrent_feature/tfl/network_aux.py:logprob (lines 147-154).

        -logprob_y = MSE * P * D_out / (kappa0 / N_muP)

    Returns (nll_data, mse) so MSE can be monitored separately.
    """
    P = x.shape[0]
    D_out = y.shape[-1]
    width = max(w.shape[1] for w in params)
    N_mup = width if mup else 1

    mse = jnp.mean((forward(params, x, mup=mup, ntp=ntp, activation=activation) - y) ** 2)
    nll_data = mse * P * D_out / (kappa0 / N_mup)
    return nll_data, mse


def loss_fn(
    params: list[jnp.ndarray],
    x: jnp.ndarray,
    y: jnp.ndarray,
    kappa0: float,
    mup: bool = False,
    sigma: float = 1.0,
    ntp: bool = True,
    terms: str = "all",
    activation: str = "linear",
) -> tuple[jnp.ndarray, tuple[jnp.ndarray, jnp.ndarray]]:
    """Negative log-posterior:  -logprob = -logprob_y + (-logprob_prior).

    Ported from recurrent_feature/tfl/network_aux.py:logprob.

    Args:
        terms: which terms to return as the scalar loss.
            'all'  — nll_data + nll_prior  (for training)
            'mse'  — raw mse only          (for evaluation)
            'nll'  — nll_prior only
    Returns:
        (scalar_loss, (mse, nll_prior))
    """
    nll_data, mse = logprob_y(params, x, y, kappa0, mup=mup, ntp=ntp, activation=activation)
    nll_prior = logprob_prior(params, mup=mup, ntp=ntp, sigma=sigma)

    if terms == "all":
        loss = nll_data + nll_prior
    elif terms == "mse":
        loss = mse
    elif terms == "nll":
        loss = nll_prior
    else:
        raise ValueError(f"Unknown terms={terms!r}, expected 'all', 'mse', or 'nll'")

    return loss, (mse, nll_prior)
