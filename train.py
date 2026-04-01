import logging
import math

import jax
import jax.numpy as jnp
import numpy as np
import wandb
from tqdm.auto import tqdm

from network_aux import get_eta, loss_fn
from plotting import kernel_images

logger = logging.getLogger(__name__)


def _ols_slope(log_t: np.ndarray, log_l: np.ndarray) -> float:
    """OLS slope of log_l vs log_t (scalar)."""
    t_mean = log_t.mean()
    l_mean = log_l.mean()
    return float(np.sum((log_t - t_mean) * (log_l - l_mean)) / (np.sum((log_t - t_mean) ** 2) + 1e-30))


def train(
    params: list[jnp.ndarray],
    x: jnp.ndarray,
    y: jnp.ndarray,
    kappa0: float = 1e-2,
    ds: float = 1.0,
    s_max: float = 1000.0,
    sigma: float = 1.0,
    ema_beta: float = 0.999,
    slope_frac: float = 0.1,
    slope_threshold: float = 0.03,
    warmup_patience: float = 100.0,
    stop_patience: int = 2,
    log_every: int = 100,
    scan_unroll: int = 10,
    x_test: jnp.ndarray | None = None,
    y_test: jnp.ndarray | None = None,
    mup: bool = False,
    ntp: bool = True,
    cossim: bool = True,
    run_name: str | None = None,
    config: dict | None = None,
) -> tuple[list[jnp.ndarray], dict]:
    """Train with early stopping based on log-log slope.

    Uses jax.lax.scan to fuse `log_every` steps on GPU per host sync.
    `scan_unroll` controls XLA unrolling depth (small constant for fast
    compile + pipelining; independent of `log_every`).
    """
    max_steps = int(s_max / ds)
    eta = get_eta(params, x, y, kappa0, mup=mup, ntp=ntp, ds=ds)

    run = wandb.init(
        project="hierarchical-compositional-generalization",
        entity="japhba-personal",
        name=run_name,
        config={
            "kappa0": kappa0,
            "ds": ds,
            "s_max": s_max,
            "eta": eta,
            "sigma": sigma,
            "max_steps": max_steps,
            "ema_beta": ema_beta,
            "slope_frac": slope_frac,
            "slope_threshold": slope_threshold,
            "warmup_patience": warmup_patience,
            "stop_patience": stop_patience,
            "log_every": log_every,
            "mup": mup,
            "ntp": ntp,
            **(config or {}),
        },
        reinit=True,
    )
    wandb.define_metric("s")
    wandb.define_metric("*", step_metric="s")

    assert jax.devices()[0].platform == "gpu", f"Expected GPU, got {jax.devices()[0].platform}"
    param_devices = {f"W{i}": w.devices() for i, w in enumerate(params)}
    print(f"Training with ds={ds:.2e}, s_max={s_max:.1f}, eta={eta:.2e}, max_steps={max_steps} | x: {x.devices()}, y: {y.devices()}, params: {param_devices}")

    if x_test is not None:
        @jax.jit
        def eval_test(params):
            _, (mse_test, _) = loss_fn(params, x_test, y_test, kappa0, mup=mup, sigma=sigma, ntp=ntp, terms="mse")
            return mse_test

    def scan_body(params, _):
        (loss, (mse, nll)), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            params, x, y, kappa0, mup=mup, sigma=sigma, ntp=ntp, terms="all",
        )
        params = [w - eta * g for w, g in zip(params, grads)]
        return params, (loss, mse, nll)

    @jax.jit
    def scan_chunk(params):
        return jax.lax.scan(scan_body, params, None, length=log_every, unroll=scan_unroll)

    n_chunks = max_steps // log_every
    remainder = max_steps % log_every

    losses = []
    mse_train_curve = []
    mse_test_curve = []
    s_curve = []
    ema = None
    slope = float("nan")
    slope_test = float("nan")
    flat_count = 0
    converged = False
    all_log_s = []
    all_log_ema = []
    all_log_mse_test = []
    pbar = tqdm(total=s_max, desc="Training", unit="s", bar_format="{l_bar}{bar}| {n:.2f}/{total:.2f}s [{elapsed}<{remaining}, {rate_fmt}{postfix}]")
    step_offset = 0
    for chunk_idx in range(n_chunks + (1 if remainder else 0)):
        is_last_chunk = chunk_idx == n_chunks  # remainder chunk
        if is_last_chunk:
            # JIT a smaller scan for the leftover steps
            @jax.jit
            def scan_remainder(params):
                return jax.lax.scan(scan_body, params, None, length=remainder, unroll=min(scan_unroll, remainder))
            params, (loss_arr, mse_arr, nll_arr) = scan_remainder(params)
            chunk_len = remainder
        else:
            params, (loss_arr, mse_arr, nll_arr) = scan_chunk(params)
            chunk_len = log_every

        # Transfer chunk metrics to host
        loss_np = np.asarray(loss_arr)
        mse_np = np.asarray(mse_arr)
        nll_np = np.asarray(nll_arr)

        for i in range(chunk_len):
            losses.append(float(loss_np[i]))

        # EMA over the chunk
        for i in range(chunk_len):
            lv = float(loss_np[i])
            ema = lv if ema is None else ema_beta * ema + (1 - ema_beta) * lv

        step_offset += chunk_len
        s = step_offset * ds

        # Use last values in chunk for logging
        loss_val = float(loss_np[-1])
        mse_val = float(mse_np[-1])
        nll_val = float(nll_np[-1])

        if s > ds and ema > 0:
            all_log_s.append(math.log(s))
            all_log_ema.append(math.log(ema))
        log_dict = {
            "loss": loss_val, "mse": mse_val, "nll_prior": nll_val,
            "log_mse": math.log(mse_val) if mse_val > 0 else float("-inf"),
            "log_nll": math.log(nll_val) if nll_val > 0 else float("-inf"),
            "ema_loss": ema, "s": s,
        }
        s_curve.append(s)
        mse_train_curve.append(mse_val)
        if x_test is not None:
            mse_test_val = float(eval_test(params))
            mse_test_curve.append(mse_test_val)
            log_dict["mse_test"] = mse_test_val
            log_dict["log_mse_test"] = math.log(mse_test_val) if mse_test_val > 0 else float("-inf")
            if s > ds and mse_test_val > 0:
                all_log_mse_test.append(math.log(mse_test_val))

        n = len(all_log_s)
        idx = max(0, n - max(10, int(n * slope_frac)))
        if n - idx >= 10:
            slope = _ols_slope(np.array(all_log_s[idx:]), np.array(all_log_ema[idx:]))
            log_dict["slope"] = slope
            if len(all_log_mse_test) >= n:
                slope_test = _ols_slope(np.array(all_log_s[idx:]), np.array(all_log_mse_test[idx:]))
                log_dict["slope_test"] = slope_test
            log_dict["stop_patience_counter"] = flat_count
            if s >= warmup_patience and abs(slope) < slope_threshold:
                flat_count += 1
            else:
                flat_count = 0
            if flat_count >= stop_patience:
                pbar.set_description("Converged")
                converged = True

        if chunk_idx % 100 == 0:
            log_dict.update(kernel_images(params, x, x_test if x_test is not None else x, mup=mup, ntp=ntp, cossim=cossim))

        wandb.log(log_dict)
        pbar.update(chunk_len * ds)
        pbar.set_postfix(mse=f"{mse_val:.2e}", ema=f"{ema:.2e}", slope=f"{slope:.2e}")

        if converged:
            break

    pbar.close()
    run.finish()
    curves = {"s": np.array(s_curve), "mse_train": np.array(mse_train_curve), "loss": losses}
    if mse_test_curve:
        curves["mse_test"] = np.array(mse_test_curve)
    return params, curves
