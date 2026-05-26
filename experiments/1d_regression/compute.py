#!/usr/bin/env python3
"""Compute cached CSV data for the 1D BNN regression KL visualization."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

PyTree = Any

DATA_DIR = Path(__file__).with_name("plot_data")

SEED = 0
N_OLD = 82
N_NEW = 44
OLD_NOISE_STD = 0.02
NEW_NOISE_STD = 0.045
PRIOR_STD = 1.0
KL_STD = 0.08
KL_WEIGHT = 1.0

HIDDEN_DIMS = (48, 48, 24)
INIT_SCALE = 0.08
MAP_STEPS = 12_000
N_CHAINS = 4
N_PARTICLES = 80
SGHMC_BURN_IN = 5_000
SGHMC_THIN = 12
SGHMC_TEMPERATURE = 0.12
ADAPT_STEPS = 8_000

X_MIN = -3.4
X_MAX = 3.4
GRID_POINTS = 500


def old_task_function(x: jnp.ndarray) -> jnp.ndarray:
    base = -0.82 + 0.05 * (x + 2.0)
    bump1 = 0.52 * jnp.exp(-((x + 1.18) / 0.22) ** 2)
    valley = -0.44 * jnp.exp(-((x + 2.02) / 0.31) ** 2)
    ripple = 0.12 * jnp.sin(2.6 * (x + 1.6))
    return base + bump1 + valley + ripple


def new_task_function(x: jnp.ndarray) -> jnp.ndarray:
    z = x - 0.45
    return 1.25 * jnp.sin(5.8 * z) + 0.85 * jnp.sin(10.8 * z) + 0.45 * jnp.cos(2.4 * z)


def generate_toy_data(
    key: jax.Array,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    k_old_x, k_old_y, k_new_x, k_new_y = jax.random.split(key, 4)
    x_old = jax.random.uniform(k_old_x, (N_OLD, 1), minval=-3.2, maxval=-0.08)
    x_new = jax.random.uniform(k_new_x, (N_NEW, 1), minval=0.28, maxval=3.2)
    y_old = old_task_function(x_old[:, 0]) + OLD_NOISE_STD * jax.random.normal(k_old_y, (N_OLD,))
    y_new = new_task_function(x_new[:, 0]) + NEW_NOISE_STD * jax.random.normal(k_new_y, (N_NEW,))
    return x_old, y_old, x_new, y_new


def init_mlp_params(key: jax.Array) -> dict[str, jnp.ndarray]:
    hidden_dim1, hidden_dim2, hidden_dim3 = HIDDEN_DIMS
    k1, k2, k3, k4 = jax.random.split(key, 4)
    return {
        "w1": INIT_SCALE * jax.random.normal(k1, (1, hidden_dim1)),
        "b1": jnp.zeros((hidden_dim1,)),
        "w2": INIT_SCALE * jax.random.normal(k2, (hidden_dim1, hidden_dim2)),
        "b2": jnp.zeros((hidden_dim2,)),
        "w3": INIT_SCALE * jax.random.normal(k3, (hidden_dim2, hidden_dim3)),
        "b3": jnp.zeros((hidden_dim3,)),
        "w4": INIT_SCALE * jax.random.normal(k4, (hidden_dim3, 1)),
        "b4": jnp.zeros((1,)),
    }


def mlp_apply(params: dict[str, jnp.ndarray], x: jnp.ndarray) -> jnp.ndarray:
    h1 = jnp.tanh(x @ params["w1"] + params["b1"])
    h2 = jnp.tanh(h1 @ params["w2"] + params["b2"])
    h3 = jnp.tanh(h2 @ params["w3"] + params["b3"])
    y = h3 @ params["w4"] + params["b4"]
    return y[:, 0]


def gaussian_log_likelihood(
    params: dict[str, jnp.ndarray],
    x: jnp.ndarray,
    y: jnp.ndarray,
    noise_std: float,
) -> jnp.ndarray:
    pred = mlp_apply(params, x)
    sq = ((y - pred) / noise_std) ** 2
    n = y.shape[0]
    return -0.5 * jnp.sum(sq) - n * jnp.log(noise_std * jnp.sqrt(2.0 * jnp.pi))


def gaussian_log_prior(params: dict[str, jnp.ndarray], prior_std: float) -> jnp.ndarray:
    sq_sum = sum(jnp.sum(v**2) for v in jax.tree_util.tree_leaves(params))
    return -0.5 * sq_sum / (prior_std**2)


def make_log_posterior_fn(
    x: jnp.ndarray,
    y: jnp.ndarray,
    noise_std: float,
    prior_std: float,
):
    def log_posterior(params: dict[str, jnp.ndarray]) -> jnp.ndarray:
        return gaussian_log_likelihood(params, x, y, noise_std) + gaussian_log_prior(params, prior_std)

    return log_posterior


def clip_tree_by_global_norm(tree: PyTree, max_norm: float) -> PyTree:
    leaves = jax.tree_util.tree_leaves(tree)
    global_norm = jnp.sqrt(sum(jnp.sum(x**2) for x in leaves) + 1e-12)
    scale = jnp.minimum(1.0, max_norm / global_norm)
    return jax.tree_util.tree_map(lambda x: x * scale, tree)


def tree_normal_like(key: jax.Array, tree: PyTree) -> PyTree:
    leaves, treedef = jax.tree_util.tree_flatten(tree)
    keys = jax.random.split(key, len(leaves))
    noise_leaves = [jax.random.normal(k, leaf.shape, leaf.dtype) for k, leaf in zip(keys, leaves)]
    return jax.tree_util.tree_unflatten(treedef, noise_leaves)


def map_optimize(
    params_init: dict[str, jnp.ndarray],
    log_posterior_fn,
    n_steps: int,
    lr_start: float = 7e-3,
    lr_end: float = 2e-4,
) -> dict[str, jnp.ndarray]:
    lrs = jnp.linspace(lr_start, lr_end, n_steps)

    def step(params, lr):
        grads = jax.grad(lambda p: -log_posterior_fn(p))(params)
        grads = clip_tree_by_global_norm(grads, max_norm=25.0)
        params = jax.tree_util.tree_map(lambda p, g: p - lr * g, params, grads)
        return params, None

    params, _ = jax.lax.scan(step, params_init, lrs)
    return params


def sghmc_sample_posterior(
    key: jax.Array,
    params_init: dict[str, jnp.ndarray],
    log_posterior_fn,
    n_particles: int,
    burn_in: int,
    thin: int,
    step_size: float = 2e-5,
    friction: float = 0.08,
    temperature: float = 0.2,
) -> dict[str, jnp.ndarray]:
    n_steps = burn_in + n_particles * thin
    momentum0 = jax.tree_util.tree_map(jnp.zeros_like, params_init)
    noise_scale = jnp.sqrt(2.0 * friction * step_size * temperature)

    def step(carry, _):
        key, params, momentum = carry
        key, key_noise = jax.random.split(key)
        grad_u = jax.grad(lambda p: -log_posterior_fn(p))(params)
        grad_u = clip_tree_by_global_norm(grad_u, max_norm=40.0)
        noise = tree_normal_like(key_noise, params)
        momentum = jax.tree_util.tree_map(
            lambda m, g, n: (1.0 - friction) * m - step_size * g + noise_scale * n,
            momentum,
            grad_u,
            noise,
        )
        params = jax.tree_util.tree_map(lambda p, m: p + m, params, momentum)
        return (key, params, momentum), params

    (_, _, _), chain = jax.lax.scan(step, (key, params_init, momentum0), None, length=n_steps)
    idx = jnp.arange(burn_in, n_steps, thin)[:n_particles]
    return jax.tree_util.tree_map(lambda x: x[idx], chain)


batched_predict = jax.vmap(mlp_apply, in_axes=(0, None))


def optimize_particles(
    params_init: dict[str, jnp.ndarray],
    loss_fn,
    n_steps: int,
    lr: float = 3.5e-3,
) -> dict[str, jnp.ndarray]:
    def step(params, _):
        grads = jax.grad(loss_fn)(params)
        grads = clip_tree_by_global_norm(grads, max_norm=40.0)
        params = jax.tree_util.tree_map(lambda p, g: p - lr * g, params, grads)
        return params, None

    params_final, _ = jax.lax.scan(step, params_init, None, length=n_steps)
    return params_final


def adapt_particles(
    params_init: dict[str, jnp.ndarray],
    x_new: jnp.ndarray,
    y_new: jnp.ndarray,
    noise_std: float,
    n_steps: int,
    lr: float = 3.5e-3,
) -> dict[str, jnp.ndarray]:
    def loss_fn(params):
        pred_new = batched_predict(params, x_new)
        return 0.5 * jnp.mean(((pred_new - y_new[None, :]) / noise_std) ** 2)

    return optimize_particles(params_init, loss_fn, n_steps=n_steps, lr=lr)


def adapt_particles_with_kl(
    params_init: dict[str, jnp.ndarray],
    x_new: jnp.ndarray,
    y_new: jnp.ndarray,
    x_kl: jnp.ndarray,
    noise_std: float,
    kl_std: float,
    kl_weight: float,
    n_steps: int,
    lr: float = 3.5e-3,
) -> dict[str, jnp.ndarray]:
    ref_pred_kl = batched_predict(params_init, x_kl)
    ref_mean = jnp.mean(ref_pred_kl, axis=0)
    ref_std = jnp.std(ref_pred_kl, axis=0) + kl_std

    def loss_fn(params):
        pred_new = batched_predict(params, x_new)
        nll = 0.5 * jnp.mean(((pred_new - y_new[None, :]) / noise_std) ** 2)
        pred_kl = batched_predict(params, x_kl)
        cur_mean = jnp.mean(pred_kl, axis=0)
        cur_std = jnp.std(pred_kl, axis=0) + kl_std
        kl = jnp.log(ref_std / cur_std) + (cur_std**2 + (cur_mean - ref_mean) ** 2) / (2.0 * ref_std**2) - 0.5
        return nll + kl_weight * jnp.mean(kl)

    return optimize_particles(params_init, loss_fn, n_steps=n_steps, lr=lr)


def predictive_interval(params_particles: dict[str, jnp.ndarray], x_grid: jnp.ndarray) -> jnp.ndarray:
    preds = batched_predict(params_particles, x_grid)
    return jnp.quantile(preds, jnp.array([0.25, 0.75]), axis=0)


def predictive_median(params_particles: dict[str, jnp.ndarray], x_grid: jnp.ndarray) -> jnp.ndarray:
    preds = batched_predict(params_particles, x_grid)
    return jnp.median(preds, axis=0)


def flatten_chain_particles(chains: dict[str, jnp.ndarray]) -> dict[str, jnp.ndarray]:
    return jax.tree_util.tree_map(lambda x: x.reshape((-1,) + x.shape[2:]), chains)


def as_numpy(value: jnp.ndarray) -> np.ndarray:
    return np.asarray(jax.device_get(value))


def compute_plot_data() -> dict[str, np.ndarray]:
    key = jax.random.PRNGKey(SEED)
    k_data, k_init, k_sghmc = jax.random.split(key, 3)
    x_old, y_old, x_new, y_new = generate_toy_data(k_data)

    log_posterior_old = make_log_posterior_fn(
        x_old,
        y_old,
        noise_std=OLD_NOISE_STD,
        prior_std=PRIOR_STD,
    )
    params_init = init_mlp_params(k_init)

    @jax.jit
    def run_map(init_params):
        return map_optimize(init_params, log_posterior_old, n_steps=MAP_STEPS)

    @jax.jit
    def run_sghmc(key_sghmc, params_start):
        return sghmc_sample_posterior(
            key_sghmc,
            params_start,
            log_posterior_old,
            n_particles=N_PARTICLES,
            burn_in=SGHMC_BURN_IN,
            thin=SGHMC_THIN,
            temperature=SGHMC_TEMPERATURE,
        )

    params_map = run_map(params_init)
    chain_keys = jax.random.split(k_sghmc, N_CHAINS)
    chain_particles = jax.vmap(run_sghmc, in_axes=(0, None))(chain_keys, params_map)
    posterior_particles_old = flatten_chain_particles(chain_particles)

    adapt_jit = jax.jit(adapt_particles, static_argnames=("n_steps",))
    adapt_with_kl_jit = jax.jit(adapt_particles_with_kl, static_argnames=("n_steps",))
    finetuned_particles = adapt_jit(
        posterior_particles_old,
        x_new,
        y_new,
        NEW_NOISE_STD,
        n_steps=ADAPT_STEPS,
    )
    kl_finetuned_particles = adapt_with_kl_jit(
        posterior_particles_old,
        x_new,
        y_new,
        x_old,
        NEW_NOISE_STD,
        KL_STD,
        KL_WEIGHT,
        n_steps=ADAPT_STEPS,
    )

    x_grid = jnp.linspace(X_MIN, X_MAX, GRID_POINTS)[:, None]
    old_curve = predictive_median(posterior_particles_old, x_grid)
    finetuned_curve = predictive_median(finetuned_particles, x_grid)
    kl_finetuned_curve = predictive_median(kl_finetuned_particles, x_grid)
    finetuned_interval = predictive_interval(finetuned_particles, x_grid)
    kl_finetuned_interval = predictive_interval(kl_finetuned_particles, x_grid)

    return {
        "curves": np.column_stack(
            [
                as_numpy(x_grid[:, 0]),
                as_numpy(old_curve),
                as_numpy(finetuned_curve),
                as_numpy(kl_finetuned_curve),
                as_numpy(finetuned_interval[0]),
                as_numpy(finetuned_interval[1]),
                as_numpy(kl_finetuned_interval[0]),
                as_numpy(kl_finetuned_interval[1]),
            ]
        ),
        "old_points": np.column_stack([as_numpy(x_old[:, 0]), as_numpy(y_old)]),
        "new_points": np.column_stack([as_numpy(x_new[:, 0]), as_numpy(y_new)]),
    }


def write_array_csv(path: Path, header: list[str], values: np.ndarray) -> None:
    np.savetxt(path, values, delimiter=",", header=",".join(header), comments="", fmt="%.17g")


def main() -> None:
    arrays = compute_plot_data()
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    curves_path = DATA_DIR / "curves.csv"
    old_points_path = DATA_DIR / "old_points.csv"
    new_points_path = DATA_DIR / "new_points.csv"

    write_array_csv(
        curves_path,
        [
            "x",
            "old_curve",
            "finetuned_curve",
            "kl_finetuned_curve",
            "finetuned_q25",
            "finetuned_q75",
            "kl_finetuned_q25",
            "kl_finetuned_q75",
        ],
        arrays["curves"],
    )
    write_array_csv(old_points_path, ["x", "y"], arrays["old_points"])
    write_array_csv(new_points_path, ["x", "y"], arrays["new_points"])

    for path in (curves_path, old_points_path, new_points_path):
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
