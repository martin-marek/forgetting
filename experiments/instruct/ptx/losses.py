from functools import partial

import jax
import jax.numpy as jnp
import optax


def get_batch_targets(tokens, mask):
    y = jnp.roll(tokens, -1, axis=1)
    mask = jnp.roll(mask, -1, axis=1).at[:, -1].set(False)
    return tokens, y, mask


def cross_entropy(logits, labels):
    # Works with explicit axes, unlike the optax version.
    _, _, vocab_size = logits.shape
    log_softmax = jax.nn.log_softmax(logits.astype(jnp.float32))
    one_hot = jax.nn.one_hot(labels, vocab_size)
    return -jnp.sum(one_hot * log_softmax, axis=-1)


@partial(jax.jit, static_argnames="forward")
def loss_fn_ntp(forward, weights, batch):
    x, y, mask = get_batch_targets(batch["tokens"], batch["mask"])
    logits = forward(x, weights)
    loss = cross_entropy(logits, y)
    loss_sum = (loss * mask).sum()
    num_tokens = mask.sum()
    avg_loss = loss_sum / num_tokens
    return avg_loss, loss_sum, num_tokens


@partial(jax.jit, static_argnames="forward")
def loss_fn_kl(forward, weights_theta, weights_pi, batch):
    x, _, mask = get_batch_targets(batch["tokens"], batch["mask"])
    logits_theta = forward(x, weights_theta)
    logits_pi = forward(x, weights_pi)
    logprobs_pi = jax.nn.log_softmax(logits_pi.astype(jnp.float32))
    logprobs_theta = jax.nn.log_softmax(logits_theta.astype(jnp.float32))
    kl = optax.losses.kl_divergence_with_log_targets(logprobs_theta, logprobs_pi)
    loss_sum = (kl * mask).sum()
    num_tokens = mask.sum()
    avg_loss = loss_sum / num_tokens
    return avg_loss, loss_sum, num_tokens


def loss_fn_reg(forward, weights_theta, weights_pi, batch_train, batch_reg, reg, dtype=jnp.float32):
    if reg.method is None or reg.coeff == 0:
        return jnp.array(0.0, dtype=dtype)
    if reg.method == "ntp_pre":
        return loss_fn_ntp(forward, weights_theta, batch_reg)[0]
    if reg.method == "kl_dwn":
        return loss_fn_kl(forward, weights_theta, weights_pi, batch_train)[0]
    if reg.method == "kl_pre":
        return loss_fn_kl(forward, weights_theta, weights_pi, batch_reg)[0]
    raise ValueError(f"Unknown regularization method: {reg.method}")

