import jax
import jax.numpy as jnp
import optax


def cross_entropy(logits, labels):
    _, _, vocab_size = logits.shape
    log_softmax = jax.nn.log_softmax(logits.astype(jnp.float32))
    one_hot = jax.nn.one_hot(labels, vocab_size)
    return -jnp.sum(one_hot * log_softmax, axis=-1)


def loss_fn_ntp(forward, weights, x, reduce=True):
    y = jnp.roll(x, -1, axis=1)
    logits, _ = forward(x, weights)
    loss = cross_entropy(logits, y)
    loss = loss.at[:, -1].set(0)
    return loss.mean() if reduce else loss


def loss_fn_kl(forward_theta, weights_theta, forward_pi, weights_pi, x):
    y = jnp.roll(x, -1, axis=1)
    logits_theta, _ = forward_theta(x, weights_theta)
    logits_pi, _ = forward_pi(x, weights_pi)
    logprobs_pi = jax.nn.log_softmax(logits_pi.astype(jnp.float32))
    logprobs_theta = jax.nn.log_softmax(logits_theta.astype(jnp.float32))
    kl = optax.losses.kl_divergence_with_log_targets(logprobs_theta, logprobs_pi)
    kl = kl.at[:, -1].set(0).mean()
    return kl


def loss_fn_reg(forward_theta, weights_theta, forward_pi, weights_pi, x_train, x_reg, method):
    match method:
        case None:
            return 0
        case "ntp_pre":
            return loss_fn_ntp(forward_theta, weights_theta, x_reg)
        case "kl_dwn":
            return loss_fn_kl(forward_theta, weights_theta, forward_pi, weights_pi, x_train)
        case "kl_pre":
            return loss_fn_kl(forward_theta, weights_theta, forward_pi, weights_pi, x_reg)
        case _:
            raise ValueError(f"Unknown regularization method: {method}")
