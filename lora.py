import math
import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P


def apply_lora(w_base, w_lora):
    w_merged = dict(w_base)
    for k, (a, b) in w_lora.items():
        delta = jnp.einsum("...ir,...rj->...ij", a, b)
        w_merged[k] = (w_merged[k] + delta).astype(w_merged[k].dtype)
    return w_merged


def make_lora(forward_fn, weights, rank=16, key=None):
    key = jax.random.key(0) if key is None else key
    weights_lora = {}
    for k, w in weights.items():
        if "embed" in k: continue
        key, subkey = jax.random.split(key)
        weights_lora[k] = _init_lora_weights(k, w, rank, subkey)

    def fwd_lora(x, weights_lora):
        return forward_fn(x, apply_lora(weights, weights_lora))

    return fwd_lora, weights_lora


def _init_lora_weights(name, w, rank, key):
    *prefix, in_dim, out_dim = w.shape
    shared_prefix = (prefix[0],) + (1,) * (len(prefix) - 1)
    # if the model einsum has multiple input axes, share B over them; if it has multiple output axes, share A over them
    a_prefix = shared_prefix if name == "qkv" else prefix  # qkv: D -> S*N*H
    b_prefix = shared_prefix if name == "out" else prefix  # out: N*H -> D
    init_std = (math.prod(a_prefix[1:]) * in_dim) ** -0.5
    a = jax.random.normal(key, (*a_prefix, in_dim, rank)) * init_std
    b = jnp.zeros((*b_prefix, rank, out_dim))
    return a, b
