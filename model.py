from functools import partial

import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P


def rms_norm(x, eps=1e-6):
    return (x / jnp.sqrt(jnp.mean(x.astype(jnp.float32) ** 2, axis=-1, keepdims=True) + eps)).astype(x.dtype)


def apply_rope(x, pos=0):
    h = x.shape[-1]
    pos = pos + jnp.broadcast_to(jnp.arange(x.shape[1])[None], x.shape[:2])
    freq = 1.0 / (10_000 ** (jnp.arange(0, h, 2, dtype=jnp.float32) / h))
    ang = jnp.einsum("bt,h->bth", pos, freq, precision=jax.lax.Precision.HIGHEST)
    sin, cos = jnp.sin(ang).astype(x.dtype)[:, :, None, :], jnp.cos(ang).astype(x.dtype)[:, :, None, :]
    x1, x2 = jnp.split(x, 2, axis=-1)
    return jnp.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin], axis=-1)


def block_forward(h, block, kv=None, pos=0, dtype=None):
    q, k, v = jnp.einsum("btd,sndh->sbtnh", rms_norm(h), block["qkv"], preferred_element_type=dtype, out_sharding=P(None, "data", None, "model", None))
    q, k = apply_rope(rms_norm(q), pos), apply_rope(rms_norm(k), pos)
    if kv is not None:
        kv = jax.lax.dynamic_update_slice(kv, jnp.stack([k, v]), (0, 0, pos, 0, 0))
        k, v = kv
    attn = jax.nn.dot_product_attention(q, k, v, mask=None if kv is None else (jnp.arange(k.shape[1]) <= pos)[None, None], is_causal=kv is None)
    o = jnp.einsum("btnh,nhd->btd", attn, block["out"], preferred_element_type=dtype, out_sharding=P("data", None, None))
    h += o
    up_act = jax.nn.gelu(jnp.einsum("btd,df->btf", rms_norm(h), block["up"], preferred_element_type=dtype, out_sharding=P("data", None, "model")))
    down_proj = jnp.einsum("btf,fd->btd", up_act, block["down"], preferred_element_type=dtype, out_sharding=P("data", None, None))
    return h + down_proj, kv


def forward(cfg, x, weights, kv=None, pos=0):  # [B, T]
    dtype = jnp.dtype(cfg.activ_dtype)
    weights = jax.tree.map(lambda w: w.astype(dtype), weights)
    h = weights["embed_in"].at[x, :].get(out_sharding=P("data", None, None)).astype(dtype)  # [B, T, D]
    layer_weights = {k:v for k, v in weights.items() if not 'embed' in k}
    block_fn = partial(block_forward, dtype=dtype)
    if cfg.remat: block_fn = jax.remat(block_fn)
    for i in range(cfg.L):
        h, kv_i = block_fn(h, {k: v[i] for k, v in layer_weights.items()}, None if kv is None else kv[i], pos)
        if kv is not None: kv[i] = kv_i

    logits = jnp.einsum("btd,vd->btv", rms_norm(h), weights["embed_out"], preferred_element_type=dtype, out_sharding=P("data", None, "model"))
    return logits, kv


def init_kv(cfg, B, T):
    N = cfg.D // cfg.H
    sharding = P(None, "data", None, "model", None)
    dtype = jnp.dtype(cfg.activ_dtype)
    return [jnp.zeros((2, B, T, N, cfg.H), dtype=dtype, out_sharding=sharding) for _ in range(cfg.L)]


def create_sharded_model(cfg, key):
    D, H, L, V = cfg.D, cfg.H, cfg.L, cfg.V
    F, N = 4 * D, D // H
    data_shard = "data" if cfg.dp_shard else None

    def init(shape, spec, scale):
        nonlocal key
        key, subkey = jax.random.split(key)
        return jax.device_put(scale * jax.random.normal(subkey, shape, jnp.float32), spec)

    return {
        "embed_in": init((V, D), P("model", data_shard), D ** -0.5),
        "embed_out": init((V, D), P("model", data_shard), D ** -0.5),
        "qkv": init((L, 3, N, D, H), P(None, None, "model", data_shard, None), D ** -0.5),
        "out": init((L, N, H, D), P(None, "model", None, data_shard), D ** -0.5),
        "up": init((L, D, F), P(None, data_shard, "model"), D ** -0.5),
        "down": init((L, F, D), P(None, "model", data_shard), F ** -0.5),
    }
