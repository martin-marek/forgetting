import json
import math
from collections import defaultdict
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Callable

import jax
import jax.numpy as jnp
import numpy as np
from huggingface_hub import snapshot_download
from jax.sharding import AxisType, PartitionSpec as P, reshard
from safetensors import safe_open
from safetensors.numpy import save_file
from transformers import AutoTokenizer


@dataclass
class Model:
    weights: dict[str, Any]
    forward: Callable
    init_kv: Callable
    tokenizer: Any
    config: dict[str, Any]
    save: Callable


def apply_rope(x, theta, pos=0):
    B, T, N, H = x.shape
    positions = pos + jnp.broadcast_to(jnp.arange(T)[None, :], [B, T])
    freq = 1.0 / (theta ** (jnp.arange(0, H, 2, dtype=jnp.float32) / H))
    inp = jnp.einsum("bt,h->bth", positions, freq, precision=jax.lax.Precision.HIGHEST)
    sin, cos = jnp.sin(inp).astype(x.dtype), jnp.cos(inp).astype(x.dtype)
    x1, x2 = x[:, :, :, :H // 2], x[:, :, :, H // 2:]
    sin, cos = sin[:, :, None, :], cos[:, :, None, :]
    return jnp.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin], axis=-1)


def rms_norm(x, gamma, eps, axis=-1):
    rms = jnp.sqrt(jnp.mean(x.astype(jnp.float32) ** 2, axis=axis, keepdims=True) + eps)
    return (gamma * x / rms).astype(x.dtype)


def forward_layer(cfg, x, w, kv=None, pos=0):
    x_norm = rms_norm(x, w["input_layernorm.weight"], cfg["rms_norm_eps"])

    q = jnp.einsum("btd,nhd->btnh", x_norm, w["self_attn.q_proj.weight"], preferred_element_type=x.dtype, out_sharding=P("data", None, "model", None))
    k = jnp.einsum("bsd,khd->bskh", x_norm, w["self_attn.k_proj.weight"], preferred_element_type=x.dtype, out_sharding=P("data", None, "model", None))
    v = jnp.einsum("bsd,khd->bskh", x_norm, w["self_attn.v_proj.weight"], preferred_element_type=x.dtype, out_sharding=P("data", None, "model", None))

    if "self_attn.q_proj.bias" in w:
        q += w["self_attn.q_proj.bias"]
        k += w["self_attn.k_proj.bias"]
        v += w["self_attn.v_proj.bias"]

    if "self_attn.q_norm.weight" in w:
        q = rms_norm(q, w["self_attn.q_norm.weight"], cfg["rms_norm_eps"])
        k = rms_norm(k, w["self_attn.k_norm.weight"], cfg["rms_norm_eps"])

    q = apply_rope(q, cfg["rope_theta"], pos)
    k = apply_rope(k, cfg["rope_theta"], pos)

    if kv is not None:
        kv = jax.lax.dynamic_update_slice(kv, jnp.stack([k, v]), (0, 0, pos, 0, 0))
        k, v = kv

    attn_mask = jnp.tri(q.shape[1], dtype=bool)[None] if kv is None else (jnp.arange(k.shape[1]) <= pos)[None, None]
    attn_out = jax.nn.dot_product_attention(q, k, v, mask=attn_mask)
    x += jnp.einsum("btnh,dnh->btd", attn_out, w["self_attn.o_proj.weight"], preferred_element_type=x.dtype, out_sharding=P("data", None, None, None))

    x_norm = rms_norm(x, w["post_attention_layernorm.weight"], cfg["rms_norm_eps"])
    gate = jax.nn.silu(jnp.einsum("btd,fd->btf", x_norm, w["mlp.gate_proj.weight"], preferred_element_type=jnp.float32, out_sharding=P("data", None, "model")))
    up = jnp.einsum("btd,fd->btf", x_norm, w["mlp.up_proj.weight"], preferred_element_type=x.dtype, out_sharding=P("data", None, "model"))
    x += jnp.einsum("btf,df->btd", gate * up, w["mlp.down_proj.weight"], preferred_element_type=x.dtype, out_sharding=P("data", None, None))
    return x, kv


def forward(cfg, x, weights, kv=None, pos=0):
    weights = jax.tree.map(lambda w: w.astype(jnp.bfloat16) if w.ndim > 1 else w, weights)
    x = reshard(x, P("data", None))
    x = weights["model.embed_tokens.weight"].at[x, :].get(out_sharding=P("data", None, None)).astype(jnp.bfloat16)

    return_kv = kv is not None
    if kv is None: kv = defaultdict(lambda: None)
    for i in range(cfg["num_hidden_layers"]):
        layer_weights = {k.replace(prefix, ""): v for k, v in weights.items() if (prefix := f"model.layers.{i}.") in k}
        x, kv[i] = jax.remat(partial(forward_layer, cfg))(x, layer_weights, kv[i], pos)

    out_embed = weights["model.embed_tokens.weight"] if cfg["tie_word_embeddings"] else weights["lm_head.weight"]
    x = rms_norm(x, weights["model.norm.weight"], cfg["rms_norm_eps"])
    logits = jnp.einsum("btd,vd->btv", x, out_embed, preferred_element_type=x.dtype, out_sharding=P("data", None, "model"))
    return (logits, kv) if return_kv else logits


def init_kv(L, K, H, B, T):
    sharding = P(None, "data", None, "model", None)
    return [jnp.zeros((2, B, T, K, H), dtype=jnp.bfloat16, out_sharding=sharding) for _ in range(L)]


def _head_dim(cfg):
    return cfg.get("head_dim", cfg["hidden_size"] // cfg["num_attention_heads"])


def _load_template(model_id, hf_ckpt_dir, allow_patterns=None):
    source = Path(model_id).expanduser()
    if not source.exists():
        source = Path(hf_ckpt_dir).expanduser() / model_id
        snapshot_download(repo_id=model_id, local_dir=source, allow_patterns=allow_patterns)
    cfg = json.loads((source / "config.json").read_text())
    if cfg.get("model_type") != "qwen3":
        raise ValueError(f"Only dense Qwen3 checkpoints are supported, got model_type={cfg.get('model_type')}")
    return source, AutoTokenizer.from_pretrained(source), cfg


def _make_random_cfg(template_cfg, hidden_size=None, num_hidden_layers=None):
    hidden_size = template_cfg["hidden_size"] if hidden_size is None else int(hidden_size)
    num_hidden_layers = template_cfg["num_hidden_layers"] if num_hidden_layers is None else int(num_hidden_layers)
    if hidden_size < 128 or hidden_size % 128 != 0:
        raise ValueError(f"hidden_size must be a positive multiple of 128, got {hidden_size}")
    if num_hidden_layers < 1:
        raise ValueError(f"num_hidden_layers must be >= 1, got {num_hidden_layers}")

    cfg = dict(template_cfg)
    cfg["hidden_size"] = hidden_size
    cfg["num_hidden_layers"] = num_hidden_layers
    cfg["num_attention_heads"] = hidden_size // 128
    cfg["num_key_value_heads"] = math.gcd(cfg["num_attention_heads"], 8)
    cfg["intermediate_size"] = 3 * hidden_size
    if "max_window_layers" in cfg: cfg["max_window_layers"] = min(cfg["max_window_layers"], num_hidden_layers)
    if "layer_types" in cfg: cfg["layer_types"] = ["full_attention"] * num_hidden_layers
    return cfg


def _get_sharding(key, dp_shard=False):
    if "norm" in key or "bias" in key: return P()
    if any(k in key for k in ("q_proj", "k_proj", "v_proj")):
        return P("model", None, "data") if dp_shard else P("model", None, None)
    if "o_proj" in key:
        return P("data", "model", None) if dp_shard else P(None, "model", None)
    if any(k in key for k in ("gate_proj", "up_proj", "embed_tokens", "lm_head")):
        return P("model", "data") if dp_shard else P("model")
    if "down_proj" in key:
        return P("data", "model") if dp_shard else P(None, "model")
    raise ValueError(f"Unrecognized key: {key}")


def _load_weights(model_ckpt_dir, cfg, dp_shard=False):
    N, K, D, H = cfg["num_attention_heads"], cfg["num_key_value_heads"], cfg["hidden_size"], _head_dim(cfg)
    weights = {}
    for file in sorted(model_ckpt_dir.glob("*.safetensors")):
        with safe_open(file, framework="numpy") as f:
            for key in f.keys():
                value = f.get_tensor(key)
                if "q_proj.bias" in key: value = value.reshape([N, H])
                if "k_proj.bias" in key or "v_proj.bias" in key: value = value.reshape([K, H])
                if "q_proj.weight" in key: value = value.reshape([N, H, D])
                if "k_proj.weight" in key or "v_proj.weight" in key: value = value.reshape([K, H, D])
                if "o_proj.weight" in key: value = value.reshape([D, N, H])
                weights[key] = jax.device_put(value, _get_sharding(key, dp_shard))
    if not weights:
        raise ValueError(f"No safetensors files found in {model_ckpt_dir}")
    return weights


def _random_weights(cfg, key, dp_shard=False):
    std = cfg["initializer_range"]
    D, F, N, K, H, V = cfg["hidden_size"], cfg["intermediate_size"], cfg["num_attention_heads"], cfg["num_key_value_heads"], _head_dim(cfg), cfg["vocab_size"]
    weights = {}

    def add(name, shape, ones=False):
        nonlocal key
        if ones:
            value = jnp.ones(shape, dtype=jnp.bfloat16)
        else:
            key, subkey = jax.random.split(key)
            value = (jax.random.normal(subkey, shape, dtype=jnp.float32) * std).astype(jnp.bfloat16)
        weights[name] = jax.device_put(value, _get_sharding(name, dp_shard))

    add("model.embed_tokens.weight", (V, D))
    add("model.norm.weight", (D,), ones=True)
    if not cfg["tie_word_embeddings"]: add("lm_head.weight", (V, D))

    specs = [
        ("input_layernorm.weight", (D,), True),
        ("post_attention_layernorm.weight", (D,), True),
        ("self_attn.q_norm.weight", (H,), True),
        ("self_attn.k_norm.weight", (H,), True),
        ("self_attn.q_proj.weight", (N, H, D), False),
        ("self_attn.k_proj.weight", (K, H, D), False),
        ("self_attn.v_proj.weight", (K, H, D), False),
        ("self_attn.o_proj.weight", (D, N, H), False),
        ("mlp.gate_proj.weight", (F, D), False),
        ("mlp.up_proj.weight", (F, D), False),
        ("mlp.down_proj.weight", (D, F), False),
    ]
    for i in range(cfg["num_hidden_layers"]):
        prefix = f"model.layers.{i}."
        for name, shape, ones in specs: add(prefix + name, shape, ones)
    return weights


def load(model_id="Qwen/Qwen3-0.6B-Base", hf_ckpt_dir="/dev/shm/ptx/weights", tp_size=1, dp_shard=False, init="pretrained", hidden_size=None, num_hidden_layers=None, key=None):
    if jax.device_count() % tp_size != 0:
        raise ValueError(f"tp_size={tp_size} does not divide device_count={jax.device_count()}")
    mesh = jax.make_mesh((jax.device_count() // tp_size, tp_size), ("data", "model"), axis_types=(AxisType.Explicit, AxisType.Explicit))
    jax.set_mesh(mesh)
    has_architecture_override = hidden_size is not None or num_hidden_layers is not None

    if init == "pretrained":
        if has_architecture_override:
            raise ValueError("hidden_size and num_hidden_layers overrides are only supported when init='random'")
        model_ckpt_dir, tokenizer, cfg = _load_template(model_id, hf_ckpt_dir)
        return Model(_load_weights(model_ckpt_dir, cfg, dp_shard), partial(forward, cfg), partial(init_kv, cfg["num_hidden_layers"], cfg["num_key_value_heads"], _head_dim(cfg)), tokenizer, cfg, save)

    if init == "random":
        if key is None:
            raise ValueError("Random initialization requires a JAX PRNG key")
        _, tokenizer, template_cfg = _load_template(model_id, hf_ckpt_dir, ["*.json", "*.txt", "*.jinja"])
        cfg = _make_random_cfg(template_cfg, hidden_size=hidden_size, num_hidden_layers=num_hidden_layers)
        return Model(_random_weights(cfg, key, dp_shard), partial(forward, cfg), partial(init_kv, cfg["num_hidden_layers"], cfg["num_key_value_heads"], _head_dim(cfg)), tokenizer, cfg, save)

    raise ValueError(f"Unknown init mode: {init}")


def save(model, output_dir):
    output_dir = Path(output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    hf = {}
    for k, v in model.weights.items():
        v = np.asarray(jax.device_get(jnp.asarray(v, dtype=jnp.bfloat16)))
        if "q_proj" in k or "k_proj" in k or "v_proj" in k: v = v.reshape(-1, v.shape[-1]) if v.ndim == 3 else v.reshape(-1)
        if "o_proj.weight" in k: v = v.reshape(v.shape[0], -1)
        hf[k] = v

    save_file(hf, str(output_dir / "model.safetensors"))
    (output_dir / "config.json").write_text(json.dumps(model.config, indent=2) + "\n")
    model.tokenizer.save_pretrained(output_dir)
