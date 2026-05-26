from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P, reshard


class SamplingState(NamedTuple):
    step: int
    key: jax.Array
    tokens: jax.Array
    kv: list[jax.Array]
    done: jax.Array


def _sample_step(state, forward, weights, temperature, pad_id, eos_id):
    key, key_sampling = jax.random.split(state.key)
    input_token = state.tokens[:, state.step, None]
    logits, kv = forward(input_token, weights, state.kv, state.step)
    sample_logits = logits[:, 0, :].astype(jnp.float32) / temperature
    sampled_token = jax.random.categorical(key_sampling, sample_logits)
    next_token = state.tokens[:, state.step + 1]
    update_token = jnp.where((~state.done) & (next_token == pad_id), sampled_token, next_token)
    tokens = state.tokens.at[:, state.step + 1].set(update_token)
    done = state.done | ((next_token == pad_id) & (sampled_token == eos_id))
    return SamplingState(state.step + 1, key, tokens, kv, done)


@partial(jax.jit, static_argnames=("forward", "init_kv"))
def sample(key, forward, init_kv, weights, tokens, eos_id, pad_id=0, temperature=1):
    B, T = tokens.shape
    tokens = reshard(tokens, P("data", None))
    state = SamplingState(
        step=0,
        key=key,
        tokens=tokens,
        kv=init_kv(B, T),
        done=jnp.zeros((B,), dtype=bool, out_sharding=P("data")),
    )
    step_fn = lambda state: _sample_step(state, forward, weights, temperature, pad_id, eos_id)
    cond_fn = lambda state: (state.step < T - 1) & jnp.any(~state.done)
    state = jax.lax.while_loop(cond_fn, step_fn, state)
    return state.tokens
