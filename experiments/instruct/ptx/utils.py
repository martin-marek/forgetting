import json
import operator as op
from pathlib import Path

import jax
import jax.numpy as jnp
from omegaconf import OmegaConf
import optax
import wandb
from jax.sharding import PartitionSpec as P

import timing
import models
from models.sampling import sample


def save_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, sort_keys=True, default=str)
        f.write("\n")


def load_training_model(config, seed=0):
    key = jax.random.key(seed)
    _, key_model = jax.random.split(key, 2)
    with timing.context("load model"):
        model = models.load(
            str(config.source),
            config.weights_dir,
            tp_size=config.tp_size,
            dp_shard=config.dp_shard,
            init=config.init,
            hidden_size=config.hidden_size,
            num_hidden_layers=config.num_hidden_layers,
            key=key_model,
        )
        weights_pi = jax.tree.map(jnp.copy, model.weights)
        model.weights = jax.tree.map(lambda x: x.astype(jnp.float32), model.weights)
        model.forward = jax.jit(model.forward)
        n_params = jax.tree.reduce_associative(op.add, jax.tree.map(lambda x: x.size, model.weights))
        print(f"{n_params=:_}")
    return model, weights_pi


def sample_synthetic_lm_batch(key, model, weights_pi, batch_size, seq_len):
    if model.config.get("model_type") != "llama":
        raise ValueError("reg.synth currently supports Llama models only")
    if batch_size is None or seq_len is None:
        raise ValueError("reg.synth requires reg.batch_size and reg.seq_len")
    bos_id = model.tokenizer.bos_token_id
    if bos_id is None:
        raise ValueError("reg.synth requires tokenizer.bos_token_id")

    tokens = jnp.full((batch_size, seq_len), model.tokenizer.pad_token_id, dtype=jnp.int32)
    tokens = tokens.at[:, 0].set(bos_id)
    tokens = sample(key, model, tokens, weights=weights_pi, eos_id=-1)
    return {"tokens": tokens, "mask": jnp.ones_like(tokens, dtype=bool)}


def sample_synthetic_lm_batches(key, model, weights_pi, batch_size, seq_len, num_batches):
    batch = sample_synthetic_lm_batch(key, model, weights_pi, batch_size * num_batches, seq_len)
    return jax.tree.map(
        lambda x: jnp.reshape(x, (num_batches, batch_size, seq_len), out_sharding=P(None, "data", None)),
        batch,
    )


def compute_training_schedule(train_len, train, opt):
    tokens_per_batch = train.batch_size * train.seq_len
    step_limit = None if opt.tokens is None else max(1, (opt.tokens + tokens_per_batch - 1) // tokens_per_batch)
    streaming_train = train_len is None
    if streaming_train:
        if opt.tokens is None:
            raise ValueError("Streaming train datasets require tokens to be set")
        total_steps = step_limit
    else:
        max_batches = train_len // train.batch_size
        total_steps = opt.n_epochs * max_batches if step_limit is None else min(opt.n_epochs * max_batches, step_limit)
    target_tokens = total_steps * tokens_per_batch if opt.tokens is None else min(opt.tokens, total_steps * tokens_per_batch)
    return tokens_per_batch, total_steps, target_tokens, streaming_train


def make_optimizer(opt, total_steps, weights, tokens_per_batch=None):
    warmup_steps = (
        (opt.warmup_tokens + tokens_per_batch - 1) // tokens_per_batch
        if "warmup_tokens" in opt else int(opt.lr_warmup * total_steps)
    )
    lr_schedule = optax.join_schedules(
        [
            optax.linear_schedule(0, opt.lr, warmup_steps),
            optax.constant_schedule(opt.lr),
        ],
        [warmup_steps],
    )
    if opt.optimizer == "sgd":
        optimizer = optax.sgd(lr_schedule)
    elif opt.optimizer == "adamw":
        optimizer = optax.adamw(lr_schedule, 0.9, opt.b2, weight_decay=0.02)
    elif opt.optimizer == "adafactor":
        optimizer = optax.adafactor(lr_schedule, decay_rate=opt.b2)
    else:
        raise ValueError(f"Unknown optimizer: {opt.optimizer}")
    return optimizer, optimizer.init(weights)


def start_run(train_config):
    train_config = OmegaConf.to_container(train_config, resolve=True)
    if jax.process_index() == 0:
        wandb.init(
            config=train_config,
            project=train_config["log"]["project"],
            mode=train_config["log"]["wandb_mode"],
            name=train_config["log"]["run_name"],
        )
        print(f"{wandb.run.id=}")
    resolved_run_name = train_config["log"]["run_name"] or (wandb.run.id if jax.process_index() == 0 else "run")
    run_dir = Path(train_config["log"]["dir"]).expanduser() / train_config["log"]["project"] / resolved_run_name
    if jax.process_index() == 0:
        run_dir.mkdir(parents=True, exist_ok=True)
        save_json(run_dir / "config.json", train_config)
        print(f"run_dir={run_dir}")
    return run_dir


def save_model(model, run_dir, name="final_model"):
    if jax.process_index() == 0:
        output_dir = Path(run_dir) / name
        model.save(model, output_dir)
        print(f"saved_model_dir={output_dir}")


def weight_norm(weights):
    return jax.tree.reduce_associative(op.add, jax.tree.map(lambda w: (w ** 2).sum(), weights)) ** 0.5


def log_metrics(step, metrics, run_dir=None):
    metrics = jax.tree.map(lambda x: x.item() if hasattr(x, "item") else x, metrics)
    if jax.process_index() == 0:
        print(f"Step {step}", metrics)
        wandb.log(metrics, step=step, commit=True)
        if run_dir is not None:
            with open(Path(run_dir) / "logs.jsonl", "a") as f:
                f.write(json.dumps({"step": step, **metrics}, default=str) + "\n")
