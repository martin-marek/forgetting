import math
import operator as op
import sys
import time
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
import wandb
from omegaconf import OmegaConf
from tqdm.auto import tqdm
from jax.sharding import AxisType, PartitionSpec as P

import loss as loss_lib
import model as model_lib
import lora as lora_lib
import sampling
import utils
from data import dataloader


DEFAULT_CFG = """
run:
  seed: 0
  tags: null
data:
  path: null
  split_seed: 0
  num_tokens_valid: 1_000_0000
mix:
  path: null
  coeff: 0
model:
  D: 256
  L: 6
  H: 64
  T: 256
  V: 4096
  lora_rank: null
  remat: false
  dp_shard: false
  tp_size: 1
  activ_dtype: bfloat16
  load_path: null
  load_dir: null
  load_idx: null
  save_dir: null
reg:
  method: null
  coeff: 1
  synth: false
  batch_size: null
  buffer: 1
opt:
  batch_size: 256
  peak_lr: 1e-3
  warmup_tokens: 10_000_000
  lr_decay: null
  optimizer: adamw
  b1: 0.9
  b2: 0.999
  weight_decay: 0.02
stop:
  num_tokens_train: null
  tokens_per_param: null
  num_epochs: null
  target_loss: null
  train_time: null
  early_stop_steps: null
log:
  every_tokens: 10_000_000
  measure_time: false
  project: ptx-mini
  mode: online
"""


def loss_fn_combined(forward_theta, forward_pi, weights_theta, weights_pi, x_train, x_reg=None, reg_method=None, reg_coeff=0):
    loss_ntp = loss_lib.loss_fn_ntp(forward_theta, weights_theta, x_train)
    loss_reg = loss_lib.loss_fn_reg(forward_theta, weights_theta, forward_pi, weights_pi, x_train, x_reg, reg_method)
    loss_optim = loss_ntp + reg_coeff * loss_reg
    return loss_optim, loss_ntp, loss_reg


@partial(jax.jit, static_argnames=("forward_theta", "forward_pi", "tx", "reg_method"), donate_argnames=("weights_theta", "opt_state"))
def train_step(forward_theta, forward_pi, tx, weights_theta, weights_pi, opt_state, x_train, x_reg, reg_method=None, reg_coeff=0):
    def loss_fn(weights_theta):
        loss_optim, loss_ntp, loss_reg = loss_fn_combined(forward_theta, forward_pi, weights_theta, weights_pi, x_train, x_reg, reg_method, reg_coeff)
        return loss_optim, (loss_ntp, loss_reg)
    (loss_optim, (loss_ntp, loss_reg)), grads = jax.value_and_grad(loss_fn, has_aux=True)(weights_theta)
    updates, opt_state = tx.update(grads, opt_state, weights_theta)
    weights_theta = optax.apply_updates(weights_theta, updates)
    return weights_theta, opt_state, loss_optim, loss_ntp, loss_reg

    
@jax.jit
def l2_norm(weights):
    sqnorm = jax.tree.reduce_associative(
        op.add,
        jax.tree.map(lambda w: jnp.square(w.astype(jnp.float32)).sum(), weights),
    )
    return jnp.sqrt(sqnorm)


@partial(jax.jit, static_argnames=("forward_theta", "forward_pi"))
def eval_step(forward_theta, forward_pi, weights_theta, weights_pi, dataset):
    def body(metric_sums, batch):
        loss = loss_lib.loss_fn_ntp(forward_theta, weights_theta, batch, reduce=False)
        loss_h2 = loss[:, loss.shape[1] // 2:]
        kl = loss_lib.loss_fn_kl(forward_theta, weights_theta, forward_pi, weights_pi, batch)
        return metric_sums + jnp.array([loss.mean(), loss_h2.mean(), kl]), None
    metric_sums, _ = jax.lax.scan(body, jnp.zeros(3), dataset)
    return metric_sums / dataset.shape[0]

    
def eval_metrics(forward_theta, forward_pi, weights_theta, weights_pi, weights_init, datasets, lora_rank=None):
    metrics = {}
    for name, dataset in datasets.items():
        ntp, ntp_h2, kl = eval_step(forward_theta, forward_pi, weights_theta, weights_pi, dataset)
        metrics[f"{name}_ntp"] = ntp
        metrics[f"{name}_ntp_h2"] = ntp_h2
        metrics[f"{name}_kl"] = kl
    if lora_rank is not None:
        weights_theta = lora_lib.apply_lora(weights_pi, weights_theta)
    metrics["weights_l2_theta"] = l2_norm(weights_theta)
    weights_pi_init_dist = l2_norm(jax.tree.map(op.sub, weights_pi, weights_init))
    metrics["weights_l2_theta_pi_over_pi_init"] = jnp.where(
        weights_pi_init_dist > 0,
        l2_norm(jax.tree.map(op.sub, weights_theta, weights_pi)) / weights_pi_init_dist,
        jnp.nan,
    )
    return metrics


def sample_synthetic_batch(key, forward, init_kv, weights, batch_size, seq_len, data_token_id):
    tokens = jnp.zeros((batch_size, seq_len), dtype=jnp.int32).at[:, 0].set(data_token_id)
    return sampling.sample(key, forward, init_kv, weights, tokens, eos_id=-1, pad_id=0)


def train_and_evaluate(c):
    c = OmegaConf.merge(OmegaConf.create(DEFAULT_CFG), c)
    run = None
    if jax.process_index() == 0:
        run = wandb.init(project=c.log.project, mode=c.log.mode, tags=c.run.tags)

    # load config of previous model if specified
    c.load_cfg = None
    weights_path = None
    load_path = c.model.load_path
    if c.model.load_dir is not None or c.model.load_idx is not None:
        if load_path is not None:
            raise ValueError("specify either model.load_path or model.load_dir/model.load_idx")
        if c.model.load_dir is None or c.model.load_idx is None:
            raise ValueError("model.load_dir and model.load_idx must be specified together")
        checkpoint_paths = sorted(Path(c.model.load_dir).expanduser().glob("*/weights.npz"))
        load_idx = int(c.model.load_idx)
        if not checkpoint_paths:
            raise ValueError(f"no checkpoints found under {c.model.load_dir}")
        if load_idx < 0 or load_idx >= len(checkpoint_paths):
            raise ValueError(f"model.load_idx={load_idx} is out of range for {c.model.load_dir} ({len(checkpoint_paths)} checkpoints)")
        load_path = str(checkpoint_paths[load_idx].parent)
        c.model.load_path = load_path
    if load_path is not None:
        load_path = Path(load_path).expanduser()
        if not load_path.is_dir():
            raise ValueError("model.load_path must be a checkpoint directory")
        weights_path = load_path / "weights.npz"
        c.load_cfg = OmegaConf.load(load_path / "config.yaml")
        for key in ("D", "L", "H", "T", "V"):
            c.model[key] = c.load_cfg.model[key]

    # upload run config to wandb
    if run is not None:
        run.config.update(utils.flatten_dict(c))

    # get model and dataset rng seed
    key = jax.random.key(c.run.seed)
    key, key_model, key_dataset, key_lora = jax.random.split(key, 4)

    # sharding
    num_fsdp_devices = jax.device_count() // c.model.tp_size
    mesh = jax.make_mesh((num_fsdp_devices, c.model.tp_size), ("data", "model"), axis_types=(AxisType.Explicit, AxisType.Explicit))
    jax.set_mesh(mesh)
    print("sharding mesh:", ", ".join(f"{k}={v}" for k, v in mesh.shape.items()))

    # model
    print("initializing model...")
    c.model.V = math.ceil(c.model.V / jax.device_count()) * jax.device_count()
    weights_init = model_lib.create_sharded_model(c.model, key_model) # random init weights
    weights_pi = jax.tree.map(jnp.copy, weights_init) # weights of teacher
    forward_pi = partial(model_lib.forward, c.model)
    forward_theta = forward_pi
    init_kv = partial(model_lib.init_kv, c.model)
    if weights_path is not None:
        print(f"loading model from {c.model.load_path}...")
        with np.load(weights_path) as weights_npz:
            weights_pi = {k: jax.device_put(weights_npz[k], v.sharding) for k, v in weights_pi.items()}
        print(f"loaded model from {weights_path}")
    if c.model.lora_rank is None:
        weights_theta = jax.tree.map(jnp.copy, weights_pi) # weights of student
    else:
        forward_theta, weights_theta = lora_lib.make_lora(forward_pi, weights_pi, c.model.lora_rank, key_lora)
        print(f"using LoRA with rank={c.model.lora_rank}")

    # get num. model parameters
    n_params = {
        "n_param_nonembed": 12 * c.model.L * c.model.D**2,
        "n_param_embed": c.model.D * c.model.V,
        "n_param_actual": jax.tree.reduce_associative(op.add, jax.tree.map(lambda x: x.size, weights_theta)),
    }
    for k, v in n_params.items():
        print(f"{k}={v:_}")
    if run is not None:
        run.summary.update(n_params)

    use_synth_reg = c.reg.synth and c.reg.method in {"ntp_pre", "kl_pre"}
    reg_data_token_id = None
    if use_synth_reg:
        reg_path = c.load_cfg.data.path if c.load_cfg is not None else c.data.path
        reg_data_token_id = dataloader.load_token_id(reg_path)

    # dataset
    train_seed = np.asarray(jax.random.key_data(key_dataset))
    length_stop_names = ("num_tokens_train", "tokens_per_param", "num_epochs")
    if sum(c.stop[name] is not None for name in length_stop_names) > 1:
        raise ValueError("specify at most one training length criterion: stop.num_tokens_train, stop.tokens_per_param, or stop.num_epochs")
    if sum(x is not None for x in c.stop.values()) == 0:
        raise ValueError("specify at least one stop criterion")
    if c.stop.num_tokens_train is not None and c.stop.num_tokens_train <= 0:
        raise ValueError("stop.num_tokens_train must be > 0")
    ds_train, ds_valid = dataloader.load_ds(c.data.split_seed, c.data.path, c.model.T, c.opt.batch_size, c.data.num_tokens_valid)
    ds_mix_train = ds_mix_valid = None
    if c.mix.path is not None:
        ds_mix_train, ds_mix_valid = dataloader.load_ds(c.data.split_seed, c.mix.path, c.model.T, c.opt.batch_size, c.data.num_tokens_valid)
        if c.mix.coeff <= 0:
            ds_mix_train = None
    tokens_per_opt_step = c.opt.batch_size * c.model.T
    tokens_per_train_epoch = len(ds_train) * tokens_per_opt_step
    if c.stop.num_tokens_train is not None:
        train_epochs = c.stop.num_tokens_train / tokens_per_train_epoch
    elif c.stop.tokens_per_param is not None:
        n_params_total = n_params["n_param_nonembed"] + n_params["n_param_embed"]
        train_epochs = c.stop.tokens_per_param * n_params_total / tokens_per_train_epoch
    else:
        train_epochs = c.stop.num_epochs
    if train_epochs is not None and train_epochs <= 0:
        raise ValueError("training length must be > 0")
    if c.stop.train_time is not None and c.stop.train_time <= 0:
        raise ValueError("stop.train_time must be > 0")
    if c.stop.early_stop_steps is not None and c.stop.early_stop_steps < 0:
        raise ValueError("stop.early_stop_steps must be >= 0")
    num_opt_steps = None if train_epochs is None else math.ceil(train_epochs * len(ds_train))
    if train_epochs is not None:
        train_epochs_str = f"{train_epochs:.2f}".rstrip("0").rstrip(".")
        print(f"training for {train_epochs_str} epochs")
    eval_datasets = {
        "valid": jax.device_put(ds_valid, P(None, "data", None)),
    }
    if ds_mix_valid is not None:
        eval_datasets["mix"] = jax.device_put(ds_mix_valid, P(None, "data", None))
    ds_prev_train = ds_train
    if c.load_cfg is not None:
        ds_prev_train, ds_prev_valid = dataloader.load_ds(c.load_cfg.data.split_seed, c.load_cfg.data.path, c.load_cfg.model.T, c.load_cfg.opt.batch_size, c.load_cfg.data.num_tokens_valid)
        eval_datasets["prev"] = jax.device_put(ds_prev_valid, P(None, "data", None))

    # optimizer
    warmup_steps = math.ceil(c.opt.warmup_tokens / tokens_per_opt_step)
    if c.opt.lr_decay is None:
        lr_schedule = optax.warmup_constant_schedule(0.0, c.opt.peak_lr, warmup_steps)
    elif num_opt_steps is not None and c.opt.lr_decay == "cosine":
        lr_schedule = optax.warmup_cosine_decay_schedule(0.0, c.opt.peak_lr, warmup_steps, num_opt_steps)
    else:
        raise ValueError("opt.lr_decay must be null or cosine, and cosine requires a fixed training length")
    if c.opt.optimizer == "adamw":
        tx = optax.adamw(lr_schedule, c.opt.b1, c.opt.b2, weight_decay=c.opt.weight_decay)
    elif c.opt.optimizer == "sgd":
        tx = optax.sgd(lr_schedule)
    else:
        raise ValueError("opt.optimizer must be adamw or sgd")
    opt_state = tx.init(weights_theta)

    # training loop
    next_log_at = 0
    best_valid_ntp = float("inf")
    valid_ntp_no_improve = 0
    train_loss_sum = train_loss_ntp_sum = train_loss_reg_sum = jnp.zeros([])
    train_time_elapsed = 0.0
    train_loss_num, step, done = 0, 0, False
    stop_reason = None
    measure_train_time = (c.stop.train_time is not None) or c.log.measure_time
    rng = np.random.default_rng(train_seed)
    train_batches = ds_train.epochs(rng)
    mix_batches = None if ds_mix_train is None else ds_mix_train.epochs(rng)
    prev_batches = ds_prev_train.epochs(rng)
    synth_reg_iter = iter(())
    pbar = tqdm(total=num_opt_steps) if jax.process_index() == 0 else None
    while not done:
        batch = next(train_batches)
        if mix_batches is not None:
            batch_mix = next(mix_batches)
            mix_rows = math.floor(c.mix.coeff * len(batch))
            batch[:mix_rows] = batch_mix[:mix_rows]
        reg_method = c.reg.method
        if c.reg.batch_size is None:
            reg_batch_size = len(batch)
        elif use_synth_reg:
            reg_batch_size = c.reg.batch_size
        elif c.reg.batch_size > len(batch):
            raise ValueError("reg.batch_size cannot exceed opt.batch_size for real-data regularization")
        else:
            reg_batch_size = c.reg.batch_size
        batch_prev = None
        if not use_synth_reg and reg_method in {"ntp_pre", "kl_pre", "replace"}:
            batch_prev = next(prev_batches)[:reg_batch_size]
        if reg_method == "replace":
            replace_rows = math.floor(c.reg.coeff * len(batch))
            batch[:replace_rows] = batch_prev[:replace_rows]
            reg_method = None

        # training step
        batch = jax.device_put(batch, P("data", None))
        if use_synth_reg:
            try:
                batch_prev = next(synth_reg_iter)
            except StopIteration:
                key, key_reg = jax.random.split(key)
                synth_reg_batches = sample_synthetic_batch(
                    key_reg,
                    forward_pi,
                    init_kv,
                    weights_pi,
                    c.reg.buffer * reg_batch_size,
                    c.model.T,
                    reg_data_token_id,
                )
                synth_reg_batches = jnp.reshape(synth_reg_batches, (c.reg.buffer, reg_batch_size, c.model.T), out_sharding=P(None, "data", None))
                synth_reg_iter = (synth_reg_batches[i] for i in range(c.reg.buffer))
                batch_prev = next(synth_reg_iter)
        elif batch_prev is not None:
            batch_prev = jax.device_put(batch_prev, P("data", None))
        should_time_step = measure_train_time and step >= 5
        step_start = time.perf_counter() if should_time_step else None
        weights_theta, opt_state, batch_loss, batch_loss_ntp, batch_loss_reg = train_step(
            forward_theta, forward_pi, tx, weights_theta, weights_pi, opt_state, batch, batch_prev, reg_method, c.reg.coeff
        )
        if should_time_step:
            jax.block_until_ready(batch_loss)
            train_time_elapsed += time.perf_counter() - step_start

        # logging
        train_loss_sum += batch_loss
        train_loss_ntp_sum += batch_loss_ntp
        train_loss_reg_sum += batch_loss_reg
        train_loss_num += 1
        step += 1
        if pbar is not None:
            pbar.update(1)
        tokens_seen = step * tokens_per_opt_step
        if tokens_seen >= next_log_at:
            while tokens_seen >= next_log_at:
                next_log_at += c.log.every_tokens
            metrics = {
                "train_loss": train_loss_sum / train_loss_num,
                "train_loss_ntp": train_loss_ntp_sum / train_loss_num,
                "train_loss_reg": train_loss_reg_sum / train_loss_num,
                "train_time_elapsed": train_time_elapsed,
                "train_tokens_seen": tokens_seen,
                "epoch": tokens_seen / tokens_per_train_epoch,
            }
            metrics |= eval_metrics(forward_theta, forward_pi, weights_theta, weights_pi, weights_init, eval_datasets, c.model.lora_rank)
            if c.stop.early_stop_steps is not None:
                valid_ntp = float(metrics["valid_ntp"])
                if valid_ntp < best_valid_ntp:
                    best_valid_ntp = valid_ntp
                    valid_ntp_no_improve = 0
                else:
                    valid_ntp_no_improve += 1
            if run is not None:
                run.log(metrics, step=step)
                pbar.set_postfix_str(f'loss={metrics["train_loss"]:.2f}')
            train_loss_sum = train_loss_ntp_sum = train_loss_reg_sum = jnp.zeros([])
            train_loss_num = 0
            if c.stop.target_loss is not None and float(metrics["valid_ntp"]) <= c.stop.target_loss:
                stop_reason = "target_loss"
            elif c.stop.early_stop_steps is not None and valid_ntp_no_improve > c.stop.early_stop_steps:
                stop_reason = "early_stop"
        if stop_reason is None and c.stop.train_time is not None and train_time_elapsed >= c.stop.train_time:
            stop_reason = "train_time"
        if stop_reason is None and num_opt_steps is not None and step >= num_opt_steps:
            stop_reason = "train_length"
        done = stop_reason is not None

    # eval at end of training
    metrics = eval_metrics(forward_theta, forward_pi, weights_theta, weights_pi, weights_init, eval_datasets, c.model.lora_rank)
    metrics["train_time_elapsed"] = train_time_elapsed
    metrics["train_tokens_seen"] = step * tokens_per_opt_step
    metrics["epoch"] = metrics["train_tokens_seen"] / tokens_per_train_epoch
    target_loss_reached = c.stop.target_loss is not None and float(metrics["valid_ntp"]) <= c.stop.target_loss
    metrics["target_loss_reached"] = float(target_loss_reached)
    metrics["target_loss_missed"] = float(c.stop.target_loss is not None and not target_loss_reached)
    if run is not None:
        run.log(metrics, step=step)
        run.summary.update({"stop_reason": stop_reason})

    # save model
    out_dir = None
    if c.model.save_dir is not None and run is not None:
        assert c.model.lora_rank is None, "saving is not supported for LoRA"
        out_dir = Path(c.model.save_dir).expanduser() / run.id
        out_dir.mkdir(parents=True, exist_ok=True)
        weights_host = jax.tree.map(np.asarray, weights_theta)
        np.savez(out_dir / "weights.npz", **weights_host)
        (out_dir / "config.yaml").write_text(OmegaConf.to_yaml(c, resolve=True))
        print(f"saved model to {out_dir}")

    if run is not None:
        run.finish()
    if pbar is not None:
        pbar.close()

    return {
        "run_id": None if run is None else run.id,
        "out_dir": None if out_dir is None else str(out_dir),
        "stop_reason": stop_reason,
        "target_loss_reached": target_loss_reached,
        "metrics": {k: float(v) for k, v in metrics.items()},
    }

def main():
    c = OmegaConf.merge(OmegaConf.create(DEFAULT_CFG), OmegaConf.from_cli(sys.argv[1:]))
    print(OmegaConf.to_yaml(c, resolve=True))
    train_and_evaluate(c)


if __name__ == "__main__":
    main()
