from functools import partial
from itertools import chain, count

import data
import jax
import jax.numpy as jnp
from omegaconf import OmegaConf
import optax
import timing
import wandb
from tqdm.auto import tqdm

from evals.run import eval_lm_valid, run_evals
import losses
import utils


DEFAULT_CFG = """
model:
  source: null
  weights_dir: /dev/shm/ptx/weights
  init: pretrained
  tp_size: 1
  dp_shard: false
  hidden_size: null
  num_hidden_layers: null
train:
  dataset: null
  split: null
  seq_len: null
  batch_size: null
  num_tokens_valid: 0
reg:
  dataset: null
  split: null
  seq_len: null
  batch_size: null
  method: null
  coeff: 0.0
  synth: false
  buffer: 1
opt:
  n_epochs: 1
  tokens: null
  optimizer: adamw
  lr: 1e-4
  b2: 0.997
  warmup_tokens: 10_000_000
  ema_halflife: null
log:
  project: ptx
  wandb_mode: online
  dir: ~/logs
  run_name: null
  save_model: false
  evals: []
  every_tokens: 1_000_000
  eval_every_tokens: 20_000_000
  num_tokens_eval: 1_000_000
  num_examples_eval: null
  verilog:
    data_dir: ~/verilog-eval/dataset_spec-to-rtl
    iverilog: ~/.local/bin/iverilog
    every: 2             # run on every Nth eval trigger
    num_problems: null   # null = all that fit max_prompt_tokens
    temperature: 0.01    # ~greedy (sampler requires temperature > 0)
    max_prompt_tokens: 896
    max_new_tokens: 512
    sim_timeout: 30
run:
  seed: 0
"""


def parse_cfg():
    cfg = OmegaConf.create(DEFAULT_CFG)
    OmegaConf.set_struct(cfg, True)
    return OmegaConf.merge(cfg, OmegaConf.from_cli())


def train(cfg) -> None:
    model, weights_pi = utils.load_training_model(config=cfg.model, seed=cfg.run.seed)
    with timing.context("load datasets"):
        train_len, make_train, make_valid = data.load(cfg.train.dataset, cfg.train.split, cfg.train.batch_size, cfg.train.seq_len, model.tokenizer, seed=cfg.run.seed, num_tokens_valid=cfg.train.num_tokens_valid)
        _, make_reg, _ = (
            (None, None, None)
            if cfg.reg.dataset is None
            else data.load(
                cfg.reg.dataset, cfg.reg.split, cfg.reg.batch_size, cfg.reg.seq_len, model.tokenizer, seed=cfg.run.seed
            )
        )
    ds_train_eval = chain.from_iterable(make_train(epoch) for epoch in count())
    ds_reg = None if make_reg is None else chain.from_iterable(make_reg(epoch) for epoch in count())
    ds_reg_eval = None if make_reg is None else chain.from_iterable(make_reg(epoch) for epoch in count())
    tokens_per_batch, total_steps, target_tokens, streaming_train = utils.compute_training_schedule(train_len, cfg.train, cfg.opt)
    optimizer, opt_state = utils.make_optimizer(cfg.opt, total_steps, model.weights, tokens_per_batch)
    run_dir = utils.start_run(cfg)
    ema_weights = None if not cfg.opt.ema_halflife else jax.tree.map(jnp.copy, model.weights)
    synth_key = jax.random.key(cfg.run.seed)
    synth_reg_iter = iter(())

    def loss_fn(weights_theta, weights_pi, batch_train, batch_reg=None):
        primary_loss_avg, primary_loss_sum, primary_num_tokens = losses.loss_fn_ntp(model.forward, weights_theta, batch_train)
        reg_loss_avg = losses.loss_fn_reg(model.forward, weights_theta, weights_pi, batch_train, batch_reg, cfg.reg, primary_loss_avg.dtype)
        optim_loss_avg = primary_loss_avg + cfg.reg.coeff * reg_loss_avg
        return optim_loss_avg, (reg_loss_avg, primary_loss_sum, primary_num_tokens)

    @partial(jax.jit, donate_argnames=("weights_theta", "opt_state"))
    def opt_step(weights_theta, weights_pi, opt_state, batch_train, batch_reg=None):
        (optim_loss_avg, (reg_loss_avg, primary_loss_sum, primary_num_tokens)), grads = jax.value_and_grad(loss_fn, argnums=0, has_aux=True)(weights_theta, weights_pi, batch_train, batch_reg)
        updates, opt_state = optimizer.update(grads, opt_state, weights_theta)
        weights_theta = optax.apply_updates(weights_theta, updates)
        batch_losses = {"optim_loss_avg": optim_loss_avg, "reg_loss_avg": reg_loss_avg, "primary_loss_sum": primary_loss_sum, "primary_num_tokens": primary_num_tokens}
        return weights_theta, opt_state, batch_losses

    @jax.jit
    def update_ema_weights(ema_weights, weights, ema_decay):
        return jax.tree.map(lambda ema, w: ema_decay * ema + (1.0 - ema_decay) * w, ema_weights, weights)

    pending_metrics = None
    step = masked_tokens_seen = total_tokens_seen = 0
    if cfg.log.every_tokens <= 0:
        raise ValueError("log.every_tokens must be positive")
    if cfg.log.eval_every_tokens <= 0:
        raise ValueError("log.eval_every_tokens must be positive")
    next_log_at = next_eval_at = 0
    interval_reg_loss_avg_sum = interval_optim_loss_avg_sum = interval_primary_loss_sum = 0.0
    interval_steps = interval_primary_tokens = epoch_tokens = code_len_model_and_data = 0
    train_batches = ((epoch, batch_train) for epoch in (count() if streaming_train else range(cfg.opt.n_epochs)) for batch_train in make_train(epoch))
    pbar = tqdm(total=total_steps, desc="Train") if jax.process_index() == 0 else None

    for epoch, batch_train in train_batches:

        # logging
        should_log = total_tokens_seen >= next_log_at
        should_eval = total_tokens_seen >= next_eval_at
        if should_log or should_eval:
            if should_log:
                while total_tokens_seen >= next_log_at:
                    next_log_at += cfg.log.every_tokens
            eval_metrics = {
                "step": step,
                "epoch": epoch,
                "masked_tokens_seen": masked_tokens_seen,
                "total_tokens_seen": total_tokens_seen,
                "weight_norm": utils.weight_norm(model.weights),
                "epiplexity": 0,
            }
            if should_log and step > 0:
                interval_reg_loss_avg = interval_reg_loss_avg_sum / interval_steps
                primary_loss_avg = interval_primary_loss_sum / interval_primary_tokens
                code_len_data_given_model = primary_loss_avg * min(masked_tokens_seen, epoch_tokens)
                code_len_model = code_len_model_and_data - code_len_data_given_model
                eval_metrics |= {
                    "loss/optim": interval_optim_loss_avg_sum / interval_steps,
                    "loss/primary": primary_loss_avg,
                    "loss/regularization": interval_reg_loss_avg,
                    "loss/regularization_weighted": cfg.reg.coeff * interval_reg_loss_avg,
                    "epiplexity": code_len_model,
                }
                interval_reg_loss_avg_sum = interval_optim_loss_avg_sum = interval_primary_loss_sum = 0.0
                interval_steps = interval_primary_tokens = 0
            if should_eval:
                while total_tokens_seen >= next_eval_at:
                    next_eval_at += cfg.log.eval_every_tokens
                eval_metrics |= run_evals(cfg.log.evals, model, weights_pi, ds_train_eval, ds_reg_eval, None, run_dir, step, epoch, tokens_per_batch, cfg.train.split, cfg.log.num_tokens_eval, cfg.log.num_examples_eval, verilog_cfg=cfg.log.verilog)
                if make_valid is not None:
                    eval_metrics |= eval_lm_valid(model, make_valid)
                if ema_weights is not None:
                    train_weights, model.weights = model.weights, ema_weights
                    ema_eval_metrics = run_evals(cfg.log.evals, model, weights_pi, ds_train_eval, ds_reg_eval, None, run_dir / "ema", step, epoch, tokens_per_batch, cfg.train.split, cfg.log.num_tokens_eval, cfg.log.num_examples_eval, verilog_cfg=cfg.log.verilog)
                    if make_valid is not None:
                        ema_eval_metrics |= eval_lm_valid(model, make_valid)
                    model.weights = train_weights
                    eval_metrics |= {f"{name}_ema": value for name, value in ema_eval_metrics.items()}
            if pending_metrics is not None: utils.log_metrics(*pending_metrics, run_dir)
            pending_metrics = step, eval_metrics

        # train step
        batch_reg = None
        if cfg.reg.method in ("ntp_pre", "kl_pre"):
            if cfg.reg.synth:
                try:
                    batch_reg = next(synth_reg_iter)
                except StopIteration:
                    synth_key, key_reg = jax.random.split(synth_key)
                    synth_reg_batches = utils.sample_synthetic_lm_batches(
                        key_reg, model, weights_pi, cfg.reg.batch_size, cfg.reg.seq_len, cfg.reg.buffer
                    )
                    synth_reg_iter = (jax.tree.map(lambda x: x[i], synth_reg_batches) for i in range(cfg.reg.buffer))
                    batch_reg = next(synth_reg_iter)
            else:
                batch_reg = next(ds_reg)
        model.weights, opt_state, batch_losses = opt_step(model.weights, weights_pi, opt_state, batch_train, batch_reg)
        if ema_weights is not None:
            ema_halflife_tokens = cfg.opt.ema_halflife * (total_tokens_seen + tokens_per_batch)
            ema_decay = 0.5 ** (tokens_per_batch / ema_halflife_tokens)
            ema_weights = update_ema_weights(ema_weights, model.weights, ema_decay)

        # logging
        interval_optim_loss_avg_sum += batch_losses["optim_loss_avg"]
        interval_reg_loss_avg_sum += batch_losses["reg_loss_avg"]
        interval_primary_loss_sum += batch_losses["primary_loss_sum"]
        interval_primary_tokens += batch_losses["primary_num_tokens"]
        if epoch == 0:
            epoch_tokens += batch_losses["primary_num_tokens"]
            code_len_model_and_data += batch_losses["primary_loss_sum"]
        step += 1
        interval_steps += 1
        masked_tokens_seen += batch_losses["primary_num_tokens"]
        total_tokens_seen += tokens_per_batch
        if pbar is not None:
            pbar.update(1)

        if total_tokens_seen >= target_tokens:
            break

    if pbar is not None:
        pbar.close()
    utils.log_metrics(*pending_metrics, run_dir)
    if cfg.log.save_model:
        utils.save_model(model, run_dir, "final_model")
        if ema_weights is not None:
            train_weights, model.weights = model.weights, ema_weights
            utils.save_model(model, run_dir, "final_model_ema")
            model.weights = train_weights
    if jax.process_index() == 0: wandb.finish()


def main() -> None:
    train(parse_cfg())


if __name__ == "__main__":
    main()
