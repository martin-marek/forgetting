import sys
from dataclasses import dataclass
from functools import partial
from pathlib import Path
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

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
import sampling
import utils


DEFAULT_CFG = """
run:
  seed: 0
  tags: null
toy:
  tasks: [add, reversal, sort, modadd]
  n_digits: 3
  modulus: 1000
  num_train: 65536
  num_valid: 4096
  steps_per_task: 2000
  eval_every: 20
model:
  D: 256
  L: 2
  H: 16
  T: null
  V: null
  remat: false
  dp_shard: false
  tp_size: 1
  activ_dtype: bfloat16
reg:
  method: null
  coeff: 10.0
opt:
  batch_size: 2048
  peak_lr: 1e-4
  warmup_steps: 200
  b1: 0.9
  b2: 0.99
  weight_decay: 0.02
log:
  project: ptx-toy
  mode: online
"""


def digits_to_int(digits, base):
    powers = base ** np.arange(digits.shape[1] - 1, -1, -1, dtype=np.int64)
    return digits.astype(np.int64) @ powers


@dataclass(frozen=True)
class ToyTokenizer:
    base: int
    pad: int
    bos_by_task: dict[str, int]
    sep: int
    eq: int
    null: int
    eos: int
    digit_offset: int

    @property
    def vocab_size(self):
        return self.digit_offset + self.base

    def encode_digits(self, digits):
        return self.digit_offset + digits.astype(np.int32)


@dataclass(frozen=True)
class ToyFormat:
    n_digits: int
    prompt_len: int
    answer_start: int
    answer_end: int
    eos_idx: int
    seq_len: int


def make_tokenizer(tasks, base):
    first = 1 + len(tasks)
    return ToyTokenizer(base, 0, {task: i + 1 for i, task in enumerate(tasks)}, first, first + 1, first + 2, first + 3, first + 4)


def make_format(n_digits):
    prompt_len = 2 * n_digits + 3
    answer_end = prompt_len + n_digits + 1
    return ToyFormat(n_digits, prompt_len, prompt_len, answer_end, answer_end, answer_end + 1)


def encode_numbers(tokenizer, values, width):
    divisors = tokenizer.base ** np.arange(width - 1, -1, -1, dtype=np.int64)
    digits = (values[:, None] // divisors) % tokenizer.base
    return tokenizer.encode_digits(digits)


def make_task_data(task, num_examples, seed, tokenizer, fmt, modulus):
    rng = np.random.default_rng(seed)
    n = fmt.n_digits
    a_slice = slice(1, 1 + n)
    sep_idx = a_slice.stop
    b_slice = slice(sep_idx + 1, sep_idx + 1 + n)
    eq_idx = b_slice.stop
    data = np.full((num_examples, fmt.seq_len), tokenizer.pad, dtype=np.int32)
    a_digits = rng.integers(0, tokenizer.base, size=(num_examples, n), dtype=np.int32)
    data[:, 0] = tokenizer.bos_by_task[task]
    data[:, a_slice] = tokenizer.encode_digits(a_digits)
    data[:, sep_idx] = tokenizer.sep
    data[:, eq_idx] = tokenizer.eq
    data[:, fmt.eos_idx] = tokenizer.eos

    if task in ("add", "modadd"):
        b_digits = rng.integers(0, tokenizer.base, size=(num_examples, n), dtype=np.int32)
        data[:, b_slice] = tokenizer.encode_digits(b_digits)
        a_values = digits_to_int(a_digits, tokenizer.base)
        b_values = digits_to_int(b_digits, tokenizer.base)
        values = a_values + b_values if task == "add" else (a_values + b_values) % modulus
        outputs = encode_numbers(tokenizer, values, n + 1)
    else:
        data[:, b_slice] = tokenizer.null
        answer_digits = a_digits[:, ::-1] if task == "reversal" else np.sort(a_digits, axis=1)
        output_digits = np.pad(answer_digits, ((0, 0), (1, 0)))
        outputs = tokenizer.encode_digits(output_digits)

    data[:, fmt.answer_start : fmt.answer_end] = outputs
    return data


def iter_batches(data, batch_size, rng=None):
    batch_indices = np.arange(len(data) // batch_size * batch_size, dtype=np.int32).reshape(-1, batch_size)
    while True:
        for batch_idx in batch_indices if rng is None else batch_indices[rng.permutation(len(batch_indices))]:
            yield data[batch_idx]
        if rng is None:
            return


@partial(jax.jit, static_argnames=("forward", "tx", "reg_method"), donate_argnames=("weights_theta", "opt_state"))
def train_step(forward, tx, weights_theta, weights_pi, opt_state, x_train, x_reg, reg_method, reg_coeff):
    def wrapped_loss(weights_theta):
        loss_ntp = loss_lib.loss_fn_ntp(forward, weights_theta, x_train)
        loss_reg = loss_lib.loss_fn_reg(forward, weights_theta, forward, weights_pi, x_train, x_reg, reg_method)
        return loss_ntp + reg_coeff * loss_reg, (loss_ntp, loss_reg)

    (loss_optim, (loss_ntp, loss_reg)), grads = jax.value_and_grad(wrapped_loss, has_aux=True)(weights_theta)
    updates, opt_state = tx.update(grads, opt_state, weights_theta)
    weights_theta = optax.apply_updates(weights_theta, updates)
    return weights_theta, opt_state, (loss_optim, loss_ntp, loss_reg)


@partial(jax.jit, static_argnames=("forward",))
def forward_logits(forward, weights, x):
    logits, _ = forward(x, weights)
    return logits


def greedy_decode_batch(forward, weights, batch, fmt, pad_token_id):
    tokens = np.full(batch.shape, pad_token_id, dtype=np.int32)
    tokens[:, : fmt.prompt_len] = batch[:, : fmt.prompt_len]
    for pos in range(fmt.prompt_len - 1, fmt.seq_len - 1):
        logits = forward_logits(forward, weights, jax.device_put(tokens, P("data", None)))
        tokens[:, pos + 1] = np.asarray(jnp.argmax(logits[:, pos, :], axis=-1), dtype=np.int32)
    return tokens


def eval_task_metrics(forward, weights, dataset, fmt, pad_token_id):
    accuracies = []
    losses = []
    target_slice = slice(fmt.answer_start, fmt.eos_idx + 1)
    for batch in dataset:
        batch_device = jax.device_put(batch, P("data", None))
        losses.append(float(loss_lib.loss_fn_ntp(forward, weights, batch_device)))
        decoded = greedy_decode_batch(forward, weights, batch, fmt, pad_token_id)
        accuracies.append(np.mean(np.all(decoded[:, target_slice] == batch[:, target_slice], axis=1)))
    return {"ntp": float(np.mean(losses)), "accuracy": float(np.mean(accuracies))}


def make_reg_batch(c, reg_tasks, rng, tokenizer, forward, init_kv, weights_pi, key):
    sizes = [len(rows) for rows in np.array_split(np.arange(c.opt.batch_size), len(reg_tasks))]
    prompt_token_ids = np.repeat([tokenizer.bos_by_task[task] for task in reg_tasks], sizes).astype(np.int32)
    key, key_reg = jax.random.split(key)
    tokens = jnp.zeros((len(prompt_token_ids), c.model.T), dtype=jnp.int32).at[:, 0].set(jnp.asarray(prompt_token_ids, dtype=jnp.int32))
    replay_batch = np.asarray(
        sampling.sample(key_reg, forward, init_kv, weights_pi, tokens, eos_id=-1, pad_id=tokenizer.pad),
        dtype=np.int32,
    )
    return replay_batch[rng.permutation(len(replay_batch))], key


def evaluate_all_tasks(forward, weights, valid_data, fmt, tokenizer, batch_size, task_name):
    metrics = {"train_task": task_name}
    for name, data in valid_data.items():
        eval_metrics = eval_task_metrics(forward, weights, iter_batches(data, batch_size), fmt, tokenizer.pad)
        metrics[f"{name}_accuracy"] = eval_metrics["accuracy"]
        metrics[f"{name}_ntp"] = eval_metrics["ntp"]
    return metrics


def train_and_evaluate(c):
    c = OmegaConf.merge(OmegaConf.create(DEFAULT_CFG), c)
    run = wandb.init(project=c.log.project, mode=c.log.mode, tags=c.run.tags) if jax.process_index() == 0 else None

    tasks = tuple(c.toy.tasks)
    tokenizer = make_tokenizer(tasks, base=10)
    fmt = make_format(c.toy.n_digits)
    c.model.T = fmt.seq_len
    c.model.V = tokenizer.vocab_size
    if run is not None:
        run.config.update(utils.flatten_dict(c))

    key = jax.random.key(c.run.seed)
    key, key_model = jax.random.split(key)

    device_count = jax.device_count()
    num_fsdp_devices = device_count // c.model.tp_size
    mesh = jax.make_mesh((num_fsdp_devices, c.model.tp_size), ("data", "model"), axis_types=(AxisType.Explicit, AxisType.Explicit))
    jax.set_mesh(mesh)
    print("sharding mesh:", ", ".join(f"{k}={v}" for k, v in mesh.shape.items()))

    start_time = time.perf_counter()

    print("building toy datasets...")
    train_data = {}
    valid_data = {}
    for task_idx, task in enumerate(tasks):
        train_data[task] = make_task_data(task, c.toy.num_train, c.run.seed + 100 * task_idx + 1, tokenizer, fmt, c.toy.modulus)
        valid_data[task] = make_task_data(task, c.toy.num_valid, c.run.seed + 100 * task_idx + 2, tokenizer, fmt, c.toy.modulus)

    c.model.V = ((c.model.V + device_count - 1) // device_count) * device_count
    print("initializing model...")
    weights_theta = model_lib.create_sharded_model(c.model, key_model)
    forward = partial(model_lib.forward, c.model)
    init_kv = partial(model_lib.init_kv, c.model)

    lr_schedule = optax.warmup_constant_schedule(0.0, c.opt.peak_lr, c.opt.warmup_steps)
    tx = optax.adamw(lr_schedule, c.opt.b1, c.opt.b2, weight_decay=c.opt.weight_decay)
    opt_state = tx.init(weights_theta)
    tokens_per_step = c.opt.batch_size * c.model.T
    tokens_per_device = tokens_per_step / device_count
    print(f"tokens/step={tokens_per_step:_}, tokens/device={tokens_per_device:_.0f}")

    loss_keys = ("train_loss", "train_loss_ntp", "train_loss_reg")
    metrics = evaluate_all_tasks(forward, weights_theta, valid_data, fmt, tokenizer, c.opt.batch_size, "init")
    metrics.update(train_tokens_seen=0, train_time_elapsed=0.0)
    if run is not None:
        run.log(metrics, step=0)
    print("initial metrics:", {k: round(v, 4) for k, v in metrics.items() if k.endswith("_accuracy")})

    global_step = 0
    rng = np.random.default_rng(c.run.seed)
    total_steps = len(tasks) * c.toy.steps_per_task
    pbar = tqdm(total=total_steps) if jax.process_index() == 0 else None

    def log_eval(step, task_name, loss_sums, loss_count):
        metrics = evaluate_all_tasks(forward, weights_theta, valid_data, fmt, tokenizer, c.opt.batch_size, task_name)
        metrics.update(dict(zip(loss_keys, (loss_sums / loss_count).tolist())))
        metrics["train_tokens_seen"] = step * tokens_per_step
        metrics["train_time_elapsed"] = time.perf_counter() - start_time
        if run is not None:
            run.log(metrics, step=step)
        summary = ", ".join(f"{name}={metrics[f'{name}_accuracy']:.3f}" for name in tasks)
        if pbar is not None:
            pbar.set_postfix_str(f"{task_name} | {summary}")
        return metrics

    for task_idx, task in enumerate(tasks):
        weights_pi = jax.tree.map(jnp.copy, weights_theta)
        current_reg_method = c.reg.method if task_idx > 0 else None
        reg_tasks = tasks[:task_idx] if current_reg_method is not None else ()
        if task_idx > 0:
            opt_state = tx.init(weights_theta)
        loss_sums = np.zeros(len(loss_keys), dtype=np.float64)
        loss_count = 0
        train_batches = iter_batches(train_data[task], c.opt.batch_size, rng)

        for task_step in range(c.toy.steps_per_task):
            batch = jax.device_put(next(train_batches), P("data", None))
            batch_prev = batch
            if reg_tasks:
                batch_prev_np, key = make_reg_batch(c, reg_tasks, rng, tokenizer, forward, init_kv, weights_pi, key)
                batch_prev = jax.device_put(batch_prev_np, P("data", None))
            weights_theta, opt_state, batch_losses = train_step(forward, tx, weights_theta, weights_pi, opt_state, batch, batch_prev, current_reg_method, c.reg.coeff)
            loss_sums += [float(loss) for loss in batch_losses]
            loss_count += 1
            global_step += 1
            if pbar is not None:
                pbar.update(1)
            if ((task_step + 1) % c.toy.eval_every == 0) or (task_step + 1 == c.toy.steps_per_task):
                metrics = log_eval(global_step, task, loss_sums, loss_count)
                loss_sums[:] = 0
                loss_count = 0

    if pbar is not None:
        pbar.close()

    elapsed = time.perf_counter() - start_time
    total_train_tokens = len(tasks) * c.toy.steps_per_task * tokens_per_step
    print(f"elapsed_sec={elapsed:.2f}, train_tokens/sec={total_train_tokens / elapsed:_.0f}")
    run_id = None if run is None else run.id
    if run is not None:
        run.finish()
    return {
        "run_id": run_id,
        "metrics": metrics,
    }


def main():
    c = OmegaConf.merge(OmegaConf.create(DEFAULT_CFG), OmegaConf.from_cli(sys.argv[1:]))
    print(OmegaConf.to_yaml(c, resolve=True))
    train_and_evaluate(c)


if __name__ == "__main__":
    main()
