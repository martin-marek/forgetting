from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp

from evals import custom, instruction
from evals.core import run_core
from evals.run_inspect import eval_inspect
from losses import cross_entropy, get_batch_targets, loss_fn_kl, loss_fn_ntp


@partial(jax.jit, static_argnames="forward")
def eval_preference_step(forward, weights, batch):
    xw, yw, mw = get_batch_targets(batch["chosen_tokens"], batch["chosen_mask"])
    xl, yl, ml = get_batch_targets(batch["rejected_tokens"], batch["rejected_mask"])
    lp_chosen = -(cross_entropy(forward(xw, weights), yw) * mw).sum(axis=1)
    lp_rejected = -(cross_entropy(forward(xl, weights), yl) * ml).sum(axis=1)
    lp_chosen_norm = lp_chosen / mw.sum(axis=1)
    lp_rejected_norm = lp_rejected / ml.sum(axis=1)
    return lp_chosen, lp_rejected, lp_chosen_norm, lp_rejected_norm


def eval_preference(model, make_valid, epoch):
    chosen_lps, rejected_lps = [], []
    chosen_lps_norm, rejected_lps_norm = [], []
    for batch in make_valid(epoch):
        lp_c, lp_r, lp_c_norm, lp_r_norm = eval_preference_step(model.forward, model.weights, batch)
        chosen_lps.append(lp_c)
        rejected_lps.append(lp_r)
        chosen_lps_norm.append(lp_c_norm)
        rejected_lps_norm.append(lp_r_norm)
    chosen_lps = jnp.concatenate(chosen_lps)
    rejected_lps = jnp.concatenate(rejected_lps)
    chosen_lps_norm = jnp.concatenate(chosen_lps_norm)
    rejected_lps_norm = jnp.concatenate(rejected_lps_norm)
    return {
        "preference/valid/chosen_logprob": chosen_lps.mean(),
        "preference/valid/rejected_logprob": rejected_lps.mean(),
        "preference/valid/accuracy": (chosen_lps > rejected_lps).mean(),
        "preference/valid/margin": (chosen_lps - rejected_lps).mean(),
        "preference/valid/chosen_logprob_norm": chosen_lps_norm.mean(),
        "preference/valid/rejected_logprob_norm": rejected_lps_norm.mean(),
        "preference/valid/accuracy_norm": (chosen_lps_norm > rejected_lps_norm).mean(),
        "preference/valid/margin_norm": (chosen_lps_norm - rejected_lps_norm).mean(),
    }


def eval_lm_losses(model, weights_pi, ds_eval, target_tokens=1_000_000):
    total_ntp_loss_sum = 0.0
    total_kl_loss_sum = 0.0
    total_tokens = 0
    while total_tokens < target_tokens:
        batch = next(ds_eval)
        if "tokens" in batch and "mask" in batch:
            lm_batch = batch
        elif "chosen_tokens" in batch and "chosen_mask" in batch:
            lm_batch = {"tokens": batch["chosen_tokens"], "mask": batch["chosen_mask"]}
        else:
            raise ValueError(f"Batch is missing LM tokens/mask fields: {batch.keys()}")
        _, batch_ntp_loss_sum, batch_tokens = loss_fn_ntp(model.forward, model.weights, lm_batch)
        _, batch_kl_loss_sum, _ = loss_fn_kl(model.forward, model.weights, weights_pi, lm_batch)
        total_ntp_loss_sum += batch_ntp_loss_sum
        total_kl_loss_sum += batch_kl_loss_sum
        total_tokens += batch_tokens
    return total_ntp_loss_sum / total_tokens, total_kl_loss_sum / total_tokens


@partial(jax.jit, static_argnames="forward")
def eval_lm_valid_step(forward, weights, batch):
    x, y, mask = get_batch_targets(batch["tokens"], batch["mask"])
    logits = forward(x, weights)
    loss_sum = (cross_entropy(logits, y) * mask).sum()
    correct = jnp.logical_and(jnp.argmax(logits, -1) == y, mask).sum()
    return loss_sum, correct, mask.sum()


def eval_lm_valid(model, make_valid):
    total_loss = total_correct = total_tokens = 0
    for batch in make_valid(0):
        loss_sum, correct, n_tokens = eval_lm_valid_step(model.forward, model.weights, batch)
        total_loss += loss_sum
        total_correct += correct
        total_tokens += n_tokens
    return {"eval/valid/ntp": total_loss / total_tokens, "eval/valid/accuracy": total_correct / total_tokens}


def run_evals(
    eval_names,
    model,
    weights_pi,
    ds_train_eval,
    ds_reg_eval,
    make_pref_valid,
    run_dir,
    step,
    epoch,
    tokens_per_batch,
    train_split,
    num_tokens_eval,
    num_examples_eval=None,
):
    if isinstance(eval_names, str):
        raise TypeError('log_evals must be a list or tuple, e.g. --log_evals=\'["core"]\'')
    eval_names = [] if eval_names is None else eval_names
    known_evals = {
        "train_lm",
        "reg_lm",
        "preference_valid",
        "inspect",
        "chat_mcq",
        "core",
        *custom.BENCHMARKS,
        *instruction.BENCHMARKS,
    }
    unknown_evals = [name for name in eval_names if name not in known_evals]
    if unknown_evals:
        raise ValueError(f"Unknown evals: {unknown_evals}")
    if num_examples_eval is not None and num_examples_eval <= 0:
        raise ValueError("log.num_examples_eval must be positive or null")

    metrics = {}
    default_num_examples = 1024 if num_examples_eval is None else num_examples_eval

    if "train_lm" in eval_names:
        train_ntp, train_kl = eval_lm_losses(model, weights_pi, ds_train_eval, target_tokens=num_tokens_eval)
        metrics |= {"eval/train/ntp": train_ntp, "eval/train/kl": train_kl}

    if "reg_lm" in eval_names:
        if ds_reg_eval is None:
            raise ValueError("reg_lm eval requires reg.dataset")
        reg_ntp, reg_kl = eval_lm_losses(model, weights_pi, ds_reg_eval, target_tokens=num_tokens_eval)
        metrics |= {"eval/reg/ntp": reg_ntp, "eval/reg/kl": reg_kl}

    if "preference_valid" in eval_names:
        if make_pref_valid is None:
            raise ValueError("preference_valid eval requires pref_valid_dataset")
        metrics |= eval_preference(model, make_pref_valid, epoch)

    if "inspect" in eval_names:
        metrics |= eval_inspect(model, Path(run_dir) / "eval_logs" / f"step_{step}", batch_size=512, seq_len=1024, limit=default_num_examples)

    if "chat_mcq" in eval_names:
        metrics |= instruction.run_set(
            model,
            instruction.CHAT_MCQ_BENCHMARKS,
            batch_size_tokens=tokens_per_batch,
            size=num_examples_eval,
            prefix="chat_mcq",
        )

    if "core" in eval_names:
        core_results = run_core(model, batch_size_tokens=tokens_per_batch, max_per_task=default_num_examples)
        metrics["core"] = core_results["core_metric"]
        metrics |= {f"core/{k}": v for k, v in core_results["results"].items()}
        metrics |= {f"core_centered/{k}": v for k, v in core_results["centered_results"].items()}

    custom_evals = [name for name in eval_names if name in custom.BENCHMARKS]
    if custom_evals:
        benchmark_kwargs = {}
        for name in ("enron_main", "enron_rephrased", "bio_qa_free", "bio_qa_comp"):
            if name in custom_evals:
                benchmark_kwargs[name] = {"load_kwargs": {"split": train_split}}
        metrics |= custom.run_set(model, custom_evals, batch_size_tokens=tokens_per_batch, size=default_num_examples, benchmark_kwargs=benchmark_kwargs)

    instruction_evals = [name for name in eval_names if name in instruction.BENCHMARKS]
    if instruction_evals:
        metrics |= instruction.run_set(model, instruction_evals, batch_size_tokens=tokens_per_batch, size=num_examples_eval)

    return metrics
