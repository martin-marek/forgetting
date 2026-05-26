"""Shared eval-style scorers used across benchmark suites.

Expected model interface:
- model.tokenizer: callable returning {'input_ids': list[list[int]]}
- model.forward(x, model.weights): logits [B, T, V]
- model.weights: model parameters consumed by `forward`
"""

import itertools

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import PartitionSpec as P


@jax.jit
def _token_logprobs(logits, y):
    """Per-token log p(y|x) without materializing full [B,T,V] float32 log_softmax."""
    logits = logits.astype(jnp.float32)
    bsz, tsz, _ = logits.shape
    gathered = logits.at[jnp.arange(bsz)[:, None], jnp.arange(tsz)[None, :], y].get(out_sharding=P("data", None))
    return gathered - jax.nn.logsumexp(logits, axis=-1)


def compare_completion_logprobs(batch, model, seq_len):
    prompts, completions, correct_idxs = zip(*batch)
    pad_id = model.tokenizer.pad_token_id
    correct_idxs = np.array(correct_idxs)
    n_comps = len(completions[0])  # num. completions per prompt

    # tokenize prompts and (flattened) completions
    prompts_token = model.tokenizer(prompts)["input_ids"]
    comps_token = model.tokenizer(sum(completions, []))["input_ids"]

    # store tokens in array
    x = np.full([len(comps_token), seq_len], pad_id)
    comp_mask = np.zeros(x.shape, dtype=bool)
    for i, prompt_tokens in enumerate(prompts_token):
        for j in range(n_comps):
            tokens = prompt_tokens + comps_token[i * n_comps + j]
            if len(tokens) >= seq_len:
                return {}
            x[i * n_comps + j, : len(tokens)] = tokens
            comp_mask[i * n_comps + j, len(prompt_tokens) - 1 : len(tokens) - 1] = True
    y = np.roll(x, -1, axis=-1)

    # get total logprob of each completion
    logits = model.forward(x, model.weights)  # [n_prompts * n_completions, seq_len, vocab_size]
    token_logprobs = _token_logprobs(logits, y)  # [n_prompts * n_completions, seq_len]
    total_logprobs = jnp.where(comp_mask, token_logprobs, 0.0).sum(-1)  # [n_prompts * n_completions]

    # mask out empty (padding) completions
    lengths = jnp.array([len(comp) for comp in sum(completions, [])])
    total_logprobs = jnp.where(lengths == 0, -jnp.inf, total_logprobs)

    # unnormalized: accuracy and likelihood from total logprobs
    lp = total_logprobs.reshape([len(prompts), -1])
    lp = lp - jax.nn.logsumexp(lp, -1, keepdims=True)
    correct = lp.argmax(-1) == correct_idxs
    loglike = lp.at[jnp.arange(len(lp)), correct_idxs].get(out_sharding=P("data"))

    # normalized: accuracy and likelihood from per-token logprobs
    norm_logprobs = jnp.where(lengths == 0, -jnp.inf, total_logprobs / lengths)
    lp_norm = norm_logprobs.reshape([len(prompts), -1])
    lp_norm = lp_norm - jax.nn.logsumexp(lp_norm, -1, keepdims=True)
    correct_norm = lp_norm.argmax(-1) == correct_idxs
    loglike_norm = lp_norm.at[jnp.arange(len(lp_norm)), correct_idxs].get(out_sharding=P("data"))

    return {
        "accuracy": correct.tolist(),
        "likelihood": loglike.tolist(),
        "accuracy_norm": correct_norm.tolist(),
        "likelihood_norm": loglike_norm.tolist(),
    }


def compare_abcd_logprobs(batch, model, seq_len):
    """Same as compare_completion_logprobs but faster for 1-token choices."""
    prompts, completions, correct_idxs = zip(*batch)
    pad_id = model.tokenizer.pad_token_id
    correct_idxs = np.array(correct_idxs)

    # tokenize completions
    # each completion must be just a single token long
    # we also assume all completions are the same, e.g. ['A', 'B', 'C', 'D']
    abcd_tokens = model.tokenizer(completions[0])["input_ids"]
    for x in abcd_tokens:
        assert len(x) == 1
    abcd_tokens = np.array([x[0] for x in abcd_tokens])  # ['A','B','C','D'] -> [32,33,34,35]

    # tokenize prompts
    prompts = model.tokenizer(prompts)["input_ids"]

    # store tokens in padded array
    x = np.full([len(prompts), seq_len], pad_id)
    comp_idx = np.zeros([len(prompts)], dtype=np.int32)
    for i, seq in enumerate(prompts):
        if len(seq) >= seq_len:
            return {}
        x[i, : len(seq)] = seq
        comp_idx[i] = len(seq)

    # get completion logprobs
    logits = model.forward(x, model.weights)  # [n_prompts, seq_len, vocab_size]
    logprobs = jax.nn.log_softmax(logits.astype(jnp.float32), -1)
    bsz, _, _ = logprobs.shape
    completion_logprobs = logprobs.at[jnp.arange(bsz), comp_idx - 1, :].get(out_sharding=P("data", "model"))
    abcd_logprobs = completion_logprobs.at[:, abcd_tokens].get(out_sharding=P("data", None))

    # normalize completion logprobs
    abcd_logprobs -= jax.nn.logsumexp(abcd_logprobs, -1, keepdims=True)

    # return stats
    correct = abcd_logprobs.argmax(-1) == correct_idxs
    loglike = abcd_logprobs.at[jnp.arange(len(abcd_logprobs)), correct_idxs].get(out_sharding=P("data"))
    return {"accuracy": correct.tolist(), "likelihood": loglike.tolist()}


def _tokenize_chat_generation_prompt(tokenizer, prompt):
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        enable_thinking=False,
        tokenize=True,
        return_dict=False,
    )


def score_chat_mcq_logprobs(examples, model, seq_len):
    """Score instruction-style MCQ prompts by logprob of answer-letter completions."""
    pad_id = model.tokenizer.pad_token_id
    valid = []
    skipped = 0

    for prompt, completions, correct_idx in examples:
        if not 0 <= correct_idx < len(completions):
            skipped += 1
            continue

        prompt_tokens = _tokenize_chat_generation_prompt(model.tokenizer, prompt)
        comp_tokens = model.tokenizer(completions, add_special_tokens=False)["input_ids"]
        if any(len(comp) == 0 for comp in comp_tokens):
            skipped += 1
            continue
        if any(len(prompt_tokens) + len(comp) >= seq_len for comp in comp_tokens):
            skipped += 1
            continue

        valid.append((prompt_tokens, comp_tokens, correct_idx))

    if not valid:
        return {"accuracy": [], "likelihood": [], "skipped": skipped}

    max_choices = max(len(comps) for _, comps, _ in valid)
    n_rows = len(valid) * max_choices
    pad_multiple = max(1, jax.device_count())
    n_padded = ((n_rows + pad_multiple - 1) // pad_multiple) * pad_multiple
    x = np.full([n_padded, seq_len], pad_id)
    comp_mask = np.zeros(x.shape, dtype=bool)
    choice_mask = np.zeros([len(valid), max_choices], dtype=bool)
    correct_idxs = np.asarray([correct_idx for _, _, correct_idx in valid])

    for i, (prompt_tokens, comp_tokens, _) in enumerate(valid):
        for j, comp in enumerate(comp_tokens):
            row = i * max_choices + j
            tokens = prompt_tokens + comp
            x[row, : len(tokens)] = tokens
            comp_mask[row, len(prompt_tokens) - 1 : len(tokens) - 1] = True
            choice_mask[i, j] = True

    y = np.roll(x, -1, axis=-1)
    logits = model.forward(x, model.weights)
    token_logprobs = _token_logprobs(logits, y)
    total_logprobs = np.asarray(jnp.where(comp_mask, token_logprobs, 0.0).sum(-1))[:n_rows]
    choice_logprobs = total_logprobs.reshape([len(valid), max_choices])
    choice_logprobs = np.where(choice_mask, choice_logprobs, -np.inf)
    normalizer = np.logaddexp.reduce(choice_logprobs, axis=-1, keepdims=True)
    lp = choice_logprobs - normalizer

    correct = lp.argmax(-1) == correct_idxs
    loglike = lp[np.arange(len(lp)), correct_idxs]
    probability = np.exp(loglike)
    return {"accuracy": correct.tolist(), "likelihood": loglike.tolist(), "probability": probability.tolist(), "skipped": skipped}


def _find_common_suffix_length(token_sequences):
    min_len = min(len(seq) for seq in token_sequences)
    for i in range(1, min_len + 1):
        if not all(seq[-i] == token_sequences[0][-i] for seq in token_sequences):
            return i - 1
    return min_len


def score_schema_common_suffix(examples, model, seq_len):
    """Score schema-style tasks by comparing mean token logprob on common suffix."""
    pad_id = model.tokenizer.pad_token_id
    correct = []
    skipped = 0

    for options_texts, gold_idx in examples:
        token_sequences = model.tokenizer(options_texts)["input_ids"]
        suffix_len = _find_common_suffix_length(token_sequences)
        if suffix_len == 0:
            skipped += 1
            continue

        # Need at least one token before the scored suffix in each option.
        if any(len(seq) <= suffix_len for seq in token_sequences):
            skipped += 1
            continue

        max_len = max(len(seq) for seq in token_sequences)
        if max_len >= seq_len:
            skipped += 1
            continue

        n_options = len(token_sequences)
        n_padded = ((n_options + 7) // 8) * 8
        x = np.full([n_padded, seq_len], pad_id)
        for i, seq in enumerate(token_sequences):
            x[i, : len(seq)] = seq
        y = np.roll(x, -1, axis=-1)

        logits = model.forward(x, model.weights)
        token_lp = np.asarray(_token_logprobs(logits, y))

        mean_lps = []
        for i, seq in enumerate(token_sequences):
            start = len(seq) - suffix_len - 1
            end = len(seq) - 1
            mean_lps.append(token_lp[i, start:end].mean())

        correct.append(int(np.argmax(mean_lps)) == gold_idx)

    return {"correct": correct, "skipped": skipped}


def score_lm_exact_match(examples, model, seq_len, batch_size):
    """Score LM-style tasks by exact argmax match on continuation tokens."""
    pad_id = model.tokenizer.pad_token_id
    correct = []
    skipped = 0

    for batch in itertools.batched(examples, batch_size):
        batch = list(batch)
        tok_without = model.tokenizer([pw for pw, _ in batch])["input_ids"]
        tok_with = model.tokenizer([pw for _, pw in batch])["input_ids"]

        valid = []
        for i, (tw, twi) in enumerate(zip(tok_without, tok_with)):
            if len(twi) >= seq_len or len(tw) >= len(twi):
                skipped += 1
                continue
            if tw != twi[: len(tw)]:
                skipped += 1
                continue
            valid.append(i)

        if not valid:
            continue

        n_valid = len(valid)
        n_padded = ((n_valid + 7) // 8) * 8
        x = np.full([n_padded, seq_len], pad_id)
        split_points = []
        end_points = []
        for j, i in enumerate(valid):
            seq = tok_with[i]
            x[j, : len(seq)] = seq
            split_points.append(len(tok_without[i]))
            end_points.append(len(seq))

        logits = model.forward(x, model.weights)
        preds = np.asarray(jnp.argmax(logits, axis=-1))

        for j, (sp, ep) in enumerate(zip(split_points, end_points)):
            predicted = preds[j, sp - 1 : ep - 1]
            actual = x[j, sp:ep]
            correct.append(bool(np.all(predicted == actual)))

    return {"correct": correct, "skipped": skipped}
