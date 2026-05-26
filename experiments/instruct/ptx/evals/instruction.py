import dataclasses
import functools
import itertools
import random

import datasets
import numpy as np
from tqdm.auto import tqdm

from evals.styles import score_chat_mcq_logprobs


SINGLE_ANSWER_TEMPLATE = """
Answer the following multiple choice question. The entire content of your response should be of the following format: 'ANSWER: $LETTER' (without quotes) where LETTER is one of {letters}.

{question}

{choices}
""".strip()

MMLU_REVISION = "c30699e8356da336a370243923dbaf21066bb9fe"
CHAT_MCQ_BENCHMARKS = (
    "mmlu_chat",
    "arc_challenge_chat",
    "commonsenseqa_chat",
)


@dataclasses.dataclass(frozen=True)
class Benchmark:
    seq_len: int
    load_fn: callable
    default_size: int | None = 1024


BENCHMARKS: dict[str, Benchmark] = {}


def benchmark(name, *, seq_len, default_size=1024):
    def decorator(fn):
        BENCHMARKS[name] = Benchmark(seq_len=seq_len, load_fn=fn, default_size=default_size)
        return fn

    return decorator


def _answer_letter(idx):
    return chr(ord("A") + idx)


def _answer_completions(n_choices):
    return [f"ANSWER: {_answer_letter(i)}" for i in range(n_choices)]


def _format_choices(choices):
    return "\n".join(f"{_answer_letter(i)}) {choice}" for i, choice in enumerate(choices))


def _format_prompt(question, choices):
    letters = ",".join(_answer_letter(i) for i in range(len(choices)))
    return SINGLE_ANSWER_TEMPLATE.format(
        question=question,
        choices=_format_choices(choices),
        letters=letters,
    )


def _shuffle_rows(rows, seed):
    rows = list(rows)
    random.Random(seed).shuffle(rows)
    return rows


def _labeled_choice_example(question, choices, answer_key):
    labels = list(choices["label"])
    texts = list(choices["text"])
    correct = labels.index(answer_key)
    return [
        _format_prompt(question, texts),
        _answer_completions(len(texts)),
        correct,
    ]


@benchmark("mmlu_chat", seq_len=512, default_size=2048)
def _load_mmlu_chat(seed=42):
    ds = datasets.load_dataset("cais/mmlu", "all", split="test", revision=MMLU_REVISION)
    rows = _shuffle_rows(ds, seed)
    return [
        [
            _format_prompt(row["question"], row["choices"]),
            _answer_completions(len(row["choices"])),
            int(row["answer"]),
        ]
        for row in rows
    ]


@benchmark("arc_challenge_chat", seq_len=256)
def _load_arc_challenge_chat(seed=42):
    ds = datasets.load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")
    rows = _shuffle_rows(ds, seed)
    return [
        _labeled_choice_example(row["question"], row["choices"], row["answerKey"])
        for row in rows
    ]


@benchmark("commonsenseqa_chat", seq_len=192)
def _load_commonsenseqa_chat(seed=42):
    ds = datasets.load_dataset("tau/commonsense_qa", split="validation")
    rows = _shuffle_rows(ds, seed)
    return [
        _labeled_choice_example(row["question"], row["choices"], row["answerKey"])
        for row in rows
    ]


@functools.cache
def load_ds(name, size=None, seed=42):
    bench = BENCHMARKS[name]
    ds = bench.load_fn(seed=seed)
    if size is None:
        size = bench.default_size
    if size is not None:
        ds = ds[: min(size, len(ds))]
    return ds, bench.seq_len


def run_benchmark(model, ds_name, batch_size=None, batch_size_tokens=None, seq_len=None, size=None, pbar=False):
    ds, default_seq_len = load_ds(ds_name, size=size)
    if seq_len is None:
        seq_len = default_seq_len

    if batch_size is None:
        assert batch_size_tokens is not None
        max_choices = max((len(completions) for _, completions, _ in ds), default=1)
        batch_size = max(8, (batch_size_tokens // (seq_len * max_choices) // 8) * 8)

    results = {}
    skipped = 0
    ds_iter = tqdm(ds, disable=(not pbar))
    for batch in itertools.batched(ds_iter, batch_size):
        batch_results = score_chat_mcq_logprobs(batch, model, seq_len)
        skipped += batch_results.pop("skipped")
        for k, v in batch_results.items():
            results.setdefault(k, []).extend(v)

    n_total = len(ds)
    skipped_frac = skipped / n_total if n_total else 0.0
    if skipped > 0:
        print(f"warning ({ds_name}): skipped {skipped_frac:.1%} examples")

    return {
        **{k: np.mean(v).item() for k, v in results.items()},
        "skipped_frac": skipped_frac,
    }


def run_set(model, benchmarks=CHAT_MCQ_BENCHMARKS, batch_size_tokens=8192, prefix=None, **kwargs):
    scores = {}
    for name in benchmarks:
        for metric, value in run_benchmark(model, name, batch_size_tokens=batch_size_tokens, **kwargs).items():
            metric_name = f"{name}/{metric}" if prefix is None else f"{prefix}/{name}/{metric}"
            scores[metric_name] = value
    return scores
