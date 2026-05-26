import jax
import numpy as np
import random
import itertools
import functools
import dataclasses
import datasets
from tqdm.auto import tqdm
from evals.styles import compare_completion_logprobs, compare_abcd_logprobs, score_lm_exact_match
random.seed(0)


@dataclasses.dataclass(frozen=True)
class Benchmark:
    style: str      # 'abcd' | 'completions' | 'lm_exact_match'
    seq_len: int
    load_fn: callable

BENCHMARKS: dict[str, Benchmark] = {}

def benchmark(name, *, style, seq_len):
    """Register a dataset loader as a benchmark."""
    def decorator(fn):
        BENCHMARKS[name] = Benchmark(style=style, seq_len=seq_len, load_fn=fn)
        return fn
    return decorator


def map_n_shot(ds, n):
    """maps 0-shot dataset to n-shot dataset"""
    out_ds = []

    # iterate over examples in the dataset
    for i, (prompt, completions, correct_idx) in enumerate(ds):

        # create n-shot prefix
        prefix = ''
        for _ in range(n):
            # sample any example except for self
            idx = random.randint(0, len(ds)-2)
            if idx == i: idx += 1

            # append example to the n-shot prefix
            ex_prompt, ex_comps, ex_corr = ds[idx]
            prefix += ex_prompt + ex_comps[ex_corr] + '\n'
            
        # prepend the n-shot prefix to the prompt, store in output dataset
        out_ds += [[prefix+prompt, completions, correct_idx]]
        
    return out_ds


@benchmark('hellaswag', style='completions', seq_len=256)
def _load_hellaswag():
    ds = datasets.load_dataset('Rowan/hellaswag', split='validation')
    return [[x['ctx'], [' '+end for end in x['endings']], int(x['label'])] for x in ds]


@benchmark('truthfulqa', style='completions', seq_len=128)
def _load_truthfulqa():
    ds = datasets.load_dataset('truthful_qa', 'multiple_choice', split='validation')
    formatted_ds = []
    for x in ds:
        # Use MC1 targets (Single True)
        formatted_ds.append([
            f'Q: {x["question"]}\nA:',
            [' ' + c for c in x['mc1_targets']['choices']],
            x['mc1_targets']['labels'].index(1)
        ])

    # TruthfulQA has variable number of completions, but the evaluator expects a rectangular shape.
    # We pad all examples to the maximum number of choices found in the dataset.
    max_choices = max(len(x[1]) for x in formatted_ds)
    return [[prompt, comps + [''] * (max_choices - len(comps)), label]
            for prompt, comps, label in formatted_ds]


@benchmark('mmlu', style='abcd', seq_len=768)
def _load_mmlu():
    # following lm-eval format
    ds = datasets.load_dataset('cais/mmlu', 'all', split='test')
    return [[
        f"{x['question']}\nA. {x['choices'][0]}\nB. {x['choices'][1]}\nC. {x['choices'][2]}\nD. {x['choices'][3]}\nAnswer:",
        [' A', ' B', ' C', ' D'],
        x['answer'],
    ] for x in ds]


def _load_enron_email(rephrased=False, **hf_kwargs):
    hf_kwargs.setdefault('split', 'train[:1000]')
    ds = datasets.load_dataset('MichaelR207/enron_qa_0922', **hf_kwargs)
    formatted_ds = []
    if rephrased:
        email_questions = ds['rephrased_questions']
        email_correct = [a[0] for a in ds['alternate_answers']]
    else:
        email_questions = ds['questions']
        email_correct = ds['gold_answers']
    email_incorrect = ds['incorrect_answers']
    for email_questions, email_gold, email_incorrect in zip(email_questions, email_correct, email_incorrect):
        for question, gold, incorrect in zip(email_questions, email_gold, email_incorrect):
            formatted_ds.append([
                f'Q: {question.strip()}\nA:',
                [' ' + gold.strip()] + [' ' + ans.strip() for ans in incorrect],
                0,
            ])
    return formatted_ds


@benchmark('enron_main', style='completions', seq_len=512)
def _load_enron_main(**load_kwargs):
    return _load_enron_email(rephrased=False, **load_kwargs)


@benchmark('enron_rephrased', style='completions', seq_len=512)
def _load_enron_rephrased(**load_kwargs):
    return _load_enron_email(rephrased=True, **load_kwargs)


@benchmark('bio_qa_free', style='lm_exact_match', seq_len=32)
def _load_bio_qa_free(**hf_kwargs):
    hf_kwargs.setdefault('split', 'train[:1000]')
    ds = datasets.load_dataset('sqvareinch/synthetic-biography-qa-v2', **hf_kwargs)
    return [
        [f'Q: {x["question"].strip()}\nA:', f'Q: {x["question"].strip()}\nA: {x["answer"].strip()}']
        for x in ds
    ]


@benchmark('bio_qa_comp', style='completions', seq_len=32)
def _load_bio_qa_comp(**hf_kwargs):
    hf_kwargs.setdefault('split', 'train[:1000]')
    ds = datasets.load_dataset('sqvareinch/synthetic-biography-qa-v2', **hf_kwargs)
    return [[
                f'Q: {x["question"]}\nA:',
                [' ' + x['answer']] + [' ' + ans for ans in x['alternate_answers']],
                0,
        ] for x in ds
    ]

@functools.cache
def load_ds(name, n_shot=0, size=None, load_kwargs=()):
    bench = BENCHMARKS[name]
    ds = bench.load_fn(**dict(load_kwargs))
    if n_shot > 0 and bench.style != 'lm_exact_match':
        ds = map_n_shot(ds, n_shot)
    elif n_shot > 0:
        raise ValueError(f'n_shot not supported for benchmark style {bench.style}')
    if size is not None:
        ds = random.sample(ds, min(size, len(ds)))
    return ds, bench.style, bench.seq_len


def run_benchmark(model, ds_name, batch_size=None, batch_size_tokens=None, seq_len=None, n_shot=0, size=None, pbar=False, load_kwargs=None):

    # load dataset
    load_kwargs = tuple(sorted(load_kwargs.items())) if load_kwargs else ()
    ds, style, default_seq_len = load_ds(ds_name, n_shot, size, load_kwargs)

    # get default seq_len
    if seq_len is None:
        seq_len = default_seq_len
    
    # get batch size
    if batch_size is None:
        assert batch_size_tokens is not None
        expansion_factor = max((len(comps) for _, comps, _ in ds), default=1) if style == 'completions' else 1
        batch_size = max(8, (batch_size_tokens // (seq_len * expansion_factor) // 8) * 8)

    if style == 'lm_exact_match':
        batch_results = score_lm_exact_match(ds, model, seq_len, batch_size)
        accuracy = np.mean(batch_results['correct']).item() if batch_results['correct'] else 0.0
        skipped_frac = batch_results['skipped'] / len(ds) if len(ds) else 0.0
        if batch_results['skipped'] > 0:
            print(f'warning ({ds_name}): skipped {skipped_frac:.1%} examples')
        return {'accuracy': accuracy, 'skipped_frac': skipped_frac}

    # eval loop
    results = {}
    compare_fn = compare_abcd_logprobs if style=='abcd' else compare_completion_logprobs
    ds_iter = tqdm(ds, disable=(not pbar))
    for batch in itertools.batched(ds_iter, batch_size):
        if len(batch) < batch_size: continue
        batch_results = compare_fn(batch, model, seq_len)
        for k, v in batch_results.items():
            results.setdefault(k, []).extend(v)

    # check if any batches were skipped (possibly due to seq_len being too short)
    n_examples = len(next(iter(results.values()))) if results else 0
    if n_examples < len(ds):
        print(f'warning ({ds_name}): skipped {1-n_examples/len(ds):.1%} examples')

    return {k: np.mean(v).item() for k, v in results.items()}


def run_set(model, benchmarks='all', batch_size_tokens=8192, benchmark_kwargs=None, **kwargs):
    scores = {}
    if benchmarks == 'all': benchmarks = BENCHMARKS.keys()
    for name in benchmarks:
        bm_kwargs = {**kwargs, **(benchmark_kwargs or {}).get(name, {})}
        for metric, value in run_benchmark(model, name, batch_size_tokens=batch_size_tokens, **bm_kwargs).items():
            scores[f'{name}/{metric}'] = value
    return scores


if __name__ == '__main__':
    from models.qwen import load

    # load model
    model = load('Qwen/Qwen3-0.6B-Base')
    model.forward = jax.jit(model.forward)

    # run evals
    scores = run_set(model, size=512)
    print(scores)
