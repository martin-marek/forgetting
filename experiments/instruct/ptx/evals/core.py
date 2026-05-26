"""
DCLM CORE benchmark evaluation (22 ICL tasks).
https://arxiv.org/abs/2406.11794
"""

import csv
import itertools
import json
import random
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import yaml
from jinja2 import Template

from evals.styles import (
    compare_completion_logprobs,
    score_lm_exact_match,
    score_schema_common_suffix,
)

# ---------------------------------------------------------------------------
# Constants

EVAL_BUNDLE_URL = "https://karpathy-public.s3.us-west-2.amazonaws.com/eval_bundle.zip"
CACHE_DIR = Path("~/.cache/ptx").expanduser()

# ---------------------------------------------------------------------------
# Jinja2 templates

# MC template renders the prompt prefix WITHOUT the choice (choice is passed
# separately to compare_completion_logprobs as a completion).
MC_TEMPLATE = Template(
    """
{%- for example in fewshot_examples -%}
{{ example.query }}{{ continuation_delimiter }}{{ example.choices[example.gold] }}

{% endfor -%}
{{ item.query }}{{ continuation_delimiter }}""".strip()
)

SCHEMA_TEMPLATE = Template(
    """
{%- for example in fewshot_examples -%}
{{ example.context_options[example.gold] }}{{ continuation_delimiter }}{{ example.continuation }}

{% endfor -%}
{{ context }}{{ continuation_delimiter }}{{ item.continuation }}""".strip()
)

LM_TEMPLATE = Template(
    """
{%- for example in fewshot_examples -%}
{{ example.context | trim }}{{ continuation_delimiter }}{{ example.continuation }}

{% endfor -%}
{{ item.context | trim }}{{ continuation_delimiter }}{% if include_continuation %}{{ item.continuation }}{% endif %}""".strip()
)

# ---------------------------------------------------------------------------
# Data download


def download_eval_bundle(cache_dir=CACHE_DIR):
    """Download and unzip eval_bundle.zip from S3 if not already cached."""
    eval_bundle_dir = cache_dir / "eval_bundle"
    if eval_bundle_dir.exists():
        return eval_bundle_dir
    cache_dir.mkdir(parents=True, exist_ok=True)
    zip_path = cache_dir / "eval_bundle.zip"
    print(f"Downloading {EVAL_BUNDLE_URL}...")
    urllib.request.urlretrieve(EVAL_BUNDLE_URL, zip_path)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(cache_dir)
    zip_path.unlink()
    return eval_bundle_dir


# ---------------------------------------------------------------------------
# Prompt rendering


def sample_fewshot(data, idx, n, seed_base=1234):
    """Deterministic fewshot sampling matching nanochat's seeding."""
    if n == 0:
        return []
    rng = random.Random(seed_base + idx)
    available = [i for i in range(len(data)) if i != idx]
    return [data[i] for i in rng.sample(available, n)]


def render_mc(item, delimiter, fewshot):
    """Render MC prompt prefix (without choice). Returns (prompt, choices, gold)."""
    ctx = {"fewshot_examples": fewshot, "continuation_delimiter": delimiter, "item": item}
    prompt = MC_TEMPLATE.render(**ctx)
    return prompt, item["choices"], item["gold"]


def render_schema(item, delimiter, fewshot):
    """Render schema prompts. Returns list of full strings (one per context option)."""
    ctx = {"fewshot_examples": fewshot, "continuation_delimiter": delimiter, "item": item}
    return [SCHEMA_TEMPLATE.render(context=opt, **ctx) for opt in item["context_options"]]


def render_lm(item, delimiter, fewshot):
    """Render LM prompts. Returns (prompt_without, prompt_with)."""
    ctx = {"fewshot_examples": fewshot, "continuation_delimiter": delimiter, "item": item}
    prompt_without = LM_TEMPLATE.render(include_continuation=False, **ctx).strip()
    prompt_with = LM_TEMPLATE.render(include_continuation=True, **ctx)
    return prompt_without, prompt_with


# ---------------------------------------------------------------------------
# Scoring by task type


def eval_multiple_choice(data, model, task_meta, seq_len, batch_size):
    """Evaluate MC task. Returns (accuracy, skipped, total)."""
    delimiter = task_meta["continuation_delimiter"]
    num_fewshot = task_meta["num_fewshot"]

    dataset = []
    for idx in range(len(data)):
        fewshot = sample_fewshot(data, idx, num_fewshot)
        prompt, choices, gold = render_mc(data[idx], delimiter, fewshot)
        dataset.append([prompt, choices, gold])

    # Pad variable-length choice lists to rectangular shape.
    max_choices = max(len(row[1]) for row in dataset)
    if any(len(row[1]) < max_choices for row in dataset):
        dataset = [[p, c + [""] * (max_choices - len(c)), g] for p, c, g in dataset]

    # Pre-filter examples that cannot fit in seq_len.
    all_prompts = [p for p, _, _ in dataset]
    all_completions = [comp for _, c, _ in dataset for comp in c]
    prompt_tok_lens = [len(t) for t in model.tokenizer(all_prompts)["input_ids"]]
    comp_tok_lens = [len(t) for t in model.tokenizer(all_completions)["input_ids"]]
    n_comps = max_choices
    total = len(dataset)
    filtered = [
        row
        for i, row in enumerate(dataset)
        if prompt_tok_lens[i] + max(comp_tok_lens[i * n_comps : (i + 1) * n_comps]) < seq_len
    ]

    skipped = total - len(filtered)
    correct = []
    for batch in itertools.batched(filtered, batch_size):
        if len(batch) < batch_size:
            skipped += len(batch)
            continue
        batch_results = compare_completion_logprobs(list(batch), model, seq_len)
        if not batch_results:
            skipped += len(batch)
            continue
        correct.extend(batch_results.get("accuracy_norm", []))

    accuracy = np.mean(correct).item() if correct else 0.0
    return accuracy, skipped, total


def eval_schema(data, model, task_meta, seq_len):
    """Evaluate schema task. Returns (accuracy, skipped, total)."""
    delimiter = task_meta["continuation_delimiter"]
    num_fewshot = task_meta["num_fewshot"]

    examples = []
    for idx in range(len(data)):
        item = data[idx]
        fewshot = sample_fewshot(data, idx, num_fewshot)
        options = render_schema(item, delimiter, fewshot)
        examples.append((options, item["gold"]))

    scores = score_schema_common_suffix(examples, model, seq_len)
    accuracy = np.mean(scores["correct"]).item() if scores["correct"] else 0.0
    return accuracy, scores["skipped"], len(examples)


def eval_language_modeling(data, model, task_meta, seq_len, batch_size):
    """Evaluate LM task. Returns (accuracy, skipped, total)."""
    delimiter = task_meta["continuation_delimiter"]
    num_fewshot = task_meta["num_fewshot"]

    examples = []
    for idx in range(len(data)):
        fewshot = sample_fewshot(data, idx, num_fewshot)
        examples.append(render_lm(data[idx], delimiter, fewshot))

    scores = score_lm_exact_match(examples, model, seq_len, batch_size)
    accuracy = np.mean(scores["correct"]).item() if scores["correct"] else 0.0
    return accuracy, scores["skipped"], len(examples)


# ---------------------------------------------------------------------------
# Entry point


def run_core(model, seq_len=2048, batch_size_tokens=8192, max_per_task=-1, max_fewshot=-1):
    """Run all DCLM CORE tasks."""
    bundle_dir = download_eval_bundle()

    with open(bundle_dir / "core.yaml") as f:
        config = yaml.safe_load(f)

    baselines = {}
    with open(bundle_dir / "eval_meta_data.csv") as f:
        for row in csv.DictReader(f):
            baselines[row["Eval Task"]] = float(row["Random baseline"])

    results = {}
    centered_results = {}

    for task in config["icl_tasks"]:
        label = task["label"]
        num_fewshot = task["num_fewshot"][0]
        if max_fewshot >= 0:
            num_fewshot = min(num_fewshot, max_fewshot)
        task_meta = {
            "task_type": task["icl_task_type"],
            "num_fewshot": num_fewshot,
            "continuation_delimiter": task.get("continuation_delimiter", " "),
        }

        # Load data
        data_path = bundle_dir / "eval_data" / task["dataset_uri"]
        with open(data_path) as f:
            data = [json.loads(line) for line in f]

        # Optional deterministic subsampling
        if max_per_task > 0 and len(data) > max_per_task:
            rng = random.Random(1337)
            rng.shuffle(data)
            data = data[:max_per_task]

        print(
            f"Evaluating: {label} ({task_meta['num_fewshot']}-shot, {task_meta['task_type']})... ",
            end="",
            flush=True,
        )

        if task_meta["task_type"] == "multiple_choice":
            n_choices = max((len(item["choices"]) for item in data), default=1)
            batch_size = max(8, (batch_size_tokens // (seq_len * n_choices) // 8) * 8)
            accuracy, skipped, total = eval_multiple_choice(data, model, task_meta, seq_len, batch_size)
        elif task_meta["task_type"] == "schema":
            accuracy, skipped, total = eval_schema(data, model, task_meta, seq_len)
        elif task_meta["task_type"] == "language_modeling":
            batch_size = max(8, (batch_size_tokens // seq_len // 8) * 8)
            accuracy, skipped, total = eval_language_modeling(data, model, task_meta, seq_len, batch_size)
        else:
            raise ValueError(f"Unknown task type: {task_meta['task_type']}")

        baseline = baselines[label]
        centered = (accuracy - 0.01 * baseline) / (1.0 - 0.01 * baseline)

        results[label] = accuracy
        centered_results[label] = centered

        if skipped > 0 and total > 0:
            print(f"skipped {skipped / total:.0%}, ", end="")
        print(f"accuracy={accuracy:.4f}, centered={centered:.4f}")

    core_metric = sum(centered_results.values()) / len(centered_results)
    print(f"\nCORE metric: {core_metric:.4f}")

    return {
        "results": results,
        "centered_results": centered_results,
        "core_metric": core_metric,
    }
