from pathlib import Path
import os
import random
import sys

import fire
from datasets import load_dataset
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, processors, trainers
from transformers import PreTrainedTokenizerFast


def _reserved_tokens(n_reserved_ids):
    return [f"<resv_{i}>" for i in range(n_reserved_ids)]


def _sample_subset(dataset, subset, split, n_words, seed, shuffle_buffer, stats):
    ds = load_dataset(dataset, subset, split=split, streaming=True).shuffle(seed=seed, buffer_size=shuffle_buffer)
    docs, seen = 0, 0
    label = subset if subset is not None else "<default>"
    for row in ds:
        text = row["text"].strip()
        if not text:
            continue
        docs += 1
        seen += len(text.split())
        stats[label] = (docs, seen)
        yield text
        if seen >= n_words:
            return


def _interleave(dataset, subsets, split, n_words, seed, shuffle_buffer, stats):
    rng = random.Random(seed)
    iters = {
        subset: iter(_sample_subset(dataset, subset, split, n_words, seed + i, shuffle_buffer, stats))
        for i, subset in enumerate(subsets)
    }
    active = list(iters)
    while active:
        subset = rng.choice(active)
        try:
            yield next(iters[subset])
        except StopIteration:
            active.remove(subset)


def main(
    dataset="allenai/c4",
    subsets=["en", "es", "fr", "pt", "de"],
    split="train",
    n_words=1_000_000,
    vocab_size=4096,
    out_dir="c4_tokenizer_4096",
    shuffle_buffer=50_000,
    seed=0,
    n_reserved_ids=10,
):
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
    os.environ.setdefault("RAYON_NUM_THREADS", str(os.cpu_count() or 1))
    if n_reserved_ids < 0:
        raise ValueError(f"n_reserved_ids must be >= 0, got {n_reserved_ids}")
    if not subsets:
        subsets = [None]
    stats = {}
    reserved_tokens = _reserved_tokens(n_reserved_ids)
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    tok.train_from_iterator(
        _interleave(dataset, subsets, split, n_words, seed, shuffle_buffer, stats),
        trainer=trainers.BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=2,
            special_tokens=[*reserved_tokens, "</s>"],
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        ),
    )
    eos_id = tok.token_to_id("</s>")
    tok.post_processor = processors.Sequence(
        [
            processors.TemplateProcessing(
                single="$A </s>",
                pair="$A </s> $B </s>",
                special_tokens=[("</s>", eos_id)],
            ),
            processors.ByteLevel(trim_offsets=True),
        ]
    )
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    PreTrainedTokenizerFast(
        tokenizer_object=tok,
        eos_token="</s>",
    ).save_pretrained(out_dir)
    n_docs = 0
    for subset in subsets:
        label = subset if subset is not None else "<default>"
        docs, seen = stats.get(label, (0, 0))
        n_docs += docs
        print(f"{label}: {docs:,} docs, {seen:,} words")
    print(f"saved {n_docs:,} docs from {dataset} to {out_dir}")


if __name__ == "__main__":
    fire.Fire(main)
    # Some streaming HF datasets crash during Python interpreter teardown after successful writes.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
