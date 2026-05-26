from pathlib import Path
import os
import sys

import fire
import numpy as np
from datasets import load_dataset
from tqdm import tqdm
from transformers import PreTrainedTokenizerFast


def load_tokenizer(tokenizer_dir):
    tok = PreTrainedTokenizerFast.from_pretrained(Path(tokenizer_dir).expanduser())
    if tok.eos_token_id is None:
        raise ValueError(f"{tokenizer_dir}: tokenizer must define EOS")
    return tok


def pack_batch(tok, texts, buffer, array, offset, bar):
    batches = tok(texts, add_special_tokens=True, return_attention_mask=False)["input_ids"]
    for ids in batches:
        buffer.extend(ids)
    take = min(len(buffer), len(array) - offset)
    if take:
        array[offset : offset + take] = np.asarray(buffer[:take], dtype=array.dtype)
        del buffer[:take]
        offset += take
        bar.update(take)
    return offset


def main(
    output_path,
    dataset="allenai/c4",
    dataset_config=None,
    split="train",
    tokens=100_000_000,
    batch_docs=4096,
    tokenizer_dir="~/datasets/c4_tokenizer_4096",
    shuffle=False,
    seed=0,
    shuffle_buffer=50_000,
):
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
    output_path = Path(output_path).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tok = load_tokenizer(tokenizer_dir)
    if tok.vocab_size > np.iinfo(np.uint16).max:
        raise ValueError(f"{tokenizer_dir}: vocab_size={tok.vocab_size} exceeds uint16 memmap capacity")
    dtype = np.uint16
    array = np.memmap(output_path, mode="w+", dtype=dtype, shape=(tokens,))
    buffer, docs, docs_seen, offset = [], [], 0, 0
    bar = tqdm(total=tokens, desc=output_path.name, unit="tok", unit_scale=True)
    ds = load_dataset(dataset, dataset_config, split=split, streaming=True)
    if shuffle:
        ds = ds.shuffle(seed=seed, buffer_size=shuffle_buffer)
    for example in ds:
        text = example["text"].strip()
        if not text:
            continue
        docs.append(text)
        docs_seen += 1
        if len(docs) < batch_docs:
            continue
        offset = pack_batch(tok, docs, buffer, array, offset, bar)
        docs.clear()
        if offset >= tokens:
            break
    if offset < tokens and docs:
        offset = pack_batch(tok, docs, buffer, array, offset, bar)
    bar.close()
    if offset < tokens:
        raise RuntimeError(f"{output_path}: only packed {offset:,} tokens from {docs_seen:,} docs")
    array.flush()
    dataset_desc = dataset if dataset_config is None else f"{dataset}/{dataset_config}"
    print(f"wrote {output_path} from {dataset_desc} split={split} with shape={array.shape} dtype={array.dtype}")


if __name__ == "__main__":
    fire.Fire(main)
    # Some streaming HF datasets crash during Python interpreter teardown after successful writes.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
