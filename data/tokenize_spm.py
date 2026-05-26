import urllib.request
from pathlib import Path

import fire
import numpy as np
import sentencepiece as spm
from datasets import load_dataset
from tqdm import tqdm

SP_URL = "https://storage.googleapis.com/t5-data/vocabs/cc_all.32000.100extra/sentencepiece.model"
SP_PATH = "~/.cache/c4_pack/sentencepiece_cc_all.32000.100extra.model"


def load_tokenizer(sp_model):
    sp_model = Path(sp_model).expanduser()
    if not sp_model.exists():
        sp_model.parent.mkdir(parents=True, exist_ok=True)
        print(f"downloading {SP_URL} -> {sp_model}")
        urllib.request.urlretrieve(SP_URL, sp_model)
    return spm.SentencePieceProcessor(model_file=str(sp_model))


def pack_batch(sp, texts, eos_id, buffer, array, offset, bar):
    batches = sp.encode(texts, out_type=int)
    for ids in batches:
        buffer.extend(ids)
        buffer.append(eos_id)
    take = min(len(buffer), len(array) - offset)
    if take:
        array[offset : offset + take] = np.asarray(buffer[:take], dtype=array.dtype)
        del buffer[:take]
        offset += take
        bar.update(take)
    return offset


def main(
    subsets=["en", "es", "fr", "pt", "de"],
    tokens=100_000_000,
    batch_docs=4096,
    out_dir="~/datasets/c4",
    sp_model=SP_PATH,
):
    out_dir = Path(out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    sp = load_tokenizer(sp_model)
    eos_id = sp.eos_id()

    for lang in subsets:
        path = out_dir / f"c4_{lang}.bin"
        array = np.memmap(path, mode="w+", dtype=np.uint16, shape=(tokens,))
        buffer, docs, offset = [], [], 0
        bar = tqdm(total=tokens, desc=lang, unit="tok", unit_scale=True)
        for example in load_dataset("allenai/c4", lang, split="train", streaming=True):
            docs.append(example["text"])
            if len(docs) < batch_docs:
                continue
            offset = pack_batch(sp, docs, eos_id, buffer, array, offset, bar)
            docs.clear()
            if offset >= tokens:
                break
        if offset < tokens and docs:
            offset = pack_batch(sp, docs, eos_id, buffer, array, offset, bar)
        bar.close()
        if offset < tokens:
            raise RuntimeError(f"{lang}: only packed {offset:,} tokens")
        array.flush()
        print(f"{lang}: wrote {path} with shape={array.shape} dtype={array.dtype}")


if __name__ == "__main__":
    fire.Fire(main)
