from dataclasses import dataclass
import os
from pathlib import Path

import numpy as np


DATASET_TOKEN_IDS = {
    "c4_en": 0,
    "c4_es": 1,
    "fineweb_edu_10B": 0,
    "nemotron_math_20M": 1,
}


def load_token_id(ds_path):
    ds_name = Path(ds_path).stem
    if ds_name not in DATASET_TOKEN_IDS:
        raise ValueError(f"unknown dataset token id for {ds_path}")
    return DATASET_TOKEN_IDS[ds_name]


@dataclass
class BatchLoader:
    data: np.memmap
    batch_indices: np.ndarray
    data_token_id: int

    def __len__(self):
        return len(self.batch_indices)

    def get_batch(self, batch_idx):
        batch = np.asarray(self.data[batch_idx]).copy()
        batch[..., 0] = self.data_token_id
        return batch

    def batches(self, rng=None):
        batch_indices = self.batch_indices if rng is None else reshuffle_batches(rng, self.batch_indices)
        for batch_idx in batch_indices:
            yield self.get_batch(batch_idx)

    def epochs(self, rng):
        batch_indices = self.batch_indices
        while True:
            for batch_idx in batch_indices:
                yield self.get_batch(batch_idx)
            batch_indices = reshuffle_batches(rng, batch_indices)


def reshuffle_batches(rng, batch_indices):
    return rng.permuted(batch_indices, axis=None)


def load_ds(split_seed, ds_path, seq_len, batch_size, n_tokens_valid):

    # get dataset size
    print('getting dataset size...')
    ds_path = os.path.expanduser(ds_path)
    data_token_id = load_token_id(ds_path)
    tokens = np.memmap(ds_path, dtype=np.uint16, mode='r')
    n_seq_dataset = len(tokens) // seq_len

    # sample a train/valid split
    n_seq_valid = n_tokens_valid // seq_len
    if n_seq_valid > n_seq_dataset:
        raise ValueError(f"requested {n_seq_valid:_} validation sequences but dataset only has {n_seq_dataset:_}")

    if n_seq_valid and (n_seq_valid < batch_size):
        raise ValueError(f"eval split must contain at least {batch_size:_} sequences")
    if n_seq_dataset - n_seq_valid < batch_size:
        raise ValueError(f"train split must contain at least {batch_size:_} sequences")

    # memmap contiguous sequences
    print('reading data...')
    data = np.memmap(ds_path, dtype=np.uint16, shape=[n_seq_dataset, seq_len], mode='r')

    # shuffle sequences, split them once, then group each split into batches
    print('shuffling data...')
    rng = np.random.default_rng(split_seed)
    seq_indices = rng.permutation(n_seq_dataset).astype(np.int32)
    seq_valid = seq_indices[:n_seq_valid]
    seq_train = seq_indices[n_seq_valid:]

    n_batch_valid = len(seq_valid) // batch_size
    n_batch_train = len(seq_train) // batch_size
    idx_valid = seq_valid[:n_batch_valid * batch_size].reshape(-1, batch_size)
    idx_train = seq_train[:n_batch_train * batch_size].reshape(-1, batch_size)

    ds_train = BatchLoader(data, idx_train, data_token_id)
    return ds_train, ds_train.get_batch(idx_valid)
