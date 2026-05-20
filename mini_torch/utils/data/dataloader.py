from __future__ import annotations
from typing import Any, Callable, Iterable, Iterator, Optional, Sequence
import numpy as np
from .dataset import Dataset
from ...tensor import Tensor


def _default_collate(batch: Sequence[Any]) -> Any:
    """Stack a list of samples into a batched Tensor."""
    if not batch:
        raise ValueError("Cannot collate an empty batch.")
    first = batch[0]
    if isinstance(first, Tensor):
        return Tensor(np.stack([b.data for b in batch], axis=0), requires_grad=False)
    if isinstance(first, (tuple, list)):
        return type(first)(_default_collate(list(col)) for col in zip(*batch))
    if isinstance(first, dict):
        return {k: _default_collate([d[k] for d in batch]) for k in first}
    raise TypeError(f"default_collate: unsupported type {type(first)}")


class DataLoader(Iterable):
    """Iterates a Dataset in mini-batches with optional shuffling."""
    def __init__(self, dataset: Dataset, batch_size: int = 1, shuffle: bool = False,
                 drop_last: bool = False, seed: Optional[int] = None,
                 collate_fn: Optional[Callable] = None):
        if batch_size <= 0:
            raise ValueError("batch_size must be > 0")
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.seed = seed
        self.collate_fn = collate_fn or _default_collate

    def __len__(self) -> int:
        n = len(self.dataset)
        return n // self.batch_size if self.drop_last else (n + self.batch_size - 1) // self.batch_size

    def __iter__(self) -> Iterator[Any]:
        n = len(self.dataset)
        indices = np.arange(n)
        if self.shuffle:
            np.random.default_rng(self.seed).shuffle(indices)
        for start in range(0, n, self.batch_size):
            end = start + self.batch_size
            if end > n and self.drop_last:
                break
            yield self.collate_fn([self.dataset[int(i)] for i in indices[start:end]])
