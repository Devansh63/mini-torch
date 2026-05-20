from __future__ import annotations
from typing import Any, Tuple
from ...tensor import Tensor


class Dataset:
    def __len__(self) -> int:
        raise NotImplementedError

    def __getitem__(self, idx: int) -> Any:
        raise NotImplementedError


class TensorDataset(Dataset):
    """Wraps one or more Tensors into a Dataset. All must share the same first dimension."""
    def __init__(self, *tensors: Tensor):
        if not tensors:
            raise ValueError("TensorDataset requires at least one Tensor.")
        for t in tensors:
            if not isinstance(t, Tensor):
                raise TypeError(f"Expected Tensor, got {type(t)}")
        size0 = tensors[0].data.shape[0]
        for t in tensors:
            if t.data.shape[0] != size0:
                raise ValueError("All tensors must have the same size in dimension 0.")
        self.tensors = tensors

    def __len__(self) -> int:
        return self.tensors[0].data.shape[0]

    def __getitem__(self, idx: int) -> Tuple[Tensor, ...]:
        return tuple(Tensor(t.data[idx], requires_grad=False) for t in self.tensors)
