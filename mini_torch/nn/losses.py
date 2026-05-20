import numpy as np
from .module import Module
from ..tensor import Tensor
from ..ops import CrossEntropy


class CrossEntropyLoss(Module):
    def forward(self, pred: Tensor, target) -> Tensor:
        tgt = target if isinstance(target, Tensor) else Tensor(np.asarray(target), requires_grad=False)
        return CrossEntropy.apply(pred, tgt)


class MSELoss(Module):
    """Mean Squared Error: (1/N) * sum((pred - target)^2)"""
    def forward(self, pred: Tensor, target) -> Tensor:
        if not isinstance(target, Tensor):
            target = Tensor(np.asarray(target, dtype=np.float32), requires_grad=False)
        diff = pred - target
        return (diff * diff).mean()
