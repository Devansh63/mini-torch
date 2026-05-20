import numpy as np
from .module import Module
from ..tensor import Tensor


class Linear(Module):
    """
    Fully connected layer: out = x @ W + b
    Weights use He initialization: N(0, sqrt(2/fan_in)) for stable variance through ReLU.
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.weight = Tensor(
            np.random.randn(in_features, out_features).astype(np.float32) * np.sqrt(2.0 / in_features),
            requires_grad=True,
        )
        self.bias = (Tensor(np.zeros(out_features, dtype=np.float32), requires_grad=True) if bias else None)

    def forward(self, x: Tensor) -> Tensor:
        out = x @ self.weight
        if self.bias is not None:
            out = out + self.bias
        return out
