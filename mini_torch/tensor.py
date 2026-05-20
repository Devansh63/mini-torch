import numpy as np
from typing import Any, Optional, Set, List


def as_tensor(x: Any) -> "Tensor":
    return x if isinstance(x, Tensor) else Tensor(x, requires_grad=False)


class Tensor:
    """
    Core tensor class. Stores data and participates in automatic differentiation
    when requires_grad=True. grad_fn points to the Function that created this tensor.
    """
    def __init__(self, data: Any, requires_grad: bool = False):
        self.data = np.array(data, dtype=np.float32)
        self.requires_grad = bool(requires_grad)
        self.grad: Optional[np.ndarray] = None
        from .ops import Function
        self.grad_fn = None

    @property
    def shape(self):
        return self.data.shape

    def zero_grad(self):
        self.grad = None

    def numpy(self) -> np.ndarray:
        if self.requires_grad:
            raise RuntimeError("Can't call numpy() on Tensor that requires grad. Use tensor.detach().numpy() instead.")
        return self.data

    def detach(self) -> "Tensor":
        return Tensor(self.data, requires_grad=False)

    def __repr__(self):
        return f"Tensor(shape={self.shape}, requires_grad={self.requires_grad})"

    def __add__(self, other):
        from .ops import Add
        return Add.apply(self, other)

    def __radd__(self, other):
        from .ops import Add
        return Add.apply(other, self)

    def __mul__(self, other):
        from .ops import Mul
        return Mul.apply(self, other)

    def __rmul__(self, other):
        from .ops import Mul
        return Mul.apply(other, self)

    def __neg__(self):
        from .ops import Neg
        return Neg.apply(self)

    def __sub__(self, other):
        from .ops import Add, Neg
        return Add.apply(self, Neg.apply(other))

    def __rsub__(self, other):
        from .ops import Add, Neg
        return Add.apply(other, Neg.apply(self))

    def __matmul__(self, other):
        from .ops import MatMul
        return MatMul.apply(self, other)

    def __pow__(self, power):
        from .ops import Pow
        return Pow.apply(self, power)

    def __truediv__(self, other):
        return self * (as_tensor(other) ** -1)

    def __rtruediv__(self, other):
        return as_tensor(other) * (self ** -1)

    def sum(self):
        from .ops import Sum
        return Sum.apply(self)

    def mean(self):
        from .ops import Mean
        return Mean.apply(self)

    def relu(self):
        from .ops import ReLU
        return ReLU.apply(self)

    def sigmoid(self):
        from .ops import Sigmoid
        return Sigmoid.apply(self)

    def backward(self):
        """
        Backpropagate gradients from this scalar Tensor.
        Traverses the computation graph in reverse topological order,
        accumulates gradients into leaf tensors, then frees the graph.
        """
        if not self.requires_grad:
            return
        if self.data.size != 1:
            raise ValueError("backward() can only be called on a scalar Tensor.")

        order: List["Tensor"] = []
        visited: Set[int] = set()

        def topo_sort(t: "Tensor") -> None:
            if id(t) in visited:
                return
            visited.add(id(t))
            if t.grad_fn is not None:
                for parent in t.grad_fn.parents:
                    topo_sort(parent)
            order.append(t)

        topo_sort(self)

        upstream: dict = {}
        upstream[id(self)] = np.ones(self.data.shape, dtype=np.float32)

        for t in reversed(order):
            if id(t) not in upstream:
                continue
            grad = upstream[id(t)]

            if t.grad_fn is None:
                if t.requires_grad:
                    t.grad = grad.copy() if t.grad is None else t.grad + grad
            else:
                node = t.grad_fn
                grads = node.backward(node.ctx, grad)
                for parent, g in zip(node.parents, grads):
                    if g is None or not parent.requires_grad:
                        continue
                    if id(parent) in upstream:
                        upstream[id(parent)] = upstream[id(parent)] + np.asarray(g, dtype=np.float32)
                    else:
                        upstream[id(parent)] = np.asarray(g, dtype=np.float32).copy()

        for t in order:
            if t.grad_fn is not None:
                t.grad_fn.ctx.saved_tensors = ()
                t.grad_fn.ctx.saved_values = ()
                t.grad_fn.ctx = None
                t.grad_fn.parents = ()
                t.grad_fn = None
