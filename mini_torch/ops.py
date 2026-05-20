import numpy as np
from typing import Any, Optional, Tuple
from .tensor import Tensor, as_tensor


class Context:
    """Stores intermediate values needed for the backward pass."""
    def __init__(self):
        self.saved_tensors: Tuple[np.ndarray, ...] = ()
        self.saved_values: Tuple[Any, ...] = ()

    def save_for_backward(self, *xs: np.ndarray) -> None:
        self.saved_tensors = tuple(xs)

    def save_values(self, *vals: Any) -> None:
        self.saved_values = tuple(vals)


def _sum_to_shape(grad: np.ndarray, shape: tuple) -> np.ndarray:
    """Sum grad over axes that were broadcast to produce the given shape."""
    if shape == ():
        return np.array(grad.sum(), dtype=np.float32)
    ndim_diff = grad.ndim - len(shape)
    sum_axes = list(range(ndim_diff))
    for i, (s_out, s_in) in enumerate(zip(grad.shape[ndim_diff:], shape)):
        if s_in == 1 and s_out != 1:
            sum_axes.append(ndim_diff + i)
    if sum_axes:
        grad = grad.sum(axis=tuple(sum_axes), keepdims=(ndim_diff == 0))
    return grad.reshape(shape).astype(np.float32)


class Function:
    """Base class for autograd operations. One instance per forward call."""
    def __init__(self, ctx: Context, parents: Tuple["Tensor", ...]):
        self.ctx = ctx
        self.parents = parents

    @staticmethod
    def forward(ctx: Context, *xs: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @staticmethod
    def backward(ctx: Context, grad_out: np.ndarray) -> Tuple[Optional[np.ndarray], ...]:
        raise NotImplementedError

    @classmethod
    def apply(cls, *inputs: Any) -> "Tensor":
        """
        Entry point for graph construction:
        1. Wrap inputs as Tensors, run forward().
        2. Build output Tensor with requires_grad set appropriately.
        3. Attach this Function node as output.grad_fn if gradient tracking is needed.
        """
        parents = tuple(as_tensor(x) for x in inputs)
        needs_grad = any(p.requires_grad for p in parents)
        ctx = Context()
        result = cls.forward(ctx, *[p.data for p in parents])
        out = Tensor(result, requires_grad=needs_grad)
        if needs_grad:
            out.grad_fn = cls(ctx, parents)
        return out


class Add(Function):
    @staticmethod
    def forward(ctx, a, b):
        ctx.save_values(a.shape, b.shape)
        return (a + b).astype(np.float32)

    @staticmethod
    def backward(ctx, grad_out):
        a_shape, b_shape = ctx.saved_values
        return _sum_to_shape(grad_out, a_shape), _sum_to_shape(grad_out, b_shape)


class Mul(Function):
    @staticmethod
    def forward(ctx, a, b):
        ctx.save_for_backward(a, b)
        ctx.save_values(a.shape, b.shape)
        return (a * b).astype(np.float32)

    @staticmethod
    def backward(ctx, grad_out):
        a, b = ctx.saved_tensors
        a_shape, b_shape = ctx.saved_values
        return (_sum_to_shape((grad_out * b).astype(np.float32), a_shape),
                _sum_to_shape((grad_out * a).astype(np.float32), b_shape))


class Neg(Function):
    @staticmethod
    def forward(ctx, x):
        return (-x).astype(np.float32)

    @staticmethod
    def backward(ctx, grad_out):
        return (-grad_out.astype(np.float32),)


class Pow(Function):
    @staticmethod
    def forward(ctx, a, b):
        out = np.power(a, b).astype(np.float32)
        ctx.save_for_backward(a, b)
        ctx.save_values(a.shape, b.shape)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        a, b = ctx.saved_tensors
        a_shape, b_shape = ctx.saved_values
        da = (b * np.power(a, b - 1) * grad_out).astype(np.float32)
        with np.errstate(divide='ignore', invalid='ignore'):
            log_a = np.where(a > 0, np.log(np.maximum(a, 1e-30)), 0.0)
        db = (np.power(a, b) * log_a * grad_out).astype(np.float32)
        return _sum_to_shape(da, a_shape), _sum_to_shape(db, b_shape)


class MatMul(Function):
    @staticmethod
    def forward(ctx, a, b):
        ctx.save_for_backward(a, b)
        ctx.save_values(a.ndim == 1, b.ndim == 1)
        return np.matmul(a, b).astype(np.float32)

    @staticmethod
    def backward(ctx, grad_out):
        a, b = ctx.saved_tensors
        a_1d, b_1d = ctx.saved_values
        a_2d = a[np.newaxis, :] if a_1d else a
        b_2d = b[:, np.newaxis] if b_1d else b
        if a_1d and b_1d:
            grad_mat = np.array([[float(grad_out)]], dtype=np.float32)
        elif a_1d:
            grad_mat = np.expand_dims(grad_out, axis=-2)
        elif b_1d:
            grad_mat = np.expand_dims(grad_out, axis=-1)
        else:
            grad_mat = grad_out
        da = grad_mat @ np.swapaxes(b_2d, -1, -2)
        db = np.swapaxes(a_2d, -1, -2) @ grad_mat
        grad_a = _sum_to_shape(da, a_2d.shape)
        grad_b = _sum_to_shape(db, b_2d.shape)
        if a_1d:
            grad_a = grad_a.squeeze(0)
        if b_1d:
            grad_b = grad_b.squeeze(-1)
        return grad_a.astype(np.float32), grad_b.astype(np.float32)


class Sum(Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_values(x.shape)
        return np.array(x.sum(), dtype=np.float32)

    @staticmethod
    def backward(ctx, grad_out):
        (shape,) = ctx.saved_values
        return (np.ones(shape, dtype=np.float32) * grad_out,)


class Mean(Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_values(x.shape)
        return np.array(x.mean(), dtype=np.float32)

    @staticmethod
    def backward(ctx, grad_out):
        (shape,) = ctx.saved_values
        n = int(np.prod(shape)) if shape else 1
        return (np.ones(shape, dtype=np.float32) * grad_out / n,)


class ReLU(Function):
    @staticmethod
    def forward(ctx, x):
        out = np.maximum(x, 0).astype(np.float32)
        ctx.save_for_backward((x > 0).astype(np.float32))
        return out

    @staticmethod
    def backward(ctx, grad_out):
        (active,) = ctx.saved_tensors
        return (grad_out * active,)


class Sigmoid(Function):
    @staticmethod
    def forward(ctx, x):
        out = np.empty_like(x, dtype=np.float32)
        pos = x >= 0
        out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
        e_x = np.exp(x[~pos])
        out[~pos] = e_x / (1.0 + e_x)
        ctx.save_for_backward(out)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        (out,) = ctx.saved_tensors
        return ((grad_out * out * (1.0 - out)).astype(np.float32),)


class CrossEntropy(Function):
    @staticmethod
    def forward(ctx, logits, target):
        if logits.ndim != 2:
            raise ValueError("CrossEntropy expects logits with shape (N, C).")
        N, C = logits.shape
        shifted = logits - np.max(logits, axis=1, keepdims=True)
        exps = np.exp(shifted)
        denom = np.sum(exps, axis=1, keepdims=True)
        probs = exps / denom
        log_probs = shifted - np.log(denom)
        if target.ndim == 1:
            labels = target.astype(np.int64)
            loss = -np.mean(log_probs[np.arange(N), labels]).astype(np.float32)
            target_probs = np.zeros((N, C), dtype=np.float32)
            target_probs[np.arange(N), labels] = 1.0
        elif target.ndim == 2:
            target_probs = target.astype(np.float32)
            loss = -np.mean(np.sum(target_probs * log_probs, axis=1)).astype(np.float32)
        else:
            raise ValueError("Target must have shape (N,) or (N, C).")
        ctx.save_for_backward(probs.astype(np.float32), target_probs.astype(np.float32))
        ctx.save_values(N)
        return np.array(loss, dtype=np.float32)

    @staticmethod
    def backward(ctx, grad_out):
        probs, target_probs = ctx.saved_tensors
        (N,) = ctx.saved_values
        grad = (probs - target_probs) / float(N)
        return (grad * grad_out).astype(np.float32), None
