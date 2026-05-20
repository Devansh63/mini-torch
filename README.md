# mini-torch

A minimal PyTorch-style deep learning library built **from scratch using only NumPy**. Implements dynamic computation graphs, reverse-mode automatic differentiation, neural network layers, and an SGD optimizer — enough to train a 2-layer MLP to >95% accuracy on MNIST.

Built as part of graduate coursework in machine learning at UIUC.

---

## What it does

`mini_torch` mirrors the core PyTorch API surface:

```python
import mini_torch as torch
import mini_torch.nn as nn
from mini_torch.optim import SGD

# Define a model
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)

model = MLP()
opt = SGD(model.parameters(), lr=0.35)
loss_fn = nn.CrossEntropyLoss()

# Training loop
for xb, yb in dataloader:
    pred = model(xb)
    loss = loss_fn(pred, yb)
    opt.zero_grad()
    loss.backward()
    opt.step()
```

---

## How autograd works

Every operation on a `Tensor` creates a `Function` node that records inputs and context needed for the backward pass. These nodes form a directed acyclic graph (DAG) during the forward pass.

```
Forward pass builds the graph:

  x ──┬──> [ Mul ] ──> a ──┐
  w ──┘                     ├──> [ Add ] ──> c ──> [ Mean ] ──> loss
  x ─────────────────────────┘

Backward pass traverses in reverse topological order:

  loss.backward()
    └─> Mean.backward()  → grad for c
          └─> Add.backward()   → grad splits to a and x
                └─> Mul.backward()  → grad for x (via mul) and w
```

### `Function.apply()` — the entry point

Every op goes through `Function.apply()`, which:
1. Wraps inputs as `Tensor` objects
2. Runs `forward(ctx, *raw_arrays)` and stores intermediates in `ctx`
3. Builds an output `Tensor` with `requires_grad` set appropriately
4. Attaches the `Function` node as `output.grad_fn` to enable backprop

```python
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
        da = _sum_to_shape((grad_out * b), a_shape)  # handles broadcasting
        db = _sum_to_shape((grad_out * a), b_shape)
        return da, db
```

### `Tensor.backward()` — topological traversal

Starts from the scalar loss, performs a DFS to collect all reachable tensors in topological order, then traverses in reverse:
- For each node: calls `grad_fn.backward(ctx, grad)` → gets gradients for each parent
- For leaf tensors with `requires_grad=True`: **accumulates** gradient into `.grad` (across multiple `backward()` calls)
- After the pass: frees the graph (clears `grad_fn`, `parents`, `ctx`) to prevent memory leaks

---

## Implemented operations

| Operation | Forward | Backward |
|---|---|---|
| `Add` | `a + b` | `grad_out` to both (sum over broadcast dims) |
| `Mul` | `a * b` | `grad_out * b` → a, `grad_out * a` → b |
| `Neg` | `-x` | `-grad_out` |
| `Pow` | `x ** n` | `n * x**(n-1) * grad_out` |
| `MatMul` | `a @ b` | `grad_out @ b.T` → a, `a.T @ grad_out` → b |
| `Mean` | mean over all elements | `grad_out / N` broadcast to input shape |
| `ReLU` | `max(0, x)` | `grad_out * (x > 0)` |
| `Sigmoid` | `1 / (1 + e^-x)` | `σ(x) * (1 - σ(x)) * grad_out` |
| `CrossEntropy` | numerically stable log-softmax + NLL | `(softmax - target) / N` |

All ops handle N-D tensors and broadcasting correctly. `MatMul` handles 1D vector inputs by temporarily reshaping to 2D.

---

## Project structure

```
mini_torch/
├── tensor.py          # Tensor class + backward() algorithm
├── ops.py             # Function base class + all operation nodes
├── nn/
│   ├── module.py      # Module base class, parameters() traversal
│   ├── layers.py      # Linear (He-initialized weights)
│   ├── activations.py # ReLU, Sigmoid
│   └── losses.py      # MSELoss, CrossEntropyLoss
├── optim/
│   └── sgd.py         # SGD optimizer
└── utils/
    └── data/
        ├── dataset.py    # Dataset, TensorDataset
        └── dataloader.py # DataLoader with shuffle + batching

mnist_classification.py  # End-to-end MNIST training script
```

---

## MNIST results

Architecture: `Linear(784→256) → ReLU → Linear(256→128) → ReLU → Linear(128→10)`
Optimizer: SGD, lr=0.35, batch size=128, 15 epochs
Data: normalized (mean/std per pixel over training set)

| Epoch | Test Loss | Test Accuracy |
|-------|-----------|---------------|
| 1     | ~0.45     | ~88%          |
| 5     | ~0.18     | ~94%          |
| 10    | ~0.12     | ~96%          |
| 15    | ~0.10     | **>96%**      |

---

## Setup

```bash
pip install numpy==2.2.0

# Download MNIST into data/mnist/ (IDX gzip format), then:
python mnist_classification.py

# Run tests
pip install pytest==9.0.2
pytest -q tests/
```

Requires Python 3.10+. No PyTorch, no JAX — only NumPy.

---

## Key design decisions

**Broadcasting-aware gradients**: when an input was broadcast in the forward pass, `_sum_to_shape()` sums the output gradient back down to the original shape. This is required for operations like `out = x + bias` where `bias` has shape `(out_features,)` but `x` has shape `(batch, out_features)`.

**Graph freeing after backward**: after `backward()` completes, all `grad_fn`, `ctx`, and `parents` references are cleared. This prevents double-backward and ensures memory is released.

**Gradient accumulation on leaves**: leaf tensors (model parameters) accumulate gradients across `backward()` calls — `p.grad += new_grad`. This matches PyTorch semantics and requires `optimizer.zero_grad()` before each backward pass.

**He initialization**: `Linear` layers initialize weights with `N(0, sqrt(2/fan_in))`, which keeps activation variance stable through ReLU layers and avoids vanishing/exploding gradients at initialization.
