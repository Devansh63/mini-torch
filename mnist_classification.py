import os
import sys
import gzip
import struct
import numpy as np

import mini_torch as torch
import mini_torch.nn as nn
from mini_torch.utils.data import TensorDataset, DataLoader
from mini_torch.optim import SGD
from mini_torch import Tensor
from mini_torch.nn import CrossEntropyLoss


def read_idx_images(path: str) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        _, n, rows, cols = struct.unpack(">IIII", f.read(16))
        return np.frombuffer(f.read(), dtype=np.uint8).reshape(n, rows, cols)


def read_idx_labels(path: str) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        _, n = struct.unpack(">II", f.read(8))
        return np.frombuffer(f.read(), dtype=np.uint8).reshape(n)


def load_mnist(root="data/mnist"):
    return (
        read_idx_images(os.path.join(root, "train-images-idx3-ubyte.gz")),
        read_idx_labels(os.path.join(root, "train-labels-idx1-ubyte.gz")),
        read_idx_images(os.path.join(root, "t10k-images-idx3-ubyte.gz")),
        read_idx_labels(os.path.join(root, "t10k-labels-idx1-ubyte.gz")),
    )


def one_hot(y: np.ndarray, num_classes=10) -> np.ndarray:
    out = np.zeros((len(y), num_classes), dtype=np.float32)
    out[np.arange(len(y)), y.astype(np.int64)] = 1.0
    return out


def accuracy(logits: np.ndarray, labels: np.ndarray) -> float:
    return float((logits.argmax(axis=1) == labels).mean())


class MNISTMLP(nn.Module):
    """3-layer MLP: 784 -> 256 -> 128 -> 10"""
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)
        self.relu = nn.ReLU()

    def forward(self, x: Tensor) -> Tensor:
        return self.fc3(self.relu(self.fc2(self.relu(self.fc1(x)))))


def main():
    np.random.seed(0)
    Xtr_img, ytr, Xte_img, yte = load_mnist("data/mnist")

    Xtr = Xtr_img.reshape(-1, 784).astype(np.float32) / 255.0
    Xte = Xte_img.reshape(-1, 784).astype(np.float32) / 255.0
    mean = Xtr.mean(axis=0, keepdims=True)
    std = Xtr.std(axis=0, keepdims=True) + 1e-6
    Xtr = (Xtr - mean) / std
    Xte = (Xte - mean) / std

    train_dl = DataLoader(TensorDataset(torch.tensor(Xtr), torch.tensor(one_hot(ytr))),
                          batch_size=128, shuffle=True)

    model = MNISTMLP()
    opt = SGD(model.parameters(), lr=0.35)
    criterion = CrossEntropyLoss()
    Xte_t = torch.tensor(Xte)
    yte_oh = one_hot(yte)

    for ep in range(1, 16):
        for xb, yb in train_dl:
            loss = criterion(model(xb), yb)
            opt.zero_grad()
            loss.backward()
            opt.step()

        logits = model(Xte_t).detach().numpy()
        shifted = logits - logits.max(axis=1, keepdims=True)
        log_probs = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))
        te_loss = -np.mean((yte_oh * log_probs).sum(axis=1))
        te_acc = accuracy(logits, yte)
        print(f"Epoch {ep:02d}/15 | loss={te_loss:.4f} | acc={te_acc:.4f}")


if __name__ == "__main__":
    main()
