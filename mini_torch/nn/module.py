from typing import Iterator, Any
from ..tensor import Tensor


class Module:
    def __init__(self):
        self.training = True

    def forward(self, *inputs):
        raise NotImplementedError

    def __call__(self, *inputs):
        return self.forward(*inputs)

    def modules(self) -> Iterator["Module"]:
        yield self
        for v in self.__dict__.values():
            yield from _iter_modules(v)

    def parameters(self) -> Iterator[Tensor]:
        seen = set()
        for p in _iter_parameters(self.__dict__):
            if id(p) not in seen:
                seen.add(id(p))
                yield p

    def zero_grad(self):
        for p in self.parameters():
            p.zero_grad()


def _iter_modules(obj: Any) -> Iterator["Module"]:
    if isinstance(obj, Module):
        for m in obj.modules():
            yield m
    elif isinstance(obj, (list, tuple)):
        for x in obj:
            yield from _iter_modules(x)
    elif isinstance(obj, dict):
        for x in obj.values():
            yield from _iter_modules(x)


def _iter_parameters(obj: Any) -> Iterator[Tensor]:
    if isinstance(obj, Tensor):
        if getattr(obj, "requires_grad", False):
            yield obj
    elif isinstance(obj, Module):
        yield from _iter_parameters(obj.__dict__)
    elif isinstance(obj, dict):
        for v in obj.values():
            yield from _iter_parameters(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            yield from _iter_parameters(v)
