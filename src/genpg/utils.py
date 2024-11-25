import numpy as np
import torch as tch
from torch import nn, Tensor


class Accum:
    def __init__(self, *names: str):
        self.names = names
        self.items = []

    def append(self, *items: float):
        assert len(self.names) == len(items), f"{self.names} != {items}"
        self.items.append(items)

    def mean(self) -> dict[str, float]:
        means = np.array(self.items).transpose().mean(1)
        return {k: v for k, v in zip(self.names, means)}


class Print(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        print(x.shape)
        return x


ToTensor = float | list | np.ndarray | Tensor


def to_tensor(value: ToTensor) -> Tensor:
    if not isinstance(value, Tensor):
        if isinstance(value, np.ndarray):
            array = value
        elif isinstance(value, list):
            array = np.array(value)
        else:
            array = np.array([value])
        value = tch.from_numpy(array)
    return value
