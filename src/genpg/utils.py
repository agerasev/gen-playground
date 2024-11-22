import numpy as np


class Stats:
    def __init__(self, *names: str):
        self.names = names
        self.items = []

    def append(self, *items: float):
        assert len(self.names) == len(items), f"{self.names} != {items}"
        self.items.append(items)

    def mean(self) -> dict[str, float]:
        means = np.array(self.items).transpose().mean(1)
        return {k: v for k, v in zip(self.names, means)}
