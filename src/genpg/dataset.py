import gzip
import numpy as np


def read_idx(path):
    with gzip.open(path, "rb") as f:
        assert list(f.read(2)) == [0, 0]
        dtype, dsize = {
            0x08: (np.uint8, 1),
            0x09: (np.int8, 1),
            0x0B: (np.int16, 2),
            0x0C: (np.int32, 4),
            0x0D: (np.float32, 4),
            0x0E: (np.float64, 8),
        }[f.read(1)[0]]
        n_dims = f.read(1)[0]
        dims = [int.from_bytes(f.read(4), "big") for _ in range(n_dims)]
        return np.frombuffer(f.read(np.prod(dims) * dsize), dtype=dtype).reshape(dims)


def read_data(prefix):
    return (
        read_idx(f"{prefix}-labels-idx1-ubyte.gz"),
        read_idx(f"{prefix}-images-idx3-ubyte.gz"),
    )


def read_train_data():
    return read_data("data/train")


def read_test_data():
    return read_data("data/t10k")
