import numpy as np
from torch import Tensor
from torch import nn


class Mlp(nn.Sequential):
    def __init__(self, dims: list[int]):
        super().__init__()
        for in_dim, out_dim, is_last in zip(
            dims[:-1], dims[1:], [i == len(dims) for i in range(len(dims) - 1)]
        ):
            self.append(nn.Linear(in_dim, out_dim))
            if not is_last:
                self.append(nn.ReLU())

    def forward(self, x: Tensor) -> Tensor:
        return super().forward(x)


class Encoder(nn.Module):
    def __init__(self, image_dims: tuple[int, int], latent_dim: int):
        super().__init__()
        hidden_dim = 2 * latent_dim
        self.latent_dim = latent_dim
        self.mlp = Mlp([np.prod(image_dims), hidden_dim, hidden_dim, 2 * latent_dim])

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        r"""
        Input: input image of dims (batch_size, image_height, image_width)
        Output: mean and variance of dims (batch_size, latent_dim)
        """
        z = self.mlp.forward(x.flatten(1))
        return z[:, : self.latent_dim], z[:, self.latent_dim :]


class Decoder(nn.Module):
    def __init__(self, latent_dim: int, image_dims: tuple[int, int]):
        super().__init__()
        hidden_dim = 2 * latent_dim
        self.image_dims = image_dims
        self.mlp = Mlp([latent_dim, hidden_dim, hidden_dim, np.prod(image_dims)])

    def forward(self, z: Tensor) -> Tensor:
        r"""
        Input: latent vector of dims (batch_size, latent_dim)
        Output: image of dims (batch_size, image_height, image_width)
        """
        x = self.mlp.forward(z)
        return x.reshape((-1, *self.image_dims))
