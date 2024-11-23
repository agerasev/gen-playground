import numpy as np
from torch import Tensor
from torch import nn


class Mlp(nn.Sequential):
    def __init__(
        self,
        dims: list[int],
        batch_norm: bool = False,
        dropout: bool = False,
    ):
        assert len(dims) >= 2
        super().__init__()
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            if batch_norm:
                self.append(nn.BatchNorm1d(in_dim))
            self.append(nn.Linear(in_dim, out_dim))
            if i + 1 < len(dims) - 1:
                self.append(nn.ReLU())
                if dropout:
                    self.append(nn.Dropout())

    def forward(self, x: Tensor) -> Tensor:
        return super().forward(x)


class ConvNet(nn.Sequential):
    def __init__(
        self,
        layers: list[list[int]],
        kernel_size: int,
        scale: float = 0.5,
    ):
        """
        Contains convolution layers separated by down- or up-sampling operation
        Args:
        + layers - list of layers of convolution channels
        + kernel_size - size of convolution kernel
        + scale - if > 1 then it is factor of upsampling, otherwise 1 / scale is max-pool size.
        """
        super().__init__()
        last_dim = layers[0][0]
        for i, n_channels in enumerate(layers):
            assert len(n_channels) >= 1
            if len(n_channels) == 1:
                assert n_channels[0] == last_dim
            for j, (in_dim, out_dim) in enumerate(zip(n_channels[:-1], n_channels[1:])):
                self.append(nn.Conv2d(in_dim, out_dim, kernel_size, padding="same"))
                if j + 1 < len(n_channels) - 1:
                    self.append(nn.ReLU())
                else:
                    last_dim = out_dim
            if i + 1 < len(layers):
                if scale > 1:
                    self.append(nn.UpsamplingBilinear2d(None, round(scale)))
                else:
                    self.append(nn.MaxPool2d(round(1.0 / scale)))

    def forward(self, x: Tensor) -> Tensor:
        return super().forward(x)


class Encoder(nn.Module):
    def __init__(
        self,
        image_dims: tuple[int, int],
        conv_channels: list[list[int]],
        kernel_size: int,
        fc_dims: list[int],
        **kws,
    ):
        hidden_shape = (
            conv_channels[-1][-1],
            *[d // 2 ** (len(conv_channels) - 1) for d in image_dims],
        )
        assert (
            np.prod(hidden_shape) == fc_dims[0]
        ), f"conv_output ({hidden_shape} = {np.prod(hidden_shape)}) != mlp_input ({fc_dims[0]})"

        super().__init__()
        self.conv = ConvNet(conv_channels, kernel_size, scale=0.5)
        self.mlp = Mlp(fc_dims, **kws)

    def forward(self, x: Tensor) -> Tensor:
        r"""
        Input: input image of dims (batch_size, image_height, image_width)
        Output: latent vector of dims (batch_size, latent_dim)
        """
        x = x.unsqueeze(1)
        x = self.conv.forward(x)
        x = x.flatten(1)
        x = self.mlp.forward(x)
        return x


class Decoder(nn.Sequential):
    def __init__(
        self,
        fc_dims: int,
        conv_channels: list[int],
        kernel_size: int,
        image_dims: tuple[int, int],
        **kws,
    ):
        hidden_shape = (
            conv_channels[0][0],
            *[d // 2 ** (len(conv_channels) - 1) for d in image_dims],
        )
        assert (
            np.prod(hidden_shape) == fc_dims[-1]
        ), f"mlp_output ({fc_dims[-1]}) != up_conv_input ({hidden_shape} = {np.prod(hidden_shape)})"
        self.hidden_shape = hidden_shape

        super().__init__()
        self.mlp = Mlp(fc_dims, **kws)
        self.up_conv = ConvNet(conv_channels, kernel_size, scale=2)

    def forward(self, x: Tensor) -> Tensor:
        r"""
        Input: latent vector of dims (batch_size, latent_dim)
        Output: image of dims (batch_size, image_height, image_width)
        """
        x = self.mlp.forward(x)
        x = x.reshape((x.shape[0], *self.hidden_shape))
        x = self.up_conv.forward(x)
        return x.squeeze(1)
