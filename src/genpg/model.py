import numpy as np
import torch as tch
from torch import Tensor
from torch import nn


class Mlp(nn.Sequential):
    def __init__(
        self,
        dims: list[int],
        batch_norm: bool = False,
        dropout: bool = False,
    ):
        super().__init__()

        assert len(dims) >= 2
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            if batch_norm:
                self.append(nn.BatchNorm1d(in_dim))
            self.append(nn.Linear(in_dim, out_dim))
            if i + 1 < len(dims) - 1:
                self.append(nn.ReLU())
                if dropout:
                    self.append(nn.Dropout())


class ConvLayer(nn.Module):
    def __init__(self, n_channels: list[int], kernel_size: int, scale: float):
        """
        Convolutions with optional down- or up-sampling operation
        Args:
        + n_channels - number of channels for convolution layers
        + kernel_size - size of convolution kernel
        + scale - if > 1 then it is factor of upsampling, if < 1 then 1 / scale is max-pool size, if == 1 then no pooling at all.
        """
        super().__init__()

        assert len(n_channels) >= 1
        self.convs = nn.Sequential()
        for j, (in_dim, out_dim) in enumerate(zip(n_channels[:-1], n_channels[1:])):
            self.convs.append(nn.Conv2d(in_dim, out_dim, kernel_size, padding="same"))
            if j + 1 < len(n_channels) - 1:
                self.convs.append(nn.ReLU())

        self.pool: nn.Module | None = None
        if scale != 1:
            if scale > 1:
                self.pool = nn.UpsamplingNearest2d(None, round(scale))
            else:
                self.pool = nn.MaxPool2d(round(1.0 / scale))

    def forward(self, x: Tensor) -> Tensor:
        x = self.convs(x)
        if self.pool is not None:
            x = self.pool(x)
        return x


class ConvNet(nn.Sequential):
    def __init__(
        self,
        layers: list[list[int]],
        kernel_size: int,
        scale: float = 0.5,
    ):
        super().__init__()
        last_dim = layers[0][0]
        for i, n_channels in enumerate(layers):
            assert n_channels[0] == last_dim
            is_last = i + 1 == len(layers)
            self.append(ConvLayer(n_channels, kernel_size, scale if not is_last else 1))
            last_dim = n_channels[-1]

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
        """
        Input: input image of dims (batch_size, image_height, image_width)
        Output: latent vector of dims (batch_size, latent_dim)
        """
        x = x.unsqueeze(1)
        x = self.conv(x)
        x = x.flatten(1)
        x = self.mlp(x)
        return x


class Decoder(nn.Module):
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
        """
        Input: latent vector of dims (batch_size, latent_dim)
        Output: image of dims (batch_size, image_height, image_width)
        """
        x = self.mlp(x)
        x = x.reshape((x.shape[0], *self.hidden_shape))
        x = self.up_conv(x)
        return x.squeeze(1)


class UNet(nn.Module):
    def __init__(
        self,
        down_layers: list[list[int]],
        middle_layer: list[int],
        kernel_size: int,
        scale: int = 2,
    ):
        super().__init__()

        self.down: nn.ModuleList[ConvLayer] = nn.ModuleList()
        for layer in down_layers:
            self.down.append(ConvLayer(layer, kernel_size, scale=(1 / scale)))

        self.middle = ConvLayer(middle_layer, kernel_size, scale=scale)

        self.up: nn.ModuleList[ConvLayer] = nn.ModuleList()
        for i, layer in enumerate(reversed(down_layers)):
            is_last = i + 1 == len(down_layers)
            layer = list(reversed(layer))
            layer[0] *= 2
            self.up.append(
                ConvLayer(
                    layer,
                    kernel_size,
                    scale=(scale if not is_last else 1),
                )
            )

    def forward(self, x: Tensor) -> Tensor:
        x = x.unsqueeze(1)

        skips = []

        for layer in self.down:
            x = layer.convs(x)
            skips.append(x)
            x = layer.pool(x)

        x = self.middle(x)

        for layer, add in zip(self.up, reversed(skips)):
            if add.shape[-2:] != x.shape[-2:]:
                x = nn.functional.pad(
                    x, (0, add.shape[-1] - x.shape[-1], 0, add.shape[-2] - x.shape[-2])
                )
            x = tch.cat((add, x), 1)
            x = layer(x)

        return x.squeeze(1)
