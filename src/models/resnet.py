"""
Unified ResNet Implementations (Flax/JAX).
"""

from typing import Any, Sequence, Tuple
from functools import partial
import jax.numpy as jnp
from flax import linen as nn

ModuleDef = Any

# --------------------------------------------------
# Modern "Wide" ResNet Architecture (ResNet18 Variant)
# Structure preserved from 'ResNet18.py'
# --------------------------------------------------


class ModernConvBlock(nn.Module):
    """
    Basic convolution block with Kaiming Normal init and Swish activation.
    Used in the Modern/Wide ResNet variant.
    """

    channels: int
    kernel_size: int
    norm: ModuleDef
    stride: int = 1
    act: bool = True

    @nn.compact
    def __call__(self, x):
        x = nn.Conv(
            self.channels,
            (self.kernel_size, self.kernel_size),
            strides=self.stride,
            padding="SAME",
            use_bias=False,
            kernel_init=nn.initializers.kaiming_normal(),
        )(x)
        x = self.norm()(x)
        if self.act:
            x = nn.swish(x)
        return x


class ModernResidualBlock(nn.Module):
    """
    Residual block with Swish, Zero-init Gamma, and specific shortcut logic.
    Used in the Modern/Wide ResNet variant.
    """

    channels: int
    conv_block: ModuleDef

    @nn.compact
    def __call__(self, x):
        channels = self.channels
        conv_block = self.conv_block

        shortcut = x

        residual = conv_block(channels, 3)(x)
        residual = conv_block(channels, 3, act=False)(residual)

        if shortcut.shape != residual.shape:
            shortcut = conv_block(channels, 1, act=False)(shortcut)

        # Zero-init gamma for the residual branch (improves signal propagation at init)
        gamma = self.param("gamma", nn.initializers.zeros, 1, jnp.float32)
        out = shortcut + gamma * residual
        out = nn.swish(out)
        return out


class ModernStage(nn.Module):
    channels: int
    num_blocks: int
    stride: int
    block: ModuleDef

    @nn.compact
    def __call__(self, x):
        stride = self.stride
        if stride > 1:
            x = nn.max_pool(x, (stride, stride), strides=(stride, stride))
        for _ in range(self.num_blocks):
            x = self.block(self.channels)(x)
        return x


class ModernBody(nn.Module):
    channel_list: Sequence[int]
    num_blocks_list: Sequence[int]
    strides: Sequence[int]
    stage: ModuleDef

    @nn.compact
    def __call__(self, x):
        for channels, num_blocks, stride in zip(
            self.channel_list, self.num_blocks_list, self.strides
        ):
            x = self.stage(channels, num_blocks, stride)(x)
        return x


class ModernStem(nn.Module):
    """
    Specialized Stem: [32, 32, 64] channel progression.
    """

    channel_list: Sequence[int]
    stride: int
    conv_block: ModuleDef

    @nn.compact
    def __call__(self, x):
        stride = self.stride
        for channels in self.channel_list:
            x = self.conv_block(channels, 3, stride=stride)(x)
            stride = 1
        return x


class ModernHead(nn.Module):
    classes: int
    dropout: ModuleDef

    @nn.compact
    def __call__(self, x):
        x = jnp.mean(x, axis=(1, 2))
        x = self.dropout()(x)
        x = nn.Dense(self.classes)(x)
        return x


class _ModernResNetBase(nn.Module):
    """
    Base implementation of the Modern ResNet logic.
    """

    classes: int
    channel_list: Sequence[int]
    num_blocks_list: Sequence[int]
    strides: Sequence[int]
    head_p_drop: float = 0.0

    @nn.compact
    def __call__(self, x, train=True):
        norm = partial(nn.BatchNorm, use_running_average=not train)
        dropout = partial(nn.Dropout, rate=self.head_p_drop, deterministic=not train)
        conv_block = partial(ModernConvBlock, norm=norm)
        residual_block = partial(ModernResidualBlock, conv_block=conv_block)
        stage = partial(ModernStage, block=residual_block)

        # Stem uses explicit channel list from original script [32, 32, 64]
        x = ModernStem([32, 32, 64], self.strides[0], conv_block)(x)
        x = ModernBody(
            self.channel_list, self.num_blocks_list, self.strides[1:], stage
        )(x)
        x = ModernHead(self.classes, dropout)(x)
        return x


class ResNet18(_ModernResNetBase):
    """
    Modern 'Wide' ResNet18.

    Configuration:
        - Blocks: [2, 2, 2, 2]
        - Channels: [64, 128, 256, 512]
        - Strides: [1, 2, 2, 2]
        - Activations: Swish
        - Init: Kaiming Normal
    """

    channel_list: Sequence[int] = (64, 128, 256, 512)
    num_blocks_list: Sequence[int] = (2, 2, 2, 2)
    strides: Sequence[int] = (1, 2, 2, 2)


# --------------------------------------------------
# Standard ResNet Architecture (ResNet34, ResNet50)
# Structure preserved from 'resnet.py'
# --------------------------------------------------


class StandardResidualBlock(nn.Module):
    """
    Standard ResNet Block (ReLU, no projection unless specified).
    Used in ResNet34.
    """

    filters: int
    strides: tuple = (1, 1)
    use_projection: bool = False

    @nn.compact
    def __call__(self, x, train: bool = True):
        residual = x
        if self.use_projection:
            residual = nn.Conv(self.filters, (1, 1), self.strides, use_bias=False)(x)
            residual = nn.BatchNorm(use_running_average=not train)(residual)

        x = nn.Conv(self.filters, (3, 3), self.strides, padding="SAME", use_bias=False)(
            x
        )
        x = nn.BatchNorm(use_running_average=not train)(x)
        x = nn.relu(x)

        x = nn.Conv(self.filters, (3, 3), padding="SAME", use_bias=False)(x)
        x = nn.BatchNorm(use_running_average=not train)(x)

        return nn.relu(x + residual)


class StandardBottleneckBlock(nn.Module):
    """
    Standard Bottleneck Block (1x1 -> 3x3 -> 1x1).
    Used in ResNet50.
    """

    filters: int
    strides: tuple = (1, 1)
    use_projection: bool = False

    @nn.compact
    def __call__(self, x, train: bool = True):
        residual = x
        if self.use_projection:
            residual = nn.Conv(self.filters * 4, (1, 1), self.strides, use_bias=False)(
                x
            )
            residual = nn.BatchNorm(use_running_average=not train)(residual)

        x = nn.Conv(self.filters, (1, 1), self.strides, use_bias=False)(x)
        x = nn.BatchNorm(use_running_average=not train)(x)
        x = nn.relu(x)

        x = nn.Conv(self.filters, (3, 3), padding="SAME", use_bias=False)(x)
        x = nn.BatchNorm(use_running_average=not train)(x)
        x = nn.relu(x)

        x = nn.Conv(self.filters * 4, (1, 1), use_bias=False)(x)
        x = nn.BatchNorm(use_running_average=not train)(x)

        return nn.relu(x + residual)


class ResNet34(nn.Module):
    """
    Standard ResNet34.
    """

    num_classes: int = 1000

    @nn.compact
    def __call__(self, x, train: bool = True):
        x = nn.Conv(64, (7, 7), padding="SAME", use_bias=False)(x)
        x = nn.BatchNorm(use_running_average=not train)(x)
        x = nn.relu(x)
        x = nn.max_pool(x, (3, 3), strides=(2, 2), padding="SAME")

        def make_layer(filters, blocks, stride):
            layers = []
            layers.append(
                StandardResidualBlock(
                    filters, strides=(stride, stride), use_projection=True
                )
            )
            for _ in range(1, blocks):
                layers.append(StandardResidualBlock(filters))
            return layers

        # ResNet-34 block configuration
        for block in make_layer(64, 3, stride=1):
            x = block(x, train)
        for block in make_layer(128, 4, stride=2):
            x = block(x, train)
        for block in make_layer(256, 6, stride=2):
            x = block(x, train)
        for block in make_layer(512, 3, stride=2):
            x = block(x, train)

        x = jnp.mean(x, axis=(1, 2))  # global average pooling
        x = nn.Dense(self.num_classes)(x)
        return x


class ResNet50(nn.Module):
    """
    Standard ResNet50.
    """

    num_classes: int = 1000

    @nn.compact
    def __call__(self, x, train: bool = True):
        x = nn.Conv(64, (7, 7), strides=(2, 2), padding="SAME", use_bias=False)(x)
        x = nn.BatchNorm(use_running_average=not train)(x)
        x = nn.relu(x)
        x = nn.max_pool(x, (3, 3), strides=(2, 2), padding="SAME")

        def make_layer(filters, blocks, stride):
            layers = []
            layers.append(
                StandardBottleneckBlock(
                    filters, strides=(stride, stride), use_projection=True
                )
            )
            for _ in range(1, blocks):
                layers.append(StandardBottleneckBlock(filters))
            return layers

        for block in make_layer(64, 3, stride=1):
            x = block(x, train)
        for block in make_layer(128, 4, stride=2):
            x = block(x, train)
        for block in make_layer(256, 6, stride=2):
            x = block(x, train)
        for block in make_layer(512, 3, stride=2):
            x = block(x, train)

        x = jnp.mean(x, axis=(1, 2))  # global average pooling
        x = nn.Dense(self.num_classes)(x)
        return x
