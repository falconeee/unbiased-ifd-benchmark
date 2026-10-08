"""Parametric model families for the controlled ablations requested by Reviewer #2.

The reviewer's objection is that AlexNet and ResNet-18 differ along several axes at once
(depth, kernel sizes, receptive field, skip connections, pooling, parameter count), so the
observed reversal on PU cannot be attributed to depth. Each class here takes one of the
paper's architectures and exposes **two** of those axes as constructor arguments, so that one
can be varied at a time while everything else stays as published:

``depth``
    how deep the network is (R2.2). The unit is the family's own: convolutional blocks for
    ``AblationCNN1D``, convolutional layers for ``AblationAlexNet1D``, weighted layers for
    ``AblationResNet1D`` (the "18" in ResNet-18).

``first_kernel``
    kernel size of the first convolutional layer, i.e. the initial receptive field (R2.3).

=====================  ==========================  ==================================
class                  built on                    reproduces the published model at
=====================  ==========================  ==================================
``AblationCNN1D``      ``src.models.CNN1D``        nowhere -- see below
``AblationAlexNet1D``  ``src.models.AlexNet1D``    ``depth=5, first_kernel=11``
``AblationResNet1D``   ``src.models.ResNet18``     ``depth=18, first_kernel=7``
=====================  ==========================  ==================================

AlexNet and ResNet-18 keep everything that is not being varied exactly as published --
channel widths, later kernels, strides, pooling, BatchNorm (or its absence), skip connections,
initialisation and the classification head -- and build their modules in the same order as the
published classes, so at the reference point they are the published model weight for weight
under the same seed. ``AblationCNN1D`` predates them and follows the response letter instead:
every kernel after the first is 3, so its ``d3_k3`` cell is ``CNN1D`` with the 7-5-3 kernel
schedule flattened to 3-3-3.

CNN1D block structure
---------------------
Each block is ``Conv1d(padding=k//2) -> BatchNorm1d -> ReLU -> pool``. Blocks ``0..depth-2``
pool with ``MaxPool1d(2)``; the **last** block pools with ``AdaptiveMaxPool1d(16)`` instead.
That is exactly what ``CNN1D`` does (its third block has no ``MaxPool1d(2)``), which makes the
``depth=3, first_kernel=3, kernel=3`` variant the paper's 1D-CNN with the kernel schedule
7-5-3 flattened to 3-3-3 -- the equivalence the response letter promises.

The alternative reading (``MaxPool1d(2)`` in *every* block, then a trailing adaptive pool)
would keep the number of pooling layers equal to ``depth`` rather than ``depth-1``, but it
would no longer reduce to the published CNN1D at depth 3, so it was not used.

Parameter count is *not* held constant across depths in any family -- it cannot be, since
adding a layer adds its weights. It is reported per variant in the result JSON
(``architecture.num_parameters``) so the manuscript can state explicitly what varies alongside
depth.

Receptive field
---------------
``receptive_field_samples`` is the theoretical receptive field of the convolutional stack
along its longest path, counting every convolution and fixed-size pooling layer. The final
adaptive pooling is excluded: it spans whatever is left of the signal, so including it would
make every variant's receptive field trivially global and hide the difference the ablation is
measuring. In ResNet the longest path runs through both convolutions of every residual branch;
the 1x1 shortcut convolutions add nothing to it.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models import AlexNet1D, BasicBlock, ResNet18, conv1x1

#: Channel width of block ``i``. Taken verbatim from the response letter (R2.2).
DEFAULT_CHANNELS: Tuple[int, ...] = (16, 32, 64, 128, 256)

#: Temporal size the last block's ``AdaptiveMaxPool1d`` collapses to (same as ``CNN1D``).
ADAPTIVE_POOL_OUTPUT = 16

#: Classification head, identical across variants (same as ``CNN1D``).
HEAD_HIDDEN_UNITS = 128
HEAD_DROPOUT = 0.3

MAX_DEPTH = len(DEFAULT_CHANNELS)

# --- AlexNet1D, as published in src/models.py -------------------------------------------
ALEXNET_CHANNELS: Tuple[int, ...] = (64, 192, 384, 256, 256)
#: Kernel of conv layers 1..4; layer 0's is ``first_kernel``.
ALEXNET_LATER_KERNELS: Tuple[int, ...] = (5, 3, 3, 3)
ALEXNET_STEM_STRIDE = 4
#: Held at the published value for every ``first_kernel``, so ``first_kernel=11`` is exact.
ALEXNET_STEM_PADDING = 2
ALEXNET_POOL_KERNEL = 3
ALEXNET_POOL_STRIDE = 2
ALEXNET_ADAPTIVE_POOL_OUTPUT = 6
ALEXNET_HEAD_HIDDEN_UNITS = 1024
ALEXNET_HEAD_DROPOUT = 0.5
ALEXNET_MAX_DEPTH = len(ALEXNET_CHANNELS)

# --- ResNet18, as published in src/models.py --------------------------------------------
RESNET_STAGE_WIDTHS: Tuple[int, ...] = (64, 128, 256, 512)
RESNET_STEM_STRIDE = 2
RESNET_BLOCK_KERNEL = 3


def _receptive_field(layers: Sequence[Tuple[int, int]]) -> int:
    """Receptive field, in input samples, of a chain of ``(kernel, stride)`` layers."""
    r, jump = 1, 1
    for kernel, stride in layers:
        r += (kernel - 1) * jump
        jump *= stride
    return r


def receptive_field(depth: int, first_kernel: int, kernel: int) -> int:
    """Theoretical receptive field of an ``AblationCNN1D``, in input samples.

    Counts the ``depth`` convolutions and the ``depth - 1`` ``MaxPool1d(2)`` layers between
    them; the trailing ``AdaptiveMaxPool1d`` is excluded (see the module docstring).
    """
    layers = []
    for block in range(depth):
        layers.append((first_kernel if block == 0 else kernel, 1))
        if block < depth - 1:
            layers.append((2, 2))  # MaxPool1d(kernel=2, stride=2)
    return _receptive_field(layers)


def resnet_depth(blocks_per_stage: int) -> int:
    """Weighted layers of a 4-stage BasicBlock ResNet: stem + 2 convs per block + fc."""
    return 2 + 2 * len(RESNET_STAGE_WIDTHS) * blocks_per_stage


def resnet_blocks_per_stage(depth: int) -> int:
    """Inverse of :func:`resnet_depth`; rejects depths no ``[n, n, n, n]`` ResNet has."""
    per_block = 2 * len(RESNET_STAGE_WIDTHS)
    if depth < 2 + per_block or (depth - 2) % per_block:
        raise ValueError(
            f"ResNet depth must be 2 + {per_block}n with n >= 1 (10, 18, 26, 34, ...), "
            f"got {depth}"
        )
    return (depth - 2) // per_block


class AblationCNN1D(nn.Module):
    """``CNN1D`` with depth and initial kernel size as constructor arguments."""

    def __init__(
        self,
        input_length: int,
        num_classes: int,
        depth: int = 3,
        first_kernel: int = 3,
        kernel: int = 3,
        channels: Sequence[int] = DEFAULT_CHANNELS,
    ):
        super().__init__()
        if not 1 <= depth <= len(channels):
            raise ValueError(f"depth must be in 1..{len(channels)}, got {depth}")

        self.depth = depth
        self.first_kernel = first_kernel
        self.kernel = kernel
        self.channels = tuple(channels[:depth])
        self.input_length = input_length
        self.num_classes = num_classes

        blocks = []
        in_channels = 1
        for block in range(depth):
            k = first_kernel if block == 0 else kernel
            out_channels = self.channels[block]
            pool = (
                nn.MaxPool1d(2)
                if block < depth - 1
                else nn.AdaptiveMaxPool1d(ADAPTIVE_POOL_OUTPUT)
            )
            blocks.append(
                nn.Sequential(
                    nn.Conv1d(in_channels, out_channels, kernel_size=k, padding=k // 2),
                    nn.BatchNorm1d(out_channels),
                    nn.ReLU(),
                    pool,
                )
            )
            in_channels = out_channels
        self.features = nn.Sequential(*blocks)

        self.flattened_size = self.channels[-1] * ADAPTIVE_POOL_OUTPUT
        self.fc1 = nn.Linear(self.flattened_size, HEAD_HIDDEN_UNITS)
        self.dropout = nn.Dropout(HEAD_DROPOUT)
        self.fc2 = nn.Linear(HEAD_HIDDEN_UNITS, num_classes)

    @property
    def kernels(self) -> Tuple[int, ...]:
        return tuple(
            self.first_kernel if block == 0 else self.kernel for block in range(self.depth)
        )

    def forward(self, x):
        # x shape: [B, L] or [B, 1, L] -- same contract as CNN1D.
        if x.ndim == 2:
            x = x.unsqueeze(1)
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)


class MPSSafeAdaptiveAvgPool1d(nn.AdaptiveAvgPool1d):
    """``AdaptiveAvgPool1d`` that also runs on Apple's MPS backend.

    MPS implements adaptive average pooling only when the input length is a multiple of the
    output size (torch 2.13 raises otherwise, pytorch/pytorch#96056), and every AlexNet
    variant reaches its ``AdaptiveAvgPool1d(6)`` with an odd length. In exactly that case the
    pool is computed as a matmul with the averaging matrix of PyTorch's own bin definition
    (bin ``i`` spans ``floor(i*L/n) .. ceil((i+1)*L/n)``) -- the same function, up to
    floating-point summation order. Everywhere else, including CPU and CUDA, it is the native
    op. It holds no parameters or buffers, so ``state_dict`` is unchanged.
    """

    def forward(self, x):
        length, bins = x.shape[-1], self.output_size
        if isinstance(bins, tuple):
            (bins,) = bins
        if x.device.type != "mps" or length % bins == 0:
            return super().forward(x)
        weights = torch.zeros(length, bins, dtype=x.dtype)
        for i in range(bins):
            start = (i * length) // bins
            end = -((-(i + 1) * length) // bins)  # ceil
            weights[start:end, i] = 1.0 / (end - start)
        return torch.matmul(x, weights.to(x.device))


class AblationAlexNet1D(AlexNet1D):
    """``AlexNet1D`` truncated to its first ``depth`` conv layers, with a variable stem kernel.

    ``depth`` keeps the first ``depth`` convolutional layers of the published stack, with their
    published widths and kernels; ``MaxPool1d(3, 2)`` follows conv 0, conv 1 and the last conv,
    which is the published pooling pattern at ``depth=5``. Stem stride and padding, the absence
    of BatchNorm, the ``AdaptiveAvgPool1d(6)`` and the dropout head are as published.

    Only the constructor differs from ``AlexNet1D``; ``forward`` is inherited. Modules are
    created in the published order, so ``depth=5, first_kernel=11`` gives the same
    ``state_dict`` keys and, after the same seed, the same initial weights as ``AlexNet1D``.
    The one substitution, :class:`MPSSafeAdaptiveAvgPool1d`, changes nothing off MPS.
    """

    def __init__(
        self,
        input_length: int,
        num_classes: int,
        depth: int = ALEXNET_MAX_DEPTH,
        first_kernel: int = 11,
        in_channel: int = 1,
    ):
        # AlexNet1D.__init__ hard-codes the published stack; build ours instead.
        nn.Module.__init__(self)
        if not 1 <= depth <= ALEXNET_MAX_DEPTH:
            raise ValueError(f"depth must be in 1..{ALEXNET_MAX_DEPTH}, got {depth}")

        self.depth = depth
        self.first_kernel = first_kernel
        self.input_length = input_length
        self.num_classes = num_classes
        self.channels = ALEXNET_CHANNELS[:depth]
        self.kernels = (first_kernel,) + ALEXNET_LATER_KERNELS[: depth - 1]

        layers = []
        self._rf_layers = []
        in_channels = in_channel
        for index, (out_channels, kernel) in enumerate(zip(self.channels, self.kernels)):
            if index == 0:
                stride, padding = ALEXNET_STEM_STRIDE, ALEXNET_STEM_PADDING
            else:
                stride, padding = 1, kernel // 2
            layers += [
                nn.Conv1d(in_channels, out_channels, kernel_size=kernel, stride=stride,
                          padding=padding),
                nn.ReLU(inplace=True),
            ]
            self._rf_layers.append((kernel, stride))
            if index < 2 or index == depth - 1:
                layers.append(
                    nn.MaxPool1d(kernel_size=ALEXNET_POOL_KERNEL, stride=ALEXNET_POOL_STRIDE)
                )
                self._rf_layers.append((ALEXNET_POOL_KERNEL, ALEXNET_POOL_STRIDE))
            in_channels = out_channels
        self.features = nn.Sequential(*layers)
        self.num_maxpool_layers = sum(isinstance(m, nn.MaxPool1d) for m in self.features)

        self.avgpool = MPSSafeAdaptiveAvgPool1d(ALEXNET_ADAPTIVE_POOL_OUTPUT)

        self.flattened_size = in_channels * ALEXNET_ADAPTIVE_POOL_OUTPUT
        self.classifier = nn.Sequential(
            nn.Dropout(ALEXNET_HEAD_DROPOUT),
            nn.Linear(self.flattened_size, ALEXNET_HEAD_HIDDEN_UNITS),
            nn.ReLU(inplace=True),
            nn.Dropout(ALEXNET_HEAD_DROPOUT),
            nn.Linear(ALEXNET_HEAD_HIDDEN_UNITS, ALEXNET_HEAD_HIDDEN_UNITS),
            nn.ReLU(inplace=True),
            nn.Linear(ALEXNET_HEAD_HIDDEN_UNITS, num_classes),
        )

    @property
    def receptive_field(self) -> int:
        return _receptive_field(self._rf_layers)


class AblationResNet1D(ResNet18):
    """``ResNet18`` with blocks per stage and the stem kernel as constructor arguments.

    ``blocks_per_stage`` sets every one of the four stages to ``[n, n, n, n]`` BasicBlocks, so
    depth (``2 + 8n`` weighted layers: ResNet-10/18/26/34) changes while stage widths, the
    three stride-2 stages, the stem, skip connections, initialisation and the head do not.
    The stem keeps stride 2 and uses ``padding = first_kernel // 2`` (3 at the published 7).

    Only the constructor differs from ``ResNet18``; ``_make_layer`` and ``forward`` are
    inherited. Modules are created in the published order, so ``blocks_per_stage=2,
    first_kernel=7`` gives the same ``state_dict`` keys and, after the same seed, the same
    initial weights as ``ResNet18``.
    """

    def __init__(
        self,
        input_length: int,
        num_classes: int,
        blocks_per_stage: int = 2,
        first_kernel: int = 7,
        in_channel: int = 1,
    ):
        # ResNet18.__init__ hard-codes layers=[2, 2, 2, 2] and the kernel-7 stem.
        nn.Module.__init__(self)
        if blocks_per_stage < 1:
            raise ValueError(f"blocks_per_stage must be >= 1, got {blocks_per_stage}")

        block = BasicBlock
        self.blocks_per_stage = blocks_per_stage
        self.first_kernel = first_kernel
        self.depth = resnet_depth(blocks_per_stage)
        self.input_length = input_length
        self.num_classes = num_classes

        self.inplanes = RESNET_STAGE_WIDTHS[0]

        self.conv1 = nn.Conv1d(in_channel, self.inplanes, kernel_size=first_kernel,
                               stride=RESNET_STEM_STRIDE, padding=first_kernel // 2, bias=False)
        self.bn1 = nn.BatchNorm1d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        widths = RESNET_STAGE_WIDTHS
        self.layer1 = self._make_layer(block, widths[0], blocks_per_stage)
        self.layer2 = self._make_layer(block, widths[1], blocks_per_stage, stride=2)
        self.layer3 = self._make_layer(block, widths[2], blocks_per_stage, stride=2)
        self.layer4 = self._make_layer(block, widths[3], blocks_per_stage, stride=2)

        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(widths[-1] * block.expansion, num_classes)

        # Same initialisation as ResNet18, over the modules in the same order.
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    @property
    def receptive_field(self) -> int:
        layers = [(self.first_kernel, RESNET_STEM_STRIDE), (3, 2)]  # stem conv, max pool
        for stage in range(len(RESNET_STAGE_WIDTHS)):
            for index in range(self.blocks_per_stage):
                stride = 2 if stage > 0 and index == 0 else 1
                layers += [(RESNET_BLOCK_KERNEL, stride), (RESNET_BLOCK_KERNEL, 1)]
        return _receptive_field(layers)


def build_ablation_model(
    family: str,
    input_length: int,
    num_classes: int,
    depth: int,
    first_kernel: int,
    kernel: int = 3,
) -> nn.Module:
    """Instantiate one variant of ``family`` (``cnn1d``, ``alexnet`` or ``resnet18``).

    ``kernel`` is the CNN1D's kernel after the first block; the other families keep their
    published later kernels and ignore it.
    """
    if family == "cnn1d":
        return AblationCNN1D(
            input_length=input_length,
            num_classes=num_classes,
            depth=depth,
            first_kernel=first_kernel,
            kernel=kernel,
        )
    if family == "alexnet":
        return AblationAlexNet1D(
            input_length=input_length,
            num_classes=num_classes,
            depth=depth,
            first_kernel=first_kernel,
        )
    if family == "resnet18":
        return AblationResNet1D(
            input_length=input_length,
            num_classes=num_classes,
            blocks_per_stage=resnet_blocks_per_stage(depth),
            first_kernel=first_kernel,
        )
    raise KeyError(f"unknown ablation family {family!r}")


def _parameter_counts(model: nn.Module, head: Sequence[nn.Module]) -> OrderedDict:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    head_params = sum(p.numel() for module in head for p in module.parameters())
    counts = OrderedDict()
    counts["num_parameters"] = int(total)
    counts["num_trainable_parameters"] = int(trainable)
    counts["num_conv_parameters"] = int(total - head_params)
    counts["num_head_parameters"] = int(head_params)
    return counts


def architecture_block(model: nn.Module, input_length: int) -> OrderedDict:
    """What varies (and what does not) across variants, for the result JSON.

    ``receptive_field_seconds`` uses the fact that the protocol segments signals into
    one-second windows, so ``input_length`` *is* the dataset's sample rate.
    """
    if isinstance(model, AblationCNN1D):
        return _cnn1d_block(model, input_length)
    if isinstance(model, AblationAlexNet1D):
        return _alexnet_block(model, input_length)
    if isinstance(model, AblationResNet1D):
        return _resnet_block(model, input_length)
    raise TypeError(f"not an ablation model: {type(model).__name__}")


def _cnn1d_block(model: AblationCNN1D, input_length: int) -> OrderedDict:
    # Key set and order are those of the results already on disk; keep them stable.
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    conv = sum(p.numel() for p in model.features.parameters())
    rf = receptive_field(model.depth, model.first_kernel, model.kernel)

    block = OrderedDict()
    block["depth"] = model.depth
    block["first_kernel"] = model.first_kernel
    block["kernel"] = model.kernel
    block["kernels_per_block"] = list(model.kernels)
    block["channels_per_block"] = list(model.channels)
    block["num_parameters"] = int(total)
    block["num_trainable_parameters"] = int(trainable)
    block["num_conv_parameters"] = int(conv)
    block["num_head_parameters"] = int(total - conv)
    block["receptive_field_samples"] = int(rf)
    # One-second windows, so input_length is the dataset's sample rate and this is both the
    # receptive field in seconds and its fraction of the window.
    block["receptive_field_seconds"] = rf / input_length
    block["adaptive_pool_output"] = ADAPTIVE_POOL_OUTPUT
    block["flattened_features"] = int(model.flattened_size)
    block["head_hidden_units"] = HEAD_HIDDEN_UNITS
    block["head_dropout"] = HEAD_DROPOUT
    block["num_maxpool_layers"] = model.depth - 1
    return block


def _alexnet_block(model: AblationAlexNet1D, input_length: int) -> OrderedDict:
    rf = model.receptive_field
    block = OrderedDict()
    block["depth"] = model.depth
    block["depth_unit"] = "conv layers"
    block["first_kernel"] = model.first_kernel
    block["kernels_per_block"] = list(model.kernels)
    block["channels_per_block"] = list(model.channels)
    block.update(_parameter_counts(model, [model.classifier]))
    block["receptive_field_samples"] = int(rf)
    block["receptive_field_seconds"] = rf / input_length
    block["stem_stride"] = ALEXNET_STEM_STRIDE
    block["stem_padding"] = ALEXNET_STEM_PADDING
    block["num_maxpool_layers"] = int(model.num_maxpool_layers)
    block["adaptive_pool_output"] = ALEXNET_ADAPTIVE_POOL_OUTPUT
    block["flattened_features"] = int(model.flattened_size)
    block["head_hidden_units"] = ALEXNET_HEAD_HIDDEN_UNITS
    block["head_dropout"] = ALEXNET_HEAD_DROPOUT
    block["batch_norm"] = False
    return block


def _resnet_block(model: AblationResNet1D, input_length: int) -> OrderedDict:
    rf = model.receptive_field
    block = OrderedDict()
    block["depth"] = model.depth
    block["depth_unit"] = "weighted layers"
    block["first_kernel"] = model.first_kernel
    block["kernel"] = RESNET_BLOCK_KERNEL
    block["blocks_per_stage"] = [model.blocks_per_stage] * len(RESNET_STAGE_WIDTHS)
    block["stage_widths"] = list(RESNET_STAGE_WIDTHS)
    block.update(_parameter_counts(model, [model.fc]))
    block["receptive_field_samples"] = int(rf)
    block["receptive_field_seconds"] = rf / input_length
    block["stem_stride"] = RESNET_STEM_STRIDE
    block["stem_padding"] = model.first_kernel // 2
    block["num_downsampling_stages"] = len(RESNET_STAGE_WIDTHS) - 1
    block["adaptive_pool_output"] = 1
    block["flattened_features"] = RESNET_STAGE_WIDTHS[-1] * BasicBlock.expansion
    block["skip_connections"] = True
    return block
