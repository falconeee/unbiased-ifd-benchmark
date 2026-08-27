"""Parametric 1D-CNN family for the controlled ablations requested by Reviewer #2.

The reviewer's objection is that AlexNet and ResNet-18 differ along several axes at once
(depth, kernel sizes, receptive field, skip connections, pooling, parameter count), so the
observed reversal on PU cannot be attributed to depth. ``AblationCNN1D`` exists to vary
**one** axis at a time on top of the ``CNN1D`` already evaluated in the paper:

``depth``
    number of convolutional blocks (R2.2). Channel widths follow ``DEFAULT_CHANNELS``.

``first_kernel``
    kernel size of the first convolutional layer, i.e. the initial receptive field (R2.3).
    Every later block keeps ``kernel``.

Everything else -- padding convention, BatchNorm, ReLU, pooling strategy, adaptive pooling
output, dropout rate and the classification head -- is fixed across variants.

Block structure (chosen to match :class:`src.models.CNN1D`)
----------------------------------------------------------
Each block is ``Conv1d(padding=k//2) -> BatchNorm1d -> ReLU -> pool``. Blocks ``0..depth-2``
pool with ``MaxPool1d(2)``; the **last** block pools with ``AdaptiveMaxPool1d(16)`` instead.
That is exactly what ``CNN1D`` does (its third block has no ``MaxPool1d(2)``), which makes the
``depth=3, first_kernel=3, kernel=3`` variant the paper's 1D-CNN with the kernel schedule
7-5-3 flattened to 3-3-3 -- the equivalence the response letter promises.

The alternative reading (``MaxPool1d(2)`` in *every* block, then a trailing adaptive pool)
would keep the number of pooling layers equal to ``depth`` rather than ``depth-1``, but it
would no longer reduce to the published CNN1D at depth 3, so it was not used.

Parameter count is *not* held constant across depths -- it cannot be, since adding a block
adds its weights and changes the flattened head input. It is reported per variant in the
result JSON (``architecture.num_parameters``) so the manuscript can state explicitly what
varies alongside depth.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

#: Channel width of block ``i``. Taken verbatim from the response letter (R2.2).
DEFAULT_CHANNELS: Tuple[int, ...] = (16, 32, 64, 128, 256)

#: Temporal size the last block's ``AdaptiveMaxPool1d`` collapses to (same as ``CNN1D``).
ADAPTIVE_POOL_OUTPUT = 16

#: Classification head, identical across variants (same as ``CNN1D``).
HEAD_HIDDEN_UNITS = 128
HEAD_DROPOUT = 0.3

MAX_DEPTH = len(DEFAULT_CHANNELS)


def receptive_field(depth: int, first_kernel: int, kernel: int) -> int:
    """Theoretical receptive field, in input samples, of the convolutional stack.

    Counts the ``depth`` convolutions and the ``depth - 1`` ``MaxPool1d(2)`` layers between
    them. The trailing ``AdaptiveMaxPool1d`` is excluded: it spans whatever is left of the
    signal, so including it would make every variant's receptive field trivially global and
    hide the difference the ablation is measuring.
    """
    r, jump = 1, 1
    for block in range(depth):
        k = first_kernel if block == 0 else kernel
        r += (k - 1) * jump
        if block < depth - 1:
            r += jump  # MaxPool1d(kernel=2, stride=2)
            jump *= 2
    return r


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


def build_ablation_model(
    input_length: int, num_classes: int, depth: int, first_kernel: int, kernel: int = 3
) -> AblationCNN1D:
    return AblationCNN1D(
        input_length=input_length,
        num_classes=num_classes,
        depth=depth,
        first_kernel=first_kernel,
        kernel=kernel,
    )


def architecture_block(model: AblationCNN1D, input_length: int) -> OrderedDict:
    """What varies (and what does not) across variants, for the result JSON.

    ``receptive_field_fraction_of_window`` uses the fact that the protocol segments signals
    into one-second windows, so ``input_length`` *is* the dataset's sample rate.
    """
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
