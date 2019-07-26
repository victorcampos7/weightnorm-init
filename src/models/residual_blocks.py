import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.init import init_and_maybe_normalize_layer


class WRNResBlock(nn.Module):
    def __init__(self, input_size, out_size, stride=1, batch_norm=False):
        super().__init__()

        self.use_bn = batch_norm
        self._initialized = False

        # ReLU non-linearity
        self.relu = nn.ReLU(inplace=True)

        # First 3x3 conv (which optionally downsamples the input through strided conv)
        self.bn1 = nn.BatchNorm2d(input_size) if self.use_bn else None
        self.conv1 = nn.Conv2d(input_size, out_size, kernel_size=3, stride=stride, padding=1)

        # Second 3x3 conv
        self.bn2 = nn.BatchNorm2d(out_size) if self.use_bn else None
        self.conv2 = nn.Conv2d(out_size, out_size, kernel_size=3, stride=1, padding=1)

        # Shortcut connection to match dimensions
        if out_size != input_size or stride > 1:
            self.shortcut = nn.Conv2d(input_size, out_size, 1, stride=stride, padding=0)
        else:
            self.shortcut = None

    def init_and_maybe_normalize(self, weight_norm, init, sample_batch=None, num_resblocks=1, resblock_id=1,
                                 resblock_conv2_g_gain_fn=None):
        sample_batch_w, sample_batch_identity = sample_batch, sample_batch

        def apply_relu(x):
            return x if x is None else self.relu(x)

        if resblock_conv2_g_gain_fn is None:
            conv2_g_gain = 1.
        else:
            conv2_g_gain = resblock_conv2_g_gain_fn(resblock_id, num_resblocks)

        # Weight path (Conv3x3 -> ReLU -> Conv3x3 -> ReLU)
        self.conv1, sample_batch_w = init_and_maybe_normalize_layer(self.conv1,
                                                                    weight_norm=weight_norm,
                                                                    init=init,
                                                                    sample_batch=sample_batch_w,
                                                                    g_gain=2)
        sample_batch_w = apply_relu(sample_batch_w)
        self.conv2, sample_batch_w = init_and_maybe_normalize_layer(self.conv2,
                                                                    weight_norm=weight_norm,
                                                                    init=init,
                                                                    sample_batch=sample_batch_w,
                                                                    g_gain=conv2_g_gain)

        # Shortcut path (identity or Conv1x1 to match dimensions)
        if self.shortcut is not None:
            self.shortcut, sample_batch_identity = init_and_maybe_normalize_layer(self.shortcut,
                                                                                  weight_norm=weight_norm,
                                                                                  init=init,
                                                                                  sample_batch=sample_batch_identity,
                                                                                  g_gain=1.  # best init
                                                                                  # g_gain=1./num_resblocks
                                                                                  )

        # Add results from both paths and apply non-linearity (for data-dependent init only)
        if sample_batch is not None:
            sample_batch = sample_batch_w + sample_batch_identity

        # Set initialized flag
        self._initialized = True

        return sample_batch

    def forward(self, x):
        # Make sure that weights were initialized properly
        assert self._initialized, "run init_and_maybe_normalize() before using the ResBlock"

        if self.use_bn:
            identity = x if self.shortcut is None else self.shortcut(self.bn1(x))

            out = self.bn1(x)
            out = self.relu(out)
            out = self.conv1(out)
            out = self.bn2(out)
            out = self.relu(out)
            out = self.conv2(out)
        else:
            identity = x if self.shortcut is None else self.shortcut(x)

            out = self.conv1(x)
            out = self.relu(out)
            out = self.conv2(out)

        out = out + identity

        return out
