import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.parameter import Parameter
from torch.nn.utils import weight_norm as WeightNorm
from torch.nn.init import _calculate_fan_in_and_fan_out
from torch.nn.init import orthogonal_, kaiming_normal_, zeros_


def _calculate_cnn_fan_in_and_fan_out(tensor):
    dimensions = tensor.ndimension()
    if dimensions < 3:
        raise ValueError("CNN fan in and fan out can not be computed for tensor with fewer than 3 dimensions")

    num_input_fmaps = tensor.size(1)
    num_output_fmaps = tensor.size(0)
    receptive_field_size = tensor[0][0].numel()

    return num_input_fmaps, num_output_fmaps, receptive_field_size


def proposed_init(layer):
    """
    He init which preserves the norm in the backwards pass for ReLU non-linearities:
        w ~ N(0, 2/fan_out)
    """
    w, b = layer.weight, layer.bias
    _, fan_out = _calculate_fan_in_and_fan_out(w)
    gain = math.sqrt(2.0)
    std = gain / math.sqrt(fan_out)
    with torch.no_grad():
        w.normal_(0, std)
        b.zero_()


def proposed_init_wn(layer):
    """
    He init which preserves the norm in the forward pass for weight-normalized ReLU networks:
        w ~ N(0, 1/fan_in)
    """
    w, b = layer.weight, layer.bias
    fan_in, _ = _calculate_fan_in_and_fan_out(w)
    gain = 1.
    std = gain / math.sqrt(fan_in)
    with torch.no_grad():
        w.normal_(0, std)
        b.zero_()


def proposed_weight_norm_g_init(wn_layer, gain=2., version=1):
    """
    Initialize WN's g to preserve the norm of the forward pass
    """
    if version == 1:
        fan_in, fan_out = _calculate_fan_in_and_fan_out(wn_layer.weight)
        wn_layer.weight_g = Parameter(torch.ones_like(wn_layer.weight_g) * math.sqrt(gain * fan_in / fan_out))
    elif version == 13:
        wn_layer.weight_g = Parameter(torch.ones_like(wn_layer.weight_g) * math.sqrt(gain))
    else:
        raise ValueError("proposed_weight_norm_g_init: version should be in {1, 13}")


def proposed_weight_norm_cnn_g_init(wn_layer, gain=2., version=2):
    """
    Initialize WN's g to preserve the norm of both forward and backward pass
    """
    i, o, rf = _calculate_cnn_fan_in_and_fan_out(wn_layer.weight)
    c1 = gain * i * rf / o
    c2 = gain * i / (o * rf)
    if version == 2:
        correction = 0.5 * math.sqrt(c1) + 0.5 * math.sqrt(c2)
    elif version == 3:
        correction = math.sqrt(0.5 * c1 + 0.5 * c2)
    elif version == 4:
        correction = math.sqrt(math.sqrt(c1 * c2))
    elif version == 5:  # new_v2
        c1 = gain * i * rf / o
        c2 = gain * o * rf / i
        correction = 0.5 * math.sqrt(c1) + 0.5 * math.sqrt(c2)
    elif version == 6:  # new_v3
        c1 = gain * i * rf / o
        c2 = gain * o * rf / i
        correction = math.sqrt(0.5 * c1 + 0.5 * c2)
    elif version == 7:  # new_v4
        correction = math.sqrt(gain * rf)
    elif version == 8:
        correction = math.sqrt(gain * i * rf / o)
    elif version == 9:
        backward_factor = gain
        correction = math.sqrt(backward_factor)
    elif version == 10:
        forward_factor = gain * i * rf / o
        backward_factor = gain
        correction = 0.5 * math.sqrt(forward_factor) + 0.5 * math.sqrt(backward_factor)
    elif version == 11:
        forward_factor = gain * i * rf / o
        backward_factor = gain
        correction = math.sqrt(0.5 * forward_factor + 0.5 * backward_factor)
    elif version == 12:
        forward_factor = gain * i * rf / o
        backward_factor = gain
        correction = math.sqrt(math.sqrt(forward_factor * backward_factor))
    else:
        raise ValueError("proposed_weight_norm_cnn_g_init: version should be in 2..12")
    wn_layer.weight_g = Parameter(torch.ones_like(wn_layer.weight_g) * correction)


def data_dependent_weight_norm_g_init(wn_layer, sample_batch):
    """
    Data-dependent init of WN's g, as in Salimans & Kingma 2016.
        1) Set g=1, so that w = v / ||v||
        2) Set b = 0
        3) Obtain pre-activations:  y = w * x + b = v * x / ||v||
        4) Compute mu, sigma = mean(y), std(y)
        5) Initialize g = 1 / sigma, b = -mu / sigma
        6) Re-compute pre-activations with properly normalized layer
        7) Return the layer and the pre-activations, so that we can propagate the batch to the next layer
    """
    with torch.no_grad():
        # Set g=1
        wn_layer.weight_g.uniform_(1, 1)

        # Set b=0
        zeros_(wn_layer.bias)

        # Forward pass
        y = wn_layer(sample_batch)

        # Compute mean and std of pre-activations
        out_size = y.size(1)
        mu = torch.mean(y.transpose(1, 0).contiguous().view(out_size, -1), dim=-1)
        sigma = (y.transpose(1, 0).contiguous().view(out_size, -1) - mu.unsqueeze(1)).pow(2).mean(dim=-1).sqrt()

        # Initialize parameters as in [Salimans & Kingma 2016] Eq (6)
        wn_layer.weight_g = Parameter((1. / sigma).view(wn_layer.weight_g.size()))
        wn_layer.bias = Parameter(-mu / sigma)

        # Re-compute pre-activations with properly normalized layer
        y_norm = wn_layer(sample_batch)
    return wn_layer, y_norm


def init_and_maybe_normalize_layer(layer, init, weight_norm, sample_batch=None, g_gain=2.):
    """
    Initialize layer and optionally weight normalize it.
    Two cases:
        (a) WN with data-dependent init: run forward pass with WN with g=1, compute proper init for g and b,
            and initialize them accordingly
        (b) No WN, or WN with default or proposed init: simply add WN and then initialize weights
    """
    # 1) Initialize layer weights. If weight_norm=False, there's nothing else to do.
    if 'traditional' in init:  # traditional, traditional_datadep
        kaiming_normal_(layer.weight)
        zeros_(layer.bias)
    elif 'orthogonal' in init:  # orthogonal, orthogonal_datadep, orthogonal_proposed
        orthogonal_(layer.weight)
        zeros_(layer.bias)
    elif init in ['proposed'] + ['proposed_v%d' % v for v in range(2, 13 + 1)]:
        if weight_norm:
            proposed_init_wn(layer)
        else:
            proposed_init(layer)
    else:
        raise ValueError('Unsupported init:', init)

    if weight_norm:
        # Add WN
        layer = WeightNorm(layer)
        # Case (a): override default WN init if needed (data-independent)
        if 'datadep' not in init:
            if 'proposed_v' in init:
                version = int(init[len("proposed_v"):].split('_')[0])
                if version == 13:
                    proposed_weight_norm_g_init(layer, gain=g_gain, version=version)
                else:
                    proposed_weight_norm_cnn_g_init(layer, gain=g_gain, version=version)
            elif 'proposed' in init:
                proposed_weight_norm_g_init(layer, gain=g_gain)
        # Case (b): data-dependent init
        else:
            assert sample_batch is not None
            layer, sample_batch = data_dependent_weight_norm_g_init(layer, sample_batch)
    return layer, sample_batch


if __name__ == "__main__":
    # Test initialization for weight normalized layers
    from torch.nn.utils import weight_norm as WeightNorm
    in_dim, out_dim = 20, 60
    linear = nn.Linear(in_dim, out_dim)
    proposed_init_wn(linear)
    linear_wn = WeightNorm(linear)
    proposed_weight_norm_g_init(linear_wn)
    assert torch.allclose(linear_wn.weight_g.mean(), torch.tensor(math.sqrt(2. * in_dim / out_dim)))
