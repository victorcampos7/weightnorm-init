import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.init import init_and_maybe_normalize_layer
from models.residual_blocks import WRNResBlock

ARCHITECTURES = ['mlp', 'cnn', 'resnet', 'wrn']


class Flatten(nn.Module):

    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        shape = torch.prod(torch.tensor(x.shape[1:])).item()
        return x.view(-1, shape)


class LayerWiseLRModule(nn.Module):

    def __init__(self):
        super().__init__()
        self.lr_mult_dict = {}

    def forward(self, *input):
        raise NotImplementedError

    def optimizer_parameters(self):
        param_groups = []
        standard_lr_params = []
        for param in self.parameters():
            if param in self.lr_mult_dict:
                param_groups.append({'params': param})
            else:
                standard_lr_params.append(param)
        param_groups.append({'params': standard_lr_params})
        return param_groups

    def get_lr_multiplier(self, param_group):
        param = param_group["params"][0]
        if param in self.lr_mult_dict:
            return self.lr_mult_dict[param]
        else:
            return 1.


class MLP(LayerWiseLRModule):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, weight_norm, init, sample_batch=None):
        super().__init__()
        self.layers = []

        input_size = np.prod(input_size)  # inputs will be flattened

        if sample_batch is not None:
            sample_batch = sample_batch.view((sample_batch.size(0), -1))

        for layer_idx in range(num_layers):
            # Create hidden layer
            layer = nn.Linear(input_size if layer_idx == 0 else hidden_size, hidden_size)

            # Initialize weights, and optionally add WeightNorm
            layer, sample_batch = init_and_maybe_normalize_layer(
                layer, init=init, weight_norm=weight_norm, sample_batch=sample_batch)

            self.layers.append(layer)
            relu = nn.ReLU()
            self.layers.append(relu)
            if sample_batch is not None:
                sample_batch = relu(sample_batch)

        final_layer = nn.Linear(hidden_size, num_classes)
        self.layers.append(final_layer)

        self.model = nn.Sequential(*self.layers)

    def forward(self, x):
        x = x.view((x.size(0), -1))
        return self.model(x)


class CNN(LayerWiseLRModule):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, weight_norm, init, kernel_size=3,
                 sample_batch=None):
        super().__init__()
        self.layers = []
        input_size = input_size[-1]  # [H, W, C]

        for layer_idx in range(num_layers):
            # Create hidden layer
            if layer_idx in [0, 1]:  # use stride in the first two layers to reduce memory footprint
                layer = nn.Conv2d(input_size if layer_idx == 0 else hidden_size, hidden_size, kernel_size,
                                  stride=2, padding=1)
            else:
                layer = nn.Conv2d(hidden_size, hidden_size, kernel_size, stride=1, padding=1)

            # Initialize weights, and optionally add WeightNorm
            layer, sample_batch = init_and_maybe_normalize_layer(
                layer, init=init, weight_norm=weight_norm, sample_batch=sample_batch)

            self.layers.append(layer)
            relu = nn.ReLU()
            self.layers.append(relu)
            if sample_batch is not None:
                sample_batch = relu(sample_batch)

        self.final_layer = nn.Linear(hidden_size, num_classes)
        self.backbone_model = nn.Sequential(*self.layers)

    def forward(self, x):
        # CNN backbone
        x = self.backbone_model(x)

        # Global average pooling
        x = torch.mean(x.view(x.size(0), x.size(1), -1), dim=2)

        # Final classification layer
        return self.final_layer(x)


class ResNet(LayerWiseLRModule):
    def __init__(self, input_size, hidden_size, num_blocks, num_classes, weight_norm, batch_norm, init, kernel_size=3,
                 sample_batch=None, init_extra_param=None):
        super().__init__()

        self.use_wn = weight_norm
        self.use_bn = batch_norm
        self.init = init

        self.layers = []
        input_size = input_size[-1]  # [H, W, C]
        nstage = 3

        # Initial conv layer, before resblocks
        layer = nn.Conv2d(input_size, hidden_size, kernel_size, stride=1, padding=1)

        # Initialize weights, and optionally add WeightNorm
        layer, sample_batch = init_and_maybe_normalize_layer(
            layer, init=init, weight_norm=weight_norm, sample_batch=sample_batch)

        self.layers.append(layer)

        bn_layer = nn.BatchNorm2d(hidden_size) if self.use_bn else nn.Sequential()
        self.layers.append(bn_layer)

        nb_filters_prev = nb_filters_cur = hidden_size
        resblock_conv2_g_gain_fn = WideResNet.get_resblock_conv2_g_gain_fn(init_extra_param)
        for stage in range(nstage):
            resblock_count = 0
            nb_filters_cur = (2 ** stage) * hidden_size
            for i in range(num_blocks):
                resblock_count += 1
                stride = 1 if (i > 0 or stage == 0) else 2
                layer = WRNResBlock(nb_filters_prev, nb_filters_cur, stride, batch_norm=batch_norm)
                sample_batch = layer.init_and_maybe_normalize(weight_norm, init,
                                                              sample_batch=sample_batch,
                                                              num_resblocks=num_blocks,
                                                              resblock_id=resblock_count,
                                                              resblock_conv2_g_gain_fn=resblock_conv2_g_gain_fn)
                self.layers.append(layer)
                nb_filters_prev = nb_filters_cur

        bn_layer = nn.BatchNorm2d(nb_filters_prev) if self.use_bn else nn.Sequential()
        self.layers.append(bn_layer)

        self.backbone_model = nn.Sequential(*self.layers)
        self.final_layer = nn.Linear(nb_filters_cur, num_classes)

    def forward(self, x):
        # CNN backbone
        x = self.backbone_model(x)

        # Global average pooling
        x = torch.mean(x.view(x.size(0), x.size(1), -1), dim=2)

        # Final classification layer
        return self.final_layer(x)


class WideResNet(LayerWiseLRModule):
    def __init__(self, input_size, num_blocks, num_classes, weight_norm, batch_norm, init, sample_batch=None, k=1,
                 reduced_memory=False, init_extra_param=None):
        """
        Create a WRN with 6*num_blocks+4 layers:
            - 3 stages, with num_blocks blocks each, where each block has 2 conv layers
            - 1 conv layer before the residual blocks
            - 1 bottleneck 1x1 conv layer per stage
        """
        super().__init__()

        self.use_wn = weight_norm
        self.use_bn = batch_norm
        self.init = init

        self.n = num_blocks
        self.n_stages = 3
        self.k = k
        self.num_layers = 6 * self.n + 4
        self.total_resblocks = self.n_stages * self.n

        self._generate_arch_description(reduced_memory)
        self._generate_init_description(init)

        self.layers = []
        input_size = input_size[-1]  # [H, W, C]
        num_resblocks = 3 * num_blocks  # 3 stages (conv2..4 in the paper), with N residual blocks each

        # First conv layer (widened by k, which seems to perform better)
        conv1_stride = 1 if not reduced_memory else 2
        self.conv1 = nn.Conv2d(input_size, 16, kernel_size=3, stride=conv1_stride, padding=1)
        self.conv1, sample_batch = init_and_maybe_normalize_layer(
            self.conv1, init=init, weight_norm=weight_norm, sample_batch=sample_batch,
            g_gain=1.
        )

        # Residual blocks
        stage_names = ['conv1', 'conv2', 'conv3', 'conv4']
        stage_size = [16, 16 * k, 32 * k, 64 * k]  # conv1..4 in the paper
        resblock_count_list = []
        resblock_conv2_g_gain_fn = self.get_resblock_conv2_g_gain_fn(init_extra_param)
        for stage_idx in range(1, 4):
            resblock_count = 0
            stage_name = stage_names[stage_idx]
            stage_layers = []
            for block_idx in range(num_blocks):
                resblock_count += 1
                stride = 2 if (stage_idx in [2, 3] and block_idx == 0) else 1  # corresponds to conv3,4 in the paper
                input_dims = stage_size[stage_idx - 1] if block_idx == 0 else stage_size[stage_idx]
                layer = WRNResBlock(input_dims, stage_size[stage_idx], stride=stride, batch_norm=self.use_bn)
                sample_batch = layer.init_and_maybe_normalize(weight_norm, init,
                                                              sample_batch=sample_batch,
                                                              num_resblocks=num_blocks,
                                                              resblock_id=resblock_count,
                                                              resblock_conv2_g_gain_fn=resblock_conv2_g_gain_fn)
                stage_layers.append(layer)
            resblock_count_list.append(resblock_count)
            self.add_module(stage_name, nn.Sequential(*stage_layers))
        assert sum(resblock_count_list) == num_resblocks

        self.final_layer = nn.Linear(stage_size[-1], num_classes)

    def forward(self, x):
        # CNN backbone
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)

        # Global average pooling
        x = torch.mean(x.view(x.size(0), x.size(1), -1), dim=2)

        # Final classification layer
        return self.final_layer(x)

    @staticmethod
    def get_resblock_conv2_g_gain_fn(init_extra_param):
        if init_extra_param == 'hanin':
            def conv2_g_gain_init_fn(resblock_id, resblocks_per_stage):
                return (0.9**resblock_id)**2
        else:  # proposed init
            def conv2_g_gain_init_fn(resblock_id, resblocks_per_stage):
                return 1. / resblocks_per_stage
        return conv2_g_gain_init_fn

    def _generate_arch_description(self, reduced_memory):
        self.arch_extra_lines = ''
        if not self.use_bn:
            self.arch_extra_lines += '\n\n=================================================================='
            self.arch_extra_lines += '\nArchitecture: WRN-%d-%d' % (self.num_layers, self.k)
            self.arch_extra_lines += '\n    No ReLU after conv1'
            self.arch_extra_lines += '\n    ResBlock F(x)=conv2(ReLU(conv1(x)))'
            self.arch_extra_lines += '\n    Downsampling through stride=2: %sconv3_1, conv4_1' % \
                                     'conv1, ' if reduced_memory else ''
            self.arch_extra_lines += '\n==================================================================\n'

    def _generate_init_description(self, init):
        self.init_extra_lines = ''
        if 'proposed' in init:
            self.init_extra_lines += '\n\n=================================================================='
            self.init_extra_lines += '\nInit: proposed (with fan-in/fan-out correction)'
            self.init_extra_lines += '\n      with stage-wise normalization'
            self.init_extra_lines += '\n    conv1: g_gain=1'
            self.init_extra_lines += '\n    resblock_conv1: g_gain=2'
            self.init_extra_lines += '\n    resblock_conv2: g_gain=1/N'
            self.init_extra_lines += '\n    resblock_shortcut: g_gain=1'
            self.init_extra_lines += '\n==================================================================\n'

    def extra_repr(self):
        return self.arch_extra_lines + self.init_extra_lines


if __name__ == '__main__':
    # Test MLP
    print('Testing MLP...', end=' ')
    x_mlp = torch.ones((64, 28, 28))
    mlp = MLP(input_size=28*28, hidden_size=32, num_layers=2, num_classes=10, weight_norm=True, init='he_proposed')
    mlp(x_mlp)
    print('Success!')

    # Test CNN
    print('Testing CNN...', end=' ')
    x_cnn = torch.ones((64, 3, 28, 28))  # [b, h, w, c]
    cnn = CNN(input_size=(28, 28, 3), hidden_size=32, num_layers=2, num_classes=10, weight_norm=True,
              init='he_proposed')
    cnn(x_cnn)
    print('Success!')

    # Test WN-WideResNet
    print('Testing WN-WideResNet...', end=' ')
    x_wrn = torch.ones((64, 3, 28, 28))  # [b, h, w, c]
    wrn = WideResNet(input_size=(28, 28, 3), num_blocks=3, num_classes=10, weight_norm=True, batch_norm=False,
                     init='he_proposed')
    wrn(x_wrn)
    print('Success!')

    # Test BN-WideResNet
    print('Testing BN-WideResNet...', end=' ')
    x_wrn = torch.ones((64, 3, 28, 28))  # [b, h, w, c]
    wrn = WideResNet(input_size=(28, 28, 3), num_blocks=3, num_classes=10, weight_norm=False, batch_norm=True,
                     init='orthogonal')
    wrn(x_wrn)
    print('Success!')

    # Test WN-ResNet
    print('Testing WN-ResNet...', end=' ')
    x_wrn = torch.ones((64, 3, 32, 32))  # [b, h, w, c]
    wrn = ResNet(input_size=(32, 32, 3), hidden_size=16, num_blocks=3, num_classes=10,
                 weight_norm=True, batch_norm=False, init='he_proposed')
    wrn(x_wrn)
    print('Success!')

    # Test BN-ResNet
    print('Testing BN-ResNet...', end=' ')
    x_wrn = torch.ones((64, 3, 32, 32))  # [b, h, w, c]
    wrn = ResNet(input_size=(32, 32, 3), hidden_size=16, num_blocks=3, num_classes=10,
                 weight_norm=False, batch_norm=True, init='orthogonal')
    wrn(x_wrn)
    print('Success!')

    # Test MLP with data-dependent init for WN
    print('Testing MLP with data-dependent init for WN...', end=' ')
    x_mlp = torch.ones((64, 28, 28))
    mlp = MLP(input_size=28 * 28, hidden_size=32, num_layers=2, num_classes=10, weight_norm=True,
              init='orthogonal_datadep', sample_batch=x_mlp)
    mlp(x_mlp)
    print('Success!')

    # Test CNN with data-dependent init for WN
    print('Testing CNN with data-dependent init for WN...', end=' ')
    x_cnn = torch.ones((13, 3, 28, 28))  # [b, h, w, c]
    cnn = CNN(input_size=(28, 28, 3), hidden_size=17, num_layers=2, num_classes=10, weight_norm=True,
              init='orthogonal_datadep', sample_batch=x_cnn)
    cnn(x_cnn)
    print('Success!')

    # Test WideResNet with data-dependent init for WN
    print('Testing WideResNet with data-dependent init for WN...', end=' ')
    x_wrn = torch.ones((13, 3, 28, 28))  # [b, h, w, c]
    wrn = WideResNet(input_size=(28, 28, 3), num_blocks=3, num_classes=10, weight_norm=True, batch_norm=False,
                     init='orthogonal_datadep', sample_batch=x_wrn)
    wrn(x_wrn)
    print('Success!')

    # Test WN-ResNet with data-dependent init for WN
    print('Testing WN-ResNet with data-dependent init for WN...', end=' ')
    x_wrn = torch.ones((64, 3, 32, 32))  # [b, h, w, c]
    wrn = ResNet(input_size=(32, 32, 3), hidden_size=16, num_blocks=3, num_classes=10,
                 weight_norm=True, batch_norm=False, init='orthogonal_datadep', sample_batch=x_wrn)
    wrn(x_wrn)
    print('Success!')
