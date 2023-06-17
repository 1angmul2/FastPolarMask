# Copyright (c) OpenMMLab. All rights reserved.
import math
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn import Conv2d, MaxPool2d, AdaptiveAvgPool2d

from mmcv.cnn import ConvModule
from mmcv.runner import BaseModule
from torch.nn.modules.batchnorm import _BatchNorm

from mmdet.models.builder import BACKBONES
from mmdet.models.utils import CSPLayer

from numbers import Integral

__all__ = ['ESNet']


def make_divisible(v, divisor=16, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def channel_shuffle(x, groups):
    batch_size, num_channels, height, width = x.shape[0:4]
    assert num_channels % groups == 0, 'num_channels should be divisible by groups'
    channels_per_group = num_channels // groups
    x = x.view((batch_size, groups, channels_per_group, height, width))
    x = x.permute(0, 2, 1, 3, 4)
    x = x.reshape((batch_size, num_channels, height, width))
    return x


class SEModule(BaseModule):
    def __init__(self, channel, reduction=4, init_cfg=None):
        super(SEModule, self).__init__(init_cfg)
        self.avg_pool = AdaptiveAvgPool2d(1)
        self.conv1 = Conv2d(
            in_channels=channel,
            out_channels=channel // reduction,
            kernel_size=1,
            stride=1,
            padding=0)
        self.conv2 = Conv2d(
            in_channels=channel // reduction,
            out_channels=channel,
            kernel_size=1,
            stride=1,
            padding=0)
        
        # export friendly
        self.hard_sigmoid = nn.Hardsigmoid()

    def forward(self, inputs):
        outputs = self.avg_pool(inputs)
        outputs = self.conv1(outputs)
        outputs = F.relu(outputs)
        outputs = self.conv2(outputs)
        outputs = self.hard_sigmoid(outputs)
        return inputs * outputs


class InvertedResidual(BaseModule):
    def __init__(self,
                 in_channels,
                 mid_channels,
                 out_channels,
                 stride,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='Relu'),
                 init_cfg=None):
        super(InvertedResidual, self).__init__(init_cfg)
        self._conv_pw = ConvModule(
            in_channels=in_channels // 2,
            out_channels=mid_channels // 2,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self._conv_dw = ConvModule(
            in_channels=mid_channels // 2,
            out_channels=mid_channels // 2,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=mid_channels // 2,
            norm_cfg=norm_cfg,
            act_cfg=None)
        self._se = SEModule(mid_channels)

        self._conv_linear = ConvModule(
            in_channels=mid_channels,
            out_channels=out_channels // 2,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)

    def forward(self, inputs):
        x1, x2 = torch.split(
            inputs,
            split_size_or_sections=[inputs.shape[1] // 2, inputs.shape[1] // 2],
            dim=1)
        x2 = self._conv_pw(x2)
        x3 = self._conv_dw(x2)
        x3 = torch.cat([x2, x3], dim=1)
        x3 = self._se(x3)
        x3 = self._conv_linear(x3)
        out = torch.cat([x1, x3], dim=1)
        return channel_shuffle(out, 2)


class InvertedResidualDS(BaseModule):
    def __init__(self,
                 in_channels,
                 mid_channels,
                 out_channels,
                 stride,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='Relu'),
                 init_cfg=None):
        super(InvertedResidualDS, self).__init__(init_cfg)

        # branch1
        self._conv_dw_1 = ConvModule(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=in_channels,
            norm_cfg=norm_cfg,
            act_cfg=None)
        self._conv_linear_1 = ConvModule(
            in_channels=in_channels,
            out_channels=out_channels // 2,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        # branch2
        self._conv_pw_2 = ConvModule(
            in_channels=in_channels,
            out_channels=mid_channels // 2,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self._conv_dw_2 = ConvModule(
            in_channels=mid_channels // 2,
            out_channels=mid_channels // 2,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=mid_channels // 2,
            norm_cfg=norm_cfg,
            act_cfg=None)
        self._se = SEModule(mid_channels // 2)
        self._conv_linear_2 = ConvModule(
            in_channels=mid_channels // 2,
            out_channels=out_channels // 2,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self._conv_dw_mv1 = ConvModule(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=out_channels,
            norm_cfg=norm_cfg,
            act_cfg=dict(type='HSwish'))
        self._conv_pw_mv1 = ConvModule(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1,
            norm_cfg=norm_cfg,
            act_cfg=dict(type='HSwish'))

    def forward(self, inputs):
        x1 = self._conv_dw_1(inputs)
        x1 = self._conv_linear_1(x1)
        x2 = self._conv_pw_2(inputs)
        x2 = self._conv_dw_2(x2)
        x2 = self._se(x2)
        x2 = self._conv_linear_2(x2)
        out = torch.cat([x1, x2], dim=1)
        out = self._conv_dw_mv1(out)
        out = self._conv_pw_mv1(out)

        return out


@ BACKBONES.register_module()
class ESNet(BaseModule):
    def __init__(self,
                 scale=1.0,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='HSwish'),
                 feature_maps=[4, 11, 14],
                 channel_ratio=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                 init_cfg=None):
        super(ESNet, self).__init__(init_cfg)
        self.scale = scale
        if isinstance(feature_maps, Integral):
            feature_maps = [feature_maps]
        self.feature_maps = feature_maps
        stage_repeats = [3, 7, 3]

        stage_out_channels = [
            -1, 24, make_divisible(128 * scale), make_divisible(256 * scale),
            make_divisible(512 * scale), 1024
        ]

        self._out_channels = []
        self._feature_idx = 0
        # 1. conv1
        self._conv1 = ConvModule(
            in_channels=3,
            out_channels=stage_out_channels[1],
            kernel_size=3,
            stride=2,
            padding=1,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self._max_pool = MaxPool2d(kernel_size=3, stride=2, padding=1)
        self._feature_idx += 1

        # 2. bottleneck sequences
        _block_list = []
        arch_idx = 0
        for stage_id, num_repeat in enumerate(stage_repeats):
            for i in range(num_repeat):
                channels_scales = channel_ratio[arch_idx]
                mid_c = make_divisible(
                    int(stage_out_channels[stage_id + 2] * channels_scales),
                    divisor=8)
                if i == 0:
                    block = (
                        str(stage_id + 2) + '_' + str(i + 1),
                        InvertedResidualDS(
                            in_channels=stage_out_channels[stage_id + 1],
                            mid_channels=mid_c,
                            out_channels=stage_out_channels[stage_id + 2],
                            stride=2,
                            act_cfg=act_cfg))
                else:
                    block = (
                        str(stage_id + 2) + '_' + str(i + 1),
                        InvertedResidual(
                            in_channels=stage_out_channels[stage_id + 2],
                            mid_channels=mid_c,
                            out_channels=stage_out_channels[stage_id + 2],
                            stride=1,
                            act_cfg=act_cfg))
                _block_list.append(block)
                arch_idx += 1
                self._feature_idx += 1
                self._update_out_channels(stage_out_channels[stage_id + 2],
                                          self._feature_idx, self.feature_maps)
        self._block_list = nn.Sequential(OrderedDict(_block_list))

    def _update_out_channels(self, channel, feature_idx, feature_maps):
        if feature_idx in feature_maps:
            self._out_channels.append(channel)

    def forward(self, inputs):
        y = self._conv1(inputs)
        y = self._max_pool(y)
        outs = []
        for i, inv in enumerate(self._block_list):
            y = inv(y)
            if i + 2 in self.feature_maps:
                outs.append(y)

        return outs
