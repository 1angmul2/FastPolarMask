# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, DepthwiseSeparableConvModule
from mmcv.runner import auto_fp16, BaseModule

from mmdet.models.builder import NECKS


class DarknetBottleneck(BaseModule):
    """The basic bottleneck block used in Darknet.

    Each Block consists of two ConvModules and the input is added to the
    final output. Each ConvModule is composed of Conv, BN, and act_cfg.
    The first convLayer has filter size of 1x1 and the second one has the
    filter size of 3x3.

    Args:
        in_channels (int): The input channels of this Module.
        out_channels (int): The output channels of this Module.
        expansion (int): The kernel size of the convolution. Default: 0.5
        add_identity (bool): Whether to add identity to the out.
            Default: True
        use_depthwise (bool): Whether to use depthwise separable convolution.
            Default: False
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 expansion=0.5,
                 add_identity=True,
                 use_depthwise=False,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='LRelu'),
                 init_cfg=None):
        super(DarknetBottleneck, self).__init__(init_cfg)
        hidden_channels = int(out_channels * expansion)
        conv_func = DepthwiseSeparableConvModule if use_depthwise else ConvModule
        self.conv1 = ConvModule(
            in_channels=in_channels,
            out_channels=hidden_channels,
            kernel_size=1,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self.conv2 = conv_func(
            in_channels=hidden_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self.add_identity = \
            add_identity and in_channels == out_channels

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.conv2(out)

        if self.add_identity:
            return out + identity
        else:
            return out


class CSPLayer(BaseModule):
    """Cross Stage Partial Layer.

    Args:
        in_channels (int): The input channels of the CSP layer.
        out_channels (int): The output channels of the CSP layer.
        expand_ratio (float): Ratio to adjust the number of channels of the
            hidden layer. Default: 0.5
        num_blocks (int): Number of blocks. Default: 1
        add_identity (bool): Whether to add identity in blocks.
            Default: True
        use_depthwise (bool): Whether to depthwise separable convolution in
            blocks. Default: False
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 expand_ratio=0.5,
                 num_blocks=1,
                 add_identity=True,
                 use_depthwise=False,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='LRelu'),
                 init_cfg=None):
        super(CSPLayer, self).__init__(init_cfg)
        mid_channels = int(out_channels * expand_ratio)
        self.main_conv = ConvModule(in_channels,
                                    mid_channels,
                                    3, 1, 1,
                                    norm_cfg=norm_cfg,
                                    act_cfg=act_cfg)
        self.short_conv = ConvModule(in_channels,
                                     mid_channels,
                                     3, 1, 1,
                                     norm_cfg=norm_cfg,
                                     act_cfg=act_cfg)
        self.final_conv = ConvModule(
            2 * mid_channels, out_channels, 3, 1, 1,
            norm_cfg=norm_cfg, act_cfg=act_cfg)

        self.blocks = nn.Sequential(* [
            DarknetBottleneck(
                mid_channels,
                mid_channels,
                kernel_size,
                1.0,
                add_identity,
                use_depthwise,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg) for _ in range(num_blocks)
        ])

    def forward(self, x):
        x_short = self.short_conv(x)

        x_main = self.main_conv(x)
        x_main = self.blocks(x_main)

        x_final = torch.cat((x_main, x_short), dim=1)
        return self.final_conv(x_final)


class Channel_T(BaseModule):
    def __init__(self,
                 in_channels=[116, 232, 464],
                 out_channels=96,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='LRelu')):
        super(Channel_T, self).__init__()
        self.convs = nn.ModuleList()
        for i in range(len(in_channels)):
            self.convs.append(
                ConvModule(
                    in_channels[i], out_channels, 1,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg))

    def forward(self, x):
        outs = [self.convs[i](x[i]) for i in range(len(x))]
        return outs


@NECKS.register_module()
class CSPPAN(BaseModule):
    """Path Aggregation Network with CSP module.

    Args:
        in_channels (List[int]): Number of input channels per scale.
        out_channels (int): Number of output channels (used at each scale)
        kernel_size (int): The conv2d kernel size of this Module.
        num_features (int): Number of output features of CSPPAN module.
        num_csp_blocks (int): Number of bottlenecks in CSPLayer. Default: 1
        use_depthwise (bool): Whether to depthwise separable convolution in
            blocks. Default: True
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=5,
                 num_features=3,
                 num_csp_blocks=1,
                 use_depthwise=True,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='HSwish'),
                 spatial_scales=[0.125, 0.0625, 0.03125],
                 init_cfg=None):
        super(CSPPAN, self).__init__(init_cfg)
        self.conv_t = Channel_T(in_channels, out_channels,
                                norm_cfg=norm_cfg, act_cfg=act_cfg)
        in_channels = [out_channels] * len(spatial_scales)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.spatial_scales = spatial_scales
        self.num_features = num_features
        conv_func = DepthwiseSeparableConvModule if use_depthwise else ConvModule

        if self.num_features == 4:
            self.first_top_conv = conv_func(
                in_channels[0], in_channels[0], kernel_size,
                stride=2, padding=kernel_size//2,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg)
            self.second_top_conv = conv_func(
                in_channels[0], in_channels[0], kernel_size,
                stride=2, padding=kernel_size//2,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg)
            self.spatial_scales.append(self.spatial_scales[-1] / 2)

        # build top-down blocks
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.top_down_blocks = nn.ModuleList()
        for idx in range(len(in_channels) - 1, 0, -1):
            self.top_down_blocks.append(
                CSPLayer(
                    in_channels[idx - 1] * 2,
                    in_channels[idx - 1],
                    kernel_size=kernel_size,
                    num_blocks=num_csp_blocks,
                    add_identity=False,
                    use_depthwise=use_depthwise,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg))

        # build bottom-up blocks
        self.downsamples = nn.ModuleList()
        self.bottom_up_blocks = nn.ModuleList()
        for idx in range(len(in_channels) - 1):
            self.downsamples.append(
                conv_func(
                    in_channels[idx],
                    in_channels[idx],
                    kernel_size=kernel_size,
                    stride=2,
                    padding=kernel_size // 2,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg))
            self.bottom_up_blocks.append(
                CSPLayer(
                    in_channels[idx] * 2,
                    in_channels[idx + 1],
                    kernel_size=kernel_size,
                    num_blocks=num_csp_blocks,
                    add_identity=False,
                    use_depthwise=use_depthwise,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg))

    @auto_fp16()
    def forward(self, inputs):
        """
        Args:
            inputs (tuple[Tensor]): input features.

        Returns:
            tuple[Tensor]: CSPPAN features.
        """
        assert len(inputs) == len(self.in_channels)
        inputs = self.conv_t(inputs)

        # top-down path
        inner_outs = [inputs[-1]]
        for idx in range(len(self.in_channels) - 1, 0, -1):
            feat_heigh = inner_outs[0]
            feat_low = inputs[idx - 1]

            upsample_feat = self.upsample(feat_heigh)

            inner_out = self.top_down_blocks[len(self.in_channels) - 1 - idx](
                torch.cat([upsample_feat, feat_low], 1))
            inner_outs.insert(0, inner_out)

        # bottom-up path
        outs = [inner_outs[0]]
        for idx in range(len(self.in_channels) - 1):
            feat_low = outs[-1]
            feat_height = inner_outs[idx + 1]
            downsample_feat = self.downsamples[idx](feat_low)
            out = self.bottom_up_blocks[idx](torch.cat(
                [downsample_feat, feat_height], 1))
            outs.append(out)

        top_features = None
        if self.num_features == 4:
            top_features = self.first_top_conv(inputs[-1])
            top_features = top_features + self.second_top_conv(outs[-1])
            outs.append(top_features)

        return tuple(outs)


