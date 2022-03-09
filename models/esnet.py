import torch
import torch.nn as nn
import torch.nn.functional as F
from numbers import Integral


def make_divisible(v, divisor=16, min_value=None):
    if min_value is None:
        min_value = divisor

    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def conv_3x3_bn(inp, oup, act, stride=1, group=1):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False, groups=group),
        nn.BatchNorm2d(oup),
        act(inplace=True)
    )


def conv_1x1_bn(inp, oup, act, group=1):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False, groups=group),
        nn.BatchNorm2d(oup),
        act(inplace=True)
    )


def channel_shuffle(x, num_groups):
    batch_size, num_channels, height, width = x.size()
    assert num_channels % num_groups == 0
    x = x.view(batch_size, num_groups, num_channels // num_groups, height, width)
    x = x.permute(0, 2, 1, 3, 4)
    return x.contiguous().view(batch_size, num_channels, height, width)


class SEModule(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SEModule, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv2d(in_channels=channel,
                               out_channels=channel // reduction,
                               kernel_size=1,
                               stride=1,
                               padding=0,
                               bias=False)
        self.conv2 = nn.Conv2d(in_channels=channel // reduction,
                               out_channels=channel,
                               kernel_size=1,
                               stride=1,
                               padding=0,
                               bias=False)

    def forward(self, inputs):
        outputs = self.avg_pool(inputs)
        outputs = F.relu(self.conv1(outputs))
        outputs = F.hardsigmoid(self.conv2(outputs))

        return inputs * outputs.expand_as(inputs)


class InvertedResidual(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, stride, act='hard_swish'):
        super(InvertedResidual, self).__init__()
        if act == 'relu':
            act = nn.ReLU
        elif act == 'relu6':
            act = nn.ReLU6
        elif act == 'leakyrelu':
            act = nn.LeakyReLU
        elif act == 'hard_swish':
            act = nn.Hardswish
        else:
            raise ValueError("the act is not available")
        self._conv_pw = conv_1x1_bn(in_channels//2, mid_channels//2, act=act)
        self._conv_dw = conv_3x3_bn(mid_channels//2, mid_channels//2, act=nn.ReLU, group=mid_channels//2, stride=stride)
        self._se = SEModule(mid_channels)

        self._conv_linear = conv_1x1_bn(mid_channels, out_channels//2, act=act)

    def forward(self, inputs):
        x1, x2 = torch.split(inputs, [inputs.shape[1]//2, inputs.shape[1]//2], dim=1)
        x2 = self._conv_pw(x2)
        x3 = self._conv_dw(x2)
        x3 = torch.cat([x2, x3], dim=1)
        x3 = self._se(x3)
        x3 = self._conv_linear(x3)
        out = torch.cat([x1, x3], dim=1)
        return channel_shuffle(out, 2)


class InvertedResidualDS(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, stride, act='hard_swish'):
        super(InvertedResidualDS, self).__init__()
        if act == 'relu':
            act = nn.ReLU
        elif act == 'relu6':
            act = nn.ReLU6
        elif act == 'leakyrelu':
            act = nn.LeakyReLU
        elif act == 'hard_swish':
            act = nn.Hardswish
        else:
            raise ValueError("the act is not available")

        self._conv_dw_1 = conv_3x3_bn(in_channels, in_channels, act=nn.ReLU, stride=stride, group=in_channels)
        self._conv_linear_1 = conv_1x1_bn(in_channels, out_channels//2, act=act)
        self._conv_pw_2 = conv_1x1_bn(in_channels, mid_channels//2, act=act)
        self._conv_dw_2 = conv_3x3_bn(mid_channels//2, mid_channels//2, stride=stride, group=mid_channels//2, act=nn.ReLU)
        self._se = SEModule(mid_channels//2)
        self._conv_linear_2 = conv_1x1_bn(mid_channels//2, out_channels//2, act=act)
        self._conv_dw_mv1 = conv_3x3_bn(out_channels, out_channels, group=out_channels, act=nn.Hardswish)
        self._conv_pw_mv1 = conv_1x1_bn(out_channels, out_channels, act=nn.Hardswish)

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


class ESNet(nn.Module):
    def __init__(self, scale=1.0, act="hard_swish", feature_maps=[4, 11, 14],
                 channel_ratio=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]):
        super(ESNet, self).__init__()
        if act == 'relu':
            act_fn = nn.ReLU
        elif act == 'relu6':
            act_fn = nn.ReLU6
        elif act == 'leakyrelu':
            act_fn = nn.LeakyReLU
        elif act == 'hard_swish':
            act_fn = nn.Hardswish
        else:
            raise ValueError("the act is not available")
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

        self._conv1 = conv_3x3_bn(3, stage_out_channels[1], stride=2, act=act_fn)
        self._max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self._feature_idx += 1

        # 2. bottle sequences
        self._block_list = []
        arch_idx = 0
        for stage_id, num_repeat in enumerate(stage_repeats):
            for i in range(num_repeat):
                channels_scales = channel_ratio[arch_idx]
                mid_c = make_divisible(
                    int(stage_out_channels[stage_id + 2] * channels_scales),
                    divisor=8)
                if i == 0:
                    block = InvertedResidualDS(
                            in_channels=stage_out_channels[stage_id + 1],
                            mid_channels=mid_c,
                            out_channels=stage_out_channels[stage_id + 2],
                            stride=2,
                            act=act)
                else:
                    block = InvertedResidual(
                            in_channels=stage_out_channels[stage_id + 2],
                            mid_channels=mid_c,
                            out_channels=stage_out_channels[stage_id + 2],
                            stride=1,
                            act=act)
                self._block_list.append(block)
                arch_idx += 1
                self._feature_idx += 1
                self._update_out_channels(stage_out_channels[stage_id + 2],
                                          self._feature_idx, self.feature_maps)

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

