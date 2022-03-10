import torch
import torch.nn as nn
import torch.nn.functional as F
from losses.gfocal_loss import DistributionFocalLoss
from losses.giou_loss import GIoULoss
from losses.varifocal_loss import VarifocalLoss
import math


class ConvNormLayer(nn.Module):
    def __init__(self,
                 ch_in=96,
                 ch_out=96,
                 filter_size=3,
                 stride=1,
                 groups=1):
        super(ConvNormLayer, self).__init__()
        self.conv = nn.Conv2d(
            in_channels=ch_in,
            out_channels=ch_out,
            kernel_size=filter_size,
            groups=groups,
            stride=stride,
            padding=(filter_size - 1) // 2,
            bias=False)
        self.bn = nn.BatchNorm2d(ch_out)

    def forward(self, x):
        x = self.bn(self.conv(x))
        return x


class DGQP(nn.Module):
    """Distribution-Guided Quality Predictor of GFocal head

    Args:
        reg_topk (int): top-k statistics of distribution to guide LQE
        reg_channels (int): hidden layer unit to generate LQE
        add_mean (bool): Whether to calculate the mean of top-k statistics
    """

    def __init__(self, reg_topk=4, reg_channels=64, add_mean=True):
        super(DGQP, self).__init__()
        self.reg_topk = reg_topk
        self.reg_channels = reg_channels
        self.add_mean = add_mean
        self.total_dim = reg_topk
        if add_mean:
            self.total_dim += 1
        self.reg_conv1 = nn.Conv2d(
                            in_channels=4 * self.total_dim,
                            out_channels=self.reg_channels,
                            kernel_size=1,)

        self.reg_conv2 = nn.Conv2d(
                in_channels=self.reg_channels,
                out_channels=1,
                kernel_size=1,)
        for m in self.modules():
            if isinstance(m ,nn.Conv2d):
                m.weight.data.normal_(mean=0., std=0.01)
                m.bias.data.fill_(0.)

    def forward(self, x):
        """Forward feature from the regression head to get integral result of
        bounding box location.
        Args:
            x (Tensor): Features of the regression head, shape (N, 4*(n+1)),
                n is self.reg_max.
        Returns:
            x (Tensor): Integral result of box locations, i.e., distance
                offsets from the box center in four directions, shape (N, 4).
        """
        N, _, H, W = x.shape[:]
        prob = F.softmax(x.reshape([N, 4, -1, H, W]), dim=2)
        prob_topk, _ = prob.topk(self.reg_topk, dim=2)
        if self.add_mean:
            stat = torch.cat([prob_topk, prob_topk.mean(axis=2, keepdim=True)], dim=2)
        else:
            stat = prob_topk
        y = F.relu(self.reg_conv1(stat.reshape([N, -1, H, W])))
        y = F.sigmoid(self.reg_conv2(y))
        return y


class PicoFeat(nn.Module):
    """
    PicoFeat of PicoDet

    Args:
        feat_in (int): The channel number of input Tensor.
        feat_out (int): The channel number of output Tensor.
        num_convs (int): The convolution number of the LiteGFLFeat.
        norm_type (str): Normalization type, 'bn'/'sync_bn'/'gn'.
    """

    def __init__(self,
                 feat_in=128,
                 feat_out=128,
                 num_fpn_stride=4,
                 num_convs=4,
                 norm_type='bn',
                 share_cls_reg=True,
                 act='hard_swish'):
        super(PicoFeat, self).__init__()
        self.num_convs = num_convs
        self.norm_type = norm_type
        self.share_cls_reg = share_cls_reg
        self.act = act
        self.cls_convs = []
        self.reg_convs = []
        for stage_idx in range(num_fpn_stride):
            cls_subnet_convs = []
            reg_subnet_convs = []
            for i in range(self.num_convs):
                in_c = feat_in if i == 0 else feat_out
                cls_conv_dw = ConvNormLayer(
                                ch_in=in_c,
                                ch_out=feat_out,
                                filter_size=5,
                                stride=1,
                                groups=feat_out)
                cls_subnet_convs.append(cls_conv_dw)
                cls_conv_pw = ConvNormLayer(
                                ch_in=in_c,
                                ch_out=feat_out,
                                filter_size=1,
                                stride=1)
                cls_subnet_convs.append(cls_conv_pw)
                if not self.share_cls_reg:
                    reg_conv_dw = ConvNormLayer(
                                    ch_in=in_c,
                                    ch_out=feat_out,
                                    filter_size=5,
                                    stride=1,
                                    groups=feat_out,)

                    reg_subnet_convs.append(reg_conv_dw)
                    reg_conv_pw = ConvNormLayer(
                                    ch_in=in_c,
                                    ch_out=feat_out,
                                    filter_size=1,
                                    stride=1,)

                    reg_subnet_convs.append(reg_conv_pw)
            self.cls_convs.append(cls_subnet_convs)
            self.reg_convs.append(reg_subnet_convs)

    def act_func(self, x):
        if self.act == "leaky_relu":
            x = F.leaky_relu(x)
        elif self.act == "hard_swish":
            x = F.hardswish(x)
        return x

    def forward(self, fpn_feat, stage_idx):
        assert stage_idx < len(self.cls_convs)
        cls_feat = fpn_feat
        reg_feat = fpn_feat
        for i in range(len(self.cls_convs[stage_idx])):
            cls_feat = self.act_func(self.cls_convs[stage_idx][i](cls_feat))
            if not self.share_cls_reg:
                reg_feat = self.act_func(self.reg_convs[stage_idx][i](reg_feat))
        return cls_feat, reg_feat


class PicoHead(nn.Module):
    def __init__(self, num_classes=80, fpn_stride=[8, 16, 32, 64], prior_prob=0.01,
                 reg_max=7, feat_in_chan=128, nms=None, nms_pre=1000, cell_offset=0.5):
        super(PicoHead, self).__init__()

        self.conv_feat = PicoFeat()
        self.dgqp_module = DGQP()
        self.num_classes = num_classes
        self.fpn_stride = fpn_stride
        self.prior_prob = prior_prob
        self.loss_vfl = VarifocalLoss()
        self.loss_dfl = DistributionFocalLoss()
        self.loss_bbox = GIoULoss()
        self.reg_max = reg_max
        self.feat_in_chan = feat_in_chan
        self.nms = nms
        self.nms_pre = nms_pre
        self.cell_offset = cell_offset
        self.use_sigmoid = self.loss_vfl.use_sigmoid
        if self.use_sigmoid:
            self.cls_out_channels = self.num_classes
        else:
            self.cls_out_channels = self.num_classes + 1
        bias_init_value = -math.log((1 - self.prior_prob) / self.prior_prob)

        self.head_cls_list = []
        self.head_reg_list = []
        for i in range(len(fpn_stride)):
            head_cls = nn.Conv2d(
                            in_channels=self.feat_in_chan,
                            out_channels=self.cls_out_channels + 4 * (self.reg_max + 1)
                            if self.conv_feat.share_cls_reg else self.cls_out_channels,
                            kernel_size=1,
                            stride=1,
                            padding=0,)
            self.head_cls_list.append(head_cls)
            nn.init.normal_(head_cls.weight.data, mean=0., std=0.01)
            nn.init.constant_(head_cls.bias.data, val=bias_init_value)
            if not self.conv_feat.share_cls_reg:
                head_reg = nn.Conv2d(
                            in_channels=self.feat_in_chan,
                            out_channels=4 * (self.reg_max + 1),
                            kernel_size=1,
                            stride=1,
                            padding=0,)
                self.head_reg_list.append(head_reg)
                nn.init.normal_(head_cls.weight.data, mean=0., std=0.01)
                nn.init.constant_(head_cls.bias.data, val=0.)

    def forward(self, fpn_feats, deploy=False):

        assert len(fpn_feats) == len(self.fpn_stride)

        cls_logits_list = []
        bboxes_reg_list = []
        for i ,fpn_feat in enumerate(fpn_feats):
            print("fpn feat shape:", fpn_feat.shape)
            conv_cls_feat, conv_reg_feat = self.conv_feat(fpn_feat, i)
            if self.conv_feat.share_cls_reg:
                cls_logits = self.head_cls_list[i](conv_cls_feat)
                cls_score, bbox_pred = torch.split(cls_logits, [self.cls_out_channels, 4*(self.reg_max+1)], dim=1)
            else:
                cls_score = self.head_cls_list[i](conv_cls_feat)
                bbox_pred = self.head_reg_list[i](conv_reg_feat)

            if self.dgqp_module:
                quality_score = self.dgqp_module(bbox_pred)
                cls_score = F.sigmoid(cls_score) * quality_score

            if deploy:
                cls_score = F.sigmoid(cls_score).reshape([1, self.cls_out_channels, -1]).transpose([0, 2, 1])
                bbox_pred = bbox_pred.reshape([1, (self.reg_max + 1) * 4, -1]).transpose([0, 2, 1])

            elif not self.training:
                cls_score = F.sigmoid(cls_score.transpose([0, 2, 3, 1]))
                bbox_pred = bbox_pred.transpose([0, 2, 3, 1])

            cls_logits_list.append(cls_score)
            bboxes_reg_list.append(bbox_pred)

        return (cls_logits_list, bboxes_reg_list)


