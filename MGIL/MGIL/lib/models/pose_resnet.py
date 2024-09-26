# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# The SimDR and SA-SimDR part:
# Written by Yanjie Li (lyj20@mails.tsinghua.edu.cn)
# ------------------------------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging

import torch
import torch.nn as nn
from einops import rearrange, repeat

BN_MOMENTUM = 0.1
logger = logging.getLogger(__name__)


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=3, stride=stride,
        padding=1, bias=False
    )



class space_to_depth(nn.Module):
    # Changing the dimension of the Tensor
    def __init__(self, dimension=1):
        super(space_to_depth,self).__init__()
        self.d = dimension

    def forward(self, x):
        return torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1)


class EfficientChannelAttention(nn.Module):           # Efficient Channel Attention module
    def __init__(self, c, b=1, gamma=2):
        super(EfficientChannelAttention, self).__init__()
        t = int(abs((math.log(c, 2) + b) / gamma))
        k = t if t % 2 else t + 1

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv1d(1, 1, kernel_size=k, padding=int(k/2), bias=False)
        self.linear =  nn.Linear(c, 2, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.softmax =nn.Softmax(dim=1)
        self.relu = nn.ReLU(inplace=True)
        self.relu_2 = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.5)  # dropout训练

    def forward(self, x):
        b, c, _, _ = x.size()

        x = self.avg_pool(x)

        x = self.conv1(x.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        x = self.relu_2(x.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        # x = self.dropout(x.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        x = self.linear(x.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        x = self.relu(x)
        out = self.softmax(x)

        # out =out.view(b, -1,1,1)
        
        return out


class spd_conv_block(nn.Module):
    # Changing the dimension of the Tensor
    def __init__(self, inchannels, outchannels):
        super(spd_conv_block, self).__init__()

        self.conv2_dila =  nn.Conv2d(inchannels, outchannels, 3, 2, 2,dilation=2, bias=False)
        # self.conv3_dila =  nn.Conv2d(inchannels, outchannels, 3, 2, 3,dilation=3, bias=False)
        self.conv1 = nn.Conv2d(inchannels, outchannels, kernel_size=3, stride=1, padding=1, bias=False)        
        self.spd = space_to_depth()
        self.bn1 = nn.BatchNorm2d(4*outchannels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(4*outchannels, 2*outchannels, kernel_size=3, padding=1, bias=False)
        # self.conv2_2 = nn.Conv2d(2*outchannels, outchannels, kernel_size=3, padding=1, bias=False)
        # self.conv2_dila = nn.Conv2d(4*outchannels, outchannels, kernel_size=3, padding=2,dilation=2, bias=False)
        self.bn2 = nn.BatchNorm2d(2*outchannels)

        self.relu_2 = nn.ReLU(inplace=True)
        self.conv2_2 = nn.Conv2d(2*outchannels, outchannels, kernel_size=3, padding=1, bias=False)
        self.bn2_2 = nn.BatchNorm2d(outchannels)
        self.relu_2_2 = nn.ReLU(inplace=True)
        self.conv2_1X1 = nn.Conv2d(outchannels, outchannels, kernel_size=1)
         
        self.conv3= nn.Conv2d(inchannels,outchannels,3, 2, 1, bias=False)                                    
        self.bn3 =  nn.BatchNorm2d(outchannels)
        # self.ECA =  eca_layer(outchannels)
        # self.channel_layer = CAModule(4*outchannels, 16)
        # self.spatial_layer = SAModule()
        # self.SE = SELayer(reduction=16)
        # self.SE = SELayer(in_channel= 2*outchannels,reduction=16).cuda()
        self.eca = EfficientChannelAttention(outchannels*2)

                                    

    def forward(self, x):
        residual = x
        # residual_2 = x

        out = self.conv1(x)
        out = self.spd(out)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu_2(out)
        out = self.conv2_2(out)
        out = self.bn2_2(out)



          

        residual = self.conv2_dila(residual)
        residual = self.conv2_1X1(residual)


        fr_feature = torch.cat([out, residual], dim=1)

        
        # fusion_score = self.SE(fr_feature)
        fusion_score = self.eca(fr_feature)

        out = fusion_score[:,0,:,:].unsqueeze(-1)*out+fusion_score[:,1,:,:].unsqueeze(-1)*residual

        out = self.relu(out)






        return out



class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion,
                                  momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class PoseResNet(nn.Module):

    def __init__(self, block, layers, cfg, **kwargs):
        super(PoseResNet, self).__init__()

        self.inplanes = 64
        extra = cfg.MODEL.EXTRA
        self.deconv_with_bias = extra.DECONV_WITH_BIAS
        self.coord_representation = cfg.MODEL.COORD_REPRESENTATION
        assert  self.coord_representation in ['heatmap', 'simdr', 'sa-simdr'], 'only heatmap or simdr or sa-simdr supported ~ '

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        # used for deconv layers
        self.deconv_layers = self._make_deconv_layer(
            extra.NUM_DECONV_LAYERS,
            extra.NUM_DECONV_FILTERS,
            extra.NUM_DECONV_KERNELS,
        )

        self.final_layer = nn.Conv2d(
            in_channels=extra.NUM_DECONV_FILTERS[-1],
            out_channels=cfg.MODEL.NUM_JOINTS,
            kernel_size=extra.FINAL_CONV_KERNEL,
            stride=1,
            padding=1 if extra.FINAL_CONV_KERNEL == 3 else 0
        )

        # head
        if self.coord_representation == 'simdr' or  self.coord_representation == 'sa-simdr':
            self.mlp_head_x = nn.Linear(cfg.MODEL.HEAD_INPUT, int(cfg.MODEL.IMAGE_SIZE[0]*cfg.MODEL.SIMDR_SPLIT_RATIO))
            self.mlp_head_y = nn.Linear(cfg.MODEL.HEAD_INPUT, int(cfg.MODEL.IMAGE_SIZE[1]*cfg.MODEL.SIMDR_SPLIT_RATIO))

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                # spd_conv_block(self.inplanes, planes * block.expansion),
                nn.BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM),
            )
        
        elif stride != 1 :
            downsample = nn.Sequential(
                #  nn.Conv2d(self.inplanes, planes * block.expansion,
                #            kernel_size=1, stride=stride, bias=False),
                spd_conv_block(self.inplanes, planes * block.expansion),
                #  nn.BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _get_deconv_cfg(self, deconv_kernel, index):
        if deconv_kernel == 4:
            padding = 1
            output_padding = 0
        elif deconv_kernel == 3:
            padding = 1
            output_padding = 1
        elif deconv_kernel == 2:
            padding = 0
            output_padding = 0

        return deconv_kernel, padding, output_padding

    def _make_deconv_layer(self, num_layers, num_filters, num_kernels):
        assert num_layers == len(num_filters), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'
        assert num_layers == len(num_kernels), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'

        layers = []
        for i in range(num_layers):
            kernel, padding, output_padding = \
                self._get_deconv_cfg(num_kernels[i], i)

            planes = num_filters[i]
            layers.append(
                nn.ConvTranspose2d(
                    in_channels=self.inplanes,
                    out_channels=planes,
                    kernel_size=kernel,
                    stride=2,
                    padding=padding,
                    output_padding=output_padding,
                    bias=self.deconv_with_bias))
            layers.append(nn.BatchNorm2d(planes, momentum=BN_MOMENTUM))
            layers.append(nn.ReLU(inplace=True))
            self.inplanes = planes

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.deconv_layers(x)

        x = self.final_layer(x)

        if self.coord_representation == 'simdr' or self.coord_representation == 'sa-simdr':
            x = rearrange(x, 'b c h w -> b c (h w)')
            pred_x = self.mlp_head_x(x)
            pred_y = self.mlp_head_y(x)
            return pred_x, pred_y
        elif self.coord_representation == 'heatmap':
            return x

    def init_weights(self, pretrained=''):
        if os.path.isfile(pretrained):
            logger.info('=> init deconv weights from normal distribution')
            for name, m in self.deconv_layers.named_modules():
                if isinstance(m, nn.ConvTranspose2d):
                    logger.info('=> init {}.weight as normal(0, 0.001)'.format(name))
                    logger.info('=> init {}.bias as 0'.format(name))
                    nn.init.normal_(m.weight, std=0.001)
                    if self.deconv_with_bias:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    logger.info('=> init {}.weight as 1'.format(name))
                    logger.info('=> init {}.bias as 0'.format(name))
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
            logger.info('=> init final conv weights from normal distribution')
            for m in self.final_layer.modules():
                if isinstance(m, nn.Conv2d):
                    logger.info('=> init {}.weight as normal(0, 0.001)'.format(name))
                    logger.info('=> init {}.bias as 0'.format(name))
                    nn.init.normal_(m.weight, std=0.001)
                    nn.init.constant_(m.bias, 0)

            pretrained_state_dict = torch.load(pretrained)
            logger.info('=> loading pretrained model {}'.format(pretrained))
            self.load_state_dict(pretrained_state_dict, strict=False)
        else:
            logger.info('=> init weights from normal distribution')
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.normal_(m.weight, std=0.001)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.ConvTranspose2d):
                    nn.init.normal_(m.weight, std=0.001)
                    if self.deconv_with_bias:
                        nn.init.constant_(m.bias, 0)


resnet_spec = {
    18: (BasicBlock, [2, 2, 2, 2]),
    34: (BasicBlock, [3, 4, 6, 3]),
    50: (Bottleneck, [3, 4, 6, 3]),
    101: (Bottleneck, [3, 4, 23, 3]),
    152: (Bottleneck, [3, 8, 36, 3])
}


def get_pose_net(cfg, is_train, **kwargs):
    num_layers = cfg.MODEL.EXTRA.NUM_LAYERS

    block_class, layers = resnet_spec[num_layers]

    model = PoseResNet(block_class, layers, cfg, **kwargs)

    if is_train and cfg.MODEL.INIT_WEIGHTS:
        model.init_weights(cfg.MODEL.PRETRAINED)
    return model
