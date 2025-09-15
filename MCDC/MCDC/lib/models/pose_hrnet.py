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
import math

import os
import logging

import torch
import torch.nn as nn
from einops import rearrange, repeat
import fvcore.nn.weight_init as weight_init

BN_MOMENTUM = 0.1
logger = logging.getLogger(__name__)

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)



class CAModule(nn.Module):
    '''Channel Attention Module'''
    def __init__(self, channels, reduction):
        super(CAModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.shared_mlp = nn.Sequential(nn.Conv2d(channels, channels // reduction, kernel_size=1, padding=0, bias=False),
                                        nn.ReLU(inplace=True),
                                        nn.Conv2d(channels // reduction, channels, kernel_size=1, padding=0, bias=False))    
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        input = x
        avg_pool = self.avg_pool(x)
        max_pool = self.max_pool(x)
        x = self.shared_mlp(avg_pool) + self.shared_mlp(max_pool)
        attn_out = x
        x = self.sigmoid(x)
        return input * x, attn_out


class SAModule(nn.Module):
    '''Spatial Attention Module'''
    def __init__(self):
        super(SAModule, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=3, padding=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        input = x
        avg_c = torch.mean(x, 1, True)
        max_c, _ = torch.max(x, 1, True)
        x = torch.cat((avg_c, max_c), 1)
        x = self.conv(x)
        attn_out = x
        x = self.sigmoid(x)
        return input * x, attn_out

class space_to_depth(nn.Module):
    # Changing the dimension of the Tensor
    def __init__(self, dimension=1):
        super(space_to_depth,self).__init__()
        self.d = dimension

    def forward(self, x):
        return torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1)


class eca_layer(nn.Module):
    """Constructs a ECA module.

    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """
    def __init__(self, channel, k_size=3):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False) 
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # feature descriptor on the global spatial information
        y = self.avg_pool(x)

        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)







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
        x = self.dropout(x.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        x = self.linear(x.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        
        x = self.relu(x)
        out = self.softmax(x)
        
        # out =out.view(b, -1,1,1)
        
        return out



class eca_block(nn.Module):

    def __init__(self, channel, gamma=2, b=1):
        super(eca_block, self).__init__()
        kernel_size = int(abs((math.log(channel, 2) + b) / gamma))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1
        # 控制根据输入的channel控制kernel_size 的大小，保证kernel_size是一个奇数 
        padding = kernel_size // 2
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.size()
        avg = self.avg_pool(x).view([b, 1, c])
        out = self.conv(avg)
        out = self.sigmoid(out).view([b, c, 1, 1])
        return out * x





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
        self.ECA =  eca_layer(outchannels)
        self.channel_layer = CAModule(4*outchannels, 16)
        self.spatial_layer = SAModule()
        # self.SE = SELayer(reduction=16)
        self.eca = EfficientChannelAttention(outchannels*2)

                                    

    def forward(self, x):
        residual = x
        # residual_2 = x

        out = self.conv1(x)
        out = self.spd(out)
        out = self.bn1(out)
        out = self.relu(out)
        # out, att_c = self.channel_layer(out)
        # out, att_c = self.channel_layer(out)
        # out, att_s = self.spatial_layer(out)
        # out = self.conv2_1(out)
        # out = self.conv2_2(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu_2(out)
        out = self.conv2_2(out)
        out = self.bn2_2(out)
        # out, att_c = self.channel_layer(out)
        # out, att_s = self.spatial_layer(out)
        # out = self.ECA(out)


        residual = self.conv2_dila(residual)
        residual = self.conv2_1X1(residual)
        # residual_2 = self.conv3_dila(residual_2)


        fr_feature = torch.cat([out, residual], dim=1)
        # fusion_score = self.SE(fr_feature).view(-1)
        # out = fusion_score[0]*out+fusion_score[1]*residual
        # in_channels = fr_feature.shape[1]
        
        # fusion_score = self.SE(fr_feature)
        fusion_score = self.eca(fr_feature)

        out = fusion_score[:,0,:,:].unsqueeze(-1)*out+fusion_score[:,1,:,:].unsqueeze(-1)*residual
        out = self.relu(out)



                                     # nn.Conv2d(num_inchannels[j], num_outchannels_conv3x3, kernel_size=3, stride=1, padding=1, bias=False),
                                    # space_to_depth(),##还可加无干扰的、附加的、有用的信息
                                    # nn.BatchNorm2d(4*num_outchannels_conv3x3),
                                    # nn.ReLU(inplace=True),
                                    # nn.Conv2d(4*num_outchannels_conv3x3, num_outchannels_conv3x3, kernel_size=3, padding=1, bias=False),##还可改为更加尽可能保留信息的维度变换操作！！！！！！！！
                                    # nn.BatchNorm2d(num_outchannels_conv3x3)


        return out


class spd_conv_block_begin(nn.Module):
    # Changing the dimension of the Tensor
    def __init__(self, inchannels, outchannels):
        super(spd_conv_block_begin, self).__init__()
        self.conv1 = nn.Conv2d(inchannels, outchannels, kernel_size=3, stride=1, padding=1, bias=False)        # add
        self.spd = space_to_depth()
        self.bn1 = nn.BatchNorm2d(4*outchannels, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(4*outchannels, 2*outchannels, kernel_size=3, padding=1, bias=False)
        self.conv2_2 = nn.Conv2d(2*outchannels, outchannels, kernel_size=3, padding=1, bias=False)
        self.conv2_dila =  nn.Conv2d(inchannels, outchannels, 3, 2, 2,dilation=2, bias=False)
        # self.conv3_dila =  nn.Conv2d(inchannels, outchannels, 3, 2, 3,dilation=3, bias=False)
        self.bn2 = nn.BatchNorm2d(2*outchannels, momentum=BN_MOMENTUM)
        self.relu_2 = nn.ReLU(inplace=True)

        self.conv2_2 = nn.Conv2d(2*outchannels, outchannels, kernel_size=3, padding=1, bias=False)
        self.bn2_2 = nn.BatchNorm2d(outchannels)
        self.relu_2_2 = nn.ReLU(inplace=True)

        self.conv3= nn.Conv2d(inchannels,outchannels,3, 2, 1, bias=False)                                    
        self.bn3 =  nn.BatchNorm2d(outchannels, momentum=BN_MOMENTUM)
        self.ECA =  eca_layer(outchannels)
        self.channel_layer = CAModule(4*outchannels, 16)
        self.spatial_layer = SAModule()

                                    

    def forward(self, x):
        residual = x
        # residual_2 = x

        out = self.conv1(x)
        out = self.spd(out)
        out = self.bn1(out)
        out = self.relu(out)
        # out, att_c = self.channel_layer(out)
        out = self.conv2(out)
        # out = self.conv2_2(out)
        # out = self.conv2_dila(out)
        out = self.bn2(out)
        out = self.relu_2(out)
        out = self.conv2_2(out)
        out = self.bn2_2(out)


        residual = self.conv2_dila(residual)
        # residual_2 = self.conv3_dila(residual_2)

        # fr_feature = torch.cat([out, residual], dim=1)
        # in_channels = fr_feature.shape[1]
        # self.SE = SELayer(in_channel=in_channels,reduction=16).cuda()
        # fusion_score = self.SE(fr_feature)
        # out = fusion_score[:,0,:,:].unsqueeze(-1)*out+fusion_score[:,1,:,:].unsqueeze(-1)*residual
        out =out +residual
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


class HighResolutionModule(nn.Module):
    def __init__(self, num_branches, blocks, num_blocks, num_inchannels,
                 num_channels, fuse_method, multi_scale_output=True):
        super(HighResolutionModule, self).__init__()
        self._check_branches(
            num_branches, blocks, num_blocks, num_inchannels, num_channels)

        self.num_inchannels = num_inchannels
        self.fuse_method = fuse_method
        self.num_branches = num_branches

        self.multi_scale_output = multi_scale_output

        self.branches = self._make_branches(
            num_branches, blocks, num_blocks, num_channels)
        self.fuse_layers = self._make_fuse_layers()
        self.relu = nn.ReLU(True)

    def _check_branches(self, num_branches, blocks, num_blocks,
                        num_inchannels, num_channels):
        if num_branches != len(num_blocks):
            error_msg = 'NUM_BRANCHES({}) <> NUM_BLOCKS({})'.format(
                num_branches, len(num_blocks))
            logger.error(error_msg)
            raise ValueError(error_msg)

        if num_branches != len(num_channels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_CHANNELS({})'.format(
                num_branches, len(num_channels))
            logger.error(error_msg)
            raise ValueError(error_msg)

        if num_branches != len(num_inchannels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_INCHANNELS({})'.format(
                num_branches, len(num_inchannels))
            logger.error(error_msg)
            raise ValueError(error_msg)

    def _make_one_branch(self, branch_index, block, num_blocks, num_channels,
                         stride=1):
        downsample = None
        if stride != 1 or \
           self.num_inchannels[branch_index] != num_channels[branch_index] * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.num_inchannels[branch_index],
                    num_channels[branch_index] * block.expansion,
                    kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(
                    num_channels[branch_index] * block.expansion,
                    momentum=BN_MOMENTUM
                ),
            )

        layers = []
        layers.append(
            block(
                self.num_inchannels[branch_index],
                num_channels[branch_index],
                stride,
                downsample
            )
        )
        self.num_inchannels[branch_index] = \
            num_channels[branch_index] * block.expansion
        for i in range(1, num_blocks[branch_index]):
            layers.append(
                block(
                    self.num_inchannels[branch_index],
                    num_channels[branch_index]
                )
            )

        return nn.Sequential(*layers)

    def _make_branches(self, num_branches, block, num_blocks, num_channels):
        branches = []

        for i in range(num_branches):
            branches.append(
                self._make_one_branch(i, block, num_blocks, num_channels)
            )

        return nn.ModuleList(branches)

    def _make_fuse_layers(self):
        if self.num_branches == 1:
            return None

        num_branches = self.num_branches
        num_inchannels = self.num_inchannels
        fuse_layers = []
        for i in range(num_branches if self.multi_scale_output else 1):
            fuse_layer = []
            for j in range(num_branches):
                if j > i:
                    fuse_layer.append(
                        nn.Sequential(
                            nn.Conv2d(
                                num_inchannels[j],
                                num_inchannels[i],
                                1, 1, 0, bias=False
                            ),
                            nn.BatchNorm2d(num_inchannels[i]),
                            nn.Upsample(scale_factor=2**(j-i), mode='nearest')
                        )
                    )
                elif j == i:
                    fuse_layer.append(None)
                else:
                    conv3x3s = []
                    for k in range(i-j):
                        if k == i - j - 1:
                            num_outchannels_conv3x3 = num_inchannels[i]
                            conv3x3s.append(
                                nn.Sequential(
                                    nn.Conv2d(
                                        num_inchannels[j],
                                        num_outchannels_conv3x3,
                                        3, 2, 1, bias=False
                                    ),
                                    nn.BatchNorm2d(num_outchannels_conv3x3)
                                )
                            )
                        else:
                            num_outchannels_conv3x3 = num_inchannels[j]
                            conv3x3s.append(
                                # nn.Sequential(
                                    # nn.Conv2d(
                                    #     num_inchannels[j],
                                    #     num_outchannels_conv3x3,
                                    #     3, 2, 1, bias=False
                                    # ),
                                    # nn.BatchNorm2d(num_outchannels_conv3x3),
                                    # nn.ReLU(True)

                                    spd_conv_block(num_inchannels[j], num_outchannels_conv3x3)
                                 #)
                            )
                    fuse_layer.append(nn.Sequential(*conv3x3s))
            fuse_layers.append(nn.ModuleList(fuse_layer))

        return nn.ModuleList(fuse_layers)

    def get_num_inchannels(self):
        return self.num_inchannels

    def forward(self, x):
        if self.num_branches == 1:
            return [self.branches[0](x[0])]

        for i in range(self.num_branches):
            x[i] = self.branches[i](x[i])

        x_fuse = []

        for i in range(len(self.fuse_layers)):
            y = x[0] if i == 0 else self.fuse_layers[i][0](x[0])
            for j in range(1, self.num_branches):
                if i == j:
                    y = y + x[j]
                else:
                    y = y + self.fuse_layers[i][j](x[j])
            x_fuse.append(self.relu(y))

        return x_fuse


blocks_dict = {
    'BASIC': BasicBlock,
    'BOTTLENECK': Bottleneck
}


class PoseHighResolutionNet(nn.Module):
    def __init__(self, cfg, **kwargs):
        self.inplanes = 64
        extra = cfg['MODEL']['EXTRA']
        super(PoseHighResolutionNet, self).__init__()
        self.coord_representation = cfg.MODEL.COORD_REPRESENTATION
        assert  cfg.MODEL.COORD_REPRESENTATION in ['simdr', 'sa-simdr', 'heatmap'], 'only simdr and sa-simdr and heatmap supported for pose_resnet_upfree'

        # stem net
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.layers_spd_1 = [
                # nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
                # space_to_depth(),
                # nn.BatchNorm2d(4*64,momentum=BN_MOMENTUM),
                # nn.ReLU(inplace=True),
                # nn.Conv2d(4*64, 64 , kernel_size=3, padding=1, bias=False),
                # nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
                spd_conv_block_begin(3,64)
            ]
        self.spd_conv_1 =torch.nn.Sequential(*self.layers_spd_1)

        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)

        self.layers_spd_2 = [
                # nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
                # space_to_depth(),
                # nn.BatchNorm2d(4*64,momentum=BN_MOMENTUM),
                # nn.ReLU(inplace=True),
                # nn.Conv2d(4*64, 64 , kernel_size=3, padding=1, bias=False),
                # nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
                spd_conv_block_begin(64,64)
            ]   
        self.spd_conv_2 =torch.nn.Sequential(*self.layers_spd_2)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(Bottleneck, 64, 4)

        self.stage2_cfg = extra['STAGE2']
        num_channels = self.stage2_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage2_cfg['BLOCK']]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))
        ]
        self.transition1 = self._make_transition_layer([256], num_channels)
        self.stage2, pre_stage_channels = self._make_stage(
            self.stage2_cfg, num_channels)

        self.stage3_cfg = extra['STAGE3']
        num_channels = self.stage3_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage3_cfg['BLOCK']]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))
        ]
        self.transition2 = self._make_transition_layer(
            pre_stage_channels, num_channels)
        self.stage3, pre_stage_channels = self._make_stage(
            self.stage3_cfg, num_channels)

        self.stage4_cfg = extra['STAGE4']
        num_channels = self.stage4_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage4_cfg['BLOCK']]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))
        ]
        self.transition3 = self._make_transition_layer(
            pre_stage_channels, num_channels)
        self.stage4, pre_stage_channels = self._make_stage(
            self.stage4_cfg, num_channels, multi_scale_output=False)

        self.final_layer = nn.Conv2d(
            in_channels=pre_stage_channels[0],
            out_channels=cfg['MODEL']['NUM_JOINTS'],
            kernel_size=extra['FINAL_CONV_KERNEL'],
            stride=1,
            padding=1 if extra['FINAL_CONV_KERNEL'] == 3 else 0
        )

        self.pretrained_layers = extra['PRETRAINED_LAYERS']
        
        # head
        if self.coord_representation == 'simdr' or self.coord_representation == 'sa-simdr':
            self.mlp_head_x = nn.Linear(cfg.MODEL.HEAD_INPUT, int(cfg.MODEL.IMAGE_SIZE[0]*cfg.MODEL.SIMDR_SPLIT_RATIO))
            self.mlp_head_y = nn.Linear(cfg.MODEL.HEAD_INPUT, int(cfg.MODEL.IMAGE_SIZE[1]*cfg.MODEL.SIMDR_SPLIT_RATIO))


    def _make_transition_layer(
            self, num_channels_pre_layer, num_channels_cur_layer):
        num_branches_cur = len(num_channels_cur_layer)
        num_branches_pre = len(num_channels_pre_layer)

        transition_layers = []
        for i in range(num_branches_cur):
            if i < num_branches_pre:
                if num_channels_cur_layer[i] != num_channels_pre_layer[i]:
                    transition_layers.append(
                        nn.Sequential(
                            nn.Conv2d(
                                num_channels_pre_layer[i],
                                num_channels_cur_layer[i],
                                3, 1, 1, bias=False
                            ),
                            nn.BatchNorm2d(num_channels_cur_layer[i]),
                            nn.ReLU(inplace=True)
                        )
                    )
                else:
                    transition_layers.append(None)
            else:
                conv3x3s = []
                for j in range(i+1-num_branches_pre):
                    inchannels = num_channels_pre_layer[-1]
                    outchannels = num_channels_cur_layer[i] \
                        if j == i-num_branches_pre else inchannels
                    conv3x3s.append(
                        # nn.Sequential(
                            # nn.Conv2d(
                            #     inchannels, outchannels, 3, 2, 1, bias=False
                            # ),
                            # nn.BatchNorm2d(outchannels),
                            # nn.ReLU(inplace=True)


                            spd_conv_block(inchannels, outchannels)
                        # )
                    )
                transition_layers.append(nn.Sequential(*conv3x3s))

        return nn.ModuleList(transition_layers)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes, planes * block.expansion,
                    kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _make_stage(self, layer_config, num_inchannels,
                    multi_scale_output=True):
        num_modules = layer_config['NUM_MODULES']
        num_branches = layer_config['NUM_BRANCHES']
        num_blocks = layer_config['NUM_BLOCKS']
        num_channels = layer_config['NUM_CHANNELS']
        block = blocks_dict[layer_config['BLOCK']]
        fuse_method = layer_config['FUSE_METHOD']

        modules = []
        for i in range(num_modules):
            # multi_scale_output is only used last module
            if not multi_scale_output and i == num_modules - 1:
                reset_multi_scale_output = False
            else:
                reset_multi_scale_output = True

            modules.append(
                HighResolutionModule(
                    num_branches,
                    block,
                    num_blocks,
                    num_inchannels,
                    num_channels,
                    fuse_method,
                    reset_multi_scale_output
                )
            )
            num_inchannels = modules[-1].get_num_inchannels()

        return nn.Sequential(*modules), num_inchannels

    def forward(self, x):
        # x = self.conv1(x)
        # x = self.bn1(x)
        # x = self.relu(x)      delete
        x = self.spd_conv_1(x)  # add  
        # x = self.conv2(x)
        # x = self.bn2(x)
        # x = self.relu(x)     # delete
        x = self.spd_conv_2(x)   # add
        x = self.layer1(x)

        x_list = []
        for i in range(self.stage2_cfg['NUM_BRANCHES']):
            if self.transition1[i] is not None:
                x_list.append(self.transition1[i](x))
            else:
                x_list.append(x)
        y_list = self.stage2(x_list)

        x_list = []
        for i in range(self.stage3_cfg['NUM_BRANCHES']):
            if self.transition2[i] is not None:
                x_list.append(self.transition2[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage3(x_list)

        x_list = []
        for i in range(self.stage4_cfg['NUM_BRANCHES']):
            if self.transition3[i] is not None:
                x_list.append(self.transition3[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage4(x_list)

        x_ = self.final_layer(y_list[0])

        if self.coord_representation == 'heatmap':
            return x_
        elif self.coord_representation == 'simdr' or self.coord_representation == 'sa-simdr':
            x = rearrange(x_, 'b c h w -> b c (h w)')
            pred_x = self.mlp_head_x(x)
            pred_y = self.mlp_head_y(x)
            return pred_x, pred_y

    def init_weights(self, pretrained=''):
        logger.info('=> init weights from normal distribution')
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
                for name, _ in m.named_parameters():
                    if name in ['bias']:
                        nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, std=0.001)
                for name, _ in m.named_parameters():
                    if name in ['bias']:
                        nn.init.constant_(m.bias, 0)

        if os.path.isfile(pretrained):
            pretrained_state_dict = torch.load(pretrained)
            logger.info('=> loading pretrained model {}'.format(pretrained))

            need_init_state_dict = {}
            for name, m in pretrained_state_dict.items():
                if name.split('.')[0] in self.pretrained_layers \
                   or self.pretrained_layers[0] is '*':
                    need_init_state_dict[name] = m
            self.load_state_dict(need_init_state_dict, strict=False)
        elif pretrained:
            logger.error('=> please download pre-trained models first!')
            raise ValueError('{} is not exist!'.format(pretrained))


def get_pose_net(cfg, is_train, **kwargs):
    model = PoseHighResolutionNet(cfg, **kwargs)

    if is_train and cfg['MODEL']['INIT_WEIGHTS']:
        model.init_weights(cfg['MODEL']['PRETRAINED'])

    return model
