# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Ke Sun (sunk@mail.ustc.edu.cn)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging
import functools
from turtle import pd

import numpy as np

import torch
import torch.nn as nn
import torch._utils
import torch.nn.functional as F

from unet import get_timestep_embedding

BN_MOMENTUM = 0.1
ALIGN_CORNERS = None

logger = logging.getLogger(__name__)

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


NUM_GROUPS = 4
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, temb_channels=512, dropout=0.1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.gn1 = nn.GroupNorm(num_groups=NUM_GROUPS, num_channels=planes, eps=1e-6, affine=True)
        
        self.conv2 = conv3x3(planes, planes)
        self.gn2 = nn.GroupNorm(num_groups=NUM_GROUPS, num_channels=planes, eps=1e-6, affine=True)
        
        self.conv3 = conv3x3(planes, planes)
        self.gn3 = nn.GroupNorm(num_groups=NUM_GROUPS, num_channels=planes, eps=1e-6, affine=True)
        
        self.conv4 = conv3x3(planes, planes)
        self.gn4 = nn.GroupNorm(num_groups=NUM_GROUPS, num_channels=planes, eps=1e-6, affine=True)
        
        self.gn5 = nn.GroupNorm(num_groups=NUM_GROUPS, num_channels=planes, eps=1e-6, affine=True)
        
        self.downsample = downsample
        self.stride = stride
        self.temb_proj = nn.Linear(temb_channels, planes)
        
        self.silu1 = nn.SiLU()
        self.silu2 = nn.SiLU()
        self.silu3 = nn.SiLU()
        self.silu4 = nn.SiLU()
        self.silu5 = nn.SiLU()       
        
        self.drop1 = nn.Dropout2d(dropout)

    def forward(self, input):
        x = input[0]
        temb = input[1]

        residual = x

        out = self.silu1(self.gn1(self.conv1(x)))
        out = self.drop1(self.silu2(self.gn2(self.conv2(out))))

        out = self.conv3(out) + self.temb_proj(self.silu3(temb))[:, :, None, None]
        out = self.silu4(self.gn3(out))

        out = self.conv4(out)
        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual
        out = self.gn5(self.silu5(self.gn4(out)))
        return (out, temb)


class Bottleneck(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, temb_channels=512):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.gn1 = nn.GroupNorm(num_groups=NUM_GROUPS, num_channels=planes, eps=1e-6, affine=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.gn2 = nn.GroupNorm(num_groups=NUM_GROUPS, num_channels=planes, eps=1e-6, affine=True)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1,
                               bias=False)
        self.gn3 = nn.GroupNorm(num_groups=NUM_GROUPS, num_channels=planes * self.expansion, eps=1e-6, affine=True)
        self.downsample = downsample
        self.stride = stride

        self.temb_proj = nn.Linear(temb_channels, planes)

        self.silu1 = nn.SiLU()
        self.silu2 = nn.SiLU()
        self.silu3 = nn.SiLU()
        self.silu4 = nn.SiLU()

    def forward(self, input):
        x = input[0]
        temb = input[1]

        residual = x

        out = self.conv1(x)
        out = self.gn1(out)
        out = self.silu1(out)

        out = self.conv2(out) + self.temb_proj(self.silu2(temb))[:, :, None, None]
        out = self.gn2(out)
        out = self.silu3(out)

        out = self.conv3(out)
        out = self.gn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual
        out = self.silu4(out)

        return (out, temb)

       
def Normalize(in_channels):
    return torch.nn.GroupNorm(num_groups=NUM_GROUPS, num_channels=in_channels, eps=1e-6, affine=True)

class AttnBlock(nn.Module):    
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.q = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.k = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.v = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.proj_out = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b, c, h, w = q.shape
        q = q.reshape(b, c, h*w)
        q = q.permute(0, 2, 1)   # b,hw,c
        k = k.reshape(b, c, h*w)  # b,c,hw
        w_ = torch.bmm(q, k)     # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        w_ = w_ * (int(c)**(-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        v = v.reshape(b, c, h*w)
        w_ = w_.permute(0, 2, 1)   # b,hw,hw (first hw of k, second of q)
        # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
        h_ = torch.bmm(v, w_)
        h_ = h_.reshape(b, c, h, w)

        h_ = self.proj_out(h_)

        return x+h_

class BasicAttentionBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None,
                        temb_channels=512, use_attn=False):
        super().__init__()
        self.basic = BasicBlock(inplanes, planes, stride=stride, downsample=downsample,
                        temb_channels=temb_channels)
        self.use_attn= use_attn
        #print("Use attention", use_attn)
        if self.use_attn:
            #print("Appending attention block")
            self.attn = AttnBlock(in_channels=planes)
    
    def forward(self, input):
        temb = input[1]
        out = self.basic(input)[0]
        #import pdb; pdb.set_trace()
        if self.use_attn:
            out = self.attn(out)
        return (out, temb)

class HighResolutionModule(nn.Module):
    def __init__(self, num_branches, blocks, num_blocks, num_inchannels,
                 num_channels, fuse_method, multi_scale_output=True):
        super(HighResolutionModule, self).__init__()
        # Branches on which to use self attention blocks
        self.attn_branch_idx = [0,1,2]
        self._check_branches(
            num_branches, blocks, num_blocks, num_inchannels, num_channels)

        self.num_inchannels = num_inchannels
        self.fuse_method = fuse_method
        self.num_branches = num_branches

        self.multi_scale_output = multi_scale_output

        self.branches = self._make_branches(
            num_branches, blocks, num_blocks, num_channels)
        self.fuse_layers = self._make_fuse_layers()

        self.silu = nn.SiLU()

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
                nn.Conv2d(self.num_inchannels[branch_index],
                          num_channels[branch_index] * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(num_channels[branch_index] * block.expansion,
                            momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(self.num_inchannels[branch_index],
                            num_channels[branch_index], stride, downsample))
        self.num_inchannels[branch_index] = \
            num_channels[branch_index] * block.expansion
        for i in range(1, num_blocks[branch_index]):
            use_attn = False
            if branch_index in self.attn_branch_idx:
                use_attn = True
            #print(block, block == BasicAttentionBlock, use_attn, branch_index)
            if block == BasicAttentionBlock:
                layers.append(block(self.num_inchannels[branch_index],
                                num_channels[branch_index], use_attn=use_attn))
            else:
                layers.append(block(self.num_inchannels[branch_index],
                                num_channels[branch_index]))

        return nn.Sequential(*layers)

    def _make_branches(self, num_branches, block, num_blocks, num_channels):
        branches = []

        for i in range(num_branches):
            branches.append(
                self._make_one_branch(i, block, num_blocks, num_channels))

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
                    fuse_layer.append(nn.Sequential(
                        nn.Conv2d(num_inchannels[j],
                                  num_inchannels[i],
                                  1,
                                  1,
                                  0,
                                  bias=False),
                        nn.BatchNorm2d(num_inchannels[i], momentum=BN_MOMENTUM, affine=False)))
                elif j == i:
                    fuse_layer.append(None)
                else:
                    conv3x3s = []
                    for k in range(i-j):
                        if k == i - j - 1:
                            num_outchannels_conv3x3 = num_inchannels[i]
                            conv3x3s.append(nn.Sequential(
                                nn.Conv2d(num_inchannels[j],
                                          num_outchannels_conv3x3,
                                          3, 2, 1, bias=False),
                                nn.BatchNorm2d(num_outchannels_conv3x3, 
                                            momentum=BN_MOMENTUM, affine=False)))
                        else:
                            num_outchannels_conv3x3 = num_inchannels[j]
                            conv3x3s.append(nn.Sequential(
                                nn.Conv2d(num_inchannels[j],
                                          num_outchannels_conv3x3,
                                          3, 2, 1, bias=False),
                                nn.BatchNorm2d(num_outchannels_conv3x3,
                                            momentum=BN_MOMENTUM, affine=False),
                                nn.SiLU()))
                    fuse_layer.append(nn.Sequential(*conv3x3s))
            fuse_layers.append(nn.ModuleList(fuse_layer))

        return nn.ModuleList(fuse_layers)

    def get_num_inchannels(self):
        return self.num_inchannels

    def forward(self, input):
        x = input[0]
        temb = input[1]

        if self.num_branches == 1:
            return [self.branches[0]((x[0], temb))[0]]

        for i in range(self.num_branches):
            x[i] = self.branches[i]((x[i], temb))[0]

        x_fuse = []
        for i in range(len(self.fuse_layers)):
            y = x[0] if i == 0 else self.fuse_layers[i][0](x[0])
            for j in range(1, self.num_branches):
                if i == j:
                    y = y + x[j]
                elif j > i:
                    width_output = x[i].shape[-1]
                    height_output = x[i].shape[-2]
                    y = y + F.interpolate(
                        self.fuse_layers[i][j](x[j]),
                        size=[height_output, width_output],
                        mode='bilinear', align_corners=ALIGN_CORNERS)
                else:
                    y = y + self.fuse_layers[i][j](x[j])
            x_fuse.append(self.silu(y))

        return (x_fuse, temb)


blocks_dict = {
    'BASIC': BasicBlock,
    'BOTTLENECK': Bottleneck,
    'ATTN': BasicAttentionBlock,
}

class HighResolutionNetV3(nn.Module):

    def __init__(self, config, **kwargs):
        global ALIGN_CORNERS
        extra = config.MODEL.EXTRA
        super(HighResolutionNetV3, self).__init__()
        ALIGN_CORNERS = True

        # temporal embeddings
        self.temb_ch = config.DIFFUSION_MODEL.CHANNELS
        self.temb = nn.Module()
        self.temb.dense = nn.ModuleList([
            torch.nn.Linear(config.DIFFUSION_MODEL.CHANNELS,
                            config.DIFFUSION_MODEL.TEMB_CHANNELS),
            torch.nn.Linear(config.DIFFUSION_MODEL.TEMB_CHANNELS,
                            config.DIFFUSION_MODEL.TEMB_CHANNELS),
        ])

        # stem net
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.gn1 = nn.GroupNorm(num_groups=NUM_GROUPS, num_channels=64, eps=1e-6, affine=True)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.gn2 = nn.GroupNorm(num_groups=NUM_GROUPS, num_channels=64, eps=1e-6, affine=True)

        self.stage1_cfg = extra['STAGE1']
        num_channels = self.stage1_cfg['NUM_CHANNELS'][0]
        block = blocks_dict[self.stage1_cfg['BLOCK']]
        num_blocks = self.stage1_cfg['NUM_BLOCKS'][0]
        self.layer1 = self._make_layer(block, 64, num_channels, num_blocks)
        stage1_out_channel = block.expansion*num_channels

        self.stage2_cfg = extra['STAGE2']
        num_channels = self.stage2_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage2_cfg['BLOCK']]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))]
        self.transition1 = self._make_transition_layer(
            [stage1_out_channel], num_channels)
        self.stage2, pre_stage_channels = self._make_stage(
            self.stage2_cfg, num_channels)

        prev_num_channels = num_channels
        self.stage3_cfg = extra['STAGE3']
        num_channels = self.stage3_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage3_cfg['BLOCK']]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))]
        self.transition2 = self._make_transition_layer(
            pre_stage_channels, num_channels)
        
        concat_num_channels = [num_channels[i] + prev_num_channels[i] for i in range(min(len(prev_num_channels), len(num_channels)))]
        self.stage3, pre_stage_channels = self._make_stage(
            self.stage3_cfg, concat_num_channels)

        prev_num_channels = num_channels
        self.stage4_cfg = extra['STAGE4']
        num_channels = self.stage4_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage4_cfg['BLOCK']]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))]
        self.transition3 = self._make_transition_layer(
            pre_stage_channels, num_channels)
        num_channels = [num_channels[i] + prev_num_channels[i] for i in range(min(len(prev_num_channels), len(num_channels)))]
        self.stage4, pre_stage_channels = self._make_stage(
            self.stage4_cfg, num_channels, multi_scale_output=False)

        last_inp_channels = pre_stage_channels[0]

        self.silu1 = nn.SiLU()
        self.silu2 = nn.SiLU()
        self.silu3 = nn.SiLU()

        self.last_layer = nn.Sequential(
            nn.Conv2d(
                in_channels=last_inp_channels,
                out_channels=last_inp_channels//2,
                kernel_size=1,
                stride=1,
                padding=0),
            nn.GroupNorm(num_groups=NUM_GROUPS, num_channels=last_inp_channels//2, eps=1e-6, affine=True),
            nn.SiLU(),
            nn.Conv2d(
                in_channels=last_inp_channels//2,
                out_channels=3,
                kernel_size=extra.FINAL_CONV_KERNEL,
                stride=1,
                padding=1 if extra.FINAL_CONV_KERNEL == 3 else 0)
        )

    def _make_transition_layer(
            self, num_channels_pre_layer, num_channels_cur_layer):
        num_branches_cur = len(num_channels_cur_layer)
        num_branches_pre = len(num_channels_pre_layer)

        transition_layers = []
        for i in range(num_branches_cur):
            if i < num_branches_pre:
                if num_channels_cur_layer[i] != num_channels_pre_layer[i]:
                    transition_layers.append(nn.Sequential(
                        nn.Conv2d(num_channels_pre_layer[i],
                                  num_channels_cur_layer[i],
                                  3,
                                  1,
                                  1,
                                  bias=False),
                        nn.GroupNorm(num_groups=NUM_GROUPS, num_channels=num_channels_cur_layer[i], eps=1e-6, affine=True),
                        nn.SiLU()))
                else:
                    transition_layers.append(None)
            else:
                conv3x3s = []
                for j in range(i+1-num_branches_pre):
                    inchannels = num_channels_pre_layer[-1]
                    outchannels = num_channels_cur_layer[i] \
                        if j == i-num_branches_pre else inchannels
                    conv3x3s.append(nn.Sequential(
                        nn.Conv2d(
                            inchannels, outchannels, 3, 2, 1, bias=False),
                        nn.GroupNorm(num_groups=NUM_GROUPS, num_channels=outchannels, eps=1e-6, affine=True),
                        nn.SiLU()))
                transition_layers.append(nn.Sequential(*conv3x3s))

        return nn.ModuleList(transition_layers)

    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.GroupNorm(num_groups=NUM_GROUPS, num_channels=planes * block.expansion, eps=1e-6, affine=True)
            )

        layers = []
        layers.append(block(inplanes, planes, stride, downsample))
        inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(inplanes, planes))

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
                HighResolutionModule(num_branches,
                                      block,
                                      num_blocks,
                                      num_inchannels,
                                      num_channels,
                                      fuse_method,
                                      reset_multi_scale_output)
            )
            num_inchannels = modules[-1].get_num_inchannels()

        return nn.Sequential(*modules), num_inchannels

    def forward(self, x, t):
        # timestep embedding
        temb = get_timestep_embedding(t, self.temb_ch)
        temb = self.temb.dense[0](temb)
        temb = self.silu1(temb)
        temb = self.temb.dense[1](temb)

        x = self.conv1(x)
        x = self.gn1(x)
        x = self.silu2(x)
        x = self.conv2(x)
        x = self.gn2(x)
        x = self.silu3(x)
        x = self.layer1((x, temb))[0] # This returns a tuple of (x, temb)

        x_list1 = []
        for i in range(self.stage2_cfg['NUM_BRANCHES']):
            if self.transition1[i] is not None:
                x_list1.append(self.transition1[i](x))
            else:
                x_list1.append(x)
        
        y_list1 = self.stage2((x_list1, temb))[0] #This returns a tuple of (x, temb)

        x_list2 = []
        for i in range(self.stage3_cfg['NUM_BRANCHES']):
            if self.transition2[i] is not None:
                if i < self.stage2_cfg['NUM_BRANCHES']:
                    x_list2.append(self.transition2[i](y_list1[i]))
                else:
                    x_list2.append(self.transition2[i](y_list1[-1]))
            else:
                x_list2.append(y_list1[i])

        for i in range(min(len(x_list2), len(x_list1))):
            x_list2[i] = torch.cat([x_list2[i], x_list1[i]], dim=1)
        
        y_list2 = self.stage3((x_list2, temb))[0]
        x_list3 = []
        for i in range(self.stage4_cfg['NUM_BRANCHES']):
            if self.transition3[i] is not None:
                if i < self.stage3_cfg['NUM_BRANCHES']:
                    x_list3.append(self.transition3[i](y_list2[i]))
                else:
                    x_list3.append(self.transition3[i](y_list2[-1]))
            else:
                x_list3.append(y_list2[i])
        
        for i in range(min(len(x_list3), len(x_list2))):
            x_list3[i] = torch.cat([x_list3[i], x_list2[i]], dim=1)

        x = self.stage4((x_list3, temb))[0]
        x = self.last_layer(x[0])
        return x

def get_diffusion_net(cfg, **kwargs):
    model = HighResolutionNetV3(cfg, **kwargs)
    return model