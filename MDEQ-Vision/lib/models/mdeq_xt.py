from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import os
import sys
import logging
import functools
from termcolor import colored

from collections import OrderedDict

import numpy as np

import torch
import torch.nn as nn
import torch._utils
import torch.nn.functional as F

sys.path.append("lib/models")
from mdeq_core_xt import MDEQDiffNet

sys.path.append("../")
from lib.layer_utils import conv3x3

BN_MOMENTUM = 0.1
logger = logging.getLogger(__name__)


def nonlinearity(x):
    # swish
    return x*torch.sigmoid(x)

def get_timestep_embedding(timesteps, embedding_dim):
    """
    This matches the implementation in Denoising Diffusion Probabilistic Models:
    From Fairseq.
    Build sinusoidal embeddings.
    This matches the implementation in tensor2tensor, but differs slightly
    from the description in Section 3.5 of "Attention Is All You Need".
    """
    assert len(timesteps.shape) == 1

    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
    emb = emb.to(device=timesteps.device)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    return emb

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        """
        A bottleneck block with receptive field only 3x3. (This is not used in MDEQ; only
        in the classifier layer).
        """
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM, affine=False)
        self.conv2 = conv3x3(planes, planes, stride=stride)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM, affine=False)
        self.conv3 = nn.Conv2d(planes, planes*self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes*self.expansion, momentum=BN_MOMENTUM, affine=False)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x, injection=None):
        if injection is None:
            injection = 0
        residual = x

        out = self.conv1(x) + injection
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

# Replace all batch norm with group norm?
class MDEQDiffusionNet(MDEQDiffNet):
    def __init__(self, cfg, **kwargs):
        """
        Build an MDEQ Segmentation model with the given hyperparameters
        """
        global BN_MOMENTUM

        self.ch = cfg.DIFFUSION_MODEL.CHANNELS

        super(MDEQDiffusionNet, self).__init__(cfg, BN_MOMENTUM=BN_MOMENTUM, **kwargs)
        self.head_channels = cfg['MODEL']['EXTRA']['FULL_STAGE']['HEAD_CHANNELS']
        self.final_chansize = cfg['MODEL']['EXTRA']['FULL_STAGE']['FINAL_CHANSIZE']
        self.out_chansize = cfg['DIFFUSION_MODEL']['OUT_CHANNELS']
        # timestep embedding
        self.temb = nn.Module()
        self.temb.dense = nn.ModuleList([
            torch.nn.Linear(cfg.DIFFUSION_MODEL.CHANNELS,
                            cfg.DIFFUSION_MODEL.TEMB_CHANNELS),
            torch.nn.Linear(cfg.DIFFUSION_MODEL.TEMB_CHANNELS,
                            cfg.DIFFUSION_MODEL.TEMB_CHANNELS),
        ])

        # Classification Head
        # self.incre_modules, self.downsamp_modules, self.final_layer = self._make_head(self.num_channels)

        # Last layer
        # self.last_layer = nn.Conv2d(self.final_chansize, self.out_chansize, kernel_size=3, 
        #                                           stride=1, padding=1)
        
        last_inp_channels = np.int(np.sum(self.num_channels))
        self.last_layer = nn.Sequential(nn.Conv2d(last_inp_channels, last_inp_channels//2, kernel_size=1),
                                        nn.BatchNorm2d(last_inp_channels//2, momentum=BN_MOMENTUM),
                                        nn.ReLU(inplace=True),
                                        nn.Conv2d(last_inp_channels//2, self.out_chansize, kernel_size=3, 
                                                  stride=1, padding=1))


    # def _make_head(self, pre_stage_channels):
    #     """
    #     Create a final prediction head that:
    #        - Increase the number of features in each resolution 
    #        - Downsample higher-resolution equilibria to the lowest-resolution and concatenate
    #        - Pass through a final FC layer for classification
    #     """
    #     head_block = Bottleneck
    #     d_model = self.init_chansize
    #     head_channels = self.head_channels
        
    #     # Increasing the number of channels on each resolution when doing classification. 
    #     incre_modules = []
    #     for i, channels  in enumerate(pre_stage_channels):
    #         incre_module = self._make_layer(head_block, channels, head_channels[i], blocks=1, stride=1)
    #         incre_modules.append(incre_module)
    #     incre_modules = nn.ModuleList(incre_modules)
            
        # Downsample the high-resolution streams to perform classification
    #     downsamp_modules = []
    #     for i in range(len(pre_stage_channels)-1):
    #         in_channels = head_channels[i] * head_block.expansion
    #         out_channels = head_channels[i+1] * head_block.expansion
    #         downsamp_module = nn.Sequential(conv3x3(in_channels, out_channels, stride=2, bias=True),
    #                                         nn.BatchNorm2d(out_channels, momentum=BN_MOMENTUM),
    #                                         nn.ReLU(inplace=True))
    #         downsamp_modules.append(downsamp_module)
    #     downsamp_modules = nn.ModuleList(downsamp_modules)

    #     # Final FC layers
    #     final_layer = nn.Sequential(nn.Conv2d(head_channels[len(pre_stage_channels)-1] * head_block.expansion,
    #                                           self.final_chansize, kernel_size=1),
    #                                 nn.BatchNorm2d(self.final_chansize, momentum=BN_MOMENTUM),
    #                                 nn.ReLU(inplace=True))
    #     return incre_modules, downsamp_modules, final_layer

    # def _make_layer(self, block, inplanes, planes, blocks, stride=1, padding=0):
    #     downsample = None
    #     if stride != 1 or inplanes != planes * block.expansion:
    #         downsample = nn.Sequential(nn.Conv2d(inplanes, planes*block.expansion, kernel_size=1, stride=stride, bias=False, padding=padding),
    #                                    nn.BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM))

    #     layers = []
    #     layers.append(block(inplanes, planes, stride, downsample))
    #     inplanes = planes * block.expansion
    #     for i in range(1, blocks):
    #         layers.append(block(inplanes, planes))

    #     return nn.Sequential(*layers)


    # def predict_noise(self, y_list):
    #     """
    #     Combine all resolutions and output noise
    #     """
    #     import pdb; pdb.set_trace()
    #     y = self.incre_modules[0](y_list[0])
    #     for i in range(len(self.downsamp_modules)):
    #         y = self.incre_modules[i+1](y_list[i+1]) + self.downsamp_modules[i](y)
    #     y = torch.nn.functional.interpolate(
    #         y, scale_factor=2.0, mode="nearest")
    #     y = self.final_layer(y)
    #     y = self.last_layer(y)
    #     return y

    def predict_noise(self, y_list):
        """
        Combine all resolutions and output noise
        """
        y0_h, y0_w = y_list[0].size(2), y_list[0].size(3)
        all_res = [y_list[0]]
        for i in range(1, self.num_branches):
            all_res.append(F.interpolate(y_list[i], size=(y0_h, y0_w), mode='bilinear', align_corners=True))

        y = torch.cat(all_res, dim=1)
        all_res = None
        y = self.last_layer(y)

        # y = self.final_layer(y)
        # y = self.last_layer(y)
        return y

    def forward(self, x, t, train_step=0, **kwargs):
        # timestep embedding
        temb = get_timestep_embedding(t, self.ch)
        temb = self.temb.dense[0](temb)
        temb = nonlinearity(temb)
        temb = self.temb.dense[1](temb)

        output, jac_loss, sradius = self._forward(x, temb, train_step, **kwargs)
        return self.predict_noise(output), jac_loss, sradius
    
    def init_weights(self, pretrained=''):
        """
        Model initialization. If pretrained weights are specified, we load the weights.
        """
        logger.info(f'=> init weights from normal distribution. PRETRAINED={pretrained}')
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.01)
                if m.bias is not None:
                    m.bias.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm2d) and m.weight is not None:
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        if os.path.isfile(pretrained):
            pretrained_dict = torch.load(pretrained)
            logger.info('=> loading pretrained model {}'.format(pretrained))
            model_dict = self.state_dict()
            
            # Just verification...
            diff_modules = set()
            for k in pretrained_dict.keys():
                if k not in model_dict.keys():
                    diff_modules.add(k.split(".")[0])
            print(colored(f"In ImageNet MDEQ but not Cityscapes MDEQ: {sorted(list(diff_modules))}", "red"))
            diff_modules = set()
            for k in model_dict.keys():
                if k not in pretrained_dict.keys():
                    diff_modules.add(k.split(".")[0])
            print(colored(f"In Cityscapes MDEQ but not ImageNet MDEQ: {sorted(list(diff_modules))}", "green"))
            
            pretrained_dict = {k: v for k, v in pretrained_dict.items()
                               if k in model_dict.keys()}
            model_dict.update(pretrained_dict)
            self.load_state_dict(model_dict)

def get_diffusion_net(config, **kwargs):
    global BN_MOMENTUM
    BN_MOMENTUM = 0.01
    model = MDEQDiffusionNet(config, **kwargs)
    model.init_weights(config.MODEL.PRETRAINED)
    return model