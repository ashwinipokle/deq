from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import logging
import functools
from termcolor import colored

from collections import OrderedDict

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch._utils
import torch.nn.functional as F
import torch.autograd as autograd

sys.path.append("lib/")
from utils.utils import get_world_size, get_rank

sys.path.append("../")
from lib.optimizations import VariationalHidDropout2d, weight_norm
from lib.solvers import anderson, broyden
from lib.jacobian import jac_loss_estimate, power_method
from lib.layer_utils import list2vec, vec2list, norm_diff, conv3x3, conv5x5


BN_MOMENTUM = 0.1
BLOCK_GN_AFFINE = True    # Don't change the value here. The value is controlled by the yaml files.
FUSE_GN_AFFINE = True     # Don't change the value here. The value is controlled by the yaml files.
POST_GN_AFFINE = True     # Don't change the value here. The value is controlled by the yaml files.
DEQ_EXPAND = 5        # Don't change the value here. The value is controlled by the yaml files.
NUM_GROUPS = 4        # Don't change the value here. The value is controlled by the yaml files.
logger = logging.getLogger(__name__)

def nonlinearity(x):
    # swish
    return x*torch.sigmoid(x)

class SwishActivation(nn.Module):
    def __init__(self) -> None:
        super(SwishActivation, self).__init__()
    
    def forward(self, x):
        return nonlinearity(x)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, n_big_kernels=0, 
                        dropout=0.0, 
                        wnorm=False, 
                        temb_channels=512):
        """
        A canonical residual block with two 3x3 convolutions and an intermediate ReLU. Corresponds to Figure 2
        in the paper.
        """
        super(BasicBlock, self).__init__()
        conv1 = conv5x5 if n_big_kernels >= 1 else conv3x3
        conv2 = conv5x5 if n_big_kernels >= 2 else conv3x3
        conv3 = conv5x5 if n_big_kernels >= 2 else conv3x3
        conv4 = conv5x5 if n_big_kernels >= 2 else conv3x3

        inner_planes = int(DEQ_EXPAND*planes)

        self.conv1 = conv1(inplanes, inner_planes)
        self.gn1 = nn.GroupNorm(NUM_GROUPS, inner_planes, affine=BLOCK_GN_AFFINE)
        
        self.conv2 = conv2(inner_planes, planes)
        self.gn2 = nn.GroupNorm(NUM_GROUPS, planes, affine=BLOCK_GN_AFFINE)

        self.conv3 = conv3(planes, planes)
        self.gn3 = nn.GroupNorm(NUM_GROUPS, planes, affine=BLOCK_GN_AFFINE)

        self.conv4 = conv4(planes, planes)
        self.gn4 = nn.GroupNorm(NUM_GROUPS, planes, affine=BLOCK_GN_AFFINE)

        self.gn5 = nn.GroupNorm(NUM_GROUPS, planes, affine=BLOCK_GN_AFFINE)

        self.downsample = downsample
        self.drop = VariationalHidDropout2d(dropout)

        self.temb_proj = nn.Linear(temb_channels, planes)

        if wnorm: self._wnorm()
    
    def _wnorm(self):
        """
        Register weight normalization
        """
        self.conv1, self.conv1_fn = weight_norm(self.conv1, names=['weight'], dim=0)
        self.conv2, self.conv2_fn = weight_norm(self.conv2, names=['weight'], dim=0)
        self.conv3, self.conv3_fn = weight_norm(self.conv3, names=['weight'], dim=0)
        self.conv4, self.conv4_fn = weight_norm(self.conv4, names=['weight'], dim=0)
        self.temb_proj, self.temb_proj_fn = weight_norm(self.temb_proj, names=['weight'], dim=0)

    def _reset(self, bsz, d, H, W):
        """
        Reset dropout mask and recompute weight via weight normalization
        """
        if 'conv1_fn' in self.__dict__:
            self.conv1_fn.reset(self.conv1)
        if 'conv2_fn' in self.__dict__:
            self.conv2_fn.reset(self.conv2)
        if 'conv3_fn' in self.__dict__:
            self.conv3_fn.reset(self.conv3)
        if 'conv4_fn' in self.__dict__:
            self.conv4_fn.reset(self.conv4)
        if 'temb_proj_fn' in self.__dict__:
            self.temb_proj_fn.reset(self.temb_proj)

        self.drop.reset_mask(bsz, d, H, W)
            
    def forward(self, x, temb, injection=None):
        if injection is None: injection = 0
        residual = x

        out = nonlinearity(self.gn1(self.conv1(x)))
        out = self.drop(self.conv2(out)) + injection
        out = self.gn2(out)

        out = self.conv3(out) + self.temb_proj(nonlinearity(temb))[:, :, None, None]
        out = nonlinearity(self.gn3(out))
        out = self.conv4(out)
        
        if self.downsample is not None:
            residual = self.downsample(x)
            
        out += residual
        out = self.gn5(nonlinearity(self.gn4(out)))
        return out
    
       
def Normalize(in_channels):
    return torch.nn.GroupNorm(num_groups=16, num_channels=in_channels, eps=1e-6, affine=True)

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

    def _wnorm(self):
        """
        Register weight normalization
        """
        self.q, self.q_fn = weight_norm(self.q, names=['weight'], dim=0)
        self.k, self.k_fn = weight_norm(self.k, names=['weight'], dim=0)
        self.v, self.v_fn = weight_norm(self.v, names=['weight'], dim=0)
        self.proj_out, self.proj_out_fn = weight_norm(self.proj_out, names=['weight'], dim=0)

    def _reset(self):
        """
        Reset dropout mask and recompute weight via weight normalization
        """
        if 'q_fn' in self.__dict__:
            self.q_fn.reset(self.q)
        if 'k_fn' in self.__dict__:
            self.k_fn.reset(self.k)
        if 'v_fn' in self.__dict__:
            self.v_fn.reset(self.v)
        if 'proj_out_fn' in self.__dict__:
            self.proj_out_fn.reset(self.proj_out)

class BasicAttentionBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1, downsample=None, n_big_kernels=0, 
                        dropout=0.0, 
                        wnorm=False, 
                        temb_channels=512):
        super().__init__()
        self.basic = BasicBlock(inplanes, planes, stride=stride, downsample=downsample, 
                        n_big_kernels=n_big_kernels, 
                        dropout=dropout, 
                        wnorm=wnorm, 
                        temb_channels=temb_channels)
        
        self.attn = AttnBlock(in_channels=planes)

        if wnorm: self._wnorm()
    
    def forward(self, x, temb, injection=None):
        out = self.basic(x, temb=temb, injection=injection)
        out = self.attn(out)
        return out
    
    def _wnorm(self):
        """
        Register weight normalization
        """
        self.basic._wnorm()
        self.attn._wnorm()

    def _reset(self, bsz, d, H, W):
        """
        Reset dropout mask and recompute weight via weight normalization
        """
        self.basic._reset(bsz, d, H, W)
        self.attn._reset()


blocks_dict = { 'BASIC': BasicBlock, 'ATTN': BasicAttentionBlock}

class BranchNet(nn.Module):
    def __init__(self, blocks):
        """
        The residual block part of each resolution stream
        """
        super().__init__()
        self.blocks = blocks
    
    def forward(self, x, temb, injection=None):
        blocks = self.blocks
        y = blocks[0](x, temb, injection)
        for i in range(1, len(blocks)):
            y = blocks[i](y, temb)
        return y
    
class DownsampleModule(nn.Module):
    def __init__(self, num_channels, in_res, out_res):
        """
        A downsample step from resolution j (with in_res) to resolution i (with out_res). A series of 2-strided convolutions.
        """
        super(DownsampleModule, self).__init__()
        # downsample (in_res=j, out_res=i)
        convs = []
        inp_chan = num_channels[in_res]
        out_chan = num_channels[out_res]
        self.level_diff = level_diff = out_res - in_res
        
        kwargs = {"kernel_size": 3, "stride": 2, "padding": 1, "bias": False}
        for k in range(level_diff):
            intermediate_out = out_chan if k == (level_diff-1) else inp_chan
            components = [('conv', nn.Conv2d(inp_chan, intermediate_out, **kwargs)), 
                          ('gnorm', nn.GroupNorm(NUM_GROUPS, intermediate_out, affine=FUSE_GN_AFFINE))]
            if k != (level_diff-1):
                components.append(('activation', SwishActivation()))
            convs.append(nn.Sequential(OrderedDict(components)))
        self.net = nn.Sequential(*convs)  
            
    def forward(self, x):
        return self.net(x)

class UpsampleModule(nn.Module):
    def __init__(self, num_channels, in_res, out_res):
        """
        An upsample step from resolution j (with in_res) to resolution i (with out_res). 
        Simply a 1x1 convolution followed by an interpolation.
        """
        super(UpsampleModule, self).__init__()
        # upsample (in_res=j, out_res=i)
        inp_chan = num_channels[in_res]
        out_chan = num_channels[out_res]
        self.level_diff = level_diff = in_res - out_res
        self.net = nn.Sequential(OrderedDict([
                        ('conv', nn.Conv2d(inp_chan, out_chan, kernel_size=1, bias=False)),
                        ('gnorm', nn.GroupNorm(NUM_GROUPS, out_chan, affine=FUSE_GN_AFFINE)),
                        ('upsample', nn.Upsample(scale_factor=2**level_diff, mode='nearest'))]))
        
    def forward(self, x):
        return self.net(x)

    
class MDEQModule(nn.Module):
    def __init__(self, num_branches, blocks, num_blocks, num_channels, big_kernels, dropout=0.0):
        """
        An MDEQ layer (note that MDEQ only has one layer). 
        """
        super(MDEQModule, self).__init__()
        self._check_branches(
            num_branches, blocks, num_blocks, num_channels, big_kernels)

        self.num_branches = num_branches
        self.num_channels = num_channels
        self.big_kernels = big_kernels

        self.branches = self._make_branches(num_branches, blocks, num_blocks, num_channels, big_kernels, dropout=dropout)
        self.fuse_layers = self._make_fuse_layers()
        self.post_fuse_layers = nn.ModuleList([
            nn.Sequential(OrderedDict([
                ('activation', SwishActivation()),
                ('conv', nn.Conv2d(num_channels[i], num_channels[i], kernel_size=1, bias=False)),
                ('gnorm', nn.GroupNorm(NUM_GROUPS // 2, num_channels[i], affine=POST_GN_AFFINE))
            ])) for i in range(num_branches)])

    def _check_branches(self, num_branches, blocks, num_blocks, num_channels, big_kernels):
        """
        To check if the config file is consistent
        """
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
        
        if num_branches != len(big_kernels):
            error_msg = 'NUM_BRANCHES({}) <> BIG_KERNELS({})'.format(
                num_branches, len(big_kernels))
            logger.error(error_msg)
            raise ValueError(error_msg)
    
    def _wnorm(self):
        """
        Apply weight normalization to the learnable parameters of MDEQ
        """
        self.post_fuse_fns = []
        for i, branch in enumerate(self.branches):
            for block in branch.blocks:
                block._wnorm()
            conv, fn = weight_norm(self.post_fuse_layers[i].conv, names=['weight'], dim=0)
            self.post_fuse_fns.append(fn)
            self.post_fuse_layers[i].conv = conv
        
        # Throw away garbage
        torch.cuda.empty_cache()
        
    def _reset(self, xs):
        """
        Reset the dropout mask and the learnable parameters (if weight normalization is applied)
        """
        for i, branch in enumerate(self.branches):
            for block in branch.blocks:
                block._reset(*xs[i].shape)
            if 'post_fuse_fns' in self.__dict__:
                self.post_fuse_fns[i].reset(self.post_fuse_layers[i].conv)    # Re-compute (...).conv.weight using _g and _v

    def _make_one_branch(self, branch_index, block, num_blocks, num_channels, big_kernels, stride=1, dropout=0.0):
        """
        Make a specific branch indexed by `branch_index`. This branch contains `num_blocks` residual blocks of type `block`.
        """
        layers = nn.ModuleList()
        n_channel = num_channels[branch_index]
        n_big_kernels = big_kernels[branch_index]
        for i in range(num_blocks[branch_index]):
            layers.append(block(n_channel, n_channel, n_big_kernels=n_big_kernels, dropout=dropout))
        return BranchNet(layers)

    def _make_branches(self, num_branches, block, num_blocks, num_channels, big_kernels, dropout=0.0):
        """
        Make the residual block (s; default=1 block) of MDEQ's f_\theta layer. Specifically,
        it returns `branch_layers[i]` gives the module that operates on input from resolution i.
        """
        branch_layers = [self._make_one_branch(i, block, num_blocks, num_channels, big_kernels, dropout=dropout) for i in range(num_branches)]
        return nn.ModuleList(branch_layers)

    def _make_fuse_layers(self):
        """
        Create the multiscale fusion layer (which does simultaneous up- and downsamplings).
        """
        if self.num_branches == 1:
            return None

        num_branches = self.num_branches
        num_channels = self.num_channels
        fuse_layers = []
        for i in range(num_branches):
            fuse_layer = []                    # The fuse modules into branch #i
            for j in range(num_branches):
                if i == j:
                    fuse_layer.append(None)    # Identity if the same branch
                else:
                    module = UpsampleModule if j > i else DownsampleModule
                    fuse_layer.append(module(num_channels, in_res=j, out_res=i))
            fuse_layers.append(nn.ModuleList(fuse_layer))

        # fuse_layers[i][j] gives the (series of) conv3x3s that convert input from branch j to branch i
        return nn.ModuleList(fuse_layers)

    def get_num_inchannels(self):
        return self.num_channels

    # Here temb is temporal embedding
    def forward(self, x, temb, injection, *args):
        """
        The two steps of a multiscale DEQ module (see paper): a per-resolution residual block and 
        a parallel multiscale fusion step.
        """
        if injection is None:
            injection = [0] * len(x)
        if self.num_branches == 1:
            return [self.branches[0](x[0], temb, injection[0])]

        # Step 1: Per-resolution residual block
        x_block = []
        for i in range(self.num_branches):
            x_block.append(self.branches[i](x[i], temb, injection[i]))
        
        # Step 2: Multiscale fusion
        x_fuse = []
        for i in range(self.num_branches):
            y = 0
            # Start fusing all #j -> #i up/down-samplings
            for j in range(self.num_branches):
                y += x_block[j] if i == j else self.fuse_layers[i][j](x_block[j])
            x_fuse.append(self.post_fuse_layers[i](y))
        return x_fuse

class Stage0Block(nn.Module):
    def __init__(self, in_channels, out_channels=None, conv_shortcut=False,
                 dropout=0.1, temb_channels=512):
        super(Stage0Block, self).__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.stage0_0 = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                                                nn.BatchNorm2d(out_channels, momentum=BN_MOMENTUM, affine=True))

        # self.temb_proj = torch.nn.Linear(temb_channels, out_channels)

        # self.stage0_1 = nn.Sequential(nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False),
        #                             nn.BatchNorm2d(out_channels, momentum=BN_MOMENTUM, affine=True),
        #                             nn.ReLU(inplace=True),
        #                             nn.Dropout(dropout))

    def forward(self, x):
        h = x
        h = nonlinearity(self.stage0_0(x))
        # h = h + self.temb_proj(nonlinearity(temb))[:, :, None, None]
        # h = self.stage0_1(h)
        return h #x+h

class MDEQDiffNet(nn.Module):

    def __init__(self, cfg, **kwargs):
        """
        Build an MDEQ model with the given hyperparameters

        Args:
            cfg ([config]): The configuration file (parsed from yaml) specifying the model settings
        """
        super(MDEQDiffNet, self).__init__()
        global BN_MOMENTUM
        BN_MOMENTUM = kwargs.get('BN_MOMENTUM', 0.1)
        self.parse_cfg(cfg)
        init_chansize = self.init_chansize

        self.downsample = nn.Sequential(
            conv3x3(3, init_chansize, stride=(2 if self.downsample_times >= 1 else 1)),
            nn.BatchNorm2d(init_chansize, momentum=BN_MOMENTUM, affine=True),
            SwishActivation(),
            conv3x3(init_chansize, init_chansize, stride=(2 if self.downsample_times >= 2 else 1)),
            nn.BatchNorm2d(init_chansize, momentum=BN_MOMENTUM, affine=True),
            SwishActivation())

        if self.downsample_times > 2:
            for i in range(3, self.downsample_times+1):
                self.downsample.add_module(f"DS{i}", conv3x3(init_chansize, init_chansize, stride=2))
                self.downsample.add_module(f"DS{i}-BN", nn.BatchNorm2d(init_chansize, momentum=BN_MOMENTUM, affine=True))
                self.downsample.add_module(f"DS{i}-ACTIVATION", SwishActivation())
        
        # PART I: Input injection module
        if self.downsample_times == 0 and self.num_branches <= 2:
            # We use the downsample module above as the injection transformation
            self.stage0 = None
        else:
            self.stage0 = Stage0Block(self.init_chansize, out_channels=self.init_chansize, dropout=0.1, temb_channels=512)
        
        # PART II: MDEQ's f_\theta layer
        self.fullstage = self._make_stage(self.fullstage_cfg, self.num_channels, dropout=self.dropout)
        self.alternative_mode = "abs" if self.stop_mode == "rel" else "rel"
        if self.wnorm:
            self.fullstage._wnorm()

        self.iodrop = VariationalHidDropout2d(0.0)
        self.hook = None
        self.hook_zm = None
        
    def parse_cfg(self, cfg):
        """
        Parse a configuration file
        """
        global DEQ_EXPAND, NUM_GROUPS, BLOCK_GN_AFFINE, FUSE_GN_AFFINE, POST_GN_AFFINE
        self.num_branches = cfg['MODEL']['EXTRA']['FULL_STAGE']['NUM_BRANCHES']
        self.num_channels = cfg['MODEL']['EXTRA']['FULL_STAGE']['NUM_CHANNELS']
        self.init_chansize = self.num_channels[0]
        self.num_layers = cfg['MODEL']['NUM_LAYERS']
        self.dropout = cfg['MODEL']['DROPOUT']
        self.wnorm = cfg['MODEL']['WNORM']
        self.num_classes = cfg['MODEL']['NUM_CLASSES']
        self.downsample_times = cfg['MODEL']['DOWNSAMPLE_TIMES']
        self.fullstage_cfg = cfg['MODEL']['EXTRA']['FULL_STAGE']   
        self.pretrain_steps = cfg['TRAIN']['PRETRAIN_STEPS']

        # DEQ related
        self.f_solver = eval(cfg['DEQ']['F_SOLVER'])
        self.b_solver = eval(cfg['DEQ']['B_SOLVER'])
        if self.b_solver is None:
            self.b_solver = self.f_solver
        self.f_thres = cfg['DEQ']['F_THRES']
        self.b_thres = cfg['DEQ']['B_THRES']
        self.stop_mode = cfg['DEQ']['STOP_MODE']
        
        # Update global variables
        DEQ_EXPAND = cfg['MODEL']['EXPANSION_FACTOR']
        NUM_GROUPS = cfg['MODEL']['NUM_GROUPS']
        BLOCK_GN_AFFINE = cfg['MODEL']['BLOCK_GN_AFFINE']
        FUSE_GN_AFFINE = cfg['MODEL']['FUSE_GN_AFFINE']
        POST_GN_AFFINE = cfg['MODEL']['POST_GN_AFFINE']
            
    def _make_stage(self, layer_config, num_channels, dropout=0.0):
        """
        Build an MDEQ block with the given hyperparameters
        """
        num_modules = layer_config['NUM_MODULES']
        num_branches = layer_config['NUM_BRANCHES']
        num_blocks = layer_config['NUM_BLOCKS']
        block_type = blocks_dict[layer_config['BLOCK']]
        big_kernels = layer_config['BIG_KERNELS']
        return MDEQModule(num_branches, block_type, num_blocks, num_channels, big_kernels, dropout=dropout)

    # Inputs are x and temporal embedding
    def _forward(self, x, temb, train_step=-1, compute_jac_loss=True, spectral_radius_mode=False, 
                    layer_loss=True,
                    writer=None, **kwargs):
        """
        The core MDEQ module. In the starting phase, we can (optionally) enter a shallow stacked f_\theta training mode
        to warm up the weights (specified by the self.pretrain_steps; see below)
        """
        num_branches = self.num_branches
        f_thres = kwargs.get('f_thres', self.f_thres)
        b_thres = kwargs.get('b_thres', self.b_thres)
        x = self.downsample(x)
        rank = get_rank()
        
        # assert self.stage0 is not None, "Temporal embeddings are not being used"
        # Inject only to the highest resolution...
        x_list = [self.stage0(x) if self.stage0 else x]
        for i in range(1, num_branches):
            bsz, _, H, W = x_list[-1].shape
            x_list.append(torch.zeros(bsz, self.num_channels[i], H//2, W//2).to(x))   # ... and the rest are all zeros
            
        z_list = [torch.zeros_like(elem) for elem in x_list]
        z1 = list2vec(z_list)
        cutoffs = [(elem.size(1), elem.size(2), elem.size(3)) for elem in z_list]
        func = lambda z: list2vec(self.fullstage(vec2list(z, cutoffs), temb, x_list))
        
        # For variational dropout mask resetting and weight normalization re-computations
        self.fullstage._reset(z_list)

        jac_loss = torch.tensor(0.0).to(x)
        sradius = torch.zeros(bsz, 1).to(x)
        deq_mode = (train_step < 0) or (train_step >= self.pretrain_steps)
        
        #import pdb; pdb.set_trace()
        # Multiscale Deep Equilibrium!
        if not deq_mode:
            for layer_ind in range(self.num_layers): 
                z1 = func(z1)
            new_z1 = z1

            if self.training:
                if compute_jac_loss:
                    z2 = z1.clone().detach().requires_grad_()
                    new_z2 = func(z2)
                    jac_loss = jac_loss_estimate(new_z2, z2)
        else:
            with torch.no_grad():
                result = self.f_solver(func, z1, threshold=f_thres, stop_mode=self.stop_mode,
                                         layer_loss=layer_loss, layer_idx= [15],
                                         name="forward")
                z1 = result['result']
                if layer_loss:
                    zm = result['zm'][0]
                if self.train_step % 5000 == 0:
                    print("Nstep ", result['nstep'], "rel_trace", min(result['rel_trace']), "abs_trace", min(result['abs_trace']))
            new_z1 = z1
            if layer_loss:
                new_zm = zm

            if (not self.training) and spectral_radius_mode:
                with torch.enable_grad():
                    new_z1 = func(z1.requires_grad_())
                _, sradius = power_method(new_z1, z1, n_iters=150)

            if self.training:
                new_z1 = func(z1.requires_grad_())
                if compute_jac_loss:
                    jac_loss = jac_loss_estimate(new_z1, z1)
                    
                def backward_hook(grad):
                    if self.hook is not None:
                        self.hook.remove()
                        torch.cuda.synchronize()
                        
                    result = self.b_solver(lambda y: autograd.grad(new_z1, z1, y, retain_graph=True)[0] + grad, torch.zeros_like(grad), 
                                          threshold=b_thres, stop_mode=self.stop_mode, name="backward")
                    return result['result']

                self.hook = new_z1.register_hook(backward_hook)

                def backward_hook_zm(grad):
                    if self.hook_zm is not None:
                        self.hook_zm.remove()
                        torch.cuda.synchronize()
                        
                    result = self.b_solver(lambda y: autograd.grad(new_zm, zm, y, retain_graph=True)[0] + grad, torch.zeros_like(grad), 
                                          threshold=b_thres, stop_mode=self.stop_mode, name="backward")
                    return result['result']
                
                self.hook_zm  = new_zm.register_hook(backward_hook_zm)

        y_list = self.iodrop(vec2list(new_z1, cutoffs))
        y_list_zm = self.iodrop(vec2list(new_zm, cutoffs))
        return y_list, jac_loss.view(1,-1), sradius.view(-1,1), y_list_zm
    
    def forward(self, x, train_step=-1, **kwargs):
        raise NotImplemented    # To be inherited & implemented by MDEQClsNet and MDEQSegNet (see mdeq.py)