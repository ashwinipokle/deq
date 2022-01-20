# Modified based on the HRNet repo.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import os
import time
import numpy as np
import sys
import random

import numpy as np
import numpy.ma as ma
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn import functional as F

from utils.utils import AverageMeter
from utils.utils import adjust_learning_rate
from utils.utils import get_world_size, get_rank

logger = logging.getLogger(__name__)

def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    def sigmoid(x):
        return 1 / (np.exp(-x) + 1)

    if beta_schedule == "quad":
        betas = (
            np.linspace(
                beta_start ** 0.5,
                beta_end ** 0.5,
                num_diffusion_timesteps,
                dtype=np.float64,
            )
            ** 2
        )
    elif beta_schedule == "linear":
        betas = np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "const":
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)

    elif beta_schedule == "jsd":  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1.0 / np.linspace(
            num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "sigmoid":
        betas = np.linspace(-6, 6, num_diffusion_timesteps)
        betas = sigmoid(betas) * (beta_end - beta_start) + beta_start

    elif beta_schedule == "geometric":
        ratio = 1 - beta_end
        betas = np.array([(ratio**n) for n in range(1, num_diffusion_timesteps+1)], dtype=np.float64)
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas

def logit_transform(image, lam=1e-6):
    image = lam + (1 - 2 * lam) * image
    return torch.log(image) - torch.log1p(-image)

def data_transform(config, X):
    if config.DATA.UNIFORM_DEQUANTIZATION:
        X = X / 256.0 * 255.0 + torch.rand_like(X) / 256.0
    if config.DATA.GAUSSIAN_DEQUANTIZATION:
        X = X + torch.randn_like(X) * 0.01

    if config.DATA.RESCALED:
        X = 2 * X - 1.0
    elif config.DATA.LOGIT_TRANSFORM:
        X = logit_transform(X)

    if config.DATA.IMAGE_MEAN:
        return X - config.DATA.IMAGE_MEAN.to(X.device)[None, ...]

    return X

def train(config, betas, num_timesteps, epoch, num_epoch, epoch_iters, base_lr, num_iters,
         trainloader, optimizer, lr_scheduler, model, writer_dict, step, ema_helper):
    
    # Training
    model.train()
    batch_time = AverageMeter()
    ave_loss = AverageMeter()
    ave_jac_loss = AverageMeter()
    tic = time.time()
    cur_iters = epoch*epoch_iters
    writer = writer_dict['writer']
    global_steps = writer_dict['train_global_steps']
    #print(f"Global steps {global_steps} Cur Iters {cur_iters} epoch {epoch} Epoch Iters {epoch_iters}")
    #assert global_steps == cur_iters, f"Step counter problem... fix this? {global_steps} {cur_iters}"
    update_freq = config.LOSS.JAC_INCREMENTAL

    # Distributed information
    rank = get_rank()
    world_size = get_world_size()

    for i_iter, batch in enumerate(trainloader):
        #print(f"Global steps {global_steps} Cur Iters {cur_iters} epoch {epoch} Epoch Iters {epoch_iters}")
        #import pdb; pdb.set_trace()
        x, labels = batch
        step += 1
        n = x.size(0)
        x = x.cuda()
        x = data_transform(config, x)
        e = torch.randn_like(x)

        t = torch.randint(
            low=0, high=num_timesteps, size=(n // 2 + 1,)
        ).cuda()
        t = torch.cat([t, num_timesteps - t - 1], dim=0)[:n]

        # compute output
        deq_steps = global_steps - config.TRAIN.PRETRAIN_STEPS
        if deq_steps < 0:
            factor = config.LOSS.PRETRAIN_JAC_LOSS_WEIGHT
        elif config.LOSS.JAC_STOP_EPOCH <= epoch:
            # If are above certain epoch, we may want to stop jacobian regularization training
            # (e.g., when the original loss is 0.01 and jac loss is 0.05, the jacobian regularization
            # will be dominating and hurt performance!)
            factor = 0
        else:
            factor = config.LOSS.JAC_LOSS_WEIGHT + 0.1 * (deq_steps // update_freq)
        compute_jac_loss = (np.random.uniform(0,1) < config.LOSS.JAC_LOSS_FREQ) and (factor > 0)
        delta_f_thres = random.randint(-config.DEQ.RAND_F_THRES_DELTA,1) if (config.DEQ.RAND_F_THRES_DELTA > 0 and compute_jac_loss) else 0
        f_thres = config.DEQ.F_THRES + delta_f_thres
        b_thres = config.DEQ.B_THRES

        a = (1-betas).cumprod(dim=0).index_select(0, t).view(-1, 1, 1, 1)
        xt = x * a.sqrt() + e * (1.0 - a).sqrt()

        output_zm = None
        if config.LOSS.USE_LAYER_LOSS:
            output, jac_loss, _, output_zm = model(xt, t.float(), train_step=global_steps, 
                                compute_jac_loss=compute_jac_loss,
                                f_thres=f_thres, b_thres=b_thres, writer=writer)
        else:
            output, jac_loss, _ = model(xt, t.float(), train_step=global_steps, 
                                       compute_jac_loss=compute_jac_loss,
                                       f_thres=f_thres, b_thres=b_thres, writer=writer)
        
        losses = (e - output).square().sum(dim=(1, 2, 3))
        if config.LOSS.USE_LAYER_LOSS and len(output_zm) > 0:
            for layer_output in output_zm:
                losses += config.LOSS.GAMMA * (layer_output - output).square().sum(dim=(1, 2, 3))
            
        loss = losses.mean(dim=0)
        jac_loss = jac_loss.mean()

        # compute gradient and do update step
        optimizer.zero_grad()
        
        if factor > 0:
            (loss + factor*jac_loss).backward()
        else:
            loss.backward()
        if config.TRAIN.CLIP > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.TRAIN.CLIP)
        optimizer.step()
        if config.TRAIN.LR_SCHEDULER == 'cosine':
            lr_scheduler.step()
            lr = optimizer.param_groups[0]['lr']
        else:
            # If LR scheduler is None
            lr = adjust_learning_rate(optimizer, base_lr, num_iters, i_iter+cur_iters)
        
        if config.DIFFUSION_MODEL.EMA:
            ema_helper.update(model)

        # update average loss
        ave_loss.update(loss.item(), x.size(0))
        if compute_jac_loss:
            ave_jac_loss.update(jac_loss.item(), x.size(0))

        # measure elapsed time (modeling + data + sync)
        batch_time.update(time.time() - tic)
        tic = time.time()

        if i_iter % config.PRINT_FREQ == 0 and rank == 0:
            print_loss = ave_loss.average() / world_size
            print_jac_loss = ave_jac_loss.average() / world_size
            msg = 'Epoch: [{}/{}] Iter:[{}/{}], Time: {:.2f}, ' \
                  'lr: {:.6f}, Loss: {:.6f}, Jac: {:.4f} ({:.4f})' .format(
                      epoch, num_epoch, i_iter, epoch_iters, 
                      batch_time.average(), lr, print_loss, print_jac_loss, factor)
            logging.info(msg)

        global_steps += 1
        writer_dict['train_global_steps'] = global_steps
        #print(f"Global steps {global_steps} Cur Iters {cur_iters} epoch {epoch} Epoch Iters {epoch_iters} cur_iter {i_iter}")

        if factor > 0 and global_steps > config.TRAIN.PRETRAIN_STEPS and deq_steps % update_freq == 0:
             logger.info(f'Note: Adding 0.1 to Jacobian regularization weight.')
    #print(writer_dict['train_global_steps'], global_steps, cur_iters, step)
    return step

