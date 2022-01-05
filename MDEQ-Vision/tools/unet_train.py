# Modified based on the HRNet repo.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
from pathlib import PureWindowsPath
import pprint
import shutil
import sys
from models import ema

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter

import numpy as np

import time 
import _init_paths
import models
from config import config
from config import update_config
from core.diffusion_function import simple_train
from utils.modelsummary import get_model_summary
from utils.utils import get_optimizer
from utils.utils import save_checkpoint
from utils.utils import create_logger
from termcolor import colored

from models.ema import EMAHelper
from models.unet import UNetModel

from core.diffusion_function import get_beta_schedule

def parse_args():
    parser = argparse.ArgumentParser(description='Train classification network')
    
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)
    parser.add_argument('--modelDir',
                        help='model directory',
                        type=str,
                        default='')
    parser.add_argument('--logDir',
                        help='log directory',
                        type=str,
                        default='')
    parser.add_argument('--dataDir',
                        help='data directory',
                        type=str,
                        default='')
    parser.add_argument('--testModel',
                        help='testModel',
                        type=str,
                        default='')
    parser.add_argument('--percent',
                        help='percentage of training data to use',
                        type=float,
                        default=1.0)
    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()
    update_config(config, args)

    return args

def torch2hwcuint8(x, clip=False):
    if clip:
        x = torch.clamp(x, -1, 1)
    x = (x + 1.0) / 2.0
    return x

def main():
    torch.manual_seed(42)
    args = parse_args()
    print(colored("Setting default tensor type to cuda.FloatTensor", "cyan"))
    torch.multiprocessing.set_start_method('spawn')
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

    logger, final_output_dir, tb_log_dir = create_logger(
        config, args.cfg, 'train')

    logger.info(pprint.pformat(args))
    logger.info(pprint.pformat(config))

    # cudnn related setting
    cudnn.benchmark = config.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = config.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = config.CUDNN.ENABLED
    
    model = UNetModel(config).cuda()
    
    if config.TRAIN.MODEL_FILE:
        model.load_state_dict(torch.load(config.TRAIN.MODEL_FILE))
        logger.info(colored('=> loading model from {}'.format(config.TRAIN.MODEL_FILE), 'red'))

    # Currently not adding EMA
    # TODO: Add EMA if needed 

    if config.DIFFUSION_MODEL.EMA:
        ema_helper = EMAHelper(mu=config.DIFFUSION_MODEL.EMA_RATE)
        ema_helper.register(model)
    else:
        ema_helper = None

    # copy model file
    this_dir = os.path.dirname(__file__)
    models_dst_dir = os.path.join(final_output_dir, 'models')
    if os.path.exists(models_dst_dir):
        shutil.rmtree(models_dst_dir)
    shutil.copytree(os.path.join(this_dir, '../lib/models'), models_dst_dir)

    writer_dict = {
        'writer': SummaryWriter(log_dir=tb_log_dir) if not config.DEBUG.DEBUG else None,
        'train_global_steps': 0,
        'valid_global_steps': 0,
    }

    gpus = list(config.GPUS)
    model = nn.DataParallel(model, device_ids=gpus).cuda()
    print("Finished constructing model!")

    # Data loading code
    dataset_name = config.DATASET.DATASET

    if dataset_name == 'cifar10':
        assert dataset_name == "cifar10", "Only CIFAR-10 and ImageNet are supported at this phase"

        augment_list = [transforms.Resize(32), transforms.RandomHorizontalFlip(p=0.5)]
        transform_train = transforms.Compose(augment_list + [
            transforms.ToTensor(),
        ])
        transform_valid = transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
        ])
        train_dataset = datasets.CIFAR10(root=f'{config.DATASET.ROOT}', train=True, download=True, transform=transform_train)
        valid_dataset = datasets.CIFAR10(root=f'{config.DATASET.ROOT}', train=False, download=True, transform=transform_valid)
    else:
        assert dataset_name == "cifar10", "Only CIFAR-10 is supported at this phase"

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.TRAIN.BATCH_SIZE_PER_GPU*len(gpus),
        shuffle=True,
        num_workers=config.WORKERS,
        pin_memory=True,
        generator=torch.Generator(device='cuda')
    )
    
    # define optimizer
    optimizer = get_optimizer(config, model)
    lr_scheduler = None

    epoch_iters = np.int(train_dataset.__len__() / config.TRAIN.BATCH_SIZE_PER_GPU / len(gpus))
    end_epoch = config.TRAIN.END_EPOCH + config.TRAIN.EXTRA_EPOCH
    num_iters = config.TRAIN.END_EPOCH * epoch_iters
    extra_iters = config.TRAIN.EXTRA_EPOCH * epoch_iters
    step = 0
    last_epoch = config.TRAIN.BEGIN_EPOCH

    if config.TRAIN.RESUME:
        model_state_file = os.path.join(final_output_dir, 'checkpoint_5474.pth.tar')
        if os.path.isfile(model_state_file):
            checkpoint = torch.load(model_state_file)
            last_epoch = checkpoint['epoch']
            step = checkpoint['step']
            model.module.load_state_dict(checkpoint['state_dict'])
            
            # Update weight decay if needed
            checkpoint['optimizer']['param_groups'][0]['weight_decay'] = config.TRAIN.WD
            optimizer.load_state_dict(checkpoint['optimizer'])
            
            writer_dict['train_global_steps'] = checkpoint['train_global_steps']
            #writer_dict['valid_global_steps'] = [checkpoint'valid_global_steps']
            ema_helper.load_state_dict(checkpoint["ema_state_dict"])

            if 'lr_scheduler' in checkpoint:
                lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_iters, 
                                  eta_min=1e-6)
                lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            logger.info("=> loaded checkpoint (epoch {})".format(checkpoint['epoch']))
   
    # Learning rate scheduler
    if lr_scheduler is None:
        if config.TRAIN.LR_SCHEDULER != 'step':
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, len(train_loader)*config.TRAIN.END_EPOCH, eta_min=1e-6)
        elif isinstance(config.TRAIN.LR_STEP, list):
            lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer, config.TRAIN.LR_STEP, config.TRAIN.LR_FACTOR,
                last_epoch-1)
        else:
            lr_scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, config.TRAIN.LR_STEP, config.TRAIN.LR_FACTOR,
                last_epoch-1)
    
    # Get alpha and beta params
    betas = get_beta_schedule(
        beta_schedule=config.DIFFUSION.BETA_SCHEDULE,
        beta_start=config.DIFFUSION.BETA_START,
        beta_end=config.DIFFUSION.BETA_END,
        num_diffusion_timesteps=config.DIFFUSION.NUM_DIFFUSIN_TIMESTEPS,
    )
    betas = torch.from_numpy(betas).float().cuda()
    num_timesteps = betas.shape[0]

    # Training code
    for epoch in range(last_epoch, config.TRAIN.END_EPOCH):

        if config.TRAIN.LR_SCHEDULER == 'step':
            lr_scheduler.step()
            
        # train for one epoch
        step = simple_train(config, betas=betas, num_timesteps=num_timesteps, 
                        epoch=epoch,
                        num_epoch=config.TRAIN.END_EPOCH,
                        epoch_iters=epoch_iters,
                        base_lr=config.TRAIN.LR,
                        num_iters=num_iters,
                        trainloader=train_loader, 
                        model=model,  
                        optimizer=optimizer, 
                        lr_scheduler=lr_scheduler,
                        writer_dict=writer_dict,
                        step=step,
                        ema_helper=ema_helper)
              
        torch.cuda.empty_cache()
        if writer_dict['writer'] is not None:
            writer_dict['writer'].flush()

        print(f"Outside train {writer_dict['train_global_steps']} {step} {config.TRAIN.CHECKPOINT_FREQ} {step % config.TRAIN.CHECKPOINT_FREQ}")
        if epoch % config.TRAIN.CHECKPOINT_FREQ == 0:
            logger.info('=> saving checkpoint to {}'.format(final_output_dir))
            save_checkpoint(states={
                'epoch': epoch + 1,
                'step': step,
                'model': config.MODEL.NAME,
                'state_dict': model.module.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'train_global_steps': writer_dict['train_global_steps'],
                'ema_state_dict': ema_helper.state_dict() if ema_helper is not None else {}
            }, is_best=False, output_dir=final_output_dir, filename=f'checkpoint_{step}.pth.tar')

    final_model_state_file = os.path.join(final_output_dir, 'final_state.pth.tar')
    logger.info('saving final model state to {}'.format(final_model_state_file))
    torch.save(model.module.state_dict(), final_model_state_file)
    if writer_dict['writer'] is not None:
        writer_dict['writer'].close()


if __name__ == '__main__':
    main()
