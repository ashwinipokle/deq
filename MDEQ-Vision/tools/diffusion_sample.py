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
import glob
import tqdm 

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

import torchvision.utils as tvu
import wandb

import numpy as np

import time 
import _init_paths
import models
from config import config
from config import update_config
from core.diffusion_function import train
from utils.modelsummary import get_model_summary
from utils.utils import get_optimizer
from utils.utils import save_checkpoint
from utils.utils import create_logger
from termcolor import colored
from models.ema import EMAHelper

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
    # Additional arguments related to sampling images from mdeq
    parser.add_argument("--fid", action="store_true")
    parser.add_argument("--interpolation", action="store_true")
    parser.add_argument(
        "--resume_training", action="store_true", help="Whether to resume training"
    )
    parser.add_argument(
        "-i",
        "--image_folder",
        type=str,
        default="images",
        help="The folder name of samples",
    )
    parser.add_argument(
        "--ni",
        action="store_true",
        help="No interaction. Suitable for Slurm Job launcher",
    )
    parser.add_argument(
        "--sample_type",
        type=str,
        default="generalized",
        help="sampling approach (generalized or ddpm_noisy)",
    )
    parser.add_argument(
        "--skip_type",
        type=str,
        default="uniform",
        help="skip according to (uniform or quadratic)",
    )
    parser.add_argument(
        "--timesteps", type=int, default=1000, help="number of steps involved"
    )
    parser.add_argument(
        "--eta",
        type=float,
        default=0.0,
        help="eta used to control the variances of sigma",
    )
    parser.add_argument("--sequence", action="store_true")

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
   
    # Create folder to save images
    if not os.path.exists(args.image_folder):
        os.mkdir(args.image_folder) 
    model = eval('models.'+ config.MODEL.NAME+'.get_diffusion_net')(config).cuda()
    
    if config.TRAIN.MODEL_FILE:
        model.load_state_dict(torch.load(config.TRAIN.MODEL_FILE))
        logger.info(colored('=> loading model from {}'.format(config.TRAIN.MODEL_FILE), 'red'))

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

    model_state_file = os.path.join(final_output_dir, 'checkpoint_63011.pth.tar')
    #model_state_file = os.path.join(final_output_dir, 'final_state.pth.tar')
    if os.path.isfile(model_state_file):
        checkpoint = torch.load(model_state_file)
        model.module.load_state_dict(checkpoint['state_dict'])
        # EMA
        if config.DIFFUSION_MODEL.EMA:
            ema_helper = EMAHelper(mu=config.DIFFUSION_MODEL.EMA_RATE)
            ema_helper.register(model)
            ema_helper.load_state_dict(checkpoint['ema_state_dict'])
            ema_helper.ema(model)
        #model.module.load_state_dict(checkpoint)
        logger.info("=> loaded checkpoint {}".format(model_state_file))
    else:
        raise ValueError("Checkpoint not loaded!")
    sample(model, args, config)


def sample(model, args, config):
    model.eval()
    if args.fid:
        sample_fid(model, args, config)
    else:
        raise NotImplementedError("Sample procedeure not defined")

def sample_fid(model, args, config):
    img_id = len(glob.glob(f"{args.image_folder}/*"))
    print(f"starting from image {img_id}")
    total_n_samples = 50000
    n_rounds = (total_n_samples - img_id) // config.SAMPLING.BATCH_SIZE

    with torch.no_grad():
        for _ in tqdm.tqdm(
            range(n_rounds), desc="Generating image samples for FID evaluation."
        ):
            n = config.SAMPLING.BATCH_SIZE
            x = torch.randn(
                n,
                config.DATA.CHANNELS,
                config.DATA.IMAGE_SIZE,
                config.DATA.IMAGE_SIZE,
            ).cuda()

            x = sample_image(x, model, args, config)
            if type(x) == dict:
                x_transformed = {}
                for t in x.keys():
                    cur_img_idx = img_id
                    x_transformed[t] = inverse_data_transform(config, x[t])
                    for i in range(n):
                        tvu.save_image(
                            x_transformed[i], os.path.join(args.image_folder, str(t), "{cur_img_idx}.png")
                        )
                        cur_img_idx += 1
            else:
                x = inverse_data_transform(config, x)

                for i in range(n):
                    tvu.save_image(
                        x[i], os.path.join(args.image_folder, f"{img_id}.png")
                    )
                    img_id += 1

def inverse_data_transform(config, X):
    if config.DATA.IMAGE_MEAN:
        X = X + config.DATA.IMAGE_MEAN.to(X.device)[None, ...]

    if config.DATA.LOGIT_TRANSFORM:
        X = torch.sigmoid(X)
    elif config.DATA.RESCALED:
        X = (X + 1.0) / 2.0

    return torch.clamp(X, 0.0, 1.0)

def sample_image(x, model, args, config, last=True):
    # Get alpha and beta params
    betas = get_beta_schedule(
        beta_schedule=config.DIFFUSION.BETA_SCHEDULE,
        beta_start=config.DIFFUSION.BETA_START,
        beta_end=config.DIFFUSION.BETA_END,
        num_diffusion_timesteps=config.DIFFUSION.NUM_DIFFUSIN_TIMESTEPS,
    )
    betas = torch.from_numpy(betas).float().cuda()
    num_timesteps = betas.shape[0]

    try:
        skip = args.skip
    except Exception:
        skip = 1

    if args.sample_type == "generalized":
        if args.skip_type == "uniform":
            skip = num_timesteps // args.timesteps
            seq = range(0, num_timesteps, skip)
        else:
            raise NotImplementedError

        logger=None
        use_wandb = False
        if use_wandb:
            wandb.init( project="DDIM-9-15", 
                        name=f"DDIM-mdeq-temb-{len(seq)}",
                        reinit=True,
                        config=config)
            logger = wandb.log

        xs = generalized_steps(x, seq, model, betas, logger=logger, print_logs=False, eta=args.eta)
        x = xs
    else:
        raise NotImplementedError
    if last:
        x = x[0][-1]
    return x

def compute_alpha(beta, t):
    beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
    a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1, 1)
    return a

def generalized_steps(x, seq, model, b, logger=None, print_logs=False, **kwargs):
    with torch.no_grad():
        n = x.size(0)
        seq_next = [-1] + list(seq[:-1])
        x0_preds = []
        xs = [x]
        image_dim = x.shape
        # print(seq)
        # print(seq_next)
        for i, j in zip(reversed(seq), reversed(seq_next)):
            t = (torch.ones(n) * i).to(x.device)
            next_t = (torch.ones(n) * j).to(x.device)
            at = compute_alpha(b, t.long())
            at_next = compute_alpha(b, next_t.long())
            xt = xs[-1].to('cuda')
            et = model(xt, t)[0]
            x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()
            x0_preds.append(x0_t.to('cpu'))
            c1 = (
                kwargs.get("eta", 0) * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
            )
            c2 = ((1 - at_next) - c1 ** 2).sqrt()
            noise_t = torch.randn_like(x)
            xt_next = at_next.sqrt() * x0_t + c1 * noise_t + c2 * et
            xs.append(xt_next.to('cpu'))

            log_dict = {
                    "alpha at": torch.mean(at),
                    "alpha at_next": torch.mean(at_next),
                    "xt": torch.norm(xt.reshape(image_dim[0], -1), -1).mean(),
                    "xt_next": torch.norm(xt_next.reshape(image_dim[0], -1), -1).mean(),
                    "coeff x0": c1.squeeze().mean(),
                    "coeff et": c2.squeeze().mean(),
                    "prediction et": torch.norm(et.reshape(image_dim[0], -1), -1).mean(),
                    "noise": torch.norm(noise_t.reshape(image_dim[0], -1), -1).mean(),
                }
            
            if logger is not None:
                if i % 50 == 0 or i < 50:
                    log_dict["samples"] = [wandb.Image(xt_next[i]) for i in range(min(xt_next.shape[0], 10))]
                logger(log_dict)
            elif print_logs:
                print(i, j, log_dict)
    return xs, x0_preds

if __name__ == '__main__':
    main()
