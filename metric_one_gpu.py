# %%
import gc
import glob
import math
import os
import shutil
import subprocess
import sys
import time
import warnings
from collections import deque
from contextlib import nullcontext
from functools import partial
from typing import List, Optional, Tuple
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import GPUtil
import colorama
import numpy as np
import torch
from torch.autograd.profiler import record_function
from torch.utils.data import DataLoader
import itertools
import dist
from utils import arg_util, misc
from utils.data import build_dataset, pil_load
from utils.data_sampler import DistInfiniteBatchSampler
from models.vqvae import VQVAE
from utils.arg_util import Args
from PIL import Image
import matplotlib.pyplot as plt
import dist
import torch.distributed as tdist
from tqdm import tqdm
from torchvision.transforms import InterpolationMode, transforms
from utils.data_loader import DIV2KData

args = Args()
# args.data = "./data/df2k_ost/GT_sub"
# args.data = "./data/df2k_ost/GT_resized"
# args.data = "./data/val_"
# args.data = "./data/brats_256_t1_new/val"
args.data = "./data/brats_256_t1_2021_pair_4x"

out_path = "./metric.txt"

args.vocab_size = 1024
args.vocab_width = 32

maxtot = 20 # -1
repeat_num = 1

args.batch_size= 4
args.device = "cpu"

def normalize_01_into_pm1(x):  # normalize x from [0, 1] to [-1, 1] by (x*2) - 1
    return x.add(x).add_(-1)

val_aug = [
        transforms.ToTensor(), normalize_01_into_pm1,
    ]
val_aug = transforms.Compose(val_aug)
val_set = DIV2KData(data_dir=os.path.join(args.data,"val"), transform=val_aug,augment=False)  # todo: junfeng; only `train_set` required, no need to create a 'validation_set'
    

ld_val = DataLoader(
    dataset=val_set, num_workers=args.workers, pin_memory=True, batch_size=args.batch_size, shuffle=False,
)
del val_set

def cal_psnr(data, rec_B3HW):
    return psnr(data, rec_B3HW)
def cal_ssim(data,rec_B3HW):
    return ssim(data, rec_B3HW, multichannel=True, channel_axis  = 2)
def cal_mse(data,rec_B3HW):
    return np.mean((data - rec_B3HW) ** 2)

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12335'
    tdist.init_process_group("nccl", rank=rank, world_size=world_size)
setup(0,1)




# ckpt_indexs = list(range(190,200,10))
# ckpt_indexs = ckpt_indexs + ["last"]
# ckpt_paths = ["local_output/ckpt-{}.pth".format(ckpt_index) for ckpt_index in ckpt_indexs]

ckpt_paths = ["local_output/ckpt-last.pth"]



for vae_ckpt in ckpt_paths:
    vae = VQVAE(vocab_size=args.vocab_size, z_channels=args.vocab_width, ch=args.ch, 
                test_mode=True, share_quant_resi=args.share_quant_resi, v_patch_nums=args.patch_nums).to(args.device).eval()
    # print(torch.load(vae_ckpt, map_location='cpu')['trainer']['vae_wo_ddp'])
    vae.load_state_dict(torch.load(vae_ckpt, map_location='cpu')['trainer']['vae_wo_ddp'])
    # vae.load_state_dict(torch.load(vae_ckpt_init, map_location='cpu'))
    vae.eval()

    ld_train_expanded = itertools.chain(*[ld_val] * repeat_num)  # 重复五次
    
    _psnr, _ssim, _mse = [],[],[]

    for index, data in tqdm(enumerate(ld_train_expanded), total=len(ld_val) * repeat_num):
        if maxtot < index and maxtot > 0:
            break
        with torch.no_grad():
            # data #[-1,1]
            data = data.to(args.device)
            rec_B3HW, usage , Lq  = vae(data) 
            rec_B3HW = rec_B3HW.clamp(-1, 1)
            data =  (data.cpu().squeeze(0).numpy() + 1.0 ) * 255.0 / 2
            rec_B3HW =  (rec_B3HW.cpu().squeeze(0).numpy() + 1.0 ) * 255.0 / 2
            data = data.astype(np.uint8)
            rec_B3HW = rec_B3HW.astype(np.uint8)
            for i in range(data.shape[0]):
                _data = data[i].transpose(1,2,0)
                _rec_B3HW = rec_B3HW[i].transpose(1,2,0)
                
                _psnr.append(cal_psnr(_data, _rec_B3HW))
                _ssim.append(cal_ssim(_data, _rec_B3HW))
                _mse.append(cal_mse(_data, _rec_B3HW))
    
    with open(out_path, "a") as f:
        f.write(f"vae_ckpt:{vae_ckpt}\n")
        f.write(f"maxPSNR:{max(_psnr)}, minPSNR:{min(_psnr)}, meanPSNR:{np.mean(_psnr)}\n")
        f.write(f"maxSSIM:{max(_ssim)}, minSSIM:{min(_ssim)}, meanSSIM:{np.mean(_ssim)}\n")
        f.write(f"maxMSE :{max(_mse)} ,  minMSE :{min(_mse)}, meanMSE :{np.mean(_mse)}\n")


# %%
