
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
import pyiqa
from skimage import io

os.environ["HTTPS_PROXY"] = "http://127.0.0.1:7890"
os.environ["HTTP_PROXY"] = "http://127.0.0.1:7890"

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
psnr_metric = pyiqa.create_metric('psnr', device=device)
ssim_metric = pyiqa.create_metric('ssim', device=device)
# fid_metric = pyiqa.create_metric('fid', device=device)
# maniqa_metric = pyiqa.create_metric('maniqa', device=device)
# lpips_iqa_metric = pyiqa.create_metric('lpips', device=device)
# clipiqa_iqa_metric = pyiqa.create_metric('clipiqa', device=device)
# musiq_iqa_metric = pyiqa.create_metric('musiq', device=device)
# dists_iqa_metric = pyiqa.create_metric('dists', device=device)
# niqe_iqa_metric = pyiqa.create_metric('niqe', device=device)

def normalize_01_into_pm1(x):  # normalize x from [0, 1] to [-1, 1] by (x*2) - 1
    return x.add(x).add_(-1)
def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12335'
    tdist.init_process_group("nccl", rank=rank, world_size=world_size)
def img2tensor(img):
    img = (img / 255.).astype('float32')
    if img.ndim ==2:
        img = np.expand_dims(np.expand_dims(img, axis = 0),axis=0)
    else:
        img = np.transpose(img, (2, 0, 1))  # C, H, W
        img = np.expand_dims(img, axis=0)
    img = np.ascontiguousarray(img, dtype=np.float32)
    tensor = torch.from_numpy(img)
    return tensor
def rgb2ycbcr_pt(img, y_only=False):
    """Convert RGB images to YCbCr images (PyTorch version).
    It implements the ITU-R BT.601 conversion for standard-definition television. See more details in
    https://en.wikipedia.org/wiki/YCbCr#ITU-R_BT.601_conversion.
    Args:
        img (Tensor): Images with shape (n, 3, h, w), the range [0, 1], float, RGB format.
         y_only (bool): Whether to only return Y channel. Default: False.
    Returns:
        (Tensor): converted images with the shape (n, 3/1, h, w), the range [0, 1], float.
    """
    if y_only:
        weight = torch.tensor([[65.481], [128.553], [24.966]]).to(img)
        out_img = torch.matmul(img.permute(0, 2, 3, 1), weight).permute(0, 3, 1, 2) + 16.0
    else:
        weight = torch.tensor([[65.481, -37.797, 112.0], [128.553, -74.203, -93.786], [24.966, 112.0, -18.214]]).to(img)
        bias = torch.tensor([16, 128, 128]).view(1, 3, 1, 1).to(img)
        out_img = torch.matmul(img.permute(0, 2, 3, 1), weight).permute(0, 3, 1, 2) + bias

    out_img = out_img / 255.
    return out_img


def get_value(x):
    return x.item() if isinstance(x, torch.Tensor) else x
def write_metrics_to_file(filename, metric_name, values):
    mean_val = sum(values) / len(values)
    max_val = max(values)
    min_val = min(values)
    with open(filename, "a") as f:
        f.write(f"{metric_name}: Mean = {get_value(mean_val)}, Max = {get_value(max_val)}, Min = {get_value(min_val)}\n")
        
def get_img(args, ld_val, maxtot, ckpt_paths):
    for vae_ckpt in ckpt_paths:
        vae = VQVAE(vocab_size=args.vocab_size, z_channels=args.vocab_width, ch=args.ch, 
                    test_mode=True, share_quant_resi=args.share_quant_resi, v_patch_nums=args.patch_nums).to(args.device).eval()
        # print(torch.load(vae_ckpt, map_location='cpu')['trainer']['vae_wo_ddp'])
        
        load_ckpt = torch.load(vae_ckpt, map_location='cpu')
        if 'trainer' in  load_ckpt.keys():
            load_ckpt = load_ckpt['trainer']['vae_wo_ddp']
        vae.load_state_dict(load_ckpt)
        # vae.load_state_dict(torch.load(vae_ckpt_init, map_location='cpu'))
        vae.eval()
        
        out_dir = os.path.join("metric_results",os.path.basename(vae_ckpt))
        predict_dir = os.path.join(out_dir,"predict")
        gt_dir = os.path.join(out_dir,"gt")
        vaex_first_rec_dir = os.path.join(out_dir,"vaex_first_rec")
        
        if os.path.exists(out_dir):
            print(f"{vae_ckpt} exist, skip!!!")
            continue
        
        os.makedirs(predict_dir,exist_ok=True)
        os.makedirs(gt_dir,exist_ok=True)
        os.makedirs(vaex_first_rec_dir,exist_ok=True)
        
        for index, data in tqdm(enumerate(ld_val), total=len(ld_val)):
            if maxtot < index and maxtot > 0:
                break
            with torch.no_grad():
                # data #[-1,1]
                data = data.to(args.device)
                rec_B3HW, usage , Lq  = vae(data) 

                vaex_first_rec = vae.decoder(vae.post_quant_conv(vae.quant_conv(vae.encoder(data))))

                rec_B3HW = rec_B3HW.clamp(-1, 1)
                vaex_first_rec = vaex_first_rec.clamp(-1, 1)
                data =  (data.cpu().squeeze(0).numpy() + 1.0 ) * 255.0 / 2
                rec_B3HW =  (rec_B3HW.cpu().squeeze(0).numpy() + 1.0 ) * 255.0 / 2
                vaex_first_rec =  (vaex_first_rec.cpu().squeeze(0).numpy() + 1.0 ) * 255.0 / 2
                data = data.astype(np.uint8)
                rec_B3HW = rec_B3HW.astype(np.uint8)
                vaex_first_rec = vaex_first_rec.astype(np.uint8)
                for i in range(data.shape[0]):
                    _data = data[i].transpose(1,2,0)
                    _rec_B3HW = rec_B3HW[i].transpose(1,2,0)
                    _vaex_first_rec = vaex_first_rec[i].transpose(1,2,0)
                    Image.fromarray(_rec_B3HW).save(os.path.join(predict_dir,f"{index*data.shape[0]+i}.png"))
                    Image.fromarray(_data).save(os.path.join(gt_dir,f"{index*data.shape[0]+i}.png"))
                    Image.fromarray(_vaex_first_rec).save(os.path.join(vaex_first_rec_dir,f"{index*data.shape[0]+i}.png"))


def metric(metric_path,ckpt_paths):

    img_preproc = transforms.Compose([
        transforms.ToTensor(),
    ])

    
    for fold in ckpt_paths:
        fold = os.path.basename(fold)
        print(f"now {fold}")
        out_dir = os.path.join("metric_results",fold)
        predict_dir = os.path.join(out_dir,"predict")
        gt_dir = os.path.join(out_dir,"gt")
        vaex_first_rec_dir = os.path.join(out_dir,"vaex_first_rec")
        
        gt_img_paths = []

        psnr_folder = []
        ssim_folder = []
        psnr_first_rec = []
        ssim_first_rec = []
        lpips_score = []
        dists_score = []
        niqe_score = []
        lpips_iqa = []
        musiq_iqa = []
        maniqa_iqa = []
        clip_iqa = []
        gt_img_paths.extend(sorted(glob.glob(f'{gt_dir}/*.png'))[:])
        
        
        for gt_img_path in tqdm(gt_img_paths):
            GT_image = img_preproc(Image.open(gt_img_path).convert('RGB'))
            prediction_img_path = gt_img_path.replace("/gt/", "/predict/")
            vaex_first_rec_path = gt_img_path.replace("/gt/", "/vaex_first_rec/")

            img1 = rgb2ycbcr_pt(img2tensor(io.imread(gt_img_path)),  y_only=True).to(torch.float64)
            img2 = rgb2ycbcr_pt(img2tensor(io.imread(prediction_img_path)),  y_only=True).to(torch.float64)
            img3 = rgb2ycbcr_pt(img2tensor(io.imread(vaex_first_rec_path)),  y_only=True).to(torch.float64)
            img1 = torch.squeeze(img1)
            img2 = torch.squeeze(img2)
            img3 = torch.squeeze(img3)

            ssim_folder.append(ssim_metric(img1.unsqueeze(0).unsqueeze(0), img2.unsqueeze(0).unsqueeze(0)))
            psnr_folder.append(psnr_metric(img1.unsqueeze(0).unsqueeze(0), img2.unsqueeze(0).unsqueeze(0)))

            ssim_first_rec.append(ssim_metric(img1.unsqueeze(0).unsqueeze(0), img3.unsqueeze(0).unsqueeze(0)))
            psnr_first_rec.append(psnr_metric(img1.unsqueeze(0).unsqueeze(0), img3.unsqueeze(0).unsqueeze(0)))
            # lpips_iqa.append(lpips_iqa_metric(prediction_img_path, gt_img_path))
            # clip_iqa.append(clipiqa_iqa_metric(prediction_img_path))
            # musiq_iqa.append(musiq_iqa_metric(prediction_img_path))
            # maniqa_iqa.append(maniqa_metric(prediction_img_path))
            # dists_score.append(dists_iqa_metric(prediction_img_path, gt_img_path))
            # niqe_score.append(niqe_iqa_metric(prediction_img_path))

        with open(metric_path, "a") as f:
            f.write(f"fold = {fold}\n")
        write_metrics_to_file(metric_path, "PSNR", psnr_folder)
        write_metrics_to_file(metric_path, "SSIM", ssim_folder)

        write_metrics_to_file(metric_path, "PSNR_first_rec", psnr_first_rec)
        write_metrics_to_file(metric_path, "SSIM_first_rec", ssim_first_rec)
        # write_metrics_to_file(metric_path, "LPIPS", lpips_iqa)
        # write_metrics_to_file(metric_path, "DISTS", dists_score)
        # write_metrics_to_file(metric_path, "NIQE", niqe_score)
        # write_metrics_to_file(metric_path, "CLIP-IQA", clip_iqa)
        # write_metrics_to_file(metric_path, "MUSIQ", musiq_iqa)
        # write_metrics_to_file(metric_path, "MANIQA", maniqa_iqa)
        # print(f"now fid")
        # fid_value = fid_metric(gt_dir, predict_dir)
        # with open(metric_path, "a") as f:
        #     f.write(f"FID = {get_value(fid_value)}\n")
        

if __name__ == "__main__":
    args = Args()
    args.data = "../data/brats_256_t1_2021_pair_4x"
    out_path = "./metric.txt"
    
    ckpt_paths = ["vae_ch160v4096z32.pth"]
    ckpt_paths = ckpt_paths + sorted(glob.glob("metric_results/ckpt*.pth"))
    print(ckpt_paths)
    # ckpt_paths = [f"local_output/ckpt-{i}.pth" for i in range(6,9)]
    
    args.vocab_size = 4096
    args.vocab_width = 32
    maxtot = -1 # -1
    args.batch_size= 4
    args.device = "cuda"

    val_aug = [
            transforms.ToTensor(), normalize_01_into_pm1,
        ]
    val_aug = transforms.Compose(val_aug)
    val_set = DIV2KData(data_dir=os.path.join(args.data,"val"), transform=val_aug,augment=False)  
    ld_val = DataLoader(
        dataset=val_set, num_workers=args.workers, pin_memory=True, batch_size=args.batch_size, shuffle=False,
    )
    del val_set

    setup(0,1)
    
    get_img(args,ld_val,maxtot,ckpt_paths)
    metric(out_path,ckpt_paths)
