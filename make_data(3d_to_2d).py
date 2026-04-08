""" Taken and adapted from https://github.com/cyclomon/3dbraingen """

import csv
import numpy as np
import torch
from torch.utils.data.dataset import Dataset
import os
from skimage.transform import resize
from nilearn import surface
import nibabel as nib
from skimage import exposure
import argparse
import pandas as pd
from torch.utils.data import DataLoader
from scipy.ndimage import zoom
import torchio as tio
# import monai
from torchvision import transforms
from tqdm import tqdm

TRAIN_TRANSFORMS = tio.Compose([
    # transforms.RandomApply(
    # [monai.transforms.RandSpatialCrop(roi_size=(128//2,128//2,128//2), random_center=True, random_size=True),
    # monai.transforms.Resize(spatial_size=(128,128,128))], p=1.0),
    tio.RandomGamma(p=0.5),
    # tio.RandomBiasField(p=0.4),
])
ALL_TRANSFORMS  = tio.Compose([
    tio.RandomFlip(axes=(0), flip_probability=0.5),
])      


class BRATSDataset_one(Dataset):
    def __init__(self, root_dir, train=True, imgtype='flair', severity='HGG', augmentation=True):
        self.augmentation = augmentation
        self.train = train
        self.severity = severity
        self.root_dir = root_dir
        self.imgtype = imgtype
        self.dataset = self.get_dataset()

    def get_dataset(self):
        if self.train:
            brats_2018 = os.listdir(self.root_dir)
        return brats_2018

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_name = self.dataset[index]
        path = os.path.join(self.root_dir, img_name)
        
        img = nib.load(os.path.join(
            path, img_name+'_' + self.imgtype+'.nii.gz'))
        gt = nib.load(os.path.join(path, img_name + '_' + 'seg.nii.gz'))

        A = np.zeros((240, 240, 166))
        G = np.zeros((240, 240, 166))
        A[:, :, 11:] = img.get_fdata()
        G[:, :, 11:] = gt.get_fdata()
        
        G[G==4] = 3
        
        x = []
        y = []
        z = []

        for i in range(240):
            if np.all(A[i, :, :] == 0):
                x.append(i)
            if np.all(A[:, i, :] == 0):
                y.append(i)
            if i < 155:
                if np.all(A[:, :, i] == 0):
                    z.append(i)

        xl, yl, zl = 0, 0, 0
        xh, yh, zh = 240, 240, 155
        for xn in x:
            if xn < 120:
                if xn > xl:
                    xl = xn
            else:
                if xn < xh:
                    xh = xn
        for yn in y:
            if yn < 120:
                if yn > yl:
                    yl = yn
            else:
                if yn < yh:
                    yh = yn
        for zn in z:
            if zn < 77:
                if zn > zl:
                    zl = zn
            else:
                if zn < zh:
                    zh = zn

        
        B = A[xl-10:xh+10, yl-10:yh+10, zl-10:zh+10]
        C = G[xl-10:xh+10, yl-10:yh+10, zl-10:zh+10]

        # img = np.resize(B, (256,256,B.shape[2]))
        zoom_factors = (256 / B.shape[0], 256 / B.shape[1], 1)
        img = zoom(B, zoom_factors, order=3)
        
        # img = B
        
        img = 1.0*img
        img = exposure.rescale_intensity(img)
        img = (img-np.min(img))/(np.max(img)-np.min(img))
        img = img * 255.0
        
        
        return img
    
from PIL import Image

gpath = '/mnt/d/DATA'
if __name__ == '__main__':
    # dataset = Teeth256Dataset(root_dir='/media/why/牙齿数据/FYC/medicaldiffusion/data/teeth_focus/crop_256/image')
    dataset = BRATSDataset_one(root_dir = os.path.join(gpath, 'Brats2021'), train=True, imgtype='t1', severity='HGG',augmentation = False)

    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    image_out_train_dir = os.path.join(gpath, 'brats_256_t1_2021/train')
    image_out_val_dir = os.path.join(gpath, 'brats_256_t1_2021/val')
    image_out_test_dir = os.path.join(gpath, 'brats_256_t1_2021/test')
    os.makedirs(image_out_train_dir, exist_ok=True)
    os.makedirs(image_out_val_dir, exist_ok=True)
    os.makedirs(image_out_test_dir, exist_ok=True)
    split_train = 0.7
    split_val = 0.15
    split_test = 0.15
    
    
    for index,batch in tqdm(enumerate(dataloader),total = len(dataloader)):
        x = batch
        for i in range(30, x.shape[-1]-30):

            pic = Image.fromarray(x[0][:,:,i].numpy()).convert("RGB")
            rand = np.random.rand()

            if rand < split_train:
                pic.save(os.path.join(image_out_train_dir, f"{index}_{i}.jpg"))
            elif rand < split_train + split_val:
                pic.save(os.path.join(image_out_val_dir, f"{index}_{i}.jpg"))
            else:
                pic.save(os.path.join(image_out_test_dir, f"{index}_{i}.jpg"))
            
        # if index < len(dataloader) * 0.8:
        #     for i in range(4):
        #         nib.save(nib.Nifti1Image(x[0][i].numpy(), np.eye(4)), f'./data/brats_128/imagesTr/{index}_000{i}.nii.gz')
        #     nib.save(nib.Nifti1Image(y[0].numpy(), np.eye(4)), f'./data/brats_128/labelsTr/{index}.nii.gz')
        # else :
        #     for i in range(4):
        #         nib.save(nib.Nifti1Image(x[0][i].numpy(), np.eye(4)), f'./data/brats_128/imagesTs/{index}_000{i}.nii.gz')
        #     nib.save(nib.Nifti1Image(y[0].numpy(), np.eye(4)), f'./data/brats_128/labelsTs/{index}.nii.gz')
        

