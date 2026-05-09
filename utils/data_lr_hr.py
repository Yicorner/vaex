"""
Data loader for LR-HR paired data
Supports two-stage training:
- Stage 1: Load LR only
- Stage 2: Load both LR and HR
"""
import os
import random
from typing import Optional, Tuple, Union

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


class LR_HR_Dataset(Dataset):
    """Dataset that loads LR and/or HR images"""
    
    def __init__(
        self, 
        data_dir: str, 
        transform: Optional[transforms.Compose] = None,
        augment: bool = False,
        load_mode: str = 'both',  # 'lr_only', 'hr_only', or 'both'
        lr_img_size: Optional[int] = None,
        return_lr_original: bool = False,
    ):
        """
        Args:
            data_dir: Root directory containing 'LR', 'HR', and optionally 'REF' subdirectories
            transform: Transform to apply to images
            augment: Whether to apply augmentation
            load_mode: What to load - 'lr_only', 'hr_only', or 'both'
        """
        self.data_dir = data_dir
        self.transform = transform
        self.augment = augment
        self.load_mode = load_mode
        self.lr_img_size = int(lr_img_size) if lr_img_size is not None else None
        self.return_lr_original = bool(return_lr_original)
        
        # Build file lists
        self.lr_files = []
        self.hr_files = []
        
        lr_dir = os.path.join(data_dir, 'LR')
        hr_dir = os.path.join(data_dir, 'HR')
        
        if load_mode in ['lr_only', 'both'] and os.path.isdir(lr_dir):
            self.lr_files = sorted([
                os.path.join(lr_dir, f) for f in os.listdir(lr_dir)
                if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))
            ])
        
        if load_mode in ['hr_only', 'both'] and os.path.isdir(hr_dir):
            self.hr_files = sorted([
                os.path.join(hr_dir, f) for f in os.listdir(hr_dir)
                if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))
            ])
        
        # Verify pairing if loading both
        if load_mode == 'both':
            if len(self.lr_files) != len(self.hr_files):
                print(f'[Warning] LR and HR file counts mismatch: {len(self.lr_files)} vs {len(self.hr_files)}')
                # Use the smaller count
                min_len = min(len(self.lr_files), len(self.hr_files))
                self.lr_files = self.lr_files[:min_len]
                self.hr_files = self.hr_files[:min_len]
        
        # Set data length
        if load_mode == 'lr_only':
            self.data_files = self.lr_files
        elif load_mode == 'hr_only':
            self.data_files = self.hr_files
        else:  # both
            self.data_files = list(zip(self.lr_files, self.hr_files))
        
        if len(self.data_files) == 0:
            raise ValueError(f'No data found in {data_dir} with mode {load_mode}')
    
    def __len__(self):
        return len(self.data_files)
    
    def __getitem__(self, idx) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Returns:
            - If load_mode == 'lr_only': lr_img tensor
            - If load_mode == 'hr_only': hr_img tensor
            - If load_mode == 'both': tuple (lr_img, hr_img)
        """
        if self.load_mode == 'lr_only':
            img = Image.open(self.lr_files[idx]).convert('RGB')
            if self.augment:
                img = self._augment(img)
            img_gt = img
            if self.lr_img_size is not None:
                img = img.resize((self.lr_img_size, self.lr_img_size), resample=Image.BICUBIC)
            if self.transform:
                img = self.transform(img)
                img_gt = self.transform(img_gt)
            if self.return_lr_original:
                return img, img_gt
            return img
        
        elif self.load_mode == 'hr_only':
            img = Image.open(self.hr_files[idx]).convert('RGB')
            if self.augment:
                img = self._augment(img)
            if self.transform:
                img = self.transform(img)
            return img
        
        else:  # both
            lr_path, hr_path = self.data_files[idx]
            lr_img = Image.open(lr_path).convert('RGB')
            hr_img = Image.open(hr_path).convert('RGB')
            
            # Apply same augmentation to both if enabled
            if self.augment:
                # Use same random state for both images
                seed = random.randint(0, 2**32 - 1)
                random.seed(seed)
                lr_img = self._augment(lr_img)
                random.seed(seed)
                hr_img = self._augment(hr_img)

            if self.lr_img_size is not None:
                lr_img = lr_img.resize((self.lr_img_size, self.lr_img_size), resample=Image.BICUBIC)
            
            if self.transform:
                lr_img = self.transform(lr_img)
                hr_img = self.transform(hr_img)
            
            return lr_img, hr_img
    
    def _augment(self, img: Image.Image) -> Image.Image:
        """Apply augmentation to image"""
        # Horizontal flip
        if random.random() > 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        return img


def normalize_01_into_pm1(x):
    """Normalize x from [0, 1] to [-1, 1]"""
    return x.add(x).add_(-1)


def build_lr_hr_dataset(
    datasets_str: str,
    load_mode: str = 'both',
    lr_img_size: Optional[int] = None,
    return_lr_original: bool = False,
):
    """
    Build LR-HR dataset for two-stage training.
    
    Args:
        datasets_str: Path to dataset root directory
        load_mode: 'lr_only', 'hr_only', or 'both'
        lr_img_size: Target LR size before tensor conversion; None keeps source resolution
        return_lr_original: For stage-1 visualization, return (resized_lr, original_lr)
    
    Returns:
        train_set, val_set
    """
    train_aug = [
        transforms.ToTensor(), 
        normalize_01_into_pm1,
    ]
    val_aug = [
        transforms.ToTensor(), 
        normalize_01_into_pm1,
    ]
    
    train_aug = transforms.Compose(train_aug)
    val_aug = transforms.Compose(val_aug)
    
    train_set = LR_HR_Dataset(
        data_dir=os.path.join(datasets_str, "train"), 
        transform=train_aug, 
        augment=True,
        load_mode=load_mode,
        lr_img_size=lr_img_size,
        return_lr_original=return_lr_original,
    )
    
    val_set = LR_HR_Dataset(
        data_dir=os.path.join(datasets_str, "val"), 
        transform=val_aug, 
        augment=False,
        load_mode=load_mode,
        lr_img_size=lr_img_size,
        return_lr_original=return_lr_original,
    )
    
    print(f'[LR-HR Dataset] mode={load_mode}, {len(train_set)=}, {len(val_set)=}')
    return train_set, val_set

