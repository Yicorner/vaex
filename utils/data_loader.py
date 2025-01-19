from torch.utils.data import Dataset, DataLoader
import torch
import os
from typing import Optional
import argparse
from torchvision import transforms
from PIL import Image



class DIV2KData(Dataset):
    def __init__(self, data_dir: str, subset_ratio:float , transform: Optional[transforms.Compose] = None):
        # todo subset_ratio
        self.data_dir = data_dir
        self.transform = transform
        self.data = os.listdir(data_dir)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = Image.open(os.path.join(self.data_dir, self.data[idx])).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img
    