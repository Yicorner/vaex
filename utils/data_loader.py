from torch.utils.data import Dataset, DataLoader
import torch
import os
from typing import Optional
import argparse
from torchvision import transforms
from PIL import Image
import random
import numpy as np
import math

def center_crop_arr(pil_image, image_size, min_crop_frac=0.9, max_crop_frac=1.0):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    min_smaller_dim_size = math.ceil(image_size / max_crop_frac)
    max_smaller_dim_size = math.ceil(image_size / min_crop_frac)
    smaller_dim_size = random.randrange(min_smaller_dim_size, max_smaller_dim_size + 1)

    # We are not on a new enough PIL to support the `reducing_gap`
    # argument, which uses BOX downsampling at powers of two first.
    # Thus, we do it by hand to improve downsample quality.
    while min(*pil_image.size) >= 2 * smaller_dim_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )
    
    scale = smaller_dim_size / min(*pil_image.size)

    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])

class DIV2KData(Dataset):
    def __init__(self, data_dir: str, transform: Optional[transforms.Compose] = None, augment=False):
        self.data_dir = data_dir
        self.transform = transform
        self.augment = augment
        
        if os.path.isdir(os.path.join(data_dir, 'HR')):
            self.data = [os.path.join(os.path.join(data_dir, 'HR'),data_name) for data_name in os.listdir(os.path.join(data_dir, 'HR'))]
            # self.data = self.data + [os.path.join(os.path.join(data_dir, 'LR'),data_name) for data_name in os.listdir(os.path.join(data_dir, 'LR'))]
        else:
            self.data = os.listdir(data_dir)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = Image.open(self.data[idx]).convert('RGB')
        # if self.target_shape:
        #     img = img.resize(self.target_shape, Image.BICUBIC)
        if self.augment:
            img = center_crop_arr(img, img.size[0])
            if random.random() > 0.5:
                if "file" in self.data[idx] or "LIDC-IDRI" in self.data[idx]:
                    img = img.transpose(0)
                else:
                    img = img.transpose(1)
        if self.transform:
            img = self.transform(img)
        return img
    
if __name__ == '__main__':
    from torchvision.transforms import InterpolationMode
    import matplotlib.pyplot as plt
    import numpy as np
    
    def normalize_01_into_pm1(x):  # normalize x from [0, 1] to [-1, 1] by (x*2) - 1
        return x.add(x).add_(-1)

    train_aug = [
        transforms.ToTensor(), normalize_01_into_pm1,
    ]
    train_aug= transforms.Compose(train_aug)
    train_set = DIV2KData(data_dir='/home/why/vaex/data/brats_256_t2_2021_pair_png_with_ref/train',transform=train_aug,augment=True)  # todo: junfeng; only `train_set` required, no need to create a 'validation_set'

    ld_train = DataLoader(
            dataset=train_set, num_workers=8, pin_memory=True ,batch_size=1
        )
    print("in")
    for data in ld_train:
        data =  (data.cpu().numpy() + 1.0 ) * 255.0 / 2
        print(data.shape)
        plt.figure()
        plt.imshow(data[0].transpose(1, 2, 0).astype(np.uint8))
        plt.savefig("temp.png")
        
        break
    
    
    # fino_ = 256
    # img_list = os.listdir("./data/brats_256_t1_2021_pair_4x")
    # for img in img_list:
    #     img_path = os.path.join("./data/df2k_ost/GT", img)
    #     im = Image.open(img_path)
        
    #     # 获取图片的宽度和高度
    #     width, height = im.size
        
    #     if width < fino_ or height < fino_:
    #         # 计算最短边
    #         if width < height:
    #             new_width = fino_
    #             new_height = int((fino_ / width) * height)  # 按照比例计算新高度
    #         else:
    #             new_height = fino_
    #             new_width = int((fino_ / height) * width)  # 按照比例计算新宽度
            
    #         # 调整大小
    #         im = im.resize((new_width, new_height))  # 使用ANTIALIAS来保持图像质量
            
    #         # 或者保存为新的文件
    #     new_img_path = img_path.replace('GT', 'GT_resized')
    #     im.save(new_img_path)
        

    
    # def normalize_01_into_pm1(x):  # normalize x from [0, 1] to [-1, 1] by (x*2) - 1
    #     return x.add(x).add_(-1)
    
    # mid_reso = 1.25
    # final_reso = 250
    # mid_reso = round(min(mid_reso, 2) * final_reso)  # first resize to mid_reso, then crop to final_reso
    # train_aug = [
    #     transforms.RandomCrop((final_reso,final_reso)),  # 随机裁剪为224x224的区域
    #     transforms.RandomResizedCrop((final_reso,final_reso), scale=(0.8, 1.2)),  # 随机缩放，scale控制缩放比例范围
    #     transforms.ToTensor(), normalize_01_into_pm1,
    # ]
    # train_aug= transforms.Compose(train_aug)
    
    # train_set = DIV2KData(data_dir="./data/DIV2K_train_HR", subset_ratio=1.0, transform=train_aug)  # todo: junfeng; only `train_set` required, no need to create a 'validation_set'
    
    # ld_train = DataLoader(
    #         dataset=train_set, num_workers=8, pin_memory=True,batch_size=1
    #     )
    
    # for data in ld_train:
    #     data =  (data.cpu().numpy() + 1.0 ) * 255.0 / 2
    #     print(data.shape)
    #     plt.figure()
    #     plt.imshow(data[0].transpose(1, 2, 0).astype(np.uint8))
    #     plt.savefig("temp.jpg")
        
    #     break
        