from torch.utils.data import Dataset, DataLoader
import torch
import os
from typing import Optional
import argparse
from torchvision import transforms
from PIL import Image
import random
import numpy as np

def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
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
        self.data = os.listdir(data_dir)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = Image.open(os.path.join(self.data_dir, self.data[idx])).convert('RGB')
        # if self.target_shape:
        #     img = img.resize(self.target_shape, Image.BICUBIC)
        if self.augment:
            img = center_crop_arr(img, img.size[0])
            if random.random() > 0.5:
                img = img.transpose(1)
        if self.transform:
            img = self.transform(img)
        return img
    
if __name__ == '__main__':
    from torchvision.transforms import InterpolationMode
    import matplotlib.pyplot as plt
    import numpy as np
    
    fino_ = 256
    img_list = os.listdir("./data/df2k_ost/GT")
    for img in img_list:
        img_path = os.path.join("./data/df2k_ost/GT", img)
        im = Image.open(img_path)
        
        # 获取图片的宽度和高度
        width, height = im.size
        
        if width < fino_ or height < fino_:
            # 计算最短边
            if width < height:
                new_width = fino_
                new_height = int((fino_ / width) * height)  # 按照比例计算新高度
            else:
                new_height = fino_
                new_width = int((fino_ / height) * width)  # 按照比例计算新宽度
            
            # 调整大小
            im = im.resize((new_width, new_height))  # 使用ANTIALIAS来保持图像质量
            
            # 或者保存为新的文件
        new_img_path = img_path.replace('GT', 'GT_resized')
        im.save(new_img_path)
        

    
    def normalize_01_into_pm1(x):  # normalize x from [0, 1] to [-1, 1] by (x*2) - 1
        return x.add(x).add_(-1)
    
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
        