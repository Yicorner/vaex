import os
from PIL import Image, ImageFilter
import shutil
from tqdm import tqdm

def generate_lr_images(root, out_root, scale_factor=4):
    # 创建输出目录
    hr_out_dir = os.path.join(out_root, 'HR')
    lr_out_dir = os.path.join(out_root, 'LR')
    
    if not os.path.exists(hr_out_dir):
        os.makedirs(hr_out_dir,exist_ok=True)
    if not os.path.exists(lr_out_dir):
        os.makedirs(lr_out_dir,exist_ok=True)
    
    # 遍历root目录下的所有文件
    for filename in tqdm(os.listdir(root)):
        hr_path = os.path.join(root, filename)
        # 只处理图片文件
        if os.path.isfile(hr_path) and filename.lower().endswith(('.png', '.jpg', '.jpeg')):

            hr_image = Image.open(hr_path)
            lr_image = generate_lr_image(hr_image, scale_factor)

            hr_image.save(os.path.join(hr_out_dir, filename))
            lr_image.save(os.path.join(lr_out_dir, filename))
            
    
    print('所有图片处理完成！')

def generate_lr_image(hr_image, scale_factor):
    # 计算LR图像的尺寸
    width, height = hr_image.size
    new_width = int(width / scale_factor)
    new_height = int(height / scale_factor)
    
    # 使用双三次插值进行下采样
    lr_image = hr_image.resize((new_width, new_height), Image.BICUBIC)
    
    # 可选：添加高斯模糊
    lr_image = lr_image.filter(ImageFilter.GaussianBlur(radius=1))
    
    # 插值回原来大小
    lr_image = lr_image.resize((width, height), Image.BICUBIC)
    return lr_image

# 示例使用方法
for str in ["train", "val","test"]:
    root = f'./data/brats_256_t1_2021/{str}'  # 替换为你的数据根目录
    out_root = f'./data/brats_256_t1_2021_pair_4x/{str}'  # 替换为输出目录
    generate_lr_images(root, out_root, scale_factor=4)