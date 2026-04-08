import PIL.Image as PImage
from PIL import ImageFile
from torchvision.transforms import InterpolationMode, transforms
from utils.data_loader import DIV2KData
PImage.MAX_IMAGE_PIXELS = (1024 * 1024 * 1024 // 4 // 3) * 5
ImageFile.LOAD_TRUNCATED_IMAGES = False
import os

def normalize_01_into_pm1(x):  # normalize x from [0, 1] to [-1, 1] by (x*2) - 1
    return x.add(x).add_(-1)


def pil_load(path: str, proposal_size):
    with open(path, 'rb') as f:
        img: PImage.Image = PImage.open(f)
        w: int = img.width
        h: int = img.height
        sh: int = min(h, w)
        if sh > proposal_size:
            ratio: float = proposal_size / sh
            w = round(ratio * w)
            h = round(ratio * h)
        img.draft('RGB', (w, h))
        img = img.convert('RGB')
    return img


def build_dataset(
    datasets_str: str
):
    train_aug = [
        transforms.ToTensor(), normalize_01_into_pm1,
    ]
    val_aug = [
        transforms.ToTensor(), normalize_01_into_pm1,
    ]

    train_aug, val_aug = transforms.Compose(train_aug), transforms.Compose(val_aug)
    
    train_set = DIV2KData(data_dir=os.path.join(datasets_str,"train"), transform=train_aug, augment=True)  # todo: junfeng; only `train_set` required, no need to create a 'validation_set'
    val_set = DIV2KData(data_dir=os.path.join(datasets_str,"val"), transform=val_aug, augment=False)  # todo: junfeng; only `train_set` required, no need to create a 'validation_set'
    
    # log dataset
    print(f'[Dataset] {len(train_set)=}')
    print(f'[Dataset] {len(val_set)=}')
    return train_set, val_set


def pil_loader(path):
    with open(path, 'rb') as f:
        img: PImage.Image = PImage.open(f).convert('RGB')
    return img


def no_transform(x): return x


def print_aug(transform, label):
    print(f'Transform {label} = ')
    if hasattr(transform, 'transforms'):
        for t in transform.transforms:
            print(t)
    else:
        print(transform)
    print('---------------------------\n')
