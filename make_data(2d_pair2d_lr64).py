import os
from PIL import Image, ImageFilter
from tqdm import tqdm


def generate_lr64_images(split_root, out_root, target_size=(64, 64), blur_radius=1):
    """Generate 64x64 LR images from HR images without upsampling back."""
    lr64_out_dir = os.path.join(out_root, "LR_64x64")
    os.makedirs(lr64_out_dir, exist_ok=True)

    for filename in tqdm(os.listdir(split_root), desc=f"Processing {split_root}"):
        img_path = os.path.join(split_root, filename)
        if not (os.path.isfile(img_path) and filename.lower().endswith((".png", ".jpg", ".jpeg"))):
            continue

        hr_image = Image.open(img_path)
        lr64_image = generate_lr64_image(hr_image, target_size=target_size, blur_radius=blur_radius)
        lr64_image.save(os.path.join(lr64_out_dir, filename))

    print(f"Done: {split_root} -> {lr64_out_dir}")


def generate_lr64_image(hr_image, target_size=(64, 64), blur_radius=1):
    lr_image = hr_image.resize(target_size, Image.BICUBIC)
    if blur_radius and blur_radius > 0:
        lr_image = lr_image.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    return lr_image


if __name__ == "__main__":
    dataset_root = r"D:\DATA\brats_256_t2_2021_pair_png_with_ref"
    splits = ["train", "val", "test"]

    for split in splits:
        split_root = os.path.join(dataset_root, split, "HR")
        out_root = os.path.join(dataset_root, split)
        generate_lr64_images(split_root, out_root, target_size=(64, 64), blur_radius=1)
