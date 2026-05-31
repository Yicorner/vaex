"""
Image saving utility for VAE training visualization.

This module provides functions to save original and reconstructed images
during training for monitoring VAE reconstruction quality.
"""
import json
import os
from typing import Any, Dict, List, Optional, Tuple
import torch
import torchvision
from PIL import Image
import numpy as np


def denormalize_image(tensor: torch.Tensor) -> torch.Tensor:
    """
    Denormalize image tensor from [-1, 1] to [0, 1] range.
    
    Args:
        tensor: Image tensor in range [-1, 1], shape [B, C, H, W]
    
    Returns:
        Denormalized tensor in range [0, 1]
    """
    return tensor.add(1).mul_(0.5).clamp_(0, 1)


def tensor_to_pil_image(tensor: torch.Tensor) -> Image.Image:
    """
    Convert a single image tensor to PIL Image.
    
    Args:
        tensor: Image tensor [C, H, W] in range [0, 1]
    
    Returns:
        PIL Image in RGB format
    """
    # Convert to numpy and transpose from CHW to HWC
    img_np = tensor.cpu().detach().numpy()
    img_np = np.transpose(img_np, (1, 2, 0))
    
    # Convert to uint8
    img_np = (img_np * 255).astype(np.uint8)
    
    # Convert to PIL Image
    if img_np.shape[2] == 1:
        # Grayscale image
        return Image.fromarray(img_np.squeeze(2), mode='L')
    else:
        # RGB image
        return Image.fromarray(img_np, mode='RGB')


def tensor_to_metric_np_01(tensor: torch.Tensor) -> np.ndarray:
    """Convert `[C,H,W]` in [-1, 1] to a PSNR/SSIM-ready numpy image.

    RGB images become HWC. Single-channel medical images become HW so skimage
    computes true grayscale SSIM instead of treating a singleton channel as RGB.
    """
    arr = ((tensor.detach().cpu().float().numpy() + 1.0) * 0.5).clip(0.0, 1.0)
    if arr.shape[0] == 1:
        return arr[0]
    return np.transpose(arr, (1, 2, 0))


def compute_psnr_ssim(pred: torch.Tensor, target: torch.Tensor) -> Dict[str, Any]:
    """Compute PSNR/SSIM for `[B,C,H,W]` tensors in [-1, 1].

    `C=1` uses grayscale HW arrays. `C=3` uses HWC arrays with `channel_axis=2`.
    """
    from skimage.metrics import peak_signal_noise_ratio, structural_similarity

    psnrs: List[float] = []
    ssims: List[float] = []
    for i in range(pred.shape[0]):
        pred_np = tensor_to_metric_np_01(pred[i])
        target_np = tensor_to_metric_np_01(target[i])
        psnrs.append(float(peak_signal_noise_ratio(target_np, pred_np, data_range=1.0)))
        if target_np.ndim == 2:
            ssims.append(float(structural_similarity(target_np, pred_np, data_range=1.0)))
        else:
            ssims.append(float(structural_similarity(target_np, pred_np, channel_axis=2, data_range=1.0)))

    return {
        "psnr_mean": float(np.mean(psnrs)) if psnrs else 0.0,
        "ssim_mean": float(np.mean(ssims)) if ssims else 0.0,
        "psnr_list": psnrs,
        "ssim_list": ssims,
    }


def save_reconstruction_comparison(
    original: torch.Tensor,
    reconstructed: torch.Tensor,
    save_dir: str,
    ep: int,
    it: int,
    max_samples: int = 4,
) -> str:
    """
    Save original and reconstructed images side by side for comparison.
    
    Args:
        original: Original image tensor [B, C, H, W] in range [-1, 1]
        reconstructed: Reconstructed image tensor [B, C, H, W] in range [-1, 1]
        save_dir: Directory to save images
        ep: Current epoch number
        it: Current iteration number
        max_samples: Maximum number of samples to save from the batch
    
    Returns:
        Path to the saved comparison image
    """
    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Denormalize images
    original_denorm = denormalize_image(original.clone())
    reconstructed_denorm = denormalize_image(reconstructed.clone())
    
    # Limit number of samples
    batch_size = original.shape[0]
    num_samples = min(batch_size, max_samples)
    
    # Create comparison grid: original and reconstructed side by side
    # Shape: [num_samples*2, C, H, W]
    comparison_images = []
    for i in range(num_samples):
        comparison_images.append(original_denorm[i])
        comparison_images.append(reconstructed_denorm[i])
    
    comparison_tensor = torch.stack(comparison_images, dim=0)
    
    # Create grid: 2 columns (original | reconstructed), num_samples rows
    grid = torchvision.utils.make_grid(
        comparison_tensor,
        nrow=2,  # 2 columns: original, reconstructed
        padding=2,
        pad_value=1.0,  # White padding
    )
    
    # Convert grid to PIL Image
    grid_pil = tensor_to_pil_image(grid)
    
    # Save image
    filename = f"ep{ep:04d}_it{it:06d}_comparison.png"
    filepath = os.path.join(save_dir, filename)
    grid_pil.save(filepath)
    
    return filepath


def save_stage3_alignment_comparison(
    lr: torch.Tensor,
    pred_scale0_img: torch.Tensor,
    target_scale0_img: torch.Tensor,
    hr: torch.Tensor,
    save_dir: str,
    ep: int,
    it: int,
    max_samples: int = 4,
) -> str:
    """Save `LR_upsampled | pred_scale0_decode | target_scale0_decode | HR`.

    This is for stage3, where the model predicts the stage2 scale[0] latent
    from LR and the frozen stage2 decoder reveals what that latent means.
    """
    os.makedirs(save_dir, exist_ok=True)

    lr_denorm = denormalize_image(lr.clone())
    pred_denorm = denormalize_image(pred_scale0_img.clone())
    target_denorm = denormalize_image(target_scale0_img.clone())
    hr_denorm = denormalize_image(hr.clone())

    target_hw = hr_denorm.shape[-2:]
    if lr_denorm.shape[-2:] != target_hw:
        lr_denorm = torch.nn.functional.interpolate(
            lr_denorm, size=target_hw, mode='bicubic', align_corners=False
        ).clamp_(0, 1)
    if pred_denorm.shape[-2:] != target_hw:
        pred_denorm = torch.nn.functional.interpolate(
            pred_denorm, size=target_hw, mode='bicubic', align_corners=False
        ).clamp_(0, 1)
    if target_denorm.shape[-2:] != target_hw:
        target_denorm = torch.nn.functional.interpolate(
            target_denorm, size=target_hw, mode='bicubic', align_corners=False
        ).clamp_(0, 1)

    num_samples = min(hr.shape[0], max_samples)
    tiles = []
    for i in range(num_samples):
        tiles.extend([lr_denorm[i], pred_denorm[i], target_denorm[i], hr_denorm[i]])

    grid = torchvision.utils.make_grid(
        torch.stack(tiles, dim=0),
        nrow=4,
        padding=2,
        pad_value=1.0,
    )
    filepath = os.path.join(save_dir, f"ep{ep:04d}_it{it:06d}_stage3_alignment.png")
    tensor_to_pil_image(grid).save(filepath)
    return filepath


def save_stage2_scale0_lr_diagnostic(
    lr: torch.Tensor,
    decode_scale0_img: torch.Tensor,
    save_dir: str,
    ep: int,
    it: int,
    max_samples: int = 4,
) -> str:
    """Save ``LR | decode_s0(HR_scale0)`` for stage-2 scale0 image alignment.

    ``decode_scale0_img`` is the scale0-only decode used by ``L_align`` (HR latent,
    shared decoder), not the full multi-scale HR reconstruction.
    """
    os.makedirs(save_dir, exist_ok=True)

    lr_denorm = denormalize_image(lr.clone())
    decode_denorm = denormalize_image(decode_scale0_img.clone())

    if lr_denorm.shape[-2:] != decode_denorm.shape[-2:]:
        lr_denorm = torch.nn.functional.interpolate(
            lr_denorm, size=decode_denorm.shape[-2:], mode='bicubic', align_corners=False
        ).clamp_(0, 1)

    num_samples = min(lr.shape[0], max_samples)
    tiles = []
    for i in range(num_samples):
        tiles.extend([lr_denorm[i], decode_denorm[i]])

    grid = torchvision.utils.make_grid(
        torch.stack(tiles, dim=0),
        nrow=2,
        padding=2,
        pad_value=1.0,
    )
    filepath = os.path.join(save_dir, f"ep{ep:04d}_it{it:06d}_scale0_lr.png")
    tensor_to_pil_image(grid).save(filepath)
    return filepath


def save_reconstruction_run_metadata(
    save_dir: str,
    args_state: Dict[str, Any],
    stage_name: str,
    frequency_description: str,
    max_samples: int,
    filename_pattern: str = "ep{epoch:04d}_it{iter:06d}_comparison.png",
    metadata_dir: Optional[str] = None,
) -> str:
    """
    Save a lightweight manifest for the training run.

    Reconstruction PNGs live under ``save_dir``; ``run_metadata.json`` is written
    to ``metadata_dir`` when provided (typically ``args.local_out_dir_path``),
    otherwise falls back to ``save_dir`` for backward compatibility.
    """
    os.makedirs(save_dir, exist_ok=True)
    out_dir = metadata_dir if metadata_dir is not None else save_dir
    os.makedirs(out_dir, exist_ok=True)
    metadata_path = os.path.join(out_dir, "run_metadata.json")
    payload = {
        "stage_name": stage_name,
        "save_dir": save_dir,
        "metadata_dir": out_dir,
        "filename_pattern": filename_pattern,
        "frequency_description": frequency_description,
        "comparison_layout": "2 columns per row: original | reconstructed",
        "max_samples_per_image": int(max_samples),
        "postprocess": [
            "denormalize tensors from [-1, 1] to [0, 1]",
            "clamp values to [0, 1]",
            "stack original and reconstruction pairs into a grid",
            "add white padding between tiles",
            "convert to uint8 PNG without extra filtering",
        ],
        "args": args_state,
    }
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=True, sort_keys=True, default=str)
    return metadata_path


def save_individual_images(
    original: torch.Tensor,
    reconstructed: torch.Tensor,
    save_dir: str,
    ep: int,
    it: int,
    max_samples: int = 4,
) -> tuple[list[str], list[str]]:
    """
    Save original and reconstructed images separately.
    
    Args:
        original: Original image tensor [B, C, H, W] in range [-1, 1]
        reconstructed: Reconstructed image tensor [B, C, H, W] in range [-1, 1]
        save_dir: Directory to save images
        ep: Current epoch number
        it: Current iteration number
        max_samples: Maximum number of samples to save from the batch
    
    Returns:
        Tuple[List[str], List[str]]: (list of original image paths, list of reconstructed image paths)
    """
    # Create save directories
    orig_dir = os.path.join(save_dir, "original")
    recon_dir = os.path.join(save_dir, "reconstructed")
    os.makedirs(orig_dir, exist_ok=True)
    os.makedirs(recon_dir, exist_ok=True)
    
    # Denormalize images
    original_denorm = denormalize_image(original.clone())
    reconstructed_denorm = denormalize_image(reconstructed.clone())
    
    # Limit number of samples
    batch_size = original.shape[0]
    num_samples = min(batch_size, max_samples)
    
    original_paths = []
    reconstructed_paths = []
    
    for i in range(num_samples):
        # Convert to PIL and save original
        orig_pil = tensor_to_pil_image(original_denorm[i])
        orig_filename = f"ep{ep:04d}_it{it:06d}_sample{i:02d}_original.png"
        orig_path = os.path.join(orig_dir, orig_filename)
        orig_pil.save(orig_path)
        original_paths.append(orig_path)
        
        # Convert to PIL and save reconstructed
        recon_pil = tensor_to_pil_image(reconstructed_denorm[i])
        recon_filename = f"ep{ep:04d}_it{it:06d}_sample{i:02d}_reconstructed.png"
        recon_path = os.path.join(recon_dir, recon_filename)
        recon_pil.save(recon_path)
        reconstructed_paths.append(recon_path)
    
    return (original_paths, reconstructed_paths)

