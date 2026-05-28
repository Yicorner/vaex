#!/usr/bin/env python3
"""Evaluate a stage3 LR->stage2 scale[0] checkpoint on paired test images."""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from models.stage3_scale0 import Stage3Scale0Encoder
from models.vqvae import VQVAE
from utils.image_saver import (
    compute_psnr_ssim,
    denormalize_image,
    save_stage3_alignment_comparison,
    tensor_to_pil_image,
)

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate stage3 LR->scale0 latent alignment.")
    parser.add_argument("--ckpt_path", type=Path, required=True, help="Path to stage3 checkpoint.")
    parser.add_argument("--stage2_ckpt", type=Path, default=None, help="Optional stage2 teacher override.")
    parser.add_argument("--test_dir", type=Path, required=True, help="Dataset split root containing LR/HR folders.")
    parser.add_argument("--lr_folder", type=str, default=None, help="LR subfolder. Defaults to ckpt args.lr_folder.")
    parser.add_argument("--hr_folder", type=str, default=None, help="HR subfolder. Defaults to ckpt args.hr_folder.")
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--img_channels", type=int, default=None)
    return parser.parse_args()


def resolve_device(raw: str) -> torch.device:
    if raw == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(raw)


def pil_mode_from_channels(img_channels: int) -> str:
    return "L" if int(img_channels) == 1 else "RGB"


def to_pm1_tensor(pil_img: Image.Image) -> torch.Tensor:
    return transforms.ToTensor()(pil_img).mul(2).add_(-1)


def read_ckpt_arg(ckpt: dict, key: str, default):
    args = ckpt.get("args", {}) if isinstance(ckpt, dict) else {}
    if isinstance(args, dict):
        return args.get(key, default)
    return default


def normalize_patch_nums(raw) -> Tuple[int, ...]:
    if raw is None:
        return (4, 5, 6, 8, 10, 13, 16)
    if isinstance(raw, str):
        return tuple(int(x) for x in raw.replace(",", " ").split())
    return tuple(int(x) for x in raw)


def extract_vae_state(ckpt: dict) -> dict:
    trainer = ckpt.get("trainer", {}) if isinstance(ckpt, dict) else {}
    if isinstance(trainer, dict):
        for key in ("vae_ema", "vae_wo_ddp", "vae"):
            state = trainer.get(key)
            if isinstance(state, dict):
                return state
    for key in ("vae_ema", "vae_wo_ddp", "state_dict"):
        state = ckpt.get(key)
        if isinstance(state, dict):
            return state
    raise KeyError("Could not locate stage2 VAE weights.")


def extract_stage3_state(ckpt: dict) -> dict:
    trainer = ckpt.get("trainer", {}) if isinstance(ckpt, dict) else {}
    if isinstance(trainer, dict) and isinstance(trainer.get("stage3_wo_ddp"), dict):
        return trainer["stage3_wo_ddp"]
    for key in ("stage3_wo_ddp", "state_dict"):
        state = ckpt.get(key)
        if isinstance(state, dict):
            return state
    raise KeyError("Could not locate stage3 weights.")


def build_teacher(stage2_ckpt_path: Path, device: torch.device) -> VQVAE:
    ckpt = torch.load(str(stage2_ckpt_path), map_location="cpu")
    patch_nums = normalize_patch_nums(read_ckpt_arg(ckpt, "patch_nums", None))
    model = VQVAE(
        vocab_size=int(read_ckpt_arg(ckpt, "vocab_size", 4096)),
        z_channels=int(read_ckpt_arg(ckpt, "vocab_width", 32)),
        ch=int(read_ckpt_arg(ckpt, "ch", 160)),
        test_mode=True,
        share_quant_resi=int(read_ckpt_arg(ckpt, "share_quant_resi", 4)),
        v_patch_nums=patch_nums,
        img_channels=int(read_ckpt_arg(ckpt, "img_channels", 3)),
    ).to(device)
    model.load_state_dict(extract_vae_state(ckpt), strict=True)
    model.eval()
    return model


def build_stage3(ckpt: dict, device: torch.device) -> Stage3Scale0Encoder:
    args = ckpt.get("args", {}) if isinstance(ckpt, dict) else {}
    if not isinstance(args, dict):
        args = {}
    model = Stage3Scale0Encoder(
        z_channels=int(args.get("vocab_width", args.get("Ct5", 32))),
        ch=int(args.get("ch", args.get("vae_ch", 128))),
        dropout=float(args.get("drop_out", 0.0)),
        latent_size=int(args.get("stage3_latent_size", 4)),
        img_channels=int(args.get("img_channels", 3)),
    ).to(device)
    model.load_state_dict(extract_stage3_state(ckpt), strict=True)
    model.eval()
    return model


def list_pairs(root: Path, lr_folder: str, hr_folder: str) -> Dict[str, Tuple[Path, Path]]:
    lr_dir = root / lr_folder
    hr_dir = root / hr_folder
    if not lr_dir.exists():
        raise FileNotFoundError(f"LR folder not found: {lr_dir}")
    if not hr_dir.exists():
        raise FileNotFoundError(f"HR folder not found: {hr_dir}")
    lr_map = {p.name: p for p in sorted(lr_dir.iterdir()) if p.suffix.lower() in IMAGE_EXTS}
    hr_map = {p.name: p for p in sorted(hr_dir.iterdir()) if p.suffix.lower() in IMAGE_EXTS}
    keys = sorted(set(lr_map) & set(hr_map))
    if not keys:
        raise ValueError(f"No paired images found under {lr_dir} and {hr_dir}")
    return {k: (lr_map[k], hr_map[k]) for k in keys}


def sample_pairs(pairs: Dict[str, Tuple[Path, Path]], n: int, seed: int):
    items = list(pairs.items())
    if n <= 0:
        raise ValueError("--num_samples must be > 0")
    if n > len(items):
        raise ValueError(f"Requested {n} samples, only {len(items)} pairs available.")
    rng = random.Random(seed)
    rng.shuffle(items)
    return items[:n]


def batch_iter(items: Sequence, batch_size: int) -> Iterable[Sequence]:
    for i in range(0, len(items), batch_size):
        yield items[i:i + batch_size]


@torch.no_grad()
def evaluate(args: argparse.Namespace) -> Dict[str, float]:
    ckpt = torch.load(str(args.ckpt_path), map_location="cpu")
    stage2_path_value = args.stage2_ckpt or read_ckpt_arg(ckpt, "stage2_ckpt", "")
    if not stage2_path_value:
        raise ValueError("--stage2_ckpt is required because it was not found in the stage3 checkpoint args.")
    stage2_path = Path(stage2_path_value)
    device = resolve_device(args.device)
    teacher = build_teacher(Path(stage2_path), device)
    stage3 = build_stage3(ckpt, device)

    img_channels = args.img_channels if args.img_channels is not None else int(read_ckpt_arg(ckpt, "img_channels", 3))
    mode = pil_mode_from_channels(img_channels)
    lr_folder = args.lr_folder or str(read_ckpt_arg(ckpt, "lr_folder", "LR_64x64"))
    hr_folder = args.hr_folder or str(read_ckpt_arg(ckpt, "hr_folder", "HR"))
    chosen = sample_pairs(list_pairs(args.test_dir, lr_folder, hr_folder), args.num_samples, args.seed)

    comparisons_dir = args.output_dir / "comparisons"
    predictions_dir = args.output_dir / "decoded_predictions"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    comparisons_dir.mkdir(parents=True, exist_ok=True)
    predictions_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, float]] = []
    sampled_lines: List[str] = []
    for bi, chunk in enumerate(batch_iter(chosen, max(1, args.batch_size))):
        names = []
        lr_tensors, hr_tensors = [], []
        for name, (lr_path, hr_path) in chunk:
            names.append(Path(name).stem)
            lr_tensors.append(to_pm1_tensor(Image.open(lr_path).convert(mode)))
            hr_tensors.append(to_pm1_tensor(Image.open(hr_path).convert(mode)))
            sampled_lines.append(f"{lr_path}\t{hr_path}")
        lr = torch.stack(lr_tensors).to(device)
        hr = torch.stack(hr_tensors).to(device)
        pred = stage3(lr)
        target, _ = teacher.img_to_scale_posterior_stats(hr, scale_index=0)
        assert pred.shape == target.shape, f"pred {pred.shape} != target {target.shape}"

        pred_img = teacher.scale_latent_to_img(pred, scale_index=0, clamp=True)
        target_img = teacher.scale_latent_to_img(target, scale_index=0, clamp=True)
        metrics = compute_psnr_ssim(pred_img, target_img)
        diff = pred.float() - target.float()
        cos = F.cosine_similarity(pred.flatten(1).float(), target.flatten(1).float(), dim=1)
        mse_per = diff.square().flatten(1).mean(dim=1)
        mae_per = diff.abs().flatten(1).mean(dim=1)
        latent_psnr_per = -10.0 * torch.log10(mse_per.clamp_min(1e-12))

        save_stage3_alignment_comparison(
            lr=lr.cpu(),
            pred_scale0_img=pred_img.cpu(),
            target_scale0_img=target_img.cpu(),
            hr=hr.cpu(),
            save_dir=str(comparisons_dir),
            ep=0,
            it=bi,
            max_samples=min(4, len(chunk)),
        )
        pred_denorm = denormalize_image(pred_img.detach().cpu().clone())
        for i, key in enumerate(names):
            tensor_to_pil_image(pred_denorm[i]).save(predictions_dir / f"{key}_stage3_scale0_decode.png")
            rows.append({
                "key": key,
                "latent_mse": float(mse_per[i].item()),
                "latent_mae": float(mae_per[i].item()),
                "latent_cosine": float(cos[i].item()),
                "latent_psnr": float(latent_psnr_per[i].item()),
                "decode_psnr": float(metrics["psnr_list"][i]),
                "decode_ssim": float(metrics["ssim_list"][i]),
            })

    with (args.output_dir / "sampled_files.txt").open("w", encoding="utf-8") as f:
        f.write("\n".join(sampled_lines) + "\n")
    with (args.output_dir / "metrics_per_image.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = {"num_samples": len(rows)}
    for key in ("latent_mse", "latent_mae", "latent_cosine", "latent_psnr", "decode_psnr", "decode_ssim"):
        vals = np.array([row[key] for row in rows], dtype=np.float64)
        summary[f"{key}_mean"] = float(vals.mean())
        summary[f"{key}_std"] = float(vals.std())
        summary[f"{key}_min"] = float(vals.min())
        summary[f"{key}_max"] = float(vals.max())
    summary["comparison_layout"] = "LR_upsampled | pred_scale0_decode | target_scale0_decode | HR"
    summary["stage2_ckpt"] = str(stage2_path)
    with (args.output_dir / "metrics_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=True)
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=True)
    with (args.output_dir / "metrics.log").open("w", encoding="utf-8") as f:
        f.write(f"num_samples={summary['num_samples']}\n")
        for key in ("latent_mse", "latent_mae", "latent_cosine", "latent_psnr", "decode_psnr", "decode_ssim"):
            f.write(f"{key}: mean={summary[f'{key}_mean']:.6f}, std={summary[f'{key}_std']:.6f}\n")
    return summary


def main() -> None:
    summary = evaluate(parse_args())
    print(
        f"[done] samples={summary['num_samples']}, "
        f"latent_mse={summary['latent_mse_mean']:.6f}, "
        f"decode_PSNR={summary['decode_psnr_mean']:.2f}, "
        f"decode_SSIM={summary['decode_ssim_mean']:.4f}"
    )


if __name__ == "__main__":
    main()
