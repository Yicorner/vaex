#!/usr/bin/env python3
"""Evaluate a stage-1 LR VAE checkpoint on a single image folder.

Outputs:
- per-image PSNR / SSIM CSV
- summary JSON + text log
- reconstruction comparisons in 4x2 layout (GT | Pred)
- single-image predictions
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from models.lr_vae import LR_VAE
from utils.image_saver import compute_psnr_ssim, denormalize_image, save_reconstruction_comparison, tensor_to_pil_image

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate stage-1 LR VAE checkpoint on one image folder.")
    parser.add_argument("--ckpt_path", type=Path, required=True, help="Path to stage-1 checkpoint (e.g., local_output/ckpt-3.pth).")
    parser.add_argument("--test_dir", type=Path, required=True, help="Folder containing images to test (direct children).")
    parser.add_argument("--output_dir", type=Path, required=True, help="Where logs and images are written.")
    parser.add_argument("--num_samples", type=int, default=100, help="Number of random images to evaluate.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling.")
    parser.add_argument("--batch_size", type=int, default=4, help="Inference batch size and comparison grid rows.")
    parser.add_argument("--device", type=str, default="auto", help="cuda | cpu | auto")
    parser.add_argument("--lr_ch", type=int, default=None, help="Optional override. Defaults to ckpt args.lr_ch or 128.")
    parser.add_argument("--lr_vocab_width", type=int, default=None, help="Optional override. Defaults to ckpt args.lr_vocab_width or 32.")
    parser.add_argument("--drop_out", type=float, default=None, help="Optional override. Defaults to ckpt args.drop_out or 0.0.")
    parser.add_argument("--lr_vq_beta", type=float, default=None, help="Optional override. Defaults to ckpt args.lr_vq_beta or 1.0.")
    parser.add_argument("--img_channels", type=int, default=None, help="Optional override. Defaults to ckpt args.img_channels or 3. Use 1 for grayscale.")
    return parser.parse_args()


def resolve_device(raw: str) -> torch.device:
    if raw == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(raw)


def list_images(folder: Path) -> Dict[str, Path]:
    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder}")
    paths = {}
    for p in sorted(folder.iterdir()):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            paths[p.stem] = p
    return paths


def sample_images(img_map: Dict[str, Path], n: int, seed: int) -> List[Path]:
    paths = list(img_map.values())
    if n <= 0:
        raise ValueError("--num_samples must be > 0.")
    if n > len(paths):
        raise ValueError(f"Requested {n} samples, but only {len(paths)} images are available.")
    rng = random.Random(seed)
    indices = list(range(len(paths)))
    rng.shuffle(indices)
    return [paths[i] for i in indices[:n]]


def to_pm1_tensor(pil_img: Image.Image) -> torch.Tensor:
    tensor = transforms.ToTensor()(pil_img)  # [0, 1]
    return tensor.add(tensor).add_(-1.0)  # [-1, 1]


def load_ckpt(ckpt_path: Path) -> dict:
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    return torch.load(str(ckpt_path), map_location="cpu")


def read_ckpt_arg(ckpt: dict, key: str, default):
    args = ckpt.get("args", {})
    if isinstance(args, dict):
        return args.get(key, default)
    return default


def extract_lr_state_dict(ckpt: dict) -> dict:
    trainer = ckpt.get("trainer", {})
    if isinstance(trainer, dict):
        for k in ("lr_vae_ema", "lr_vae_wo_ddp"):
            state = trainer.get(k)
            if isinstance(state, dict):
                return state
    for k in ("lr_vae_ema", "lr_vae_wo_ddp", "state_dict"):
        state = ckpt.get(k)
        if isinstance(state, dict):
            return state
    raise KeyError("Could not locate LR VAE weights in checkpoint.")


def build_model_from_ckpt(ckpt: dict, args: argparse.Namespace, device: torch.device) -> LR_VAE:
    lr_ch = args.lr_ch if args.lr_ch is not None else int(read_ckpt_arg(ckpt, "lr_ch", 128))
    lr_vocab_width = args.lr_vocab_width if args.lr_vocab_width is not None else int(read_ckpt_arg(ckpt, "lr_vocab_width", 32))
    drop_out = args.drop_out if args.drop_out is not None else float(read_ckpt_arg(ckpt, "drop_out", 0.0))
    lr_vq_beta = args.lr_vq_beta if args.lr_vq_beta is not None else float(read_ckpt_arg(ckpt, "lr_vq_beta", 1.0))
    img_channels = args.img_channels if args.img_channels is not None else int(read_ckpt_arg(ckpt, "img_channels", 3))

    model = LR_VAE(
        z_channels=lr_vocab_width,
        ch=lr_ch,
        dropout=drop_out,
        beta=lr_vq_beta,
        test_mode=True,
        img_channels=img_channels,
    ).to(device)
    state = extract_lr_state_dict(ckpt)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"[ckpt load] missing={len(missing)}, unexpected={len(unexpected)}")
    model.eval()
    return model


def batch_iter(items: Sequence[Path], batch_size: int) -> Iterable[Sequence[Path]]:
    for i in range(0, len(items), batch_size):
        yield items[i:i + batch_size]


def evaluate(
    model: LR_VAE,
    samples: Sequence[Path],
    output_dir: Path,
    batch_size: int,
    device: torch.device,
) -> Dict[str, float]:
    output_dir.mkdir(parents=True, exist_ok=True)
    comparisons_dir = output_dir / "comparisons"
    predictions_dir = output_dir / "predictions"
    comparisons_dir.mkdir(parents=True, exist_ok=True)
    predictions_dir.mkdir(parents=True, exist_ok=True)

    per_image_csv = output_dir / "metrics_per_image.csv"
    metrics_log = output_dir / "metrics.log"
    sampled_list_path = output_dir / "sampled_files.txt"

    rows: List[Dict[str, float]] = []
    sampled_lines: List[str] = []
    comp_idx = 0

    with torch.no_grad():
        for chunk in batch_iter(samples, batch_size):
            inp_tensors = []
            keys: List[str] = []
            image_mode = "L" if getattr(model, "img_channels", 3) == 1 else "RGB"
            for img_path in chunk:
                img = Image.open(img_path).convert(image_mode)
                inp_tensors.append(to_pm1_tensor(img))
                keys.append(img_path.stem)
                sampled_lines.append(str(img_path))

            inp_batch = torch.stack(inp_tensors, dim=0)
            pred_batch, _, _ = model(inp_batch.to(device, non_blocking=True))
            pred_batch = pred_batch.detach().cpu().clamp(-1, 1)

            save_reconstruction_comparison(
                original=inp_batch,
                reconstructed=pred_batch,
                save_dir=str(comparisons_dir),
                ep=0,
                it=comp_idx,
                max_samples=min(4, len(chunk)),
            )
            comp_idx += 1

            pred_denorm = denormalize_image(pred_batch.clone())  # [0, 1]
            for idx, key in enumerate(keys):
                cur_metrics = compute_psnr_ssim(pred_batch[idx:idx + 1], inp_batch[idx:idx + 1])
                cur_psnr = float(cur_metrics["psnr_list"][0])
                cur_ssim = float(cur_metrics["ssim_list"][0])
                rows.append({"key": key, "psnr": cur_psnr, "ssim": cur_ssim})

                pred_img = tensor_to_pil_image(pred_denorm[idx])
                pred_img.save(predictions_dir / f"{key}_pred.png")

    with sampled_list_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(sampled_lines) + "\n")

    with per_image_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["index", "key", "psnr", "ssim"])
        for i, row in enumerate(rows):
            writer.writerow([i, row["key"], f'{row["psnr"]:.6f}', f'{row["ssim"]:.6f}'])

    psnr_vals = [row["psnr"] for row in rows]
    ssim_vals = [row["ssim"] for row in rows]
    summary = {
        "num_samples": len(rows),
        "psnr_mean": float(np.mean(psnr_vals)),
        "psnr_std": float(np.std(psnr_vals)),
        "psnr_min": float(np.min(psnr_vals)),
        "psnr_max": float(np.max(psnr_vals)),
        "ssim_mean": float(np.mean(ssim_vals)),
        "ssim_std": float(np.std(ssim_vals)),
        "ssim_min": float(np.min(ssim_vals)),
        "ssim_max": float(np.max(ssim_vals)),
        "comparison_layout": "4 rows x 2 cols (GT | Pred), batched by up to 4 samples per image",
        "comparison_images": int(math.ceil(len(rows) / max(1, batch_size))),
        "prediction_images": len(rows),
    }

    with (output_dir / "metrics_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=True)

    with metrics_log.open("w", encoding="utf-8") as f:
        f.write(f"num_samples={summary['num_samples']}\n")
        f.write(f"PSNR mean/std/min/max = {summary['psnr_mean']:.6f} / {summary['psnr_std']:.6f} / {summary['psnr_min']:.6f} / {summary['psnr_max']:.6f}\n")
        f.write(f"SSIM mean/std/min/max = {summary['ssim_mean']:.6f} / {summary['ssim_std']:.6f} / {summary['ssim_min']:.6f} / {summary['ssim_max']:.6f}\n")
        f.write(f"comparisons_dir={comparisons_dir}\n")
        f.write(f"predictions_dir={predictions_dir}\n")
        f.write(f"per_image_csv={per_image_csv}\n")
        f.write(f"sampled_files={sampled_list_path}\n")

    return summary


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)

    image_map = list_images(args.test_dir)
    chosen_images = sample_images(image_map, args.num_samples, args.seed)

    ckpt = load_ckpt(args.ckpt_path)
    model = build_model_from_ckpt(ckpt, args, device)

    summary = evaluate(
        model=model,
        samples=chosen_images,
        output_dir=args.output_dir,
        batch_size=max(1, args.batch_size),
        device=device,
    )

    print(
        f"[done] samples={summary['num_samples']}, "
        f"PSNR={summary['psnr_mean']:.4f}, SSIM={summary['ssim_mean']:.4f}, "
        f"output_dir={args.output_dir}"
    )


if __name__ == "__main__":
    main()
