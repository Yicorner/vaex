import os
import shutil
import sys
import time
import warnings
from collections import deque
from functools import partial
from typing import Dict, Optional, Tuple

import GPUtil
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import dist
from models import Stage3Scale0Encoder, VQVAE
from utils import arg_util, misc
from utils.amp_opt import AmpOptimizer
from utils.data_lr_hr import build_lr_hr_dataset
from utils.data_sampler import DistInfiniteBatchSampler, EvalDistributedSampler
from utils.image_saver import (
    compute_psnr_ssim,
    save_stage3_alignment_comparison,
    save_reconstruction_run_metadata,
)
from utils.lr_control import lr_wd_annealing

warnings.filterwarnings("ignore", category=FutureWarning)


class NullDDP(torch.nn.Module):
    def __init__(self, module, *args, **kwargs):
        super().__init__()
        self.module = module
        self.require_backward_grad_sync = False

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)


def _read_ckpt_arg(ckpt: dict, key: str, default):
    args_state = ckpt.get('args', {}) if isinstance(ckpt, dict) else {}
    if isinstance(args_state, dict):
        return args_state.get(key, default)
    return default


def _normalize_patch_nums(raw) -> Tuple[int, ...]:
    if raw is None:
        return (4, 5, 6, 8, 10, 13, 16)
    if isinstance(raw, str):
        return tuple(int(x) for x in raw.replace(',', ' ').split())
    return tuple(int(x) for x in raw)


def _extract_stage2_vae_state(ckpt: dict) -> dict:
    trainer = ckpt.get('trainer', {}) if isinstance(ckpt, dict) else {}
    if isinstance(trainer, dict):
        for key in ('vae_ema', 'vae_wo_ddp', 'vae'):
            state = trainer.get(key)
            if isinstance(state, dict):
                print(f"[stage2_ckpt] using trainer['{key}']")
                return state
    for key in ('vae_ema', 'vae_wo_ddp', 'state_dict'):
        state = ckpt.get(key)
        if isinstance(state, dict):
            print(f"[stage2_ckpt] using top-level ['{key}']")
            return state
    raise KeyError('Could not locate stage2 VAE weights in checkpoint.')


def build_stage2_teacher(args: arg_util.Args) -> VQVAE:
    assert args.stage2_ckpt, '--stage2_ckpt is required for stage3.'
    ckpt = torch.load(args.stage2_ckpt, map_location='cpu')
    args_state = ckpt.get('args', {}) if isinstance(ckpt, dict) else {}
    if not isinstance(args_state, dict):
        args_state = {}

    # Prefer the stage2 checkpoint configuration; command-line overrides only
    # exist for emergency recovery of very old checkpoints.
    args.img_channels = int(args_state.get('img_channels', args.img_channels))
    args.ch = int(args_state.get('ch', args.ch))
    args.vocab_width = int(args_state.get('vocab_width', args.vocab_width))
    args.vocab_size = int(args_state.get('vocab_size', args.vocab_size))
    args.share_quant_resi = int(args_state.get('share_quant_resi', args.share_quant_resi))
    args.patch_nums = _normalize_patch_nums(args_state.get('patch_nums', args.patch_nums))

    teacher = VQVAE(
        vocab_size=args.vocab_size,
        z_channels=args.vocab_width,
        ch=args.ch,
        test_mode=True,
        share_quant_resi=args.share_quant_resi,
        v_patch_nums=args.patch_nums,
        img_channels=args.img_channels,
    ).to(args.device)
    teacher.load_state_dict(_extract_stage2_vae_state(ckpt), strict=True)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    assert int(teacher.quantize.v_patch_nums[0]) == int(args.stage3_latent_size), (
        f'stage2 scale[0]={teacher.quantize.v_patch_nums[0]} must match '
        f'--stage3_latent_size={args.stage3_latent_size}.'
    )
    return teacher


class Stage3Trainer:
    def __init__(
        self,
        stage3: torch.nn.Module,
        stage3_wo_ddp: Stage3Scale0Encoder,
        teacher_vae: VQVAE,
        stage3_opt: AmpOptimizer,
        args: arg_util.Args,
    ):
        self.stage3 = stage3
        self.stage3_wo_ddp = stage3_wo_ddp
        self.teacher_vae = teacher_vae
        self.stage3_opt = stage3_opt
        self.args = args
        self._metadata_written = False

    def _target_s0(self, inp_hr: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            target, _ = self.teacher_vae.img_to_scale_posterior_stats(inp_hr, scale_index=0)
        return target.detach().contiguous()

    def _decode_s0(self, s0: torch.Tensor, grad: bool) -> torch.Tensor:
        if grad:
            return self.teacher_vae.scale_latent_to_img(s0, scale_index=0, clamp=False)
        with torch.no_grad():
            return self.teacher_vae.scale_latent_to_img(s0, scale_index=0, clamp=False).detach()

    def _losses(self, inp_lr: torch.Tensor, inp_hr: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        pred = self.stage3(inp_lr)
        target = self._target_s0(inp_hr)
        assert pred.shape == target.shape, (
            f'stage3 pred {tuple(pred.shape)} != target {tuple(target.shape)}'
        )
        n = int(self.args.stage3_latent_size)
        assert pred.shape[-2:] == (n, n), f'expected stage3 latent {n}x{n}, got {tuple(pred.shape[-2:])}'

        latent_mse = F.mse_loss(pred.float(), target.float())
        smooth_l1 = F.smooth_l1_loss(pred.float(), target.float())

        pred_img = self._decode_s0(pred, grad=True)
        target_img = self._decode_s0(target, grad=False)
        lr_target = inp_lr
        if lr_target.shape[-2:] != pred_img.shape[-2:]:
            lr_target = F.interpolate(lr_target, size=pred_img.shape[-2:], mode='bicubic', align_corners=False)
        pixel_lr = F.l1_loss(pred_img, lr_target)
        pixel_target = F.l1_loss(pred_img, target_img)

        total = (
            self.args.stage3_latent_mse_weight * latent_mse
            + self.args.stage3_smooth_l1_weight * smooth_l1
            + self.args.stage3_pixel_lr_weight * pixel_lr
            + self.args.stage3_pixel_target_weight * pixel_target
        )
        return total, {
            'latent_mse': latent_mse.detach(),
            'smooth_l1': smooth_l1.detach(),
            'pixel_lr': pixel_lr.detach(),
            'pixel_target': pixel_target.detach(),
            'pred': pred.detach(),
            'target': target.detach(),
            'pred_img': pred_img.detach(),
            'target_img': target_img.detach(),
        }

    def train_step(
        self,
        ep: int,
        it: int,
        g_it: int,
        stepping: bool,
        metric_lg: misc.MetricLogger,
        inp_lr: torch.Tensor,
        inp_hr: torch.Tensor,
    ):
        self.stage3.require_backward_grad_sync = stepping
        with self.stage3_opt.amp_ctx:
            loss, parts = self._losses(inp_lr, inp_hr)
        grad_norm, scale_log2 = self.stage3_opt.backward_clip_step(stepping=stepping, loss=loss)

        if it == 0 or it in metric_lg.log_iters:
            metric_lg.update(
                loss=float(loss.item()),
                latent_mse=float(parts['latent_mse'].item()),
                smooth_l1=float(parts['smooth_l1'].item()),
                pixel_lr=float(parts['pixel_lr'].item()),
                pixel_target=float(parts['pixel_target'].item()),
                gnm=grad_norm.item() if hasattr(grad_norm, 'item') else (0.0 if grad_norm is None else float(grad_norm)),
            )
            if dist.is_master() and self.args.save_reconstruction_images:
                self._save_visuals(ep, it, inp_lr, inp_hr, parts)
        return grad_norm, scale_log2

    @torch.no_grad()
    def eval_ep(self, ld_val: DataLoader, max_batches: Optional[int] = None):
        self.stage3_wo_ddp.eval()
        total = 0
        sums = torch.zeros(6, device=dist.get_device())
        for bi, batch in enumerate(ld_val):
            if max_batches is not None and bi >= max_batches:
                break
            inp_lr, inp_hr = [x.to(dist.get_device(), non_blocking=True) for x in batch]
            loss, parts = self._losses(inp_lr, inp_hr)
            pred, target = parts['pred'], parts['target']
            pred_img, target_img = parts['pred_img'], parts['target_img']
            metrics = compute_psnr_ssim(pred_img, target_img)
            B = inp_lr.shape[0]
            sums += torch.tensor([
                float(loss.item()) * B,
                float(parts['latent_mse'].item()) * B,
                float((pred.float() - target.float()).abs().mean().item()) * B,
                float(F.cosine_similarity(pred.flatten(1).float(), target.flatten(1).float(), dim=1).mean().item()) * B,
                metrics['psnr_mean'] * B,
                metrics['ssim_mean'] * B,
            ], device=dist.get_device())
            total += B
        totals = torch.tensor([float(total)], device=dist.get_device())
        dist.allreduce(sums)
        dist.allreduce(totals)
        denom = max(float(totals.item()), 1.0)
        self.stage3_wo_ddp.train()
        return {
            'loss': float(sums[0].item() / denom),
            'latent_mse': float(sums[1].item() / denom),
            'latent_mae': float(sums[2].item() / denom),
            'latent_cosine': float(sums[3].item() / denom),
            'decode_psnr': float(sums[4].item() / denom),
            'decode_ssim': float(sums[5].item() / denom),
            'num_samples': int(round(denom)),
        }

    def _save_visuals(self, ep: int, it: int, inp_lr: torch.Tensor, inp_hr: torch.Tensor, parts: Dict[str, torch.Tensor]):
        save_dir = os.path.join(self.args.local_out_dir_path, self.args.reconstruction_dir_name or 'stage3_alignment')
        save_stage3_alignment_comparison(
            lr=inp_lr,
            pred_scale0_img=parts['pred_img'],
            target_scale0_img=parts['target_img'],
            hr=inp_hr,
            save_dir=save_dir,
            ep=ep,
            it=it,
            max_samples=self.args.reconstruction_max_samples,
        )
        if not self._metadata_written and self.args.record_reconstruction_metadata:
            save_reconstruction_run_metadata(
                save_dir=save_dir,
                args_state=self.args.state_dict(key_ordered=False),
                stage_name='stage3_lr_to_stage2_scale0',
                frequency_description='save stage3 LR/scale0 alignment comparisons on log iterations',
                max_samples=self.args.reconstruction_max_samples,
                filename_pattern='ep{epoch:04d}_it{iter:06d}_stage3_alignment.png',
            )
            self._metadata_written = True

    def state_dict(self):
        return {
            'stage3_wo_ddp': self.stage3_wo_ddp.state_dict(),
            'stage3_opt': self.stage3_opt.state_dict(),
            'config': {
                'stage3_latent_size': int(self.args.stage3_latent_size),
                'stage2_ckpt': self.args.stage2_ckpt,
                'img_channels': int(self.args.img_channels),
                'Cvae': int(self.stage3_wo_ddp.Cvae),
                'ch': int(self.stage3_wo_ddp.ch),
            },
        }

    def load_state_dict(self, state: dict, strict: bool = True):
        self.stage3_wo_ddp.load_state_dict(state['stage3_wo_ddp'], strict=strict)
        if 'stage3_opt' in state:
            self.stage3_opt.load_state_dict(state['stage3_opt'], strict=False)


def create_tb_lg(args: arg_util.Args):
    if dist.is_master():
        os.makedirs(args.tb_log_dir_path, exist_ok=True)
        tb_lg = misc.DistLogger(misc.TensorboardLogger(log_dir=args.tb_log_dir_path, filename_suffix=f'_{misc.time_str("%m%d_%H%M")}'))
        tb_lg.flush()
    else:
        tb_lg = misc.DistLogger(None)
    dist.barrier()
    return tb_lg


def maybe_resume(args: arg_util.Args):
    if not args.resume:
        return [], 0, 0, {}
    info = [f'[resume] load stage3 checkpoint from {args.resume}']
    ckpt = torch.load(args.resume, map_location='cpu')
    return info, int(ckpt.get('epoch', 0)), int(ckpt.get('iter', 0)), ckpt.get('trainer', {})


def build_everything(args: arg_util.Args):
    resume_info, start_ep, start_it, trainer_state = maybe_resume(args)
    tb_lg = create_tb_lg(args)
    print(f'initial args:\n{str(args)}')
    for line in resume_info:
        print(line)

    args.training_stage = 3
    args.stage3_latent_size = int(args.stage3_latent_size)
    teacher = build_stage2_teacher(args)

    dataset_train, dataset_val = build_lr_hr_dataset(
        args.data,
        load_mode='both',
        lr_folder=args.lr_folder,
        hr_folder=args.hr_folder,
        img_channels=args.img_channels,
    )
    ld_train = DataLoader(
        dataset_train,
        num_workers=args.workers,
        pin_memory=True,
        generator=args.get_different_generator_for_each_rank(),
        batch_sampler=DistInfiniteBatchSampler(
            dataset_len=len(dataset_train),
            glb_batch_size=args.bs,
            same_seed_for_all_ranks=args.same_seed_for_all_ranks,
            shuffle=True,
            fill_last=True,
            rank=dist.get_rank(),
            world_size=dist.get_world_size(),
            start_ep=start_ep,
            start_it=start_it,
        ),
    )
    ld_val = DataLoader(
        dataset_val,
        num_workers=0,
        pin_memory=True,
        batch_size=args.lbs,
        sampler=EvalDistributedSampler(dataset_val, num_replicas=dist.get_world_size(), rank=dist.get_rank()),
        shuffle=False,
        drop_last=False,
    )
    iters_train = len(ld_train)
    args.iters_per_ep = iters_train
    ld_train = iter(ld_train)
    print(f'[stage3 dataloader] gbs={args.bs}, lbs={args.lbs}, iters_train={iters_train}')

    stage3_wo_ddp = Stage3Scale0Encoder(
        z_channels=teacher.Cvae,
        ch=teacher.ch if hasattr(teacher, 'ch') else args.ch,
        dropout=getattr(teacher, 'dropout', args.drop_out),
        latent_size=args.stage3_latent_size,
        img_channels=args.img_channels,
    ).to(args.device)
    stage3_wo_ddp.initialize_from_stage2_vae(teacher)

    ddp_class = torch.nn.parallel.DistributedDataParallel if dist.initialized() else NullDDP
    stage3 = ddp_class(
        stage3_wo_ddp,
        device_ids=[dist.get_local_rank()] if torch.cuda.is_available() else None,
        find_unused_parameters=False,
        broadcast_buffers=False,
    )

    trainable_params = [p for p in stage3_wo_ddp.parameters() if p.requires_grad]
    opt_clz = partial(torch.optim.AdamW, betas=tuple(map(float, args.vae_opt_beta.split('_'))), fused=args.fuse_opt)
    stage3_opt = AmpOptimizer(
        's30',
        model_maybe_fsdp=stage3_wo_ddp,
        fp16=args.fp16,
        bf16=args.bf16,
        zero=args.zero,
        optimizer=opt_clz(params=trainable_params, lr=args.vae_lr, weight_decay=args.vae_wd),
        grad_clip=args.grad_clip,
        n_gradient_accumulation=args.grad_accu,
    )

    trainer = Stage3Trainer(stage3, stage3_wo_ddp, teacher, stage3_opt, args)
    if trainer_state:
        trainer.load_state_dict(trainer_state, strict=False)
    return tb_lg, trainer, start_ep, start_it, iters_train, ld_train, ld_val


def save_checkpoint(args: arg_util.Args, trainer: Stage3Trainer, epoch: int, is_best: bool = False):
    if not dist.is_local_master():
        dist.barrier()
        return
    os.makedirs(args.local_out_dir_path, exist_ok=True)
    ckpt = {
        'epoch': epoch,
        'iter': 0,
        'trainer': trainer.state_dict(),
        'args': args.state_dict(),
    }
    last_path = os.path.join(args.local_out_dir_path, 'ckpt-last.pth')
    torch.save(ckpt, last_path)
    torch.save(ckpt, os.path.join(args.local_out_dir_path, f'ckpt-{epoch - 1}.pth'))
    if is_best:
        shutil.copyfile(last_path, os.path.join(args.local_out_dir_path, 'ckpt-best.pth'))
    dist.barrier()


g_speed_ls = deque(maxlen=128)


def train_one_ep(ep, start_it, args, tb_lg, ld_or_itrt, iters_train, trainer: Stage3Trainer):
    me = misc.MetricLogger(delimiter='  ')
    for key in ['vlr', 'loss', 'latent_mse', 'smooth_l1', 'pixel_lr', 'pixel_target', 'gnm']:
        me.add_meter(key, misc.SmoothedValue(window_size=1, fmt='{value:.4g}'))
    header = f'[Stage3 Ep]: [{ep:4d}/{args.ep}]'
    g_it, wp_it, max_it = ep * iters_train, args.warmup_ep * iters_train, args.ep * iters_train
    log_points = int(getattr(args, 'train_log_points_per_epoch', 0) or max(10, iters_train // 1000))

    last_t_perf = time.perf_counter()
    FREQ = min(50, max(iters_train // 2 - 1, 1))
    for it, batch in me.log_every(start_it, iters_train, ld_or_itrt, log_points, header):
        if (it + 1) % FREQ == 0:
            g_speed_ls.append((time.perf_counter() - last_t_perf) / FREQ)
            args.iter_speed = float(np.median(g_speed_ls))
            try:
                args.max_nvidia_smi = max(args.max_nvidia_smi, max(gpu.memoryUsed for gpu in GPUtil.getGPUs()) / 1024)
            except Exception:
                pass
            last_t_perf = time.perf_counter()

        if it < start_it:
            continue
        inp_lr, inp_hr = [x.to(args.device, non_blocking=True) for x in batch]
        g_it = ep * iters_train + it
        args.cur_it = f'{it + 1}/{iters_train}'
        min_lr, max_lr, _, _ = lr_wd_annealing(
            args.sche, trainer.stage3_opt.optimizer,
            args.vae_lr, args.vae_wd, g_it, wp_it, max_it,
            wp0=args.wp0, wpe=args.sche_end,
        )
        stepping = (g_it + 1) % args.grad_accu == 0
        grad_norm, scale_log2 = trainer.train_step(
            ep=ep, it=it, g_it=g_it, stepping=stepping,
            metric_lg=me, inp_lr=inp_lr, inp_hr=inp_hr,
        )
        me.update(vlr=max_lr)
        tb_lg.set_step(step=g_it)
        tb_lg.update(head='stage3_opt_lr/lr_max', lr=max_lr)
        tb_lg.update(head='stage3_opt_lr/lr_min', lr=min_lr)
        if grad_norm is not None:
            tb_lg.update(head='stage3_opt_grad/grad', grad_norm=grad_norm)
        if scale_log2 is not None:
            tb_lg.update(head='stage3_opt_grad/fp16', scale_log2=scale_log2)

    me.synchronize_between_processes()
    return {k: meter.global_avg for k, meter in me.meters.items()}, me.iter_time.time_preds(max_it - (g_it + 1))


def main_training():
    args: arg_util.Args = arg_util.init_dist_and_get_args()
    tb_lg, trainer, start_ep, start_it, iters_train, ld_train, ld_val = build_everything(args)
    trainer.stage3_opt.log_param(ep=-1, tb_lg=tb_lg)

    best_val = float('inf')
    start_time = time.time()
    for ep in range(start_ep, args.ep):
        tb_lg.set_step(ep * iters_train)
        stats, (_, remain_time, finish_time) = train_one_ep(
            ep, start_it if ep == start_ep else 0, args, tb_lg, ld_train, iters_train, trainer
        )
        print(f'[*][stage3 ep{ep}] train_loss={stats["loss"]:.6f}, remain={remain_time}, finish={finish_time}')
        if (ep + 1) % args.val_and_saving_per_ep == 0 or (ep + 1) == args.ep:
            val_stats = trainer.eval_ep(ld_val)
            print(
                f'[*][stage3 val ep{ep}] loss={val_stats["loss"]:.6f}, '
                f'latent_mse={val_stats["latent_mse"]:.6f}, '
                f'latent_cos={val_stats["latent_cosine"]:.4f}, '
                f'decode_PSNR={val_stats["decode_psnr"]:.2f}, '
                f'decode_SSIM={val_stats["decode_ssim"]:.4f}'
            )
            is_best = val_stats['loss'] < best_val
            best_val = min(best_val, val_stats['loss'])
            save_checkpoint(args, trainer, ep + 1, is_best=is_best)
        args.cur_ep = f'{ep + 1}/{args.ep}'
        args.remain_time, args.finish_time = remain_time, finish_time

    print(f'[*][stage3 finished] total={(time.time() - start_time) / 3600:.2f}h best_val={best_val:.6f}')
    tb_lg.flush()
    tb_lg.close()
    dist.barrier()


if __name__ == '__main__':
    try:
        main_training()
    finally:
        dist.finalize()
        if isinstance(sys.stdout, dist.BackupStreamToFile) and isinstance(sys.stderr, dist.BackupStreamToFile):
            sys.stdout.close(), sys.stderr.close()
