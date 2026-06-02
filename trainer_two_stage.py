"""
Two-stage VAE trainer for LR-HR alignment.

Stage 1:
- train LR VAE only

Stage 2:
- train HR multi-scale VAE
- optional: set stage2_use_kl=False to train deterministic AE (posterior mean, KL=0)
- default: decode the first HR scale through the shared decoder and align it
  with the LR image in image space
- optional legacy path: freeze LR VAE and align the first HR scale token with
  LR posterior mean
"""
import os
import math
from copy import deepcopy
from typing import Callable, Dict, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

import dist
from models import DinoDisc, LR_VAE, VQVAE
from utils import arg_util, misc
from utils.amp_opt import AmpOptimizer
from utils.diffaug import DiffAug
from utils.image_saver import (
    compute_psnr_ssim,
    save_reconstruction_comparison,
    save_reconstruction_run_metadata,
    save_stage2_multiscale_diagnostic,
    save_stage2_scale0_lr_diagnostic,
)
from utils.loss import hinge_loss, linear_loss, softplus_loss
from utils.lpips import LPIPS

FTen = torch.Tensor
BatchInput = Union[FTen, Tuple[FTen, FTen]]


def freeze_model(model: nn.Module):
    for param in model.parameters():
        param.requires_grad_(False)
    model.eval()
    print(f'[Freeze] Froze {type(model).__name__}')


def unfreeze_model(model: nn.Module):
    for param in model.parameters():
        param.requires_grad_(True)
    model.train()
    print(f'[Unfreeze] Unfroze {type(model).__name__}')


class TwoStageVAETrainer(object):
    def __init__(
        self,
        is_visualizer: bool,
        vae: DDP,
        vae_wo_ddp: VQVAE,
        lr_vae: DDP,
        lr_vae_wo_ddp: LR_VAE,
        disc: DDP,
        disc_wo_ddp: DinoDisc,
        vae_opt: AmpOptimizer,
        lr_vae_opt: AmpOptimizer,
        disc_opt: AmpOptimizer,
        ema_ratio: float,
        dcrit: str,
        daug=1.0,
        lpips_loss: LPIPS = None,
        lp_reso=64,
        wei_l1=1.0,
        wei_l2=0.0,
        wei_entropy=0.0,
        wei_lpips=0.5,
        wei_disc=0.6,
        adapt_type=1,
        bcr=5.0,
        bcr_cut=0.5,
        reg=0.0,
        reg_every=16,
        disc_grad_ckpt=False,
        training_stage: int = 1,
        use_alignment_loss: bool = False,
        alignment_loss_type: str = 'scale0_image',
        alignment_loss_weight: float = 1.0,
        alignment_loss_warmup_ep: float = 0.0,
        alignment_scale_index: int = 0,
        stage2_use_kl: bool = True,
        use_stage2_mid_scale_loss: bool = False,
        stage2_mid_scale_indices: Sequence[int] = (1, 2),
        stage2_mid_scale_weights: Sequence[float] = (0.05, 0.05),
        dbg_unused=False,
        dbg_nan=False,
    ):
        super().__init__()
        self.dbg_unused, self.dbg_nan = dbg_unused, dbg_nan

        self.vae = vae
        self.vae_wo_ddp: VQVAE = vae_wo_ddp
        self.lr_vae = lr_vae
        self.lr_vae_wo_ddp: LR_VAE = lr_vae_wo_ddp
        self.disc = disc
        self.disc_wo_ddp: DinoDisc = disc_wo_ddp
        self.disc_params: Tuple[nn.Parameter, ...] = tuple(self.disc_wo_ddp.parameters())

        self.vae_opt = vae_opt
        self.lr_vae_opt = lr_vae_opt
        self.disc_opt = disc_opt

        self.ema_ratio = ema_ratio
        self.is_visualizer = is_visualizer
        self.using_ema = is_visualizer
        if self.using_ema:
            self.vae_ema: VQVAE = deepcopy(vae_wo_ddp).eval()
            self.lr_vae_ema: LR_VAE = deepcopy(lr_vae_wo_ddp).eval()
        else:
            self.vae_ema = None
            self.lr_vae_ema = None

        self.dcrit = dcrit
        self.d_criterion: Callable = {
            'hg': hinge_loss, 'hinge': hinge_loss,
            'sp': softplus_loss, 'softplus': softplus_loss,
            'ln': linear_loss, 'lin': linear_loss, 'linear': linear_loss,
        }[dcrit]

        self.daug = DiffAug(prob=daug, cutout=0.2)
        self.wei_l1, self.wei_l2, self.wei_entropy = wei_l1, wei_l2, wei_entropy
        self.lpips_loss: LPIPS = lpips_loss
        self.lp_reso = lp_reso
        self.adapt_wei_disc = wei_disc > 0
        self.adapt_type = adapt_type
        self.wei_lpips, self.wei_disc = wei_lpips * 2, abs(wei_disc)
        self.ema_gada: Optional[torch.Tensor] = None

        self.reg = 0.5 * reg * reg_every
        self.bcr = bcr * 2
        if self.bcr > 0:
            self.bcr_strong_aug = DiffAug(prob=1, cutout=bcr_cut)
        self.disc_grad_ckpt = disc_grad_ckpt

        self.training_stage = training_stage
        self.use_alignment_loss = use_alignment_loss
        self.alignment_loss_type = self._normalize_alignment_loss_type(alignment_loss_type)
        self.alignment_loss_weight = alignment_loss_weight
        self.alignment_loss_warmup_ep = max(float(alignment_loss_warmup_ep), 0.0)
        self.alignment_scale_index = alignment_scale_index
        self.stage2_use_kl = self._normalize_bool(stage2_use_kl)
        self.configure_stage2_mid_scale_loss(
            use_stage2_mid_scale_loss,
            stage2_mid_scale_indices,
            stage2_mid_scale_weights,
        )

        self.usage_max = 0.0
        self._run_metadata_written = False
        self._lr_posterior_log_printed = 0

        self.set_training_stage(training_stage)

    def _get_reconstruction_save_dir(self, args: arg_util.Args) -> str:
        default_dir_name = 'reconstruction_samples_lr' if self.training_stage == 1 else 'reconstruction_samples_hr'
        dir_name = args.reconstruction_dir_name.strip() or default_dir_name
        return os.path.join(args.local_out_dir_path, dir_name)

    def _get_diagnostic_save_dir(self, args: arg_util.Args) -> str:
        return os.path.join(args.local_out_dir_path, 'diagnostic')

    def _should_save_reconstruction(self, it: int, metric_lg: misc.MetricLogger, args: arg_util.Args) -> bool:
        interval = max(int(getattr(args, 'reconstruction_save_interval', 0)), 0)
        if interval > 0:
            return it % interval == 0
        return it == 0 or it in metric_lg.log_iters

    def _record_reconstruction_metadata(self, save_dir: str, args: arg_util.Args) -> None:
        if not getattr(args, 'record_reconstruction_metadata', True):
            return
        if self._run_metadata_written:
            return

        interval = max(int(getattr(args, 'reconstruction_save_interval', 0)), 0)
        if interval > 0:
            frequency_description = f'save one comparison image every {interval} training iterations'
        else:
            frequency_description = 'save on legacy log iterations (evenly spaced log points within each epoch, plus iteration 0)'

        stage_name = 'stage1_lr_vae' if self.training_stage == 1 else 'stage2_hr_vae'
        save_reconstruction_run_metadata(
            save_dir=save_dir,
            metadata_dir=args.local_out_dir_path,
            args_state=args.state_dict(key_ordered=False),
            stage_name=stage_name,
            frequency_description=frequency_description,
            max_samples=getattr(args, 'reconstruction_max_samples', 4),
        )
        self._run_metadata_written = True

    def _assert_finite(self, name: str, value, ep: int, it: int, stage: str) -> None:
        """Fail fast on NaN/Inf when dbg_nan is enabled."""
        if not self.dbg_nan:
            return

        if torch.is_tensor(value):
            tensor = value.detach()
            if torch.isfinite(tensor).all():
                return
            nan_count = torch.isnan(tensor).sum().item()
            posinf_count = torch.isposinf(tensor).sum().item()
            neginf_count = torch.isneginf(tensor).sum().item()
            total = tensor.numel()
            raise RuntimeError(
                f'[NaN Debug][{stage}] [ep{ep}] [it{it}] `{name}` is non-finite: '
                f'nan={nan_count}, +inf={posinf_count}, -inf={neginf_count}, total={total}'
            )

        scalar = float(value)
        if math.isfinite(scalar):
            return
        raise RuntimeError(
            f'[NaN Debug][{stage}] [ep{ep}] [it{it}] `{name}` is non-finite: {scalar}'
        )

    def _assert_finite(self, name: str, value, ep: int, it: int, stage: str) -> None:
        """Fail fast on NaN/Inf when dbg_nan is enabled."""
        if not self.dbg_nan:
            return

        if torch.is_tensor(value):
            tensor = value.detach()
            finite_mask = torch.isfinite(tensor)
            if finite_mask.all():
                return
            nan_count = torch.isnan(tensor).sum().item()
            posinf_count = torch.isposinf(tensor).sum().item()
            neginf_count = torch.isneginf(tensor).sum().item()
            total = tensor.numel()
            raise RuntimeError(
                f'[NaN Debug][{stage}] [ep{ep}] [it{it}] `{name}` is non-finite: '
                f'nan={nan_count}, +inf={posinf_count}, -inf={neginf_count}, total={total}'
            )

        scalar = float(value)
        if math.isfinite(scalar):
            return
        raise RuntimeError(
            f'[NaN Debug][{stage}] [ep{ep}] [it{it}] `{name}` is non-finite: {scalar}'
        )

    def _compute_adaptive_weight(self, nll_loss: torch.Tensor, g_loss: torch.Tensor, last_layer_weight: torch.Tensor) -> torch.Tensor:
        nll_grads = torch.autograd.grad(nll_loss, last_layer_weight, retain_graph=True)[0]
        g_grads = torch.autograd.grad(g_loss, last_layer_weight, retain_graph=True)[0]

        d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-7)
        d_weight = torch.nan_to_num(d_weight, nan=0.0, posinf=1e4, neginf=0.0)
        d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()

        if self.ema_gada is None or not torch.isfinite(self.ema_gada).all().item():
            self.ema_gada = d_weight
        else:
            self.ema_gada = self.ema_gada * 0.9 + d_weight * 0.1
        self.ema_gada = torch.nan_to_num(self.ema_gada, nan=0.0, posinf=1e4, neginf=0.0)
        return self.ema_gada * self.wei_disc

    @staticmethod
    def _as_rgb_for_pretrained(img: torch.Tensor) -> torch.Tensor:
        if img.shape[1] == 1:
            return img.repeat(1, 3, 1, 1)
        return img

    def _disc_input_pm1(self, img: torch.Tensor) -> torch.Tensor:
        return self._as_rgb_for_pretrained(img).float().clamp(-1.0, 1.0)

    def _ema_update(self, ema_model: nn.Module, model: nn.Module):
        with torch.no_grad():
            for ema_param, param in zip(ema_model.parameters(), model.parameters()):
                ema_param.data.mul_(self.ema_ratio).add_(param.data, alpha=1 - self.ema_ratio)

    def _disc_forward(self, real_img: torch.Tensor, fake_img: torch.Tensor, fade_blur_schedule: float):
        real_aug = self.daug.aug(self._disc_input_pm1(real_img), fade_blur_schedule)
        fake_aug = self.daug.aug(self._disc_input_pm1(fake_img), fade_blur_schedule)
        return self.disc(torch.cat((real_aug, fake_aug), dim=0)).split([real_img.shape[0], fake_img.shape[0]], dim=0)

    def _generator_adv_loss(self, fake_img: torch.Tensor, fade_blur_schedule: float) -> torch.Tensor:
        disc_training = self.disc_wo_ddp.training
        requires_grad_states = tuple(param.requires_grad for param in self.disc_params)
        try:
            for param in self.disc_params:
                param.requires_grad_(False)
            self.disc_wo_ddp.eval()
            fake_logits = self.disc_wo_ddp(
                self.daug.aug(self._disc_input_pm1(fake_img), fade_blur_schedule),
                grad_ckpt=self.disc_grad_ckpt,
            )
        finally:
            for param, requires_grad in zip(self.disc_params, requires_grad_states):
                param.requires_grad_(requires_grad)
            self.disc_wo_ddp.train(disc_training)
        return self.d_criterion(is_real_pred=True, logits=fake_logits, for_g=True)

    @torch.no_grad()
    def _deterministic_reconstruction(
        self,
        model: nn.Module,
        inp: torch.Tensor,
        use_kl: Optional[bool] = None,
    ) -> torch.Tensor:
        """Run a forward pass in eval mode so posterior uses its mean (no sampling noise).

        This is what `eval_ep` uses, and it's what the saved comparison images should show —
        otherwise early-training snapshots are dominated by latent sampling noise.
        """
        was_training = model.training
        model.eval()
        try:
            if use_kl is None:
                rec, _, _ = model(inp)
            else:
                rec, _, _ = model(inp, use_kl=use_kl)
        finally:
            model.train(was_training)
        return rec.detach()

    @torch.no_grad()
    def _log_lr_posterior_stats(self, inp_lr: torch.Tensor, ep: int, it: int,
                                 kl_weight: float, Lrec_for_log: torch.Tensor,
                                 Lpip: torch.Tensor) -> None:
        """Print posterior-mean magnitude, posterior std, per-dim KL. Rate-limited."""
        if self._lr_posterior_log_printed >= 0 and (it == 0 or self._lr_posterior_log_printed < 1e9):
            pass
        mean, logvar = self.lr_vae_wo_ddp.encode_to_posterior_stats(inp_lr)
        logvar = logvar.clamp(-10.0, 5.0)
        std = (0.5 * logvar).exp()
        kl_per_dim = 0.5 * (mean.pow(2) + logvar.exp() - 1.0 - logvar)
        abs_mean = mean.abs().mean().item()
        std_mean = std.mean().item()
        kl_pd = kl_per_dim.mean().item()
        kl_total = kl_per_dim.sum(dim=[1, 2, 3]).mean().item()
        nll_scalar = Lrec_for_log.item() + Lpip.item()
        print(
            f'[LR Posterior][ep{ep}][it{it}] |mean|={abs_mean:.4f}  std={std_mean:.4f}  '
            f'kl_per_dim={kl_pd:.4f}  kl_sum={kl_total:.4f}  kl_w={kl_weight:.4f}  '
            f'Lkl_effective={kl_total * kl_weight:.4f}  Lnll≈{nll_scalar:.4f}',
            flush=True,
        )
        self._lr_posterior_log_printed += 1

    @staticmethod
    def _normalize_bool(value) -> bool:
        if isinstance(value, str):
            return value.strip().lower() not in {'0', 'false', 'no', 'off'}
        return bool(value)

    def configure_stage2_mid_scale_loss(
        self,
        use_stage2_mid_scale_loss,
        stage2_mid_scale_indices: Sequence[int],
        stage2_mid_scale_weights: Sequence[float],
    ) -> None:
        self.stage2_mid_scale_indices = tuple(int(i) for i in stage2_mid_scale_indices)
        self.stage2_mid_scale_weights = tuple(float(w) for w in stage2_mid_scale_weights)
        if len(self.stage2_mid_scale_indices) != len(self.stage2_mid_scale_weights):
            raise ValueError(
                'stage2_mid_scale_indices and stage2_mid_scale_weights must have the same length: '
                f'{self.stage2_mid_scale_indices=} vs {self.stage2_mid_scale_weights=}'
            )
        self.use_stage2_mid_scale_loss = (
            self._normalize_bool(use_stage2_mid_scale_loss)
            and sum(self.stage2_mid_scale_weights) > 0
        )

    @staticmethod
    def _normalize_alignment_loss_type(loss_type: str) -> str:
        key = str(loss_type or 'scale0_image').strip().lower().replace('-', '_')
        aliases = {
            'scale0_image': 'scale0_image',
            'scale0_img': 'scale0_image',
            'image': 'scale0_image',
            'img': 'scale0_image',
            'latent': 'latent',
            'lr_latent': 'latent',
            'stage1_latent': 'latent',
        }
        if key not in aliases:
            raise ValueError(
                f'Invalid alignment_loss_type={loss_type!r}. '
                'Expected scale0_image or latent.'
            )
        return aliases[key]

    @staticmethod
    def _resize_image_like(src: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        if src.shape[-2:] == ref.shape[-2:]:
            return src
        return F.interpolate(src, size=ref.shape[-2:], mode='bicubic', align_corners=False)

    def _alignment_weight_multiplier(self, ep: int, it: int, args: arg_util.Args) -> float:
        warmup_ep = self.alignment_loss_warmup_ep
        if warmup_ep <= 0:
            return 1.0

        iters_per_ep = max(int(getattr(args, 'iters_per_ep', 0) or 0), 0)
        if iters_per_ep > 0:
            cur = ep * iters_per_ep + it + 1
            total = max(warmup_ep * iters_per_ep, 1.0)
            return min(1.0, cur / total)
        return min(1.0, (ep + 1) / warmup_ep)

    def _get_alignment_targets(
        self,
        inp_lr: torch.Tensor,
        hr_mean: torch.Tensor,
        lr_model: Optional[LR_VAE] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        lr_model = lr_model or self.lr_vae_wo_ddp
        with torch.no_grad():
            lr_mean = lr_model.encode_to_posterior_mean(inp_lr)

        if hr_mean.shape != lr_mean.shape:
            raise ValueError(
                'Alignment shape mismatch: '
                f'HR first-scale mean {tuple(hr_mean.shape)} vs LR posterior mean {tuple(lr_mean.shape)}. '
                f'Expected args.patch_nums[{self.alignment_scale_index}] to match LR latent size '
                f'(usually lr_img_size / 16).'
            )
        return lr_mean, hr_mean

    def _compute_alignment_loss(
        self,
        inp_lr: torch.Tensor,
        hr_scale_latent: torch.Tensor,
        vae_model: Optional[VQVAE] = None,
        lr_model: Optional[LR_VAE] = None,
    ) -> torch.Tensor:
        if self.alignment_loss_type == 'latent':
            lr_mean, hr_mean = self._get_alignment_targets(inp_lr, hr_scale_latent, lr_model=lr_model)
            return F.mse_loss(hr_mean, lr_mean)

        vae_model = vae_model or self.vae_wo_ddp
        scale0_img = vae_model.scale_latent_to_img(
            hr_scale_latent,
            scale_index=self.alignment_scale_index,
            clamp=False,
        )
        lr_img_target = self._resize_image_like(inp_lr, scale0_img).detach()
        return F.l1_loss(scale0_img, lr_img_target)

    @staticmethod
    def _bandlimited_hr_target(inp_hr: torch.Tensor, patch_num: int, max_patch_num: int) -> torch.Tensor:
        """Downsample HR to the frequency band of one cumulative scale, then upsample back."""
        h, w = inp_hr.shape[-2:]
        size = max(1, int(round(min(h, w) * patch_num / max_patch_num)))
        down = F.interpolate(inp_hr, size=(size, size), mode='area')
        return F.interpolate(down, size=(h, w), mode='bicubic', align_corners=False)

    def _compute_mid_scale_loss(
        self,
        inp_hr: torch.Tensor,
        mid_scale_recs: Dict[int, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[int, torch.Tensor]]:
        """Return weighted total and per-scale-index unweighted L1 losses."""
        patch_nums = tuple(self.vae_wo_ddp.quantize.v_patch_nums)
        max_patch_num = patch_nums[-1]
        loss = inp_hr.new_zeros(())
        per_scale: Dict[int, torch.Tensor] = {}
        for si, weight in zip(self.stage2_mid_scale_indices, self.stage2_mid_scale_weights):
            if weight <= 0:
                continue
            if si < 0 or si >= len(patch_nums):
                raise IndexError(f'{si=} out of range for patch_nums={patch_nums}')
            if si not in mid_scale_recs:
                raise KeyError(f'missing cumulative reconstruction for scale index {si}')
            target = self._bandlimited_hr_target(inp_hr, patch_nums[si], max_patch_num).detach()
            l1 = F.l1_loss(mid_scale_recs[si], target)
            per_scale[si] = l1
            loss = loss + weight * l1
        return loss, per_scale

    def _forward_hr_vae_stage2(
        self,
        inp_hr: torch.Tensor,
        *,
        ret_usages: bool,
        use_kl: bool,
    ) -> Tuple[torch.Tensor, Optional[list], torch.Tensor, Optional[torch.Tensor], Optional[Dict[int, torch.Tensor]]]:
        need_align = self.use_alignment_loss
        need_mid = self.use_stage2_mid_scale_loss
        out = self.vae(
            inp_hr,
            ret_usages=ret_usages,
            ret_scale_posterior_stats=need_align,
            scale_index=self.alignment_scale_index,
            use_kl=use_kl,
            ret_mid_scale_recs=need_mid,
            mid_scale_indices=self.stage2_mid_scale_indices,
        )
        hr_scale_latent = None
        mid_scale_recs = None
        if need_align and need_mid:
            rec_B3HW, usage, Lkl, hr_scale_latent, _, mid_scale_recs = out
        elif need_align:
            rec_B3HW, usage, Lkl, hr_scale_latent, _ = out
        elif need_mid:
            rec_B3HW, usage, Lkl, mid_scale_recs = out
        else:
            rec_B3HW, usage, Lkl = out
        return rec_B3HW, usage, Lkl, hr_scale_latent, mid_scale_recs

    @torch.no_grad()
    def _save_stage2_scale0_diagnostic(
        self,
        inp_lr: torch.Tensor,
        hr_scale_latent: torch.Tensor,
        save_dir: str,
        ep: int,
        it: int,
        max_samples: int,
    ) -> None:
        scale0_img = self.vae_wo_ddp.scale_latent_to_img(
            hr_scale_latent.detach(),
            scale_index=self.alignment_scale_index,
            clamp=True,
        )
        save_stage2_scale0_lr_diagnostic(
            lr=inp_lr,
            decode_scale0_img=scale0_img,
            save_dir=save_dir,
            ep=ep,
            it=it,
            max_samples=max_samples,
        )

    @torch.no_grad()
    def _cumulative_scale_reconstructions(self, inp_hr: FTen) -> list:
        """HR images after each cumulative multi-scale latent (eval / posterior mean)."""
        vae = self.vae_wo_ddp
        was_training = vae.training
        vae.eval()
        try:
            return vae.img_to_reconstructed_img(inp_hr, last_one=False)
        finally:
            vae.train(was_training)

    @torch.no_grad()
    def _stage2_mid_scale_targets(self, inp_hr: FTen) -> Dict[int, FTen]:
        """Band-limited targets matching the configured mid-scale supervision."""
        if not self.use_stage2_mid_scale_loss:
            return {}

        patch_nums = tuple(self.vae_wo_ddp.quantize.v_patch_nums)
        max_patch_num = patch_nums[-1]
        targets: Dict[int, FTen] = {}
        for si, weight in zip(self.stage2_mid_scale_indices, self.stage2_mid_scale_weights):
            if weight <= 0:
                continue
            if si < 0 or si >= len(patch_nums):
                raise IndexError(f'{si=} out of range for patch_nums={patch_nums}')
            targets[si] = self._bandlimited_hr_target(inp_hr, patch_nums[si], max_patch_num)
        return targets

    @torch.no_grad()
    def _save_stage2_multiscale_diagnostic(
        self,
        inp_lr: FTen,
        inp_hr: FTen,
        save_dir: str,
        ep: int,
        it: int,
        max_samples: int,
    ) -> None:
        hr_by_scale = self._cumulative_scale_reconstructions(inp_hr)
        hr_targets_by_scale = self._stage2_mid_scale_targets(inp_hr)
        save_stage2_multiscale_diagnostic(
            lr=inp_lr,
            hr_by_scale=hr_by_scale,
            hr_gt=inp_hr,
            save_dir=save_dir,
            ep=ep,
            it=it,
            max_samples=max_samples,
            hr_targets_by_scale=hr_targets_by_scale,
        )

    def train_step_stage1(
        self,
        ep: int,
        it: int,
        g_it: int,
        stepping: bool,
        regularizing: bool,
        metric_lg: misc.MetricLogger,
        logging_params: bool,
        tb_lg: misc.TensorboardLogger,
        inp_lr: FTen,
        warmup_disc_schedule: float,
        fade_blur_schedule: float,
        maybe_record_function: Callable,
        args: arg_util.Args,
    ) -> Tuple[Optional[torch.Tensor], Optional[float], Optional[torch.Tensor], Optional[float]]:
        del g_it, regularizing, logging_params
        loggable = tb_lg.loggable() and stepping
        if warmup_disc_schedule < 1e-6:
            warmup_disc_schedule = 0.0

        iters_per_ep = max(int(getattr(args, 'iters_per_ep', 0) or 0), 0)
        warmup_ep = max(float(getattr(args, 'lr_kl_warmup_ep', 0.0) or 0.0), 0.0)
        if warmup_ep > 0 and iters_per_ep > 0:
            kl_schedule = min(1.0, (ep * iters_per_ep + it) / max(warmup_ep * iters_per_ep, 1.0))
        else:
            kl_schedule = 1.0

        with maybe_record_function('LR_VAE_rec'):
            with self.lr_vae_opt.amp_ctx:
                rec_B3HW, _, Lkl = self.lr_vae(inp_lr)
                self._assert_finite('rec_B3HW', rec_B3HW, ep, it, 'stage1')
                self._assert_finite('Lkl', Lkl, ep, it, 'stage1')
                B = rec_B3HW.shape[0]
                inp_rec_no_grad = torch.cat((inp_lr, rec_B3HW.detach()), dim=0)

                Lrec = F.l1_loss(rec_B3HW, inp_lr)
                Lrec_for_log = Lrec.detach().clone()
                Lrec = Lrec * self.wei_l1
                if self.wei_l2 > 0:
                    Lrec = Lrec + F.mse_loss(rec_B3HW, inp_lr) * self.wei_l2

                using_lpips = inp_lr.shape[-2] >= self.lp_reso and self.wei_lpips > 0
                if using_lpips:
                    Lpip = torch.mean(self.lpips_loss(
                        self._as_rgb_for_pretrained(inp_lr),
                        self._as_rgb_for_pretrained(rec_B3HW),
                    ))
                    Lnll = Lrec + self.wei_lpips * Lpip
                else:
                    Lpip = inp_lr.new_zeros(())
                    Lnll = Lrec

                Lkl_effective = Lkl * kl_schedule
                Lg = Lnll + Lkl_effective
                self._assert_finite('Lnll', Lnll, ep, it, 'stage1')
                self._assert_finite('Lg_pre_adv', Lg, ep, it, 'stage1')

        if warmup_disc_schedule > 0:
            with maybe_record_function('LR_disc'):
                with self.disc_opt.amp_ctx:
                    inp_d_logits, rec_d_logits = self._disc_forward(inp_rec_no_grad[:B], inp_rec_no_grad[B:], fade_blur_schedule)
                    self._assert_finite('inp_d_logits', inp_d_logits, ep, it, 'stage1')
                    self._assert_finite('rec_d_logits', rec_d_logits, ep, it, 'stage1')

            acc_real = (inp_d_logits > 0).float().mean().item()
            acc_fake = (rec_d_logits < 0).float().mean().item()
            Ld_real = self.d_criterion(is_real_pred=True, logits=inp_d_logits)
            Ld_fake = self.d_criterion(is_real_pred=False, logits=rec_d_logits)
            Ld = Ld_real + Ld_fake
            self._assert_finite('Ld', Ld, ep, it, 'stage1')
            with maybe_record_function('LR_VAE_disc'):
                with self.disc_opt.amp_ctx:
                    Lg_adv = self._generator_adv_loss(rec_B3HW, fade_blur_schedule)
                    self._assert_finite('Lg_adv', Lg_adv, ep, it, 'stage1')

            if self.adapt_wei_disc:
                wei_g = self._compute_adaptive_weight(Lnll, Lg_adv, self.lr_vae_wo_ddp.decoder.conv_out.weight)
            else:
                wei_g = inp_lr.new_tensor(self.wei_disc)
            self._assert_finite('wei_g', wei_g, ep, it, 'stage1')
            Lg = Lg + wei_g * Lg_adv * warmup_disc_schedule
            self._assert_finite('Lg', Lg, ep, it, 'stage1')
            grad_norm_d, scale_log2_d = self.disc_opt.backward_clip_step(stepping=stepping, loss=Ld)
        else:
            Ld = inp_lr.new_zeros(())
            wei_g = inp_lr.new_zeros(())
            acc_real = 0.0
            acc_fake = 0.0
            grad_norm_d, scale_log2_d = 0.0, None

        grad_norm_g, scale_log2_g = self.lr_vae_opt.backward_clip_step(stepping=stepping, loss=Lg)

        if self.using_ema and stepping:
            self._ema_update(self.lr_vae_ema, self.lr_vae_wo_ddp)

        if it == 0 or it in metric_lg.log_iters:
            metric_lg.update(
                L1=Lrec_for_log.item(),
                NLL=Lrec_for_log.item() + Lpip.item(),
                Lkl=Lkl.item() if isinstance(Lkl, torch.Tensor) else Lkl,
                Ld=Ld.item(),
                Wg=wei_g.item() if hasattr(wei_g, 'item') else wei_g,
                acc_real=acc_real,
                acc_fake=acc_fake,
                gnm=grad_norm_g,
                dnm=grad_norm_d,
                usage=0.0,
            )
            if self._lr_posterior_log_printed < getattr(args, 'lr_posterior_log_limit', 0):
                self._log_lr_posterior_stats(inp_lr, ep, it, kl_schedule, Lrec_for_log, Lpip)
            if args.save_reconstruction_images and dist.is_master() and self._should_save_reconstruction(it, metric_lg, args):
                try:
                    save_dir = self._get_reconstruction_save_dir(args)
                    self._record_reconstruction_metadata(save_dir, args)
                    rec_for_vis = self._deterministic_reconstruction(self.lr_vae_wo_ddp, inp_lr)
                    save_reconstruction_comparison(
                        original=inp_lr,
                        reconstructed=rec_for_vis,
                        save_dir=save_dir,
                        ep=ep,
                        it=it,
                        max_samples=args.reconstruction_max_samples,
                    )
                except Exception as e:
                    print(f'[Warning] Failed to save LR reconstruction images: {e}', flush=True)

        return grad_norm_g, scale_log2_g, grad_norm_d, scale_log2_d

    def train_step_stage2(
        self,
        ep: int,
        it: int,
        g_it: int,
        stepping: bool,
        regularizing: bool,
        metric_lg: misc.MetricLogger,
        logging_params: bool,
        tb_lg: misc.TensorboardLogger,
        inp_lr: FTen,
        inp_hr: FTen,
        warmup_disc_schedule: float,
        fade_blur_schedule: float,
        maybe_record_function: Callable,
        args: arg_util.Args,
    ) -> Tuple[Optional[torch.Tensor], Optional[float], Optional[torch.Tensor], Optional[float]]:
        del g_it, regularizing, logging_params
        loggable = tb_lg.loggable() and stepping
        if warmup_disc_schedule < 1e-6:
            warmup_disc_schedule = 0.0

        with maybe_record_function('HR_VAE_rec'):
            with self.vae_opt.amp_ctx:
                rec_B3HW, usage, Lkl, hr_scale_latent, mid_scale_recs = self._forward_hr_vae_stage2(
                    inp_hr,
                    ret_usages=loggable,
                    use_kl=self.stage2_use_kl,
                )
                self._assert_finite('rec_B3HW', rec_B3HW, ep, it, 'stage2')
                self._assert_finite('Lkl', Lkl, ep, it, 'stage2')
                if loggable and usage is not None:
                    self.usage_max = max(self.usage_max, max(usage))
                B = rec_B3HW.shape[0]
                inp_rec_no_grad = torch.cat((inp_hr, rec_B3HW.detach()), dim=0)

                Lrec = F.l1_loss(rec_B3HW, inp_hr)
                Lrec_for_log = Lrec.detach().clone()
                Lrec = Lrec * self.wei_l1
                if self.wei_l2 > 0:
                    Lrec = Lrec + F.mse_loss(rec_B3HW, inp_hr) * self.wei_l2

                using_lpips = inp_hr.shape[-2] >= self.lp_reso and self.wei_lpips > 0
                if using_lpips:
                    Lpip = torch.mean(self.lpips_loss(
                        self._as_rgb_for_pretrained(inp_hr),
                        self._as_rgb_for_pretrained(rec_B3HW),
                    ))
                    Lnll = Lrec + self.wei_lpips * Lpip
                else:
                    Lpip = inp_hr.new_zeros(())
                    Lnll = Lrec

                if self.use_alignment_loss:
                    L_align = self._compute_alignment_loss(inp_lr, hr_scale_latent)
                    align_weight_mult = self._alignment_weight_multiplier(ep, it, args)
                else:
                    L_align = inp_hr.new_zeros(())
                    align_weight_mult = 0.0

                effective_align_weight = self.alignment_loss_weight * align_weight_mult
                mid_scale_per_si: Dict[int, torch.Tensor] = {}
                if self.use_stage2_mid_scale_loss and mid_scale_recs is not None:
                    L_mid, mid_scale_per_si = self._compute_mid_scale_loss(inp_hr, mid_scale_recs)
                else:
                    L_mid = inp_hr.new_zeros(())
                Lg = Lnll + Lkl + effective_align_weight * L_align + L_mid
                self._assert_finite('Lnll', Lnll, ep, it, 'stage2')
                self._assert_finite('L_align', L_align, ep, it, 'stage2')
                self._assert_finite('L_mid', L_mid, ep, it, 'stage2')
                self._assert_finite('Lg_pre_adv', Lg, ep, it, 'stage2')

        if warmup_disc_schedule > 0:
            with maybe_record_function('HR_disc'):
                with self.disc_opt.amp_ctx:
                    inp_d_logits, rec_d_logits = self._disc_forward(inp_rec_no_grad[:B], inp_rec_no_grad[B:], fade_blur_schedule)
                    self._assert_finite('inp_d_logits', inp_d_logits, ep, it, 'stage2')
                    self._assert_finite('rec_d_logits', rec_d_logits, ep, it, 'stage2')

            acc_real = (inp_d_logits > 0).float().mean().item()
            acc_fake = (rec_d_logits < 0).float().mean().item()
            Ld_real = self.d_criterion(is_real_pred=True, logits=inp_d_logits)
            Ld_fake = self.d_criterion(is_real_pred=False, logits=rec_d_logits)
            Ld = Ld_real + Ld_fake
            self._assert_finite('Ld', Ld, ep, it, 'stage2')
            with maybe_record_function('HR_VAE_disc'):
                with self.disc_opt.amp_ctx:
                    Lg_adv = self._generator_adv_loss(rec_B3HW, fade_blur_schedule)
                    self._assert_finite('Lg_adv', Lg_adv, ep, it, 'stage2')

            if self.adapt_wei_disc:
                wei_g = self._compute_adaptive_weight(Lnll, Lg_adv, self.vae_wo_ddp.decoder.conv_out.weight)
            else:
                wei_g = inp_hr.new_tensor(self.wei_disc)
            self._assert_finite('wei_g', wei_g, ep, it, 'stage2')
            Lg = Lg + wei_g * Lg_adv * warmup_disc_schedule
            self._assert_finite('Lg', Lg, ep, it, 'stage2')
            grad_norm_d, scale_log2_d = self.disc_opt.backward_clip_step(stepping=stepping, loss=Ld)
        else:
            Ld = inp_hr.new_zeros(())
            wei_g = inp_hr.new_zeros(())
            acc_real = 0.0
            acc_fake = 0.0
            grad_norm_d, scale_log2_d = 0.0, None

        grad_norm_g, scale_log2_g = self.vae_opt.backward_clip_step(stepping=stepping, loss=Lg)

        if self.using_ema and stepping:
            self._ema_update(self.vae_ema, self.vae_wo_ddp)

        if it == 0 or it in metric_lg.log_iters:
            log_kw: Dict[str, float] = dict(
                L1=Lrec_for_log.item(),
                NLL=Lrec_for_log.item() + Lpip.item(),
                Lkl=Lkl.item() if isinstance(Lkl, torch.Tensor) else Lkl,
                Ld=Ld.item(),
                Wg=wei_g.item() if hasattr(wei_g, 'item') else wei_g,
                acc_real=acc_real,
                acc_fake=acc_fake,
                gnm=grad_norm_g,
                dnm=grad_norm_d,
                usage=self.usage_max,
                L_align=L_align.item(),
                W_align=effective_align_weight,
                L_mid=L_mid.item(),
            )
            if self.use_stage2_mid_scale_loss and mid_scale_per_si:
                patch_nums = tuple(self.vae_wo_ddp.quantize.v_patch_nums)
                weight_by_si = dict(zip(self.stage2_mid_scale_indices, self.stage2_mid_scale_weights))
                for si, l1 in mid_scale_per_si.items():
                    pn = patch_nums[si]
                    log_kw[f'L_mid_pn{pn}'] = l1.item()
                    log_kw[f'L_midw_pn{pn}'] = (weight_by_si[si] * l1).item()
            metric_lg.update(**log_kw)
            should_save_vis = (
                args.save_reconstruction_images
                and dist.is_master()
                and self._should_save_reconstruction(it, metric_lg, args)
            )
            if should_save_vis:
                try:
                    save_dir = self._get_reconstruction_save_dir(args)
                    self._record_reconstruction_metadata(save_dir, args)
                    rec_for_vis = self._deterministic_reconstruction(
                        self.vae_wo_ddp,
                        inp_hr,
                        use_kl=self.stage2_use_kl,
                    )
                    save_reconstruction_comparison(
                        original=inp_hr,
                        reconstructed=rec_for_vis,
                        save_dir=save_dir,
                        ep=ep,
                        it=it,
                        max_samples=args.reconstruction_max_samples,
                    )
                except Exception as e:
                    print(f'[Warning] Failed to save HR reconstruction images: {e}', flush=True)
                try:
                    self._save_stage2_multiscale_diagnostic(
                        inp_lr=inp_lr,
                        inp_hr=inp_hr,
                        save_dir=self._get_diagnostic_save_dir(args),
                        ep=ep,
                        it=it,
                        max_samples=args.reconstruction_max_samples,
                    )
                except Exception as e:
                    print(f'[Warning] Failed to save stage2 multiscale diagnostic images: {e}', flush=True)
            if (
                should_save_vis
                and self.use_alignment_loss
                and self.alignment_loss_type == 'scale0_image'
                and hr_scale_latent is not None
            ):
                try:
                    self._save_stage2_scale0_diagnostic(
                        inp_lr=inp_lr,
                        hr_scale_latent=hr_scale_latent,
                        save_dir=self._get_diagnostic_save_dir(args),
                        ep=ep,
                        it=it,
                        max_samples=args.reconstruction_max_samples,
                    )
                except Exception as e:
                    print(f'[Warning] Failed to save stage2 scale0 LR diagnostic images: {e}', flush=True)

        return grad_norm_g, scale_log2_g, grad_norm_d, scale_log2_d

    def train_step(
        self,
        ep: int,
        it: int,
        g_it: int,
        stepping: bool,
        regularizing: bool,
        metric_lg: misc.MetricLogger,
        logging_params: bool,
        tb_lg: misc.TensorboardLogger,
        inp: BatchInput,
        warmup_disc_schedule: float,
        fade_blur_schedule: float,
        maybe_record_function: Callable,
        args: arg_util.Args,
    ) -> Tuple[Optional[torch.Tensor], Optional[float], Optional[torch.Tensor], Optional[float]]:
        if self.training_stage == 1:
            if isinstance(inp, tuple):
                inp = inp[0]
            return self.train_step_stage1(
                ep=ep,
                it=it,
                g_it=g_it,
                stepping=stepping,
                regularizing=regularizing,
                metric_lg=metric_lg,
                logging_params=logging_params,
                tb_lg=tb_lg,
                inp_lr=inp,
                warmup_disc_schedule=warmup_disc_schedule,
                fade_blur_schedule=fade_blur_schedule,
                maybe_record_function=maybe_record_function,
                args=args,
            )

        if not isinstance(inp, tuple) or len(inp) != 2:
            raise TypeError('Stage 2 expects a batch of (inp_lr, inp_hr).')
        inp_lr, inp_hr = inp
        return self.train_step_stage2(
            ep=ep,
            it=it,
            g_it=g_it,
            stepping=stepping,
            regularizing=regularizing,
            metric_lg=metric_lg,
            logging_params=logging_params,
            tb_lg=tb_lg,
            inp_lr=inp_lr,
            inp_hr=inp_hr,
            warmup_disc_schedule=warmup_disc_schedule,
            fade_blur_schedule=fade_blur_schedule,
            maybe_record_function=maybe_record_function,
            args=args,
        )

    @torch.no_grad()
    def eval_ep(self, ld_val, max_batches=None):
        tot = 0
        rec_loss = 0.0
        psnr_sum = 0.0
        ssim_sum = 0.0
        align_loss_sum = 0.0

        if self.training_stage == 1:
            eval_model = self.lr_vae_ema if self.using_ema else self.lr_vae_wo_ddp
            eval_model.eval()
        else:
            eval_model = self.vae_ema if self.using_ema else self.vae_wo_ddp
            lr_eval_model = self.lr_vae_ema if self.using_ema else self.lr_vae_wo_ddp
            eval_model.eval()
            lr_eval_model.eval()

        for i, batch in enumerate(ld_val):
            if max_batches is not None and i >= max_batches:
                break

            if self.training_stage == 1:
                inp = batch.to(next(eval_model.parameters()).device)
                rec, _, _ = eval_model(inp)
            else:
                inp_lr, inp_hr = batch
                inp_lr = inp_lr.to(next(lr_eval_model.parameters()).device)
                inp_hr = inp_hr.to(next(eval_model.parameters()).device)
                inp = inp_hr

                if self.use_alignment_loss:
                    rec, _, _, hr_scale_latent, _ = eval_model.forward_with_scale_posterior_stats(
                        inp_hr,
                        scale_index=self.alignment_scale_index,
                        use_kl=self.stage2_use_kl,
                    )
                    L_align_eval = self._compute_alignment_loss(
                        inp_lr,
                        hr_scale_latent,
                        vae_model=eval_model,
                        lr_model=lr_eval_model,
                    )
                    align_loss_sum += L_align_eval.item() * inp.shape[0]
                else:
                    rec, _, _ = eval_model(inp_hr, use_kl=self.stage2_use_kl)

            rec_loss += F.l1_loss(rec, inp, reduction='sum').item()
            metrics = compute_psnr_ssim(rec, inp)
            psnr_sum += metrics['psnr_mean'] * inp.shape[0]
            ssim_sum += metrics['ssim_mean'] * inp.shape[0]

            tot += inp.shape[0]

        device = next(eval_model.parameters()).device
        stats = torch.tensor(
            [rec_loss, psnr_sum, ssim_sum, align_loss_sum, float(tot)],
            device=device,
            dtype=torch.float64,
        )
        dist.allreduce(stats)
        tot_g = max(round(stats[4].item()), 1)
        rec_loss_mean = stats[0].item() / tot_g
        psnr_mean = stats[1].item() / tot_g
        ssim_mean = stats[2].item() / tot_g
        align_loss_mean = stats[3].item() / tot_g

        if self.training_stage == 1:
            eval_model.train()
        else:
            eval_model.train()
            lr_eval_model.train()

        if self.training_stage == 1:
            return rec_loss_mean, psnr_mean, ssim_mean
        return rec_loss_mean, psnr_mean, ssim_mean, align_loss_mean

    def state_dict(self):
        state = {
            'vae_wo_ddp': self.vae_wo_ddp.state_dict(),
            'lr_vae_wo_ddp': self.lr_vae_wo_ddp.state_dict(),
            'disc_wo_ddp': self.disc_wo_ddp.state_dict(),
            'vae_opt': self.vae_opt.state_dict(),
            'lr_vae_opt': self.lr_vae_opt.state_dict(),
            'disc_opt': self.disc_opt.state_dict(),
            'training_stage': self.training_stage,
            'use_alignment_loss': self.use_alignment_loss,
            'alignment_loss_type': self.alignment_loss_type,
            'alignment_loss_weight': self.alignment_loss_weight,
            'alignment_loss_warmup_ep': self.alignment_loss_warmup_ep,
            'alignment_scale_index': self.alignment_scale_index,
            'stage2_use_kl': self.stage2_use_kl,
            'use_stage2_mid_scale_loss': self.use_stage2_mid_scale_loss,
            'stage2_mid_scale_indices': self.stage2_mid_scale_indices,
            'stage2_mid_scale_weights': self.stage2_mid_scale_weights,
        }
        if self.using_ema:
            state['vae_ema'] = self.vae_ema.state_dict()
            state['lr_vae_ema'] = self.lr_vae_ema.state_dict()
        if self.ema_gada is not None:
            state['ema_gada'] = self.ema_gada
        return state

    def load_state_dict(self, state, strict=True):
        misc.try_load_state_dict('vae_wo_ddp', self.vae_wo_ddp, state.get('vae_wo_ddp'), strict=strict)
        misc.try_load_state_dict('lr_vae_wo_ddp', self.lr_vae_wo_ddp, state.get('lr_vae_wo_ddp'), strict=strict)
        misc.try_load_state_dict('disc_wo_ddp', self.disc_wo_ddp, state.get('disc_wo_ddp'), strict=strict)
        self.vae_opt.load_state_dict(state.get('vae_opt'))
        self.lr_vae_opt.load_state_dict(state.get('lr_vae_opt'))
        self.disc_opt.load_state_dict(state.get('disc_opt'))
        self.training_stage = state.get('training_stage', self.training_stage)
        self.use_alignment_loss = state.get('use_alignment_loss', self.use_alignment_loss)
        self.alignment_loss_type = self._normalize_alignment_loss_type(state.get('alignment_loss_type', self.alignment_loss_type))
        self.alignment_loss_weight = state.get('alignment_loss_weight', self.alignment_loss_weight)
        self.alignment_loss_warmup_ep = max(float(state.get('alignment_loss_warmup_ep', self.alignment_loss_warmup_ep)), 0.0)
        self.alignment_scale_index = state.get('alignment_scale_index', self.alignment_scale_index)
        self.stage2_use_kl = self._normalize_bool(state.get('stage2_use_kl', self.stage2_use_kl))
        self.configure_stage2_mid_scale_loss(
            state.get('use_stage2_mid_scale_loss', self.use_stage2_mid_scale_loss),
            state.get('stage2_mid_scale_indices', self.stage2_mid_scale_indices),
            state.get('stage2_mid_scale_weights', self.stage2_mid_scale_weights),
        )
        if self.using_ema:
            misc.try_load_state_dict('vae_ema', self.vae_ema, state.get('vae_ema'), strict=strict)
            misc.try_load_state_dict('lr_vae_ema', self.lr_vae_ema, state.get('lr_vae_ema'), strict=strict)
        if 'ema_gada' in state:
            self.ema_gada = state['ema_gada']

    def set_training_stage(self, stage: int):
        if stage not in (1, 2):
            raise ValueError(f'Invalid training stage: {stage}. Must be 1 or 2.')

        self.training_stage = stage
        if stage == 1:
            unfreeze_model(self.lr_vae_wo_ddp)
            freeze_model(self.vae_wo_ddp)
            print('[Stage 1] Training LR VAE, HR VAE frozen')
        else:
            freeze_model(self.lr_vae_wo_ddp)
            unfreeze_model(self.vae_wo_ddp)
            if self.use_alignment_loss:
                print(
                    f'[Stage 2] Training HR VAE with {self.alignment_loss_type} alignment '
                    f'(weight={self.alignment_loss_weight}, warmup_ep={self.alignment_loss_warmup_ep}, '
                    f'stage2_use_kl={self.stage2_use_kl})'
                )
            else:
                print(f'[Stage 2] Training HR VAE without auxiliary alignment (stage2_use_kl={self.stage2_use_kl})')
            if self.use_stage2_mid_scale_loss:
                patch_nums = tuple(self.vae_wo_ddp.quantize.v_patch_nums)
                pairs = [
                    f'pn={patch_nums[si]} w={w}'
                    for si, w in zip(self.stage2_mid_scale_indices, self.stage2_mid_scale_weights)
                ]
                print(f'[Stage 2] Mid-scale band-limited supervision: {", ".join(pairs)}')
