"""
Two-stage VAE trainer for LR-HR alignment.

Stage 1:
- train LR VAE only

Stage 2:
- freeze LR VAE
- train HR multi-scale VAE
- align the first HR scale token with LR posterior mean
"""
import os
from copy import deepcopy
from typing import Callable, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

import dist
from models import DinoDisc, LR_VAE, VQVAE
from utils import arg_util, misc
from utils.amp_opt import AmpOptimizer
from utils.diffaug import DiffAug
from utils.image_saver import save_reconstruction_comparison
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
        alignment_loss_weight: float = 1.0,
        alignment_scale_index: int = 0,
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
        self.alignment_loss_weight = alignment_loss_weight
        self.alignment_scale_index = alignment_scale_index

        self.usage_max = 0.0
        self._debug_loss_printed = 0

        self.set_training_stage(training_stage)

    def _compute_adaptive_weight(self, nll_loss: torch.Tensor, g_loss: torch.Tensor, last_layer_weight: torch.Tensor) -> torch.Tensor:
        nll_grads = torch.autograd.grad(nll_loss, last_layer_weight, retain_graph=True)[0]
        g_grads = torch.autograd.grad(g_loss, last_layer_weight, retain_graph=True)[0]

        d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-7)
        d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()

        if self.ema_gada is None:
            self.ema_gada = d_weight
        else:
            self.ema_gada = self.ema_gada * 0.9 + d_weight * 0.1
        return self.ema_gada * self.wei_disc

    def _ema_update(self, ema_model: nn.Module, model: nn.Module):
        with torch.no_grad():
            for ema_param, param in zip(ema_model.parameters(), model.parameters()):
                ema_param.data.mul_(self.ema_ratio).add_(param.data, alpha=1 - self.ema_ratio)

    def _disc_forward(self, real_img: torch.Tensor, fake_img: torch.Tensor, fade_blur_schedule: float):
        real_aug = self.daug.aug(real_img, fade_blur_schedule)
        fake_aug = self.daug.aug(fake_img, fade_blur_schedule)
        return self.disc(torch.cat((real_aug, fake_aug), dim=0)).split([real_img.shape[0], fake_img.shape[0]], dim=0)

    def _get_alignment_targets(self, inp_lr: torch.Tensor, inp_hr: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            lr_mean = self.lr_vae_wo_ddp.encode_to_posterior_mean(inp_lr)

        hr_mean, _ = self.vae_wo_ddp.img_to_scale_posterior_stats(inp_hr, scale_index=self.alignment_scale_index)

        if hr_mean.shape != lr_mean.shape:
            raise ValueError(
                'Alignment shape mismatch: '
                f'HR first-scale mean {tuple(hr_mean.shape)} vs LR posterior mean {tuple(lr_mean.shape)}. '
                f'Expected args.patch_nums[{self.alignment_scale_index}] to match LR latent size '
                f'(usually lr_img_size / 16).'
            )
        return lr_mean, hr_mean

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

        with maybe_record_function('LR_VAE_rec'):
            with self.lr_vae_opt.amp_ctx:
                rec_B3HW, _, Lkl = self.lr_vae(inp_lr)
                B = rec_B3HW.shape[0]
                inp_rec_no_grad = torch.cat((inp_lr, rec_B3HW.detach()), dim=0)

                Lrec = F.l1_loss(rec_B3HW, inp_lr)
                Lrec_for_log = Lrec.detach().clone()
                Lrec = Lrec * self.wei_l1
                if self.wei_l2 > 0:
                    Lrec = Lrec + F.mse_loss(rec_B3HW, inp_lr) * self.wei_l2

                using_lpips = inp_lr.shape[-2] >= self.lp_reso and self.wei_lpips > 0
                if using_lpips:
                    Lpip = torch.mean(self.lpips_loss(inp_lr, rec_B3HW))
                    Lnll = Lrec + self.wei_lpips * Lpip
                else:
                    Lpip = inp_lr.new_zeros(())
                    Lnll = Lrec

                Lg = Lnll + Lkl

        with maybe_record_function('LR_disc'):
            with self.disc_opt.amp_ctx:
                inp_d_logits, rec_d_logits = self._disc_forward(inp_rec_no_grad[:B], inp_rec_no_grad[B:], fade_blur_schedule)

        acc_real = (inp_d_logits > 0).float().mean().item()
        acc_fake = (rec_d_logits < 0).float().mean().item()
        Ld_real = self.d_criterion(is_real_pred=True, logits=inp_d_logits)
        Ld_fake = self.d_criterion(is_real_pred=False, logits=rec_d_logits)
        Ld = Ld_real + Ld_fake
        Lg_adv = self.d_criterion(is_real_pred=True, logits=rec_d_logits, for_g=True)

        if self.adapt_wei_disc:
            wei_g = self._compute_adaptive_weight(Lnll, Lg_adv, self.lr_vae_wo_ddp.decoder.conv_out.weight)
        else:
            wei_g = inp_lr.new_tensor(self.wei_disc)
        Lg = Lg + wei_g * Lg_adv * warmup_disc_schedule

        grad_norm_g, scale_log2_g = self.lr_vae_opt.backward_clip_step(stepping=stepping, loss=Lg)
        grad_norm_d, scale_log2_d = self.disc_opt.backward_clip_step(stepping=stepping, loss=Ld)

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
            if args.save_reconstruction_images and dist.is_master():
                try:
                    save_reconstruction_comparison(
                        original=inp_lr,
                        reconstructed=rec_B3HW.detach(),
                        save_dir=os.path.join(args.local_out_dir_path, 'reconstruction_samples_lr'),
                        ep=ep,
                        it=it,
                        max_samples=4,
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

        with maybe_record_function('HR_VAE_rec'):
            with self.vae_opt.amp_ctx:
                rec_B3HW, usage, Lkl = self.vae(inp_hr, ret_usages=loggable)
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
                    Lpip = torch.mean(self.lpips_loss(inp_hr, rec_B3HW))
                    Lnll = Lrec + self.wei_lpips * Lpip
                else:
                    Lpip = inp_hr.new_zeros(())
                    Lnll = Lrec

                if self.use_alignment_loss:
                    lr_mean, hr_mean = self._get_alignment_targets(inp_lr, inp_hr)
                    L_align = F.mse_loss(hr_mean, lr_mean)
                else:
                    L_align = inp_hr.new_zeros(())

                Lg = Lnll + Lkl + self.alignment_loss_weight * L_align

        with maybe_record_function('HR_disc'):
            with self.disc_opt.amp_ctx:
                inp_d_logits, rec_d_logits = self._disc_forward(inp_rec_no_grad[:B], inp_rec_no_grad[B:], fade_blur_schedule)

        acc_real = (inp_d_logits > 0).float().mean().item()
        acc_fake = (rec_d_logits < 0).float().mean().item()
        Ld_real = self.d_criterion(is_real_pred=True, logits=inp_d_logits)
        Ld_fake = self.d_criterion(is_real_pred=False, logits=rec_d_logits)
        Ld = Ld_real + Ld_fake
        Lg_adv = self.d_criterion(is_real_pred=True, logits=rec_d_logits, for_g=True)

        if self.adapt_wei_disc:
            wei_g = self._compute_adaptive_weight(Lnll, Lg_adv, self.vae_wo_ddp.decoder.conv_out.weight)
        else:
            wei_g = inp_hr.new_tensor(self.wei_disc)
        Lg = Lg + wei_g * Lg_adv * warmup_disc_schedule

        grad_norm_g, scale_log2_g = self.vae_opt.backward_clip_step(stepping=stepping, loss=Lg)
        grad_norm_d, scale_log2_d = self.disc_opt.backward_clip_step(stepping=stepping, loss=Ld)

        if self.using_ema and stepping:
            self._ema_update(self.vae_ema, self.vae_wo_ddp)

        if self._debug_loss_printed < args.debug_loss_printed_limit:
            print(
                f'[Stage2 Debug] [Ep {ep}] [It {it}] '
                f'Lrec={Lrec_for_log.item():.4f}, '
                f'Lkl={(Lkl.item() if isinstance(Lkl, torch.Tensor) else Lkl):.6f}, '
                f'L_align={L_align.item():.6f}',
                flush=True,
            )
            self._debug_loss_printed += 1

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
                usage=self.usage_max,
                L_align=L_align.item(),
            )
            if args.save_reconstruction_images and dist.is_master():
                try:
                    save_reconstruction_comparison(
                        original=inp_hr,
                        reconstructed=rec_B3HW.detach(),
                        save_dir=os.path.join(args.local_out_dir_path, 'reconstruction_samples_hr'),
                        ep=ep,
                        it=it,
                        max_samples=4,
                    )
                except Exception as e:
                    print(f'[Warning] Failed to save HR reconstruction images: {e}', flush=True)

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
        from skimage.metrics import peak_signal_noise_ratio, structural_similarity

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
                rec, _, _ = eval_model(inp_hr)
                inp = inp_hr

                if self.use_alignment_loss:
                    lr_mean = lr_eval_model.encode_to_posterior_mean(inp_lr)
                    hr_mean, _ = eval_model.img_to_scale_posterior_stats(inp_hr, scale_index=self.alignment_scale_index)
                    align_loss_sum += F.mse_loss(hr_mean, lr_mean).item() * inp.shape[0]

            rec_loss += F.l1_loss(rec, inp, reduction='sum').item()
            inp_np = inp.cpu().numpy()
            rec_np = rec.cpu().numpy()

            for b in range(inp.shape[0]):
                img_gt = np.transpose((inp_np[b] + 1) / 2, (1, 2, 0))
                img_rec = np.transpose(np.clip((rec_np[b] + 1) / 2, 0, 1), (1, 2, 0))
                psnr_sum += peak_signal_noise_ratio(img_gt, img_rec, data_range=1.0)
                ssim_sum += structural_similarity(img_gt, img_rec, channel_axis=2, data_range=1.0)

            tot += inp.shape[0]

        rec_loss_mean = rec_loss / tot
        psnr_mean = psnr_sum / tot
        ssim_mean = ssim_sum / tot
        if self.training_stage == 1:
            return rec_loss_mean, psnr_mean, ssim_mean
        return rec_loss_mean, psnr_mean, ssim_mean, align_loss_sum / max(tot, 1)

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
            'alignment_loss_weight': self.alignment_loss_weight,
            'alignment_scale_index': self.alignment_scale_index,
        }
        if self.using_ema:
            state['vae_ema'] = self.vae_ema.state_dict()
            state['lr_vae_ema'] = self.lr_vae_ema.state_dict()
        if self.ema_gada is not None:
            state['ema_gada'] = self.ema_gada
        return state

    def load_state_dict(self, state, strict=True):
        self.vae_wo_ddp.load_state_dict(state['vae_wo_ddp'], strict=strict)
        self.lr_vae_wo_ddp.load_state_dict(state['lr_vae_wo_ddp'], strict=strict)
        self.disc_wo_ddp.load_state_dict(state['disc_wo_ddp'], strict=strict)
        self.vae_opt.load_state_dict(state['vae_opt'])
        self.lr_vae_opt.load_state_dict(state['lr_vae_opt'])
        self.disc_opt.load_state_dict(state['disc_opt'])
        self.training_stage = state.get('training_stage', self.training_stage)
        self.use_alignment_loss = state.get('use_alignment_loss', self.use_alignment_loss)
        self.alignment_loss_weight = state.get('alignment_loss_weight', self.alignment_loss_weight)
        self.alignment_scale_index = state.get('alignment_scale_index', self.alignment_scale_index)
        if self.using_ema:
            if 'vae_ema' in state:
                self.vae_ema.load_state_dict(state['vae_ema'], strict=strict)
            if 'lr_vae_ema' in state:
                self.lr_vae_ema.load_state_dict(state['lr_vae_ema'], strict=strict)
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
            print('[Stage 2] Training HR VAE with alignment, LR VAE frozen')

