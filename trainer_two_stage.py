"""
Two-Stage VAE Trainer for LR-HR alignment
Stage 1: Train LR VAE (single-scale 5x5)
Stage 2: Train HR VAE (multi-scale) with LR-HR alignment
"""
import sys
import os
from copy import deepcopy
from typing import Callable, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
import dist

from models import VQVAE, LR_VAE, DinoDisc
from utils import arg_util, misc
from utils.amp_opt import AmpOptimizer
from utils.diffaug import DiffAug
from utils.loss import hinge_loss, linear_loss, softplus_loss
from utils.lpips import LPIPS
from utils.image_saver import save_reconstruction_comparison

# Import the original VAETrainer
from trainer import VAETrainer

FTen = torch.Tensor


def freeze_model(model: nn.Module):
    """Freeze all parameters in a model"""
    for param in model.parameters():
        param.requires_grad = False
    model.eval()
    print(f'[Freeze] Froze {type(model).__name__}')


def unfreeze_model(model: nn.Module):
    """Unfreeze all parameters in a model"""
    for param in model.parameters():
        param.requires_grad = True
    model.train()
    print(f'[Unfreeze] Unfroze {type(model).__name__}')


class TwoStageVAETrainer(object):
    """
    Two-stage VAE trainer:
    - Stage 1: Train LR VAE to encode LR images to 5x5
    - Stage 2: Train HR VAE with alignment loss to match LR's 5x5
    """
    
    def __init__(
        self, is_visualizer: bool,
        # HR VAE
        vae: DDP, vae_wo_ddp: VQVAE, 
        # LR VAE
        lr_vae: DDP, lr_vae_wo_ddp: LR_VAE,
        # Discriminator
        disc: DDP, disc_wo_ddp: DinoDisc, 
        # Optimizers
        vae_opt: AmpOptimizer, lr_vae_opt: AmpOptimizer, disc_opt: AmpOptimizer,
        # Training config
        ema_ratio: float, dcrit: str,
        daug=1.0, lpips_loss: LPIPS = None, lp_reso=64, 
        wei_l1=1.0, wei_l2=0.0, wei_entropy=0.0, wei_lpips=0.5, wei_disc=0.6, 
        adapt_type=1, bcr=5.0, bcr_cut=0.5, reg=0.0, reg_every=16,
        disc_grad_ckpt=False,
        # Two-stage specific
        training_stage: int = 1,
        alignment_loss_weight: float = 1.0,
        dbg_unused=False, dbg_nan=False,
    ):
        super(TwoStageVAETrainer, self).__init__()
        self.dbg_unused, self.dbg_nan = dbg_unused, dbg_nan
        self.training_stage = training_stage
        self.alignment_loss_weight = alignment_loss_weight
        
        # Models
        self.vae, self.disc = vae, disc
        self.lr_vae = lr_vae
        self.vae_wo_ddp: VQVAE = vae_wo_ddp
        self.lr_vae_wo_ddp: LR_VAE = lr_vae_wo_ddp
        self.disc_wo_ddp: DinoDisc = disc_wo_ddp
        
        # Optimizers
        self.vae_opt = vae_opt
        self.lr_vae_opt = lr_vae_opt
        self.disc_opt = disc_opt
        
        # EMA for visualization
        self.ema_ratio = ema_ratio
        self.is_visualizer = is_visualizer
        self.using_ema = is_visualizer
        if self.using_ema:
            self.vae_ema: VQVAE = deepcopy(vae_wo_ddp).eval()
            self.lr_vae_ema: LR_VAE = deepcopy(lr_vae_wo_ddp).eval()
        else:
            self.vae_ema = None
            self.lr_vae_ema = None
        
        # Discriminator criterion
        self.dcrit = dcrit
        self.d_criterion: Callable = {
            'hg': hinge_loss, 'hinge': hinge_loss,
            'sp': softplus_loss, 'softplus': softplus_loss,
            'ln': linear_loss, 'lin': linear_loss, 'linear': linear_loss
        }[dcrit]
        
        # Augmentation and losses
        self.daug = DiffAug(prob=daug, cutout=0.2)
        self.wei_l1, self.wei_l2, self.wei_entropy = wei_l1, wei_l2, wei_entropy
        self.lpips_loss: LPIPS = lpips_loss
        self.lp_reso = lp_reso
        self.adapt_wei_disc = wei_disc > 0
        self.adapt_type = adapt_type
        self.ema_gada: torch.Tensor = None
        self.wei_lpips, self.wei_disc = wei_lpips * 2, abs(wei_disc)
        self.reg = 0.5 * reg * reg_every
        self.bcr = bcr * 2
        if self.bcr > 0:
            self.bcr_strong_aug = DiffAug(prob=1, cutout=bcr_cut)
        self.disc_grad_ckpt = disc_grad_ckpt
        
        self.usage_max = 0.0
        
        # Debug counters
        self._debug_loss_printed = 0
    
    def train_step_stage1(
        self, ep: int, it: int, g_it: int, stepping: bool, regularizing: bool,
        metric_lg: misc.MetricLogger, logging_params: bool, tb_lg: misc.TensorboardLogger,
        inp_lr: FTen, warmup_disc_schedule: float, fade_blur_schedule: float,
        maybe_record_function: Callable,
        args: arg_util.Args,
    ) -> Tuple[torch.Tensor, Optional[float], Optional[torch.Tensor], Optional[float]]:
        """
        Stage 1 training step: Train LR VAE only
        
        Args:
            inp_lr: LR input images [B, 3, lr_img_size, lr_img_size]
        """
        loggable = tb_lg.loggable() and stepping
        
        # ==================== Train LR VAE ====================
        with maybe_record_function('LR_VAE_rec'):
            with self.lr_vae_opt.amp_ctx:
                # Forward pass through LR VAE
                rec_B3HW, f_5x5, Lkl = self.lr_vae(inp_lr)
                
                B = rec_B3HW.shape[0]
                inp_rec_no_grad = torch.cat((inp_lr, rec_B3HW.data), dim=0)
            
            # Reconstruction loss (L1 + optional L2)
            Lrec = F.l1_loss(rec_B3HW, inp_lr)
            Lrec_for_log = Lrec.data.clone()
            Lrec *= self.wei_l1
            if self.wei_l2 > 0:
                Lrec += F.mse_loss(rec_B3HW, inp_lr).mul_(self.wei_l2)
            
            # LPIPS loss
            using_lpips = inp_lr.shape[-2] >= self.lp_reso and self.wei_lpips > 0
            if using_lpips:
                Lpip = self.lpips_loss(inp_lr, rec_B3HW)
                Lpip = torch.mean(Lpip)
                Lnll = Lrec + self.wei_lpips * Lpip
            else:
                Lpip = torch.tensor(0.)
                Lnll = Lrec
            
            # Total VAE loss
            Lkl_for_log = Lkl.item() if isinstance(Lkl, torch.Tensor) else Lkl
            Lg = Lnll + Lkl
        
        # ==================== Discriminator ====================
        with maybe_record_function('LR_disc'):
            with self.disc_opt.amp_ctx:
                inp_da = self.daug(inp_rec_no_grad[:B], fade_blur_schedule=fade_blur_schedule)
                rec_da = self.daug(inp_rec_no_grad[B:], fade_blur_schedule=fade_blur_schedule)
                inp_d_logits, rec_d_logits = self.disc(torch.cat((inp_da, rec_da), dim=0)).split([B, B], dim=0)
        
        acc_real = (inp_d_logits > 0).float().mean().item()
        acc_fake = (rec_d_logits < 0).float().mean().item()
        
        Ld_real, Ld_fake = self.d_criterion(is_real_pred=True, logits=inp_d_logits), self.d_criterion(is_real_pred=False, logits=rec_d_logits)
        Ld = Ld_real + Ld_fake
        
        # Adversarial loss for generator
        Lg_adv = self.d_criterion(is_real_pred=True, logits=rec_d_logits, for_g=True)
        
        # Adaptive weighting
        if self.adapt_wei_disc:
            wei_g = self._compute_adaptive_weight(Lnll, Lg_adv, self.vae_wo_ddp.decoder.conv_out.weight)
        else:
            wei_g = self.wei_disc
        
        Lg = Lg + wei_g * Lg_adv * warmup_disc_schedule
        
        # ==================== Backward and optimize ====================
        grad_norm_g = self.lr_vae_opt.backward_clip_step(Lg / self.lr_vae_opt.grad_accu, stepping)
        grad_norm_d = self.disc_opt.backward_clip_step(Ld / self.disc_opt.grad_accu, stepping)
        
        scale_log2_g = self.lr_vae_opt.get_scale_log2() if stepping else None
        scale_log2_d = self.disc_opt.get_scale_log2() if stepping else None
        
        # EMA update
        if self.using_ema and stepping:
            self._ema_update(self.lr_vae_ema, self.lr_vae_wo_ddp)
        
        # Logging
        if it == 0 or it in metric_lg.log_iters:
            Lpip_val = Lpip.item() if isinstance(Lpip, torch.Tensor) else Lpip
            Lnll_val = Lrec_for_log.item() + Lpip_val
            metric_lg.update(L1=Lrec_for_log.item(), NLL=Lnll_val, Lkl=Lkl_for_log, 
                           Ld=Ld.item(), Wg=wei_g, acc_real=acc_real, acc_fake=acc_fake, 
                           gnm=grad_norm_g, dnm=grad_norm_d, usage=0.0)
            
            # Save reconstruction images
            if args.save_reconstruction_images and dist.is_master():
                try:
                    save_dir = os.path.join(args.local_out_dir_path, 'reconstruction_samples_lr')
                    save_reconstruction_comparison(
                        original=inp_lr,
                        reconstructed=rec_B3HW.detach(),
                        save_dir=save_dir,
                        ep=ep, it=it, max_samples=4,
                    )
                except Exception as e:
                    print(f'[Warning] Failed to save LR reconstruction images: {e}', flush=True)
        
        return grad_norm_g, scale_log2_g, grad_norm_d, scale_log2_d
    
    def train_step_stage2(
        self, ep: int, it: int, g_it: int, stepping: bool, regularizing: bool,
        metric_lg: misc.MetricLogger, logging_params: bool, tb_lg: misc.TensorboardLogger,
        inp_lr: FTen, inp_hr: FTen, warmup_disc_schedule: float, fade_blur_schedule: float,
        maybe_record_function: Callable,
        args: arg_util.Args,
    ) -> Tuple[torch.Tensor, Optional[float], Optional[torch.Tensor], Optional[float]]:
        """
        Stage 2 training step: Train HR VAE with LR-HR alignment
        
        Args:
            inp_lr: LR input images [B, 3, lr_img_size, lr_img_size]
            inp_hr: HR input images [B, 3, hr_img_size, hr_img_size]
        """
        loggable = tb_lg.loggable() and stepping
        
        # ==================== Get LR 5x5 representation (frozen) ====================
        with torch.no_grad():
            _, lr_f_5x5, _ = self.lr_vae(inp_lr)  # [B, C, 5, 5]
        
        # ==================== Train HR VAE ====================
        with maybe_record_function('HR_VAE_rec'):
            with self.vae_opt.amp_ctx:
                # Forward pass through HR VAE
                rec_B3HW, usage, Lkl = self.vae(inp_hr, ret_usages=loggable)
                
                if loggable and usage is not None:
                    self.usage_max = max(self.usage_max, max(usage))
                
                B = rec_B3HW.shape[0]
                inp_rec_no_grad = torch.cat((inp_hr, rec_B3HW.data), dim=0)
            
            # Reconstruction loss (L1 + optional L2)
            Lrec = F.l1_loss(rec_B3HW, inp_hr)
            Lrec_for_log = Lrec.data.clone()
            Lrec *= self.wei_l1
            if self.wei_l2 > 0:
                Lrec += F.mse_loss(rec_B3HW, inp_hr).mul_(self.wei_l2)
            
            # LPIPS loss
            using_lpips = inp_hr.shape[-2] >= self.lp_reso and self.wei_lpips > 0
            if using_lpips:
                Lpip = self.lpips_loss(inp_hr, rec_B3HW)
                Lpip = torch.mean(Lpip)
                Lnll = Lrec + self.wei_lpips * Lpip
            else:
                Lpip = torch.tensor(0.)
                Lnll = Lrec
            
            # KL loss
            Lkl_for_log = Lkl.item() if isinstance(Lkl, torch.Tensor) else Lkl
            
            # ==================== LR-HR Alignment Loss ====================
            # Get HR's first scale (5x5) latent representation
            # We need to extract the 5x5 representation from HR VAE
            with torch.no_grad():
                # Encode HR image
                f_hr_encoded = self.vae_wo_ddp.encoder(inp_hr)
                f_hr_encoded = self.vae_wo_ddp.quant_conv(f_hr_encoded)
                
                # Get first scale (5x5) from multi-scale quantizer
                # Downsample to 5x5
                hr_f_5x5 = F.interpolate(f_hr_encoded, size=(5, 5), mode='area')
                
                # Pass through quantizer's mean_logvar_conv to get latent
                moments_hr = self.vae_wo_ddp.quantize.mean_logvar_conv(hr_f_5x5)
                # Extract mean (first half of channels)
                hr_f_5x5_mean = moments_hr[:, :self.vae_wo_ddp.Cvae, :, :]  # [B, C, 5, 5]
            
            # Compute alignment loss (MSE between LR and HR's 5x5 representations)
            L_align = F.mse_loss(hr_f_5x5_mean, lr_f_5x5)
            
            # Total VAE loss
            Lg = Lnll + Lkl + self.alignment_loss_weight * L_align
            
            # Debug logging
            if self._debug_loss_printed < args.debug_loss_printed_limit:
                print(f'[Stage2 Debug] [Ep {ep}] [It {it}] Lrec={Lrec_for_log.item():.4f}, '
                      f'Lkl={Lkl_for_log:.6f}, L_align={L_align.item():.6f}', flush=True)
                self._debug_loss_printed += 1
        
        # ==================== Discriminator ====================
        with maybe_record_function('HR_disc'):
            with self.disc_opt.amp_ctx:
                inp_da = self.daug(inp_rec_no_grad[:B], fade_blur_schedule=fade_blur_schedule)
                rec_da = self.daug(inp_rec_no_grad[B:], fade_blur_schedule=fade_blur_schedule)
                inp_d_logits, rec_d_logits = self.disc(torch.cat((inp_da, rec_da), dim=0)).split([B, B], dim=0)
        
        acc_real = (inp_d_logits > 0).float().mean().item()
        acc_fake = (rec_d_logits < 0).float().mean().item()
        
        Ld_real, Ld_fake = self.d_criterion(is_real_pred=True, logits=inp_d_logits), self.d_criterion(is_real_pred=False, logits=rec_d_logits)
        Ld = Ld_real + Ld_fake
        
        # Adversarial loss for generator
        Lg_adv = self.d_criterion(is_real_pred=True, logits=rec_d_logits, for_g=True)
        
        # Adaptive weighting
        if self.adapt_wei_disc:
            wei_g = self._compute_adaptive_weight(Lnll, Lg_adv, self.vae_wo_ddp.decoder.conv_out.weight)
        else:
            wei_g = self.wei_disc
        
        Lg = Lg + wei_g * Lg_adv * warmup_disc_schedule
        
        # ==================== Backward and optimize ====================
        grad_norm_g = self.vae_opt.backward_clip_step(Lg / self.vae_opt.grad_accu, stepping)
        grad_norm_d = self.disc_opt.backward_clip_step(Ld / self.disc_opt.grad_accu, stepping)
        
        scale_log2_g = self.vae_opt.get_scale_log2() if stepping else None
        scale_log2_d = self.disc_opt.get_scale_log2() if stepping else None
        
        # EMA update
        if self.using_ema and stepping:
            self._ema_update(self.vae_ema, self.vae_wo_ddp)
        
        # Logging
        if it == 0 or it in metric_lg.log_iters:
            Lpip_val = Lpip.item() if isinstance(Lpip, torch.Tensor) else Lpip
            Lnll_val = Lrec_for_log.item() + Lpip_val
            metric_lg.update(L1=Lrec_for_log.item(), NLL=Lnll_val, Lkl=Lkl_for_log, 
                           Ld=Ld.item(), Wg=wei_g, acc_real=acc_real, acc_fake=acc_fake, 
                           gnm=grad_norm_g, dnm=grad_norm_d, usage=self.usage_max,
                           L_align=L_align.item())
            
            # Save reconstruction images
            if args.save_reconstruction_images and dist.is_master():
                try:
                    save_dir = os.path.join(args.local_out_dir_path, 'reconstruction_samples_hr')
                    save_reconstruction_comparison(
                        original=inp_hr,
                        reconstructed=rec_B3HW.detach(),
                        save_dir=save_dir,
                        ep=ep, it=it, max_samples=4,
                    )
                except Exception as e:
                    print(f'[Warning] Failed to save HR reconstruction images: {e}', flush=True)
        
        return grad_norm_g, scale_log2_g, grad_norm_d, scale_log2_d
    
    def _compute_adaptive_weight(self, nll_loss, g_loss, last_layer_weight):
        """Compute adaptive discriminator weight"""
        nll_grads = torch.autograd.grad(nll_loss, last_layer_weight, retain_graph=True)[0]
        g_grads = torch.autograd.grad(g_loss, last_layer_weight, retain_graph=True)[0]
        
        d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-7)
        d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
        
        # EMA smoothing
        if self.ema_gada is None:
            self.ema_gada = d_weight
        else:
            self.ema_gada = self.ema_gada * 0.9 + d_weight * 0.1
        
        return self.ema_gada * self.wei_disc
    
    def _ema_update(self, ema_model, model):
        """Update EMA model"""
        with torch.no_grad():
            for ema_param, param in zip(ema_model.parameters(), model.parameters()):
                ema_param.data.mul_(self.ema_ratio).add_(param.data, alpha=1 - self.ema_ratio)
    
    @torch.no_grad()
    def eval_ep(self, ld_val, max_batches=None):
        """
        Evaluate on validation set.
        
        Args:
            ld_val: Validation dataloader (returns LR or HR or both depending on stage)
            max_batches: Maximum number of batches to evaluate
        
        Returns:
            Depending on training stage:
            - Stage 1: (lr_rec_loss, lr_psnr, lr_ssim)
            - Stage 2: (hr_rec_loss, hr_psnr, hr_ssim, alignment_loss)
        """
        from skimage.metrics import peak_signal_noise_ratio, structural_similarity
        
        tot = 0
        rec_loss = 0
        psnr_sum = 0.0
        ssim_sum = 0.0
        align_loss_sum = 0.0
        
        # Choose which model to evaluate based on training stage
        if self.training_stage == 1:
            eval_model = self.lr_vae_ema if self.using_ema else self.lr_vae_wo_ddp
            eval_model.eval()
        else:  # Stage 2
            eval_model = self.vae_ema if self.using_ema else self.vae_wo_ddp
            eval_model.eval()
            lr_eval_model = self.lr_vae_ema if self.using_ema else self.lr_vae_wo_ddp
            lr_eval_model.eval()
        
        for i, batch in enumerate(ld_val):
            if max_batches is not None and i >= max_batches:
                break
            
            if self.training_stage == 1:
                # Stage 1: only LR data
                inp = batch.to(next(eval_model.parameters()).device)
                rec, _, _ = eval_model(inp)
            else:
                # Stage 2: both LR and HR data
                inp_lr, inp_hr = batch
                inp_lr = inp_lr.to(next(lr_eval_model.parameters()).device)
                inp_hr = inp_hr.to(next(eval_model.parameters()).device)
                
                # Evaluate HR reconstruction
                rec, _, _ = eval_model(inp_hr)
                inp = inp_hr
                
                # Compute alignment loss
                _, lr_f_5x5, _ = lr_eval_model(inp_lr)
                f_hr_encoded = eval_model.encoder(inp_hr)
                f_hr_encoded = eval_model.quant_conv(f_hr_encoded)
                hr_f_5x5 = F.interpolate(f_hr_encoded, size=(5, 5), mode='area')
                moments_hr = eval_model.quantize.mean_logvar_conv(hr_f_5x5)
                hr_f_5x5_mean = moments_hr[:, :eval_model.Cvae, :, :]
                align_loss_sum += F.mse_loss(hr_f_5x5_mean, lr_f_5x5).item() * inp.shape[0]
            
            # Compute reconstruction loss
            rec_loss += F.l1_loss(rec, inp, reduction='sum').item()
            
            # Compute PSNR and SSIM
            inp_np = inp.cpu().numpy()
            rec_np = rec.cpu().numpy()
            
            for b in range(inp.shape[0]):
                # Convert from [-1, 1] to [0, 1]
                img_gt = (inp_np[b] + 1) / 2
                img_rec = (rec_np[b] + 1) / 2
                img_rec = np.clip(img_rec, 0, 1)
                
                # Transpose from CHW to HWC for skimage
                img_gt = np.transpose(img_gt, (1, 2, 0))
                img_rec = np.transpose(img_rec, (1, 2, 0))
                
                # PSNR
                psnr = peak_signal_noise_ratio(img_gt, img_rec, data_range=1.0)
                psnr_sum += psnr
                
                # SSIM
                ssim = structural_similarity(img_gt, img_rec, channel_axis=2, data_range=1.0)
                ssim_sum += ssim
            
            tot += inp.shape[0]
        
        # Aggregate results
        rec_loss_mean = rec_loss / tot
        psnr_mean = psnr_sum / tot
        ssim_mean = ssim_sum / tot
        
        if self.training_stage == 1:
            return rec_loss_mean, psnr_mean, ssim_mean
        else:
            align_loss_mean = align_loss_sum / tot
            return rec_loss_mean, psnr_mean, ssim_mean, align_loss_mean
    
    def state_dict(self):
        """Save trainer state"""
        state = {
            'vae_wo_ddp': self.vae_wo_ddp.state_dict(),
            'lr_vae_wo_ddp': self.lr_vae_wo_ddp.state_dict(),
            'disc_wo_ddp': self.disc_wo_ddp.state_dict(),
            'vae_opt': self.vae_opt.state_dict(),
            'lr_vae_opt': self.lr_vae_opt.state_dict(),
            'disc_opt': self.disc_opt.state_dict(),
            'training_stage': self.training_stage,
        }
        if self.using_ema:
            state['vae_ema'] = self.vae_ema.state_dict()
            state['lr_vae_ema'] = self.lr_vae_ema.state_dict()
        if self.ema_gada is not None:
            state['ema_gada'] = self.ema_gada
        return state
    
    def load_state_dict(self, state, strict=True):
        """Load trainer state"""
        self.vae_wo_ddp.load_state_dict(state['vae_wo_ddp'], strict=strict)
        self.lr_vae_wo_ddp.load_state_dict(state['lr_vae_wo_ddp'], strict=strict)
        self.disc_wo_ddp.load_state_dict(state['disc_wo_ddp'], strict=strict)
        self.vae_opt.load_state_dict(state['vae_opt'])
        self.lr_vae_opt.load_state_dict(state['lr_vae_opt'])
        self.disc_opt.load_state_dict(state['disc_opt'])
        self.training_stage = state.get('training_stage', 1)
        if self.using_ema:
            if 'vae_ema' in state:
                self.vae_ema.load_state_dict(state['vae_ema'], strict=strict)
            if 'lr_vae_ema' in state:
                self.lr_vae_ema.load_state_dict(state['lr_vae_ema'], strict=strict)
        if 'ema_gada' in state:
            self.ema_gada = state['ema_gada']
    
    def set_training_stage(self, stage: int):
        """
        Switch between training stages.
        
        Args:
            stage: 1 for LR VAE training, 2 for HR VAE training with alignment
        """
        if stage not in [1, 2]:
            raise ValueError(f'Invalid training stage: {stage}. Must be 1 or 2.')
        
        self.training_stage = stage
        
        if stage == 1:
            # Stage 1: Train LR VAE, freeze HR VAE
            unfreeze_model(self.lr_vae_wo_ddp)
            freeze_model(self.vae_wo_ddp)
            print('[Stage 1] Training LR VAE, HR VAE frozen')
        else:
            # Stage 2: Train HR VAE, freeze LR VAE
            freeze_model(self.lr_vae_wo_ddp)
            unfreeze_model(self.vae_wo_ddp)
            print('[Stage 2] Training HR VAE with alignment, LR VAE frozen')

