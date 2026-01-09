from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch import distributed as tdist, nn as nn
from torch.nn import functional as F

import dist


# this file provides the ContinuousMultiScaleQuantizer for continuous VAE
__all__ = ['ContinuousMultiScaleQuantizer',]


class DiagonalGaussianDistribution(object):
    """Diagonal Gaussian distribution for reparameterization trick"""
    def __init__(self, parameters, deterministic=False):
        self.parameters = parameters
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=1)
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)
        if self.deterministic:
            self.var = self.std = torch.zeros_like(self.mean).to(device=self.parameters.device)
    
    def sample(self):
        x = self.mean + self.std * torch.randn(self.mean.shape).to(device=self.parameters.device)
        return x
    
    def kl(self, other=None):
        if self.deterministic:
            return torch.Tensor([0.])
        else:
            if other is None:
                return 0.5 * torch.sum(torch.pow(self.mean, 2)
                                       + self.var - 1.0 - self.logvar,
                                       dim=[1, 2, 3])
            else:
                return 0.5 * torch.sum(
                    torch.pow(self.mean - other.mean, 2) / other.var
                    + self.var / other.var - 1.0 - self.logvar + other.logvar,
                    dim=[1, 2, 3])
    
    def mode(self):
        return self.mean


class ContinuousMultiScaleQuantizer(nn.Module):
    """Continuous Multi-Scale VAE Quantizer using Gaussian distributions"""
    def __init__(
        self, Cvae, beta: float = 1.0,  # beta is now kl_weight
        default_qresi_counts=0, v_patch_nums=None, quant_resi=0.5, share_quant_resi=4,
        codebook_drop=0.1
    ):
        super().__init__()
        self.codebook_drop = codebook_drop
        self.Cvae: int = Cvae
        self.v_patch_nums: Tuple[int] = v_patch_nums
        self.kl_weight: float = beta  # reuse beta as kl_weight
        
        # quant_resi: feature refinement, still useful for continuous VAE
        self.quant_resi_ratio = quant_resi
        if share_quant_resi == 0:   # non-shared: \phi_{1 to K} for K scales
            self.quant_resi = PhiNonShared([(Phi(Cvae, quant_resi) if abs(quant_resi) > 1e-6 else nn.Identity()) for _ in range(default_qresi_counts or len(self.v_patch_nums))])
        elif share_quant_resi == 1: # fully shared: only a single \phi for K scales
            self.quant_resi = PhiShared(Phi(Cvae, quant_resi) if abs(quant_resi) > 1e-6 else nn.Identity())
        else:                       # partially shared: \phi_{1 to share_quant_resi} for K scales
            self.quant_resi = PhiPartiallyShared(nn.ModuleList([(Phi(Cvae, quant_resi) if abs(quant_resi) > 1e-6 else nn.Identity()) for _ in range(share_quant_resi)]))
        
        # For continuous VAE: conv layer to predict mean and logvar (output 2*Cvae channels)
        # Each scale will use this shared predictor
        self.mean_logvar_conv = nn.Conv2d(Cvae, 2 * Cvae, kernel_size=1, stride=1, padding=0)
        self._init_mean_logvar_conv()
        
        # only used for progressive training (not supported yet)
        self.prog_si = -1
    
    def _init_mean_logvar_conv(self):
        """
        Initialize mean_logvar_conv with proper strategy for VAE:
        - Weight: xavier_normal_ initialization (same as other conv layers)
        - Bias for mean channels: 0 (mean should start near 0)
        - Bias for logvar channels: -2.0 (logvar should start small, std ≈ 0.37)
        This prevents initial KL explosion and helps stable training.
        """
        # Initialize weight with xavier_normal (gain=1.0)
        nn.init.xavier_normal_(self.mean_logvar_conv.weight.data, gain=1.0)
        
        if self.mean_logvar_conv.bias is not None:
            # Split bias into mean and logvar parts
            Cvae = self.Cvae
            # Mean channels: initialize to 0
            self.mean_logvar_conv.bias.data[:Cvae].zero_()
            # Logvar channels: initialize to -2.0 (so initial std ≈ exp(-1) ≈ 0.37)
            # This is a common practice in VAE to prevent initial KL explosion
            self.mean_logvar_conv.bias.data[Cvae:].fill_(-2.0)
    
    def extra_repr(self) -> str:
        return f'{self.v_patch_nums}, kl_weight={self.kl_weight}  |  S={len(self.v_patch_nums)}, quant_resi={self.quant_resi_ratio}'
    
    # ===================== `forward` is only used in VAE training =====================
    def forward(self, f_BChw: torch.Tensor, ret_usages=False, dropout=None) -> Tuple[torch.Tensor, List[float], torch.Tensor]:
        """
        Forward pass for continuous multi-scale VAE.
        
        Args:
            f_BChw: encoder output features [B, C, H, W]
            ret_usages: whether to return usage statistics (kept for compatibility, returns None)
            dropout: dropout for stochastic depth (optional, for multi-scale dropout)
        
        Returns:
            f_hat: reconstructed features [B, C, H, W]
            usages: None (kept for compatibility with training loop)
            kl_loss: KL divergence loss aggregated across all scales
        """
        dtype = f_BChw.dtype
        if dtype != torch.float32: f_BChw = f_BChw.float()
        B, C, H, W = f_BChw.shape
        
        # For continuous VAE with multi-scale: process residuals at each scale
        f_rest = f_BChw.clone()  # clone to avoid in-place modification, but keep gradient flow
        f_hat = torch.zeros_like(f_rest)
        
        with torch.amp.autocast('cuda', enabled=False):
            total_kl_loss = 0.0
            SN = len(self.v_patch_nums)
            
            # Stochastic depth: randomly skip some scales during training
            max_n = SN
            if self.training and dropout is not None and self.codebook_drop > 0:
                n_quantizers = torch.full((B,), max_n, dtype=torch.long, device=f_BChw.device)
                n_dropout = np.arange(B)[np.random.rand(B) < self.codebook_drop]
                # Ensure dropout is on the same device as n_quantizers
                if isinstance(dropout, torch.Tensor):
                    dropout = dropout.to(device=f_BChw.device)
                n_quantizers[n_dropout] = dropout[n_dropout]
            else:
                n_quantizers = torch.full((B,), max_n, dtype=torch.long, device=f_BChw.device)
            
            # Multi-scale processing: from small to large
            for si, pn in enumerate(self.v_patch_nums):
                # Downsample residual to current scale
                if si != SN - 1:
                    rest_scale = F.interpolate(f_rest, size=(pn, pn), mode='area')
                else:
                    rest_scale = f_rest
                
                # Predict mean and logvar for Gaussian distribution
                moments = self.mean_logvar_conv(rest_scale)  # [B, 2*C, pn, pn]
                posterior = DiagonalGaussianDistribution(moments, deterministic=not self.training)
                
                # Sample from the distribution (reparameterization trick)
                if self.training:
                    h_scale = posterior.sample()  # [B, C, pn, pn]
                else:
                    h_scale = posterior.mode()  # use mean during inference
                
                # Compute KL divergence for this scale
                kl_loss_scale = posterior.kl()  # [B]
                kl_loss_scale = torch.mean(kl_loss_scale)  # scalar
                
                # Upsample to original resolution and apply feature refinement
                if si != SN - 1:
                    h_BChw = F.interpolate(h_scale, size=(H, W), mode='bicubic').contiguous()
                else:
                    h_BChw = h_scale.contiguous()
                
                h_BChw = self.quant_resi[si/(SN-1)](h_BChw)
                
                # Apply stochastic depth mask
                mask = (torch.full((B,), fill_value=si, device=h_BChw.device) < n_quantizers)[:, None, None, None].float()
                
                # Accumulate features
                f_hat = f_hat + h_BChw * mask
                f_rest = f_rest - h_BChw  # update residual
                
                # Accumulate KL loss (weighted by mask ratio for stochastic depth)
                ratio = mask.sum() / B if mask.sum() > 0 else 1.0
                total_kl_loss += kl_loss_scale / ratio
            
            # Average KL loss across scales and apply weight
            total_kl_loss = total_kl_loss / SN * self.kl_weight
        
        # For continuous VAE, no straight-through estimator needed - reparameterization trick handles gradients
        # Return the accumulated features directly
        
        # usages is None for continuous VAE (no discrete codebook)
        usages = None if not ret_usages else [None] * SN
        
        return f_hat.to(dtype), usages, total_kl_loss
    # ===================== `forward` is only used in VAE training =====================
    
    def embed_to_fhat(self, ms_h_BChw: List[torch.Tensor], all_to_max_scale=True, last_one=False) -> Union[List[torch.Tensor], torch.Tensor]:
        """
        Combine multi-scale features into final feature map(s).
        For continuous VAE, ms_h_BChw contains sampled latent features at different scales.
        """
        ls_f_hat_BChw = []
        B = ms_h_BChw[0].shape[0]
        H = W = self.v_patch_nums[-1]
        SN = len(self.v_patch_nums)
        if all_to_max_scale:
            f_hat = ms_h_BChw[0].new_zeros(B, self.Cvae, H, W, dtype=torch.float32)
            for si, pn in enumerate(self.v_patch_nums): # from small to large
                h_BChw = ms_h_BChw[si]
                if si < len(self.v_patch_nums) - 1:
                    h_BChw = F.interpolate(h_BChw, size=(H, W), mode='bicubic')
                h_BChw = self.quant_resi[si/(SN-1)](h_BChw)
                f_hat.add_(h_BChw)
                if last_one: ls_f_hat_BChw = f_hat
                else: ls_f_hat_BChw.append(f_hat.clone())
        else:
            f_hat = ms_h_BChw[0].new_zeros(B, self.Cvae, self.v_patch_nums[0], self.v_patch_nums[0], dtype=torch.float32)
            for si, pn in enumerate(self.v_patch_nums): # from small to large
                f_hat = F.interpolate(f_hat, size=(pn, pn), mode='bicubic')
                h_BChw = self.quant_resi[si/(SN-1)](ms_h_BChw[si])
                f_hat.add_(h_BChw)
                if last_one: ls_f_hat_BChw = f_hat
                else: ls_f_hat_BChw.append(f_hat)
        
        return ls_f_hat_BChw
    
    def f_to_fhat_multiscale(self, f_BChw: torch.Tensor, v_patch_nums: Optional[Sequence[Union[int, Tuple[int, int]]]] = None) -> List[torch.Tensor]:
        """
        Convert encoder features to multi-scale reconstructed features (inference mode).
        For continuous VAE, this samples from the learned Gaussian distributions.
        
        Args:
            f_BChw: encoder output features [B, C, H, W]
            v_patch_nums: optional patch numbers for each scale
        
        Returns:
            List of accumulated feature maps at each scale
        """
        B, C, H, W = f_BChw.shape
        f_rest = f_BChw.clone()
        f_hat = torch.zeros_like(f_rest)
        
        ls_f_hat: List[torch.Tensor] = []
        
        patch_hws = [(pn, pn) if isinstance(pn, int) else (pn[0], pn[1]) for pn in (v_patch_nums or self.v_patch_nums)]
        assert patch_hws[-1][0] == H and patch_hws[-1][1] == W, f'{patch_hws[-1]=} != ({H=}, {W=})'
        
        SN = len(patch_hws)
        for si, (ph, pw) in enumerate(patch_hws):
            if 0 <= self.prog_si < si: break  # progressive training not supported
            
            # Downsample residual to current scale
            if si != SN - 1:
                rest_scale = F.interpolate(f_rest, size=(ph, pw), mode='area')
            else:
                rest_scale = f_rest
            
            # Predict mean and logvar, then sample (use mode during inference)
            moments = self.mean_logvar_conv(rest_scale)
            posterior = DiagonalGaussianDistribution(moments, deterministic=True)  # use mode for inference
            h_scale = posterior.mode()
            
            # Upsample and refine
            if si != SN - 1:
                h_BChw = F.interpolate(h_scale, size=(H, W), mode='bicubic').contiguous()
            else:
                h_BChw = h_scale.contiguous()
            
            h_BChw = self.quant_resi[si/(SN-1)](h_BChw)
            f_hat = f_hat + h_BChw
            f_rest = f_rest - h_BChw
            
            ls_f_hat.append(f_hat.clone())
        
        return ls_f_hat
    
    # ===================== idxBl_to_var_input: only used in VAR training, for getting teacher-forcing input =====================
    def idxBl_to_var_input(self, gt_ms_idx_Bl: List[torch.Tensor]) -> torch.Tensor:
        next_scales = []
        B = gt_ms_idx_Bl[0].shape[0]
        C = self.Cvae
        H = W = self.v_patch_nums[-1]
        SN = len(self.v_patch_nums)
        
        f_hat = gt_ms_idx_Bl[0].new_zeros(B, C, H, W, dtype=torch.float32)
        pn_next: int = self.v_patch_nums[0]
        for si in range(SN-1):
            if self.prog_si == 0 or (0 <= self.prog_si-1 < si): break   # progressive training: not supported yet, prog_si always -1
            h_BChw = F.interpolate(self.embedding(gt_ms_idx_Bl[si]).transpose_(1, 2).view(B, C, pn_next, pn_next), size=(H, W), mode='bicubic')
            f_hat.add_(self.quant_resi[si/(SN-1)](h_BChw))
            pn_next = self.v_patch_nums[si+1]
            next_scales.append(F.interpolate(f_hat, size=(pn_next, pn_next), mode='area').view(B, C, -1).transpose(1, 2))
        return torch.cat(next_scales, dim=1) if len(next_scales) else None    # cat BlCs to BLC, this should be float32
    
    # ===================== get_next_autoregressive_input: only used in VAR inference, for getting next step's input =====================
    def get_next_autoregressive_input(self, si: int, SN: int, f_hat: torch.Tensor, h_BChw: torch.Tensor) -> Tuple[Optional[torch.Tensor], torch.Tensor]: # only used in VAR inference
        HW = self.v_patch_nums[-1]
        if si != SN-1:
            h = self.quant_resi[si/(SN-1)](F.interpolate(h_BChw, size=(HW, HW), mode='bicubic'))     # conv after upsample
            f_hat.add_(h)
            return f_hat, F.interpolate(f_hat, size=(self.v_patch_nums[si+1], self.v_patch_nums[si+1]), mode='area')
        else:
            h = self.quant_resi[si/(SN-1)](h_BChw)
            f_hat.add_(h)
            return f_hat, f_hat


class Phi(nn.Conv2d):
    def __init__(self, embed_dim, quant_resi):
        ks = 3
        super().__init__(in_channels=embed_dim, out_channels=embed_dim, kernel_size=ks, stride=1, padding=ks//2)
        self.resi_ratio = abs(quant_resi)
    
    def forward(self, h_BChw):
        return h_BChw.mul(1-self.resi_ratio) + super().forward(h_BChw).mul_(self.resi_ratio)


class PhiShared(nn.Module):
    def __init__(self, qresi: Phi):
        super().__init__()
        self.qresi: Phi = qresi
    
    def __getitem__(self, _) -> Phi:
        return self.qresi


class PhiPartiallyShared(nn.Module):
    def __init__(self, qresi_ls: nn.ModuleList):
        super().__init__()
        self.qresi_ls = qresi_ls
        K = len(qresi_ls)
        self.ticks = np.linspace(1/3/K, 1-1/3/K, K) if K == 4 else np.linspace(1/2/K, 1-1/2/K, K)
    
    def __getitem__(self, at_from_0_to_1: float) -> Phi:
        return self.qresi_ls[np.argmin(np.abs(self.ticks - at_from_0_to_1)).item()]
    
    def extra_repr(self) -> str:
        return f'ticks={self.ticks}'


class PhiNonShared(nn.ModuleList):
    def __init__(self, qresi: List):
        super().__init__(qresi)
        # self.qresi = qresi
        K = len(qresi)
        self.ticks = np.linspace(1/3/K, 1-1/3/K, K) if K == 4 else np.linspace(1/2/K, 1-1/2/K, K)
    
    def __getitem__(self, at_from_0_to_1: float) -> Phi:
        return super().__getitem__(np.argmin(np.abs(self.ticks - at_from_0_to_1)).item())
    
    def extra_repr(self) -> str:
        return f'ticks={self.ticks}'
