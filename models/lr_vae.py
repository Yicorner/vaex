"""
Single-scale VAE for Low-Resolution images.
This VAE encodes LR images into 5x5 tokens using simple KL divergence.
"""
from typing import Tuple
import torch
import torch.nn as nn

from .basic_vae import Encoder, Decoder
from .quant import DiagonalGaussianDistribution


class LR_VAE(nn.Module):
    """Single-scale continuous VAE for Low-Resolution images (output: 5x5 tokens)"""
    
    def __init__(
        self, 
        z_channels=32, 
        ch=128, 
        dropout=0.0,
        beta=1.0,  # KL loss weight
        quant_conv_ks=3,
        test_mode=False,
    ):
        super().__init__()
        self.test_mode = test_mode
        self.Cvae = z_channels
        self.kl_weight = beta
        
        # Encoder and Decoder configuration
        # Same as HR VAE but we'll ensure output is 5x5
        ddconfig = dict(
            dropout=dropout, 
            ch=ch, 
            z_channels=z_channels,
            in_channels=3, 
            ch_mult=(1, 1, 2, 2, 4),  # 5 levels, downsample 2^4 = 16x
            num_res_blocks=2,
            using_sa=True, 
            using_mid_sa=True,
        )
        
        self.encoder = Encoder(double_z=False, **ddconfig)
        self.decoder = Decoder(**ddconfig)
        
        self.downsample = 2 ** (len(ddconfig['ch_mult']) - 1)  # 16
        
        # Conv layers for quantization
        self.quant_conv = nn.Conv2d(self.Cvae, self.Cvae, quant_conv_ks, stride=1, padding=quant_conv_ks//2)
        self.post_quant_conv = nn.Conv2d(self.Cvae, self.Cvae, quant_conv_ks, stride=1, padding=quant_conv_ks//2)
        
        # Mean and logvar prediction (output 2*Cvae channels)
        self.mean_logvar_conv = nn.Conv2d(self.Cvae, 2 * self.Cvae, kernel_size=1, stride=1, padding=0)
        self._init_mean_logvar_conv()
        
        if self.test_mode:
            self.eval()
            [p.requires_grad_(False) for p in self.parameters()]
    
    def _init_mean_logvar_conv(self):
        """Initialize mean_logvar_conv for stable VAE training"""
        nn.init.xavier_normal_(self.mean_logvar_conv.weight.data, gain=1.0)
        if self.mean_logvar_conv.bias is not None:
            Cvae = self.Cvae
            self.mean_logvar_conv.bias.data[:Cvae].zero_()  # mean channels
            self.mean_logvar_conv.bias.data[Cvae:].fill_(-2.0)  # logvar channels
    
    def forward(self, inp: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass for LR VAE training.
        
        Args:
            inp: LR input images [B, 3, H, W] where H=W should be 80 (to get 5x5 after 16x downsample)
        
        Returns:
            rec_B3HW: reconstructed images [B, 3, H, W]
            f_5x5: latent features at 5x5 resolution [B, C, 5, 5]
            kl_loss: KL divergence loss (scalar)
        """
        # Encode
        f_encoded = self.encoder(inp)  # [B, C, 5, 5] if input is 80x80
        f_encoded = self.quant_conv(f_encoded)  # [B, C, 5, 5]
        
        # Predict mean and logvar
        moments = self.mean_logvar_conv(f_encoded)  # [B, 2*C, 5, 5]
        posterior = DiagonalGaussianDistribution(moments, deterministic=not self.training)
        
        # Sample using reparameterization trick
        if self.training:
            f_5x5 = posterior.sample()  # [B, C, 5, 5]
        else:
            f_5x5 = posterior.mode()  # use mean during inference
        
        # Compute KL divergence (simple version, no compression)
        kl_loss = posterior.kl()  # [B] - sum over C×H×W dimensions
        kl_loss = torch.mean(kl_loss) * self.kl_weight  # average over batch
        
        # Decode
        rec_B3HW = self.decoder(self.post_quant_conv(f_5x5))
        
        return rec_B3HW, f_5x5, kl_loss

    def encode_to_posterior_stats(self, inp: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode LR image and return posterior mean/logvar at the latent scale.

        Returns:
            mean: [B, C, h, w]
            logvar: [B, C, h, w]
        """
        f_encoded = self.quant_conv(self.encoder(inp))
        moments = self.mean_logvar_conv(f_encoded)
        posterior = DiagonalGaussianDistribution(moments, deterministic=not self.training)
        return posterior.mean, posterior.logvar

    def encode_to_posterior_mean(self, inp: torch.Tensor) -> torch.Tensor:
        """
        Encode LR image and return posterior mean.
        This is the recommended representation for cross-model alignment.
        """
        mean, _ = self.encode_to_posterior_stats(inp)
        return mean
    
    def encode_to_5x5(self, inp: torch.Tensor) -> torch.Tensor:
        """
        Encode LR image to 5x5 latent (for inference/alignment).
        
        Args:
            inp: LR input images [B, 3, H, W]
        
        Returns:
            f_5x5: latent features [B, C, 5, 5]
        """
        with torch.no_grad():
            return self.encode_to_posterior_mean(inp)
    
    def decode_from_5x5(self, f_5x5: torch.Tensor) -> torch.Tensor:
        """
        Decode from 5x5 latent to image.
        
        Args:
            f_5x5: latent features [B, C, 5, 5]
        
        Returns:
            rec: reconstructed images [B, 3, H, W]
        """
        return self.decoder(self.post_quant_conv(f_5x5)).clamp(-1, 1)

