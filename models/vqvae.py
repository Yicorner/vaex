"""
References:
- VectorQuantizer2: https://github.com/CompVis/taming-transformers/blob/3ba01b241669f5ade541ce990f7650a3b8f65318/taming/modules/vqvae/quantize.py#L110
- GumbelQuantize: https://github.com/CompVis/taming-transformers/blob/3ba01b241669f5ade541ce990f7650a3b8f65318/taming/modules/vqvae/quantize.py#L213
- VQVAE (VQModel): https://github.com/CompVis/stable-diffusion/blob/21f890f9da3cfbeaba8e2ac3c425ee9e998d5229/ldm/models/autoencoder.py#L14
"""
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn

from .basic_vae import Encoder, Decoder
from .quant import ContinuousMultiScaleQuantizer


class VQVAE(nn.Module):
    """Continuous Multi-Scale VAE (formerly VQ-VAE)"""
    def __init__(
        self, z_channels=32, ch=128, dropout=0.0,
        beta=1.0,               # KL loss weight (formerly commitment loss weight)
        quant_conv_ks=3,        # quant conv kernel size
        quant_resi=0.5,         # 0.5 means \phi(x) = 0.5conv(x) + (1-0.5)x
        share_quant_resi=4,     # use 4 \phi layers for K scales: partially-shared \phi
        default_qresi_counts=0, # if is 0: automatically set to len(v_patch_nums)
        v_patch_nums=(1, 2, 3, 4, 5, 6, 8, 10, 13, 16), # number of patches for each scale
        test_mode=True,
        start_drop=3,
        # Legacy parameters kept for compatibility but not used:
        vocab_size=None,        # not used in continuous VAE
        using_znorm=None,       # not used in continuous VAE
    ):
        super().__init__()
        self.v_patch_nums = v_patch_nums
        self.start_drop = start_drop
        self.test_mode = test_mode
        self.Cvae = z_channels
        # ddconfig is copied from https://github.com/CompVis/latent-diffusion/blob/e66308c7f2e64cb581c6d27ab6fbeb846828253b/models/first_stage_models/vq-f16/config.yaml
        ddconfig = dict(
            dropout=dropout, ch=ch, z_channels=z_channels,
            in_channels=3, ch_mult=(1, 1, 2, 2, 4), num_res_blocks=2,   # from vq-f16/config.yaml above
            using_sa=True, using_mid_sa=True,                           # from vq-f16/config.yaml above
            # resamp_with_conv=True,   # always True, removed.
        )
        ddconfig.pop('double_z', None)  # we don't use double_z in encoder, mean/logvar handled in quantizer
        self.encoder = Encoder(double_z=False, **ddconfig)
        self.decoder = Decoder(**ddconfig)
        
        self.downsample = 2 ** (len(ddconfig['ch_mult'])-1)
        self.quantize: ContinuousMultiScaleQuantizer = ContinuousMultiScaleQuantizer(
            Cvae=self.Cvae, beta=beta,
            default_qresi_counts=default_qresi_counts, v_patch_nums=v_patch_nums, 
            quant_resi=quant_resi, share_quant_resi=share_quant_resi,
        )
        self.quant_conv = torch.nn.Conv2d(self.Cvae, self.Cvae, quant_conv_ks, stride=1, padding=quant_conv_ks//2)
        self.post_quant_conv = torch.nn.Conv2d(self.Cvae, self.Cvae, quant_conv_ks, stride=1, padding=quant_conv_ks//2)
        
        if self.test_mode:
            self.eval()
            [p.requires_grad_(False) for p in self.parameters()]
    
    # ===================== `forward` is only used in VAE training =====================
    def forward(self, inp, ret_usages=False):   # -> rec_B3HW, usages, kl_loss
        """
        Forward pass for continuous multi-scale VAE training.
        
        Args:
            inp: input images [B, 3, H, W]
            ret_usages: whether to return usage statistics (kept for compatibility)
        
        Returns:
            rec_B3HW: reconstructed images [B, 3, H, W]
            usages: usage statistics (None for continuous VAE)
            kl_loss: KL divergence loss
        """
        B = inp.shape[0]
        # Stochastic depth: randomly skip some scales during training
        dropout_rand = torch.randint(self.start_drop, len(self.v_patch_nums) + 1, (B,))
        
        # Encode, quantize (sample from Gaussian), and decode
        f_encoded = self.encoder(inp)
        f_hat, usages, kl_loss = self.quantize(self.quant_conv(f_encoded), ret_usages=ret_usages, dropout=dropout_rand)
        rec_B3HW = self.decoder(self.post_quant_conv(f_hat))
        
        return rec_B3HW, usages, kl_loss
    # ===================== `forward` is only used in VAE training =====================
    
    def fhat_to_img(self, f_hat: torch.Tensor):
        return self.decoder(self.post_quant_conv(f_hat)).clamp_(-1, 1)
    
    def img_to_fhat_multiscale(self, inp_img_no_grad: torch.Tensor, v_patch_nums: Optional[Sequence[Union[int, Tuple[int, int]]]] = None) -> List[torch.Tensor]:
        """Convert image to multi-scale latent representations (continuous VAE)."""
        f = self.quant_conv(self.encoder(inp_img_no_grad))
        return self.quantize.f_to_fhat_multiscale(f, v_patch_nums=v_patch_nums)
    
    def embed_to_img(self, ms_h_BChw: List[torch.Tensor], all_to_max_scale: bool, last_one=False) -> Union[List[torch.Tensor], torch.Tensor]:
        if last_one:
            return self.decoder(self.post_quant_conv(self.quantize.embed_to_fhat(ms_h_BChw, all_to_max_scale=all_to_max_scale, last_one=True))).clamp_(-1, 1)
        else:
            return [self.decoder(self.post_quant_conv(f_hat)).clamp_(-1, 1) for f_hat in self.quantize.embed_to_fhat(ms_h_BChw, all_to_max_scale=all_to_max_scale, last_one=False)]
    
    def img_to_reconstructed_img(self, x, v_patch_nums: Optional[Sequence[Union[int, Tuple[int, int]]]] = None, last_one=False) -> Union[torch.Tensor, List[torch.Tensor]]:
        """Reconstruct image through continuous multi-scale VAE."""
        f = self.quant_conv(self.encoder(x))
        ls_f_hat_BChw = self.quantize.f_to_fhat_multiscale(f, v_patch_nums=v_patch_nums)
        if last_one:
            return self.decoder(self.post_quant_conv(ls_f_hat_BChw[-1])).clamp_(-1, 1)
        else:
            return [self.decoder(self.post_quant_conv(f_hat)).clamp_(-1, 1) for f_hat in ls_f_hat_BChw]
    
    def load_state_dict(self, state_dict: Dict[str, Any], strict=True, assign=False):
        # Remove legacy VQ-VAE parameters if loading from old checkpoint
        legacy_keys = ['quantize.ema_vocab_hit_SV', 'quantize.embedding.weight', 'quantize.vocab_size']
        for key in legacy_keys:
            if key in state_dict:
                print(f"Removing legacy VQ-VAE parameter: {key}")
                del state_dict[key]
        return super().load_state_dict(state_dict=state_dict, strict=strict, assign=assign)
