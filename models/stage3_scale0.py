from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .basic_vae import Encoder


class Stage3Scale0Encoder(nn.Module):
    """Encode LR images into the frozen stage2 VAE's first-scale latent space.

    The trainable path is intentionally small and deterministic:

        LR -> encoder -> quant_conv -> adaptive_pool(n,n) -> frozen mean head -> mean

    `mean_logvar_conv` is copied from the stage2 VAE quantizer and frozen so the
    predicted latent has the same meaning as `VQVAE.img_to_scale_posterior_stats`.
    """

    def __init__(
        self,
        z_channels: int = 32,
        ch: int = 128,
        dropout: float = 0.0,
        quant_conv_ks: int = 3,
        latent_size: int = 4,
        img_channels: int = 3,
        freeze_mean_head: bool = True,
    ):
        super().__init__()
        self.Cvae = int(z_channels)
        self.ch = int(ch)
        self.dropout = float(dropout)
        self.quant_conv_ks = int(quant_conv_ks)
        self.latent_size = int(latent_size)
        self.img_channels = int(img_channels)

        ddconfig = dict(
            dropout=self.dropout,
            ch=self.ch,
            z_channels=self.Cvae,
            in_channels=self.img_channels,
            ch_mult=(1, 1, 2, 2, 4),
            num_res_blocks=2,
            using_sa=True,
            using_mid_sa=True,
        )
        self.encoder = Encoder(double_z=False, **ddconfig)
        self.quant_conv = nn.Conv2d(
            self.Cvae,
            self.Cvae,
            self.quant_conv_ks,
            stride=1,
            padding=self.quant_conv_ks // 2,
        )
        self.mean_logvar_conv = nn.Conv2d(self.Cvae, 2 * self.Cvae, kernel_size=1)
        if freeze_mean_head:
            self.freeze_mean_head()

    def freeze_mean_head(self) -> None:
        for p in self.mean_logvar_conv.parameters():
            p.requires_grad_(False)
        self.mean_logvar_conv.eval()

    def initialize_from_stage2_vae(self, stage2_vae: nn.Module) -> None:
        """Copy compatible stage2 weights.

        The encoder and quant conv become the stage3 trainable initialization.
        The Gaussian moment head is copied and kept frozen.
        """
        self.encoder.load_state_dict(stage2_vae.encoder.state_dict())
        self.quant_conv.load_state_dict(stage2_vae.quant_conv.state_dict())
        self.mean_logvar_conv.load_state_dict(stage2_vae.quantize.mean_logvar_conv.state_dict())
        self.freeze_mean_head()

    def encode_feature(self, inp: torch.Tensor) -> torch.Tensor:
        f = self.quant_conv(self.encoder(inp))
        if f.shape[-2:] != (self.latent_size, self.latent_size):
            f = F.adaptive_avg_pool2d(f, output_size=(self.latent_size, self.latent_size))
        return f

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        f = self.encode_feature(inp)
        with torch.amp.autocast('cuda', enabled=False):
            moments = self.mean_logvar_conv(f.float())
            mean, _ = torch.chunk(moments, 2, dim=1)
        return mean.to(dtype=f.dtype)

    def extra_repr(self) -> str:
        return (
            f'Cvae={self.Cvae}, ch={self.ch}, latent_size={self.latent_size}, '
            f'img_channels={self.img_channels}, quant_conv_ks={self.quant_conv_ks}'
        )


def stage3_config_from_ckpt_args(args_state: dict, latent_size: int = 4) -> Tuple[int, int, float, int, int]:
    """Return `(Cvae, ch, dropout, img_channels, latent_size)` from a ckpt args dict."""
    Cvae = int(args_state.get('vocab_width', args_state.get('Ct5', 32)))
    ch = int(args_state.get('ch', args_state.get('vae_ch', 128)))
    dropout = float(args_state.get('drop_out', 0.0))
    img_channels = int(args_state.get('img_channels', 3))
    return Cvae, ch, dropout, img_channels, int(latent_size)
