from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.arg_util import Args
from .quant import ContinuousMultiScaleQuantizer
from .vqvae import VQVAE
from .lr_vae import LR_VAE
from .dino import DinoDisc
from .basic_vae import Encoder


def build_vae_disc(args: Args) -> Tuple[VQVAE, DinoDisc]:
    # disable built-in initialization for speed
    for clz in (
        nn.Linear, nn.Embedding,
        nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d,
        nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm, nn.GroupNorm, nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d,
    ):
        setattr(clz, 'reset_parameters', lambda self: None)
    
    # build models
    vae = VQVAE(
        vocab_size=args.vocab_size,
        z_channels=args.vocab_width,
        ch=args.ch,
        beta=args.vq_beta,
        test_mode=False,
        share_quant_resi=args.share_quant_resi,
        v_patch_nums=args.patch_nums,
        debug_kl_count_limit=args.debug_kl_count_limit,
        img_channels=args.img_channels,
    ).to(args.device)
    disc = DinoDisc(
        device=args.device, depth=args.dino_depth, key_depths=(2, 5, 8, 11),
        ks=args.dino_kernel_size, norm_type=args.disc_norm, using_spec_norm=args.disc_spec_norm, norm_eps=1e-6,
    ).to(args.device)
    
    # init weights for HR VAE
    need_init = [
        vae.quant_conv,
        vae.quantize,
        vae.post_quant_conv,
        vae.decoder,
    ]
    if isinstance(vae.encoder, Encoder):
        need_init.insert(0, vae.encoder)
    for vv in need_init:
        init_weights(vv, args.vae_init)
    init_weights(disc, args.disc_init)
    return vae, disc


def build_two_stage_models(args: Args) -> Tuple[VQVAE, DinoDisc, LR_VAE]:
    vae, disc = build_vae_disc(args)
    lr_vae = LR_VAE(
        z_channels=args.lr_vocab_width,
        ch=args.lr_ch,
        dropout=args.drop_out,
        beta=args.lr_vq_beta,
        test_mode=False,
        img_channels=args.img_channels,
    ).to(args.device)

    lr_need_init = [
        lr_vae.encoder,
        lr_vae.quant_conv,
        lr_vae.post_quant_conv,
        lr_vae.decoder,
    ]
    for vv in lr_need_init:
        init_weights(vv, args.vae_init)

    return vae, disc, lr_vae


def _init_conv_weight(weight: torch.Tensor, conv_std_or_gain: float):
    if conv_std_or_gain > 0:
        nn.init.trunc_normal_(weight, std=conv_std_or_gain)
    else:
        nn.init.xavier_normal_(weight, gain=-conv_std_or_gain)


def _reset_spectral_norm_state(module: nn.Module):
    if not all(hasattr(module, name) for name in ('weight_orig', 'weight_u', 'weight_v')):
        return

    with torch.no_grad():
        weight = module.weight_orig
        if not torch.isfinite(weight).all():
            raise RuntimeError(f'Non-finite spectral-norm weight in {type(module).__name__}')
        module.weight_u.copy_(F.normalize(torch.randn_like(module.weight_u), dim=0, eps=1e-12))
        module.weight_v.copy_(F.normalize(torch.randn_like(module.weight_v), dim=0, eps=1e-12))


def init_weights(model, conv_std_or_gain):
    print(f'[init_weights] {type(model).__name__} with {"std" if conv_std_or_gain > 0 else "gain"}={abs(conv_std_or_gain):g}')
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight.data, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias.data, 0.)
        elif isinstance(m, nn.Embedding):
            nn.init.trunc_normal_(m.weight.data, std=0.02)
            if m.padding_idx is not None:
                m.weight.data[m.padding_idx].zero_()
        elif isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d)):
            weight = m.weight_orig if hasattr(m, 'weight_orig') else m.weight
            _init_conv_weight(weight.data, conv_std_or_gain)
            _reset_spectral_norm_state(m)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm, nn.GroupNorm, nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d)):
            if m.bias is not None: nn.init.constant_(m.bias.data, 0.)
            if m.weight is not None: nn.init.constant_(m.weight.data, 1.)
