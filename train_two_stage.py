import os
import shutil
import sys
import time
import warnings
from collections import deque
from contextlib import nullcontext
from functools import partial
from typing import List, Optional, Tuple

import GPUtil
import numpy as np
import torch
from torch.utils.data import DataLoader

import dist
from trainer_two_stage import TwoStageVAETrainer
from utils import arg_util, misc
from utils.amp_opt import AmpOptimizer
from utils.data_lr_hr import build_lr_hr_dataset
from utils.data_sampler import DistInfiniteBatchSampler, EvalDistributedSampler
from utils.lpips import LPIPS
from utils.lr_control import filter_params, lr_wd_annealing

warnings.filterwarnings("ignore", category=FutureWarning)


class NullDDP(torch.nn.Module):
    def __init__(self, module, *args, **kwargs):
        super().__init__()
        self.module = module
        self.require_backward_grad_sync = False

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)


def create_tb_lg(args: arg_util.Args):
    with_tb_lg = dist.is_master()
    if with_tb_lg:
        os.makedirs(args.tb_log_dir_path, exist_ok=True)
        tb_lg = misc.DistLogger(misc.TensorboardLogger(log_dir=args.tb_log_dir_path, filename_suffix=f'_{misc.time_str("%m%d_%H%M")}'))
        tb_lg.flush()
    else:
        tb_lg = misc.DistLogger(None)
    dist.barrier()
    return tb_lg


def maybe_auto_resume(args: arg_util.Args):
    info = []
    if not args.resume:
        return info, 0, 0, {}, {}

    info.append(f'[auto_resume] load from args.resume @ {args.resume} ...')
    ckpt = torch.load(args.resume, map_location='cpu')
    dist.barrier()
    ep, it = ckpt.get('epoch', 0), ckpt.get('iter', 0)
    info.append(f'[auto_resume success] resume from ep{ep}, it{it}')
    return info, ep, it, ckpt.get('trainer', {}), ckpt.get('args', {})


def print_title(args: arg_util.Args):
    if args.exp_note and dist.is_master():
        border = '=' * 80
        print('\n' + border, flush=True)
        print(f'  TRAINING TITLE: {args.exp_note}', flush=True)
        print(f'  Experiment: {args.exp_name}', flush=True)
        print(border + '\n', flush=True)


def build_two_stage_trainer(args: arg_util.Args):
    auto_resume_info, start_ep, start_it, trainer_state, _ = maybe_auto_resume(args)
    tb_lg = create_tb_lg(args)
    print(f'global bs={args.bs}, local bs={args.lbs}')
    print(f'initial args:\n{str(args)}')
    [print(l) for l in auto_resume_info]

    load_mode = 'lr_only' if args.training_stage == 1 else 'both'
    dataset_train, dataset_val = build_lr_hr_dataset(args.data, load_mode=load_mode)
    ld_train = DataLoader(
        dataset=dataset_train,
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
    ld_train = iter(ld_train)
    print(f'[dataloader] gbs={args.bs}, lbs={args.lbs}, iters_train={iters_train}, mode={load_mode}')

    from torch.nn.parallel import DistributedDataParallel as DDP
    from models import DinoDisc, LR_VAE, VQVAE, build_two_stage_models
    from utils import optimizer

    vae_wo_ddp, disc_wo_ddp, lr_vae_wo_ddp = build_two_stage_models(args)
    vae_wo_ddp: VQVAE
    disc_wo_ddp: DinoDisc
    lr_vae_wo_ddp: LR_VAE

    print(f'[PT] HR VAE = {vae_wo_ddp}\n')
    print(f'[PT] LR VAE = {lr_vae_wo_ddp}\n')
    print(f'[PT] Disc model = {disc_wo_ddp}\n')

    count_p = lambda m: f'{sum(p.numel() for p in m.parameters()) / 1e6:.2f}'
    print(f'[PT][#para] HR_VAE={count_p(vae_wo_ddp)}, LR_VAE={count_p(lr_vae_wo_ddp)}, Disc={count_p(disc_wo_ddp)}')

    expected_lr_latent = args.lr_img_size // lr_vae_wo_ddp.downsample
    expected_hr_first_scale = args.patch_nums[0]
    if expected_lr_latent != expected_hr_first_scale:
        raise ValueError(
            f'Alignment requires lr_img_size / {lr_vae_wo_ddp.downsample} == patch_nums[0], '
            f'but got {args.lr_img_size} / {lr_vae_wo_ddp.downsample} = {expected_lr_latent} '
            f'and patch_nums[0] = {expected_hr_first_scale}.'
        )

    optimizers: List[AmpOptimizer] = []
    optimizer_specs = [
        ('vae', vae_wo_ddp, args.vae_opt_beta, args.vae_lr, args.vae_wd, args.grad_clip),
        ('lrv', lr_vae_wo_ddp, args.vae_opt_beta, args.vae_lr, args.vae_wd, args.grad_clip),
        ('dis', disc_wo_ddp, args.disc_opt_beta, args.disc_lr, args.disc_wd, args.grad_clip),
    ]
    for model_name, model_wo_ddp, opt_beta, lr, wd, clip in optimizer_specs:
        for p in model_wo_ddp.parameters():
            if p.requires_grad:
                dist.broadcast(p.data, src_rank=0)
        ndim_dict = {name: para.ndim for name, para in model_wo_ddp.named_parameters() if para.requires_grad}
        nowd_keys = {
            'cls_token', 'start_token', 'task_token', 'cfg_uncond',
            'pos_embed', 'pos_1LC', 'pos_start', 'start_pos', 'lvl_embed',
            'gamma', 'beta',
            'ada_gss', 'moe_bias',
            'class_emb', 'embedding',
            'norm_scale',
        }
        _, _, para_groups = filter_params(model_wo_ddp, ndim_dict, nowd_keys=nowd_keys)
        beta1, beta2 = map(float, opt_beta.split('_'))
        opt_clz = {
            'adam': partial(torch.optim.AdamW, betas=(beta1, beta2), fused=args.fuse_opt),
            'adamw': partial(torch.optim.AdamW, betas=(beta1, beta2), fused=args.fuse_opt),
            'lamb': partial(optimizer.LAMBtimm, betas=(beta1, beta2), max_grad_norm=clip),
            'lion': partial(optimizer.Lion, betas=(beta1, beta2), max_grad_norm=clip),
        }[args.opt]
        opt_kw = dict(lr=lr, weight_decay=0)
        if args.oeps:
            opt_kw['eps'] = args.oeps
        optimizers.append(
            AmpOptimizer(
                model_name,
                model_maybe_fsdp=None,
                fp16=args.fp16,
                bf16=args.bf16,
                zero=args.zero,
                optimizer=opt_clz(params=para_groups, **opt_kw),
                grad_clip=clip,
                n_gradient_accumulation=args.grad_accu,
            )
        )

    vae_optim, lr_vae_optim, disc_optim = optimizers
    vae_wo_ddp = args.compile_model(vae_wo_ddp, args.compile_vae)
    lr_vae_wo_ddp = args.compile_model(lr_vae_wo_ddp, args.compile_vae)
    disc_wo_ddp = args.compile_model(disc_wo_ddp, args.compile_disc)
    lpips_loss: LPIPS = args.compile_model(LPIPS(args.lpips_path).to(args.device), fast=args.compile_lpips)

    ddp_class = DDP if dist.initialized() else NullDDP
    vae = ddp_class(vae_wo_ddp, device_ids=[dist.get_local_rank()], find_unused_parameters=False, static_graph=args.ddp_static, broadcast_buffers=False)
    lr_vae = ddp_class(lr_vae_wo_ddp, device_ids=[dist.get_local_rank()], find_unused_parameters=False, static_graph=args.ddp_static, broadcast_buffers=False)
    disc = ddp_class(disc_wo_ddp, device_ids=[dist.get_local_rank()], find_unused_parameters=False, static_graph=args.ddp_static, broadcast_buffers=False)

    vae_optim.model_maybe_fsdp = vae if args.zero else vae_wo_ddp
    lr_vae_optim.model_maybe_fsdp = lr_vae if args.zero else lr_vae_wo_ddp
    disc_optim.model_maybe_fsdp = disc if args.zero else disc_wo_ddp

    trainer = TwoStageVAETrainer(
        is_visualizer=dist.is_master(),
        vae=vae,
        vae_wo_ddp=vae_wo_ddp,
        lr_vae=lr_vae,
        lr_vae_wo_ddp=lr_vae_wo_ddp,
        disc=disc,
        disc_wo_ddp=disc_wo_ddp,
        vae_opt=vae_optim,
        lr_vae_opt=lr_vae_optim,
        disc_opt=disc_optim,
        ema_ratio=args.ema,
        dcrit=args.dcrit,
        daug=args.disc_aug_prob,
        lpips_loss=lpips_loss,
        lp_reso=args.lpr,
        wei_l1=args.l1,
        wei_l2=args.l2,
        wei_entropy=args.le,
        wei_lpips=args.lp,
        wei_disc=args.ld,
        adapt_type=args.gada,
        bcr=args.bcr,
        bcr_cut=args.bcr_cut,
        reg=args.reg,
        reg_every=args.reg_every,
        disc_grad_ckpt=args.disc_grad_ckpt,
        training_stage=args.training_stage,
        use_alignment_loss=args.use_lr_hr_alignment,
        alignment_loss_weight=args.alignment_loss_weight,
        alignment_scale_index=0,
        dbg_unused=args.dbg_unused,
        dbg_nan=args.dbg_nan,
    )

    if trainer_state:
        trainer.load_state_dict(trainer_state, strict=False)

    if args.training_stage == 2 and args.lr_vae_resume and not trainer_state:
        ckpt = torch.load(args.lr_vae_resume, map_location='cpu')
        lr_state = ckpt.get('trainer', {}).get('lr_vae_ema') or ckpt.get('trainer', {}).get('lr_vae_wo_ddp')
        if lr_state is None:
            raise KeyError(f'Cannot find LR VAE weights in {args.lr_vae_resume}')
        trainer.lr_vae_wo_ddp.load_state_dict(lr_state, strict=False)
        if trainer.using_ema:
            trainer.lr_vae_ema.load_state_dict(lr_state, strict=False)
        print(f'[stage2 warm start] loaded LR VAE from {args.lr_vae_resume}')

    trainer.set_training_stage(args.training_stage)
    if args.training_stage == 2:
        args.lr_vae_frozen = True

    return tb_lg, trainer, start_ep, start_it, iters_train, ld_train, ld_val


def save_checkpoint(args: arg_util.Args, trainer: TwoStageVAETrainer, epoch: int, is_best: bool = False):
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


def train_one_ep(ep: int, is_first_ep: bool, start_it: int, args: arg_util.Args, tb_lg: misc.TensorboardLogger, ld_or_itrt, iters_train: int, trainer: TwoStageVAETrainer, logging_params_milestone, ld_val=None):
    step_cnt = 0
    me = misc.MetricLogger(delimiter='  ')
    [me.add_meter(x, misc.SmoothedValue(window_size=1, fmt='{value:.2g}')) for x in ['glr', 'dlr']]
    [me.add_meter(x, misc.SmoothedValue(window_size=1, fmt='{median:.2f} ({global_avg:.2f})')) for x in ['gnm', 'dnm']]
    for l in ['L1', 'NLL', 'Ld', 'Wg', 'L_align']:
        me.add_meter(l, misc.SmoothedValue(fmt='{median:.3f} ({global_avg:.3f})'))
    me.add_meter("usage", misc.SmoothedValue(fmt='{median:.2f} ({global_avg:.2f})'))
    me.add_meter("Lkl", misc.SmoothedValue(fmt='{median:.2e} ({global_avg:.2e})'))
    header = f'[Ep]: [{ep:4d}/{args.ep}]'

    if is_first_ep:
        warnings.filterwarnings('ignore', category=DeprecationWarning)
        warnings.filterwarnings('ignore', category=UserWarning)

    g_it, wp_it, max_it = ep * iters_train, args.warmup_ep * iters_train, args.ep * iters_train
    disc_start = args.disc_start_ep * iters_train
    disc_wp_it, disc_max_it = args.disc_warmup_ep * iters_train, max_it - disc_start
    maybe_record_function = nullcontext

    last_t_perf = time.perf_counter()
    speed_ls: deque = g_speed_ls
    FREQ = min(50, max(iters_train // 2 - 1, 1))

    for it, batch in me.log_every(start_it, iters_train, ld_or_itrt, max(10, iters_train // 1000), header):
        if (it + 1) % FREQ == 0:
            speed_ls.append((time.perf_counter() - last_t_perf) / FREQ)
            args.iter_speed = float(np.median(speed_ls))
            args.img_per_day = args.bs / args.iter_speed * 3600 * 24 / 1e6
            args.max_nvidia_smi = max(args.max_nvidia_smi, max(gpu.memoryUsed for gpu in GPUtil.getGPUs()) / 1024)
            last_t_perf = time.perf_counter()

        if it < start_it:
            continue

        if trainer.training_stage == 1:
            batch = batch.to(args.device, non_blocking=True)
        else:
            batch = tuple(x.to(args.device, non_blocking=True) for x in batch)

        g_it = ep * iters_train + it
        disc_g_it = g_it - disc_start
        args.cur_it = f'{it + 1}/{iters_train}'
        active_opt = trainer.lr_vae_opt.optimizer if trainer.training_stage == 1 else trainer.vae_opt.optimizer
        min_glr, max_glr, min_gwd, max_gwd = lr_wd_annealing(args.sche, active_opt, args.vae_lr, args.vae_wd, g_it, wp_it, max_it, wp0=args.wp0, wpe=args.sche_end)
        if disc_g_it >= 0:
            min_dlr, max_dlr, min_dwd, max_dwd = lr_wd_annealing(args.sche, trainer.disc_opt.optimizer, args.disc_lr, args.disc_wd, disc_g_it, disc_wp_it, disc_max_it, wp0=args.wp0, wpe=args.sche_end)
        else:
            min_dlr = max_dlr = min_dwd = max_dwd = 0

        stepping = (g_it + 1) % args.grad_accu == 0
        step_cnt += int(stepping)
        warmup_disc_schedule = 0 if disc_g_it < 0 else min(1.0, disc_g_it / max(disc_wp_it, 1))
        fade_blur_schedule = 0 if disc_g_it < 0 else min(1.0, disc_g_it / max(disc_wp_it * 2, 1))
        fade_blur_schedule = 1 - fade_blur_schedule

        grad_norm_g, scale_log2_g, grad_norm_d, scale_log2_d = trainer.train_step(
            ep=ep,
            it=it,
            g_it=g_it,
            stepping=stepping,
            regularizing=args.reg > 0 and (g_it % args.reg_every == 0),
            metric_lg=me,
            logging_params=stepping and step_cnt == 1 and (ep < 4 or ep in logging_params_milestone),
            tb_lg=tb_lg,
            inp=batch,
            warmup_disc_schedule=warmup_disc_schedule,
            fade_blur_schedule=fade_blur_schedule,
            maybe_record_function=maybe_record_function,
            args=args,
        )

        me.update(glr=max_glr, dlr=max_dlr)
        tb_lg.set_step(step=g_it)
        if tb_lg.loggable():
            tb_lg.update(head='PT_opt_lr/lr_max', sche_glr=max_glr, sche_dlr=max_dlr)
            tb_lg.update(head='PT_opt_lr/lr_min', sche_glr=min_glr, sche_dlr=min_dlr)
            tb_lg.update(head='PT_opt_wd/wd_max', sche_gwd=max_gwd, sche_dwd=max_dwd)
            tb_lg.update(head='PT_opt_wd/wd_min', sche_gwd=min_gwd, sche_dwd=min_dwd)
            if scale_log2_g is not None:
                tb_lg.update(head='PT_opt_grad/fp16', scale_log2_g=scale_log2_g, scale_log2_d=scale_log2_d)
            tb_lg.update(head='PT_opt_grad/grad', grad_norm_g=grad_norm_g, grad_norm_d=grad_norm_d)

        if ld_val is not None and it in me.log_iters:
            val_ret = trainer.eval_ep(ld_val, max_batches=10)
            if trainer.training_stage == 1:
                val_L_rec_mean, val_psnr_mean, val_ssim_mean = val_ret
                print(f' [*] [ep{ep}] [it{it}] val_L_rec_mean: {val_L_rec_mean:.4f}, PSNR: {val_psnr_mean:.4f}, SSIM: {val_ssim_mean:.4f}')
            else:
                val_L_rec_mean, val_psnr_mean, val_ssim_mean, val_align_mean = val_ret
                print(f' [*] [ep{ep}] [it{it}] val_L_rec_mean: {val_L_rec_mean:.4f}, PSNR: {val_psnr_mean:.4f}, SSIM: {val_ssim_mean:.4f}, Align: {val_align_mean:.4f}')

    me.synchronize_between_processes()
    remain_steps = max_it - (g_it + 1)
    return {k: meter.global_avg for k, meter in me.meters.items()}, me.iter_time.time_preds(remain_steps + (args.ep - ep) * 15)


def main_training():
    args: arg_util.Args = arg_util.init_dist_and_get_args()
    if args.dbg_unused:
        torch.autograd.set_detect_anomaly(True)

    print_title(args)
    tb_lg, trainer, start_ep, start_it, iters_train, ld_train, ld_val = build_two_stage_trainer(args)

    start_time = time.time()
    val_min_L_rec = float('inf')
    logging_params_milestone: List[int] = np.linspace(1, args.ep, 11, dtype=int).tolist()

    trainer.vae_opt.log_param(ep=-1, tb_lg=tb_lg)
    trainer.lr_vae_opt.log_param(ep=-1, tb_lg=tb_lg)
    trainer.disc_opt.log_param(ep=-1, tb_lg=tb_lg)

    for ep in range(start_ep, args.ep):
        tb_lg.set_step(ep * iters_train)
        with nullcontext():
            stats, (sec, remain_time, finish_time) = train_one_ep(
                ep,
                ep == start_ep,
                start_it if ep == start_ep else 0,
                args,
                tb_lg,
                ld_train,
                iters_train,
                trainer,
                logging_params_milestone,
                ld_val=ld_val,
            )

        Lnll = stats['NLL']
        L1 = stats['L1']
        Ld = stats.get('Ld', 0.0)
        wei_g = stats.get('Wg', 0.0)
        args.last_Lnll, args.last_L1, args.last_Ld, args.last_wei_g = Lnll, L1, Ld, wei_g
        args.cur_phase = 'PT2'
        args.cur_ep = f'{ep + 1}/{args.ep}'
        args.remain_time, args.finish_time = remain_time, finish_time

        print(f'  [*] [ep{ep}] Remain: {remain_time}, Finish: {finish_time}')
        tb_lg.update(head='PT_ep_loss', step=ep + 1, L1rec=L1, Lnll=Lnll, Ld=Ld, wei_g=wei_g, L_align=stats.get('L_align', 0.0))
        tb_lg.update(head='PT_z_burnout', step=ep + 1, rest_hours=round(sec / 60 / 60, 2))

        is_val_and_also_saving = (ep + 1) % args.val_and_saving_per_ep == 0 or (ep + 1) == args.ep
        if is_val_and_also_saving:
            val_ret = trainer.eval_ep(ld_val)
            if trainer.training_stage == 1:
                val_L_rec_mean, val_psnr_mean, val_ssim_mean = val_ret
                print(f' [*] [ep{ep}] val_L_rec_mean: {val_L_rec_mean:.4f}, PSNR: {val_psnr_mean:.4f}, SSIM: {val_ssim_mean:.4f}')
            else:
                val_L_rec_mean, val_psnr_mean, val_ssim_mean, val_align_mean = val_ret
                print(f' [*] [ep{ep}] val_L_rec_mean: {val_L_rec_mean:.4f}, PSNR: {val_psnr_mean:.4f}, SSIM: {val_ssim_mean:.4f}, Align: {val_align_mean:.4f}')
            is_best = val_L_rec_mean < val_min_L_rec
            val_min_L_rec = min(val_min_L_rec, val_L_rec_mean)
            save_checkpoint(args, trainer, ep + 1, is_best=is_best)

    total_time = f'{(time.time() - start_time) / 60 / 60:.1f}h'
    print(f'  [*] [finished] Total Time: {total_time}')
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
