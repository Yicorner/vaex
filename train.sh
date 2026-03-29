#!/bin/bash
DATA_PATH_SMALL="/home/featurize/work/brats_256_t2_2021_pair_png_with_ref_small"
DATA_PATH_LARGE="/home/featurize/data/brats_256_t2_2021_pair_png_with_ref"

export CUDA_VISIBLE_DEVICES=0

# ==================== Two-Stage Training Example ====================
#
# Stage 1: Train LR VAE (单尺度，输出5x5)
# torchrun --nproc_per_node=1 --nnodes=1 --node_rank=0 --master_addr=127.0.0.1 --master_port=13333 train.py \
# --exp_name="stage1_lr_vae" --bed="myvaex_stage1" \
# --exp_note="Stage 1: Training LR VAE to encode LR images to 5x5 tokens" \
# --lbs=8 --vae_lr=1e-4 --disc_lr=1e-4 \
# --data="$DATA_PATH_LARGE" \
# --val_and_saving_per_ep=5 --ep=100 \
# --training_stage=1 \
# --lr_img_size=80 --lr_ch=128 --lr_vocab_width=32 --lr_vq_beta=1.0 \
# --ld=0.4 --disc_start_ep=20 \
# --save_reconstruction_images=True \
# --debug_loss_printed_limit=10 --debug_kl_count_limit=10
#
# Stage 2: Train HR VAE with LR-HR alignment (多尺度，与LR的5x5对齐)
# torchrun --nproc_per_node=1 --nnodes=1 --node_rank=0 --master_addr=127.0.0.1 --master_port=13333 train.py \
# --exp_name="stage2_hr_vae_aligned" --bed="myvaex_stage2" \
# --exp_note="Stage 2: Training HR multi-scale VAE with LR-HR 5x5 alignment" \
# --lbs=4 --vae_lr=1e-4 --disc_lr=1e-4 \
# --data="$DATA_PATH_LARGE" \
# --val_and_saving_per_ep=5 --ep=150 \
# --training_stage=2 \
# --use_lr_hr_alignment=True --alignment_loss_weight=1.0 \
# --vocab_size=1024 --vocab_width=32 --patch_nums 5 6 8 10 13 16 \
# --ld=0.4 --disc_start_ep=30 \
# --save_reconstruction_images=True \
# --debug_loss_printed_limit=10 --debug_kl_count_limit=10 \
# --resume="myvaex_stage1/ckpt-best.pth"  # Load pretrained LR VAE from Stage 1
#
# ====================================================================

# Original single-stage training (HR only, for reference)
torchrun --nproc_per_node=1 --nnodes=1 --node_rank=0 --master_addr=127.0.0.1 --master_port=13333 train.py \
--exp_name="brats_256_t2_2021_pair_png_with_ref" --bed="myvaex" \
--exp_note="测试新的patch_nums配置: 5,6,8,10,13,16" \
--lbs=4 --vae_lr=1e-4 --disc_lr=1e-4 \
--data="$DATA_PATH_LARGE"  \
--val_and_saving_per_ep=1 \
--ep=2 \
--vocab_size=1024 --vocab_width=32 \
--ld=0 --disc_start_ep=1000 \
--save_reconstruction_images=True \
--debug_loss_printed_limit=10 \
--debug_kl_count_limit=10

# --data='./data/brats_256_t1_2021_pair_4x'  \

# --resume="local_output/ckpt_last.pth"
# --data='./data/df2k_ost/GT_sub'
# --data='./data/DIV2K_train_HR' 