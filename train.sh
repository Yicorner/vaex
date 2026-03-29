#!/bin/bash
set -e

DATA_PATH_SMALL="/home/featurize/work/brats_256_t2_2021_pair_png_with_ref_small"
DATA_PATH_LARGE="/home/featurize/data/brats_256_t2_2021_pair_png_with_ref"

export CUDA_VISIBLE_DEVICES=0

# 用法:
#   bash train.sh            # 默认跑 stage 1
#   STAGE=1 bash train.sh    # 跑 stage 1
#   STAGE=2 bash train.sh    # 跑 stage 2

STAGE=${STAGE:-1}
PORT=${PORT:-13333}
DATA_PATH=${DATA_PATH:-$DATA_PATH_LARGE}

# 训练配置
PATCH_NUMS=(5 6 8 10 13 16)
LR_IMG_SIZE=80
LR_CH=128
LR_VOCAB_WIDTH=32
LR_VQ_BETA=1.0
HR_VOCAB_WIDTH=32
VAE_LR=1e-4
DISC_LR=1e-4

# 输出目录
STAGE1_BED="myvaex_stage1_lr_vae"
STAGE2_BED="myvaex_stage2_hr_aligned"
STAGE1_CKPT="${STAGE1_BED}/ckpt-best.pth"

if [ "$STAGE" = "1" ]; then
  torchrun --nproc_per_node=1 --nnodes=1 --node_rank=0 --master_addr=127.0.0.1 --master_port="$PORT" train_two_stage.py \
  --exp_name="stage1_lr_vae" --bed="$STAGE1_BED" \
  --exp_note="Stage 1: train LR VAE to posterior mean tokens" \
  --data="$DATA_PATH" \
  --training_stage=1 \
  --lbs=8 \
  --ep=100 \
  --val_and_saving_per_ep=5 \
  --lr_img_size="$LR_IMG_SIZE" \
  --lr_ch="$LR_CH" \
  --lr_vocab_width="$LR_VOCAB_WIDTH" \
  --lr_vq_beta="$LR_VQ_BETA" \
  --vae_lr="$VAE_LR" \
  --disc_lr="$DISC_LR" \
  --ld=0.4 \
  --disc_start_ep=20 \
  --save_reconstruction_images=True \
  --debug_loss_printed_limit=10 \
  --debug_kl_count_limit=10
elif [ "$STAGE" = "2" ]; then
  torchrun --nproc_per_node=1 --nnodes=1 --node_rank=0 --master_addr=127.0.0.1 --master_port="$PORT" train_two_stage.py \
  --exp_name="stage2_hr_vae_aligned" --bed="$STAGE2_BED" \
  --exp_note="Stage 2: train HR multi-scale VAE with first-scale posterior mean alignment" \
  --data="$DATA_PATH" \
  --training_stage=2 \
  --use_lr_hr_alignment=True \
  --alignment_loss_weight=1.0 \
  --lr_vae_resume="$STAGE1_CKPT" \
  --lbs=4 \
  --ep=150 \
  --val_and_saving_per_ep=5 \
  --lr_img_size="$LR_IMG_SIZE" \
  --lr_ch="$LR_CH" \
  --lr_vocab_width="$LR_VOCAB_WIDTH" \
  --lr_vq_beta="$LR_VQ_BETA" \
  --vocab_width="$HR_VOCAB_WIDTH" \
  --patch_nums "${PATCH_NUMS[@]}" \
  --vae_lr="$VAE_LR" \
  --disc_lr="$DISC_LR" \
  --ld=0.4 \
  --disc_start_ep=30 \
  --save_reconstruction_images=True \
  --debug_loss_printed_limit=10 \
  --debug_kl_count_limit=10
else
  echo "Unknown STAGE=$STAGE, expected 1 or 2"
  exit 1
fi