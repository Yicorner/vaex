#!/bin/bash
set -e

# 用法:
#   bash train.sh                      # 默认 FEATURIZE，跑 stage 1
#   STAGE=2 bash train.sh              # FEATURIZE，跑 stage 2
#   TRAIN_ENV=HOME bash train.sh       # HOME，跑 stage 1
#   TRAIN_ENV=HOME STAGE=2 bash train.sh

TRAIN_ENV=${TRAIN_ENV:-FEATURIZE}

case "$TRAIN_ENV" in
  FEATURIZE)
    DATA_PATH_SMALL="/home/featurize/work/brats_256_t2_2021_pair_png_with_ref_small"
    DATA_PATH_LARGE="/home/featurize/data/brats_256_t2_2021_pair_png_with_ref"
    ;;
  HOME)
    DATA_PATH_SMALL="/mnt/d/DATA/brats_256_t2_2021_pair_png_with_ref_small"
    DATA_PATH_LARGE="/mnt/d/DATA/brats_256_t2_2021_pair_png_with_ref"
    ;;
  *)
    echo "Unknown TRAIN_ENV=$TRAIN_ENV, expected HOME or FEATURIZE"
    exit 1
    ;;
esac

export CUDA_VISIBLE_DEVICES=0

STAGE=${STAGE:-1}
PORT=${PORT:-13333}

# Choose dataset path without typing full path.
# Priority:
# 1) Explicit DATA_PATH (full path) if provided
# 2) DATA=small|large (or DATASET=small|large), default "small"
DATA=${DATA:-${DATASET:-small}}
case "$DATA" in
  small|SMALL)
    DEFAULT_DATA_PATH="$DATA_PATH_SMALL"
    ;;
  large|LARGE)
    DEFAULT_DATA_PATH="$DATA_PATH_LARGE"
    ;;
  *)
    echo "Unknown DATA=$DATA, expected small or large"
    exit 1
    ;;
esac
DATA_PATH=${DATA_PATH:-$DEFAULT_DATA_PATH}

# 训练配置
PATCH_NUMS=(5 6 8 10 13 16)
LR_IMG_SIZE=80
LR_CH=128
LR_VOCAB_WIDTH=32
LR_VQ_BETA=${LR_VQ_BETA:-1e-3}           # 32x5x5=800 dim latent, KL is summed, 1.0 会直接把 posterior 压塌，建议 1e-4 ~ 1e-3
LR_KL_WARMUP_EP=${LR_KL_WARMUP_EP:-1.0}  # 对 KL 权重做 N epoch 的线性 warmup，避免早期 posterior collapse
HR_VOCAB_WIDTH=32
VAE_LR=1e-4
DISC_LR=1e-4
RECON_SAVE_INTERVAL=${RECON_SAVE_INTERVAL:-0}
RECON_MAX_SAMPLES=${RECON_MAX_SAMPLES:-4}
RECON_DIR_NAME=${RECON_DIR_NAME:-}

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
  --val_and_saving_per_ep=25 \
  --lr_img_size="$LR_IMG_SIZE" \
  --lr_ch="$LR_CH" \
  --lr_vocab_width="$LR_VOCAB_WIDTH" \
  --lr_vq_beta="$LR_VQ_BETA" \
  --lr_kl_warmup_ep="$LR_KL_WARMUP_EP" \
  --vae_lr="$VAE_LR" \
  --disc_lr="$DISC_LR" \
  --ld=0.4 \
  --disc_start_ep=20 \
  --save_reconstruction_images=True \
  --reconstruction_save_interval="$RECON_SAVE_INTERVAL" \
  --reconstruction_max_samples="$RECON_MAX_SAMPLES" \
  --reconstruction_dir_name="$RECON_DIR_NAME" \
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
  --val_and_saving_per_ep=25 \
  --lr_img_size="$LR_IMG_SIZE" \
  --lr_ch="$LR_CH" \
  --lr_vocab_width="$LR_VOCAB_WIDTH" \
  --lr_vq_beta="$LR_VQ_BETA" \
  --lr_kl_warmup_ep="$LR_KL_WARMUP_EP" \
  --vocab_width="$HR_VOCAB_WIDTH" \
  --patch_nums "${PATCH_NUMS[@]}" \
  --vae_lr="$VAE_LR" \
  --disc_lr="$DISC_LR" \
  --ld=0.4 \
  --disc_start_ep=30 \
  --save_reconstruction_images=True \
  --reconstruction_save_interval="$RECON_SAVE_INTERVAL" \
  --reconstruction_max_samples="$RECON_MAX_SAMPLES" \
  --reconstruction_dir_name="$RECON_DIR_NAME" \
  --debug_loss_printed_limit=10 \
  --debug_kl_count_limit=10
else
  echo "Unknown STAGE=$STAGE, expected 1 or 2"
  exit 1
fi