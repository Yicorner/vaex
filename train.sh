#!/bin/bash
set -e

# 用法:
#   bash train.sh                      # 默认 FEATURIZE，跑 stage 1
#   STAGE=2 bash train.sh              # FEATURIZE，跑 stage 2（默认 scale0 图像空间对齐，不依赖 stage1 latent）
#   TRAIN_ENV=HOME bash train.sh       # HOME，跑 stage 1
#   TRAIN_ENV=HOME STAGE=2 bash train.sh
#   VAL_AND_SAVING_PER_EP=5 bash train.sh   # 每 N epoch 验证与存 ckpt，默认 2（也可用 val_and_saving_per_ep）
#   EP=80 bash train.sh                     # 两阶段训练 epoch 数（也可用 ep）；stage 默认 100 / 150
#   STAGE1_EP=120 bash train.sh             # 仅 stage 1；stage2 用 STAGE2_EP（或 stage1_ep / stage2_ep）
#   STAGE=2 STAGE2_USE_KL=False bash train.sh  # stage2 训练成 deterministic AE，不采样、不加 KL

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

# 数据子目录（可按需覆盖）
LR_FOLDER=${LR_FOLDER:-${lr_folder:-LR}}
HR_FOLDER=${HR_FOLDER:-${hr_folder:-HR}}

# 训练配置（支持环境变量覆盖，便于不同数据配置复用）
PATCH_NUMS_STR=${PATCH_NUMS:-${patch_nums:-"5 6 8 10 13 16"}}
read -r -a PATCH_NUMS <<< "$PATCH_NUMS_STR"
LR_IMG_SIZE=${LR_IMG_SIZE:-${lr_img_size:-80}}
IMG_CHANNELS=${IMG_CHANNELS:-${img_channels:-3}}
LR_CH=128
LR_VOCAB_WIDTH=32
LR_VQ_BETA=${LR_VQ_BETA:-${lr_vq_beta:-1e-3}}           # 32x5x5=800 dim latent, KL is summed, 1.0 会直接把 posterior 压塌，建议 1e-4 ~ 1e-3
LR_KL_WARMUP_EP=${LR_KL_WARMUP_EP:-${lr_kl_warmup_ep:-1.0}}  # 对 KL 权重做 N epoch 的线性 warmup，避免早期 posterior collapse
HR_VOCAB_WIDTH=32
VAE_LR=1e-4
DISC_LR=1e-4
DISC_NORM=${DISC_NORM:-${disc_norm:-sbn}}
DISC_SPEC_NORM=${DISC_SPEC_NORM:-${disc_spec_norm:-True}}
DISC_AUG_PROB=${DISC_AUG_PROB:-${disc_aug_prob:-1.0}}
DISC_GRAD_CKPT=${DISC_GRAD_CKPT:-${disc_grad_ckpt:-False}}
DBG_NAN=${DBG_NAN:-${dbg_nan:-False}}
HR_VQ_BETA=${HR_VQ_BETA:-${VQ_BETA:-${vq_beta:-0.25}}}
# Epochs: STAGE1_EP / STAGE2_EP（或 stage1_ep / stage2_ep）优先；否则用 EP / ep；再否则 stage 默认 100 / 150
STAGE1_EP=${STAGE1_EP:-${stage1_ep:-}}
STAGE2_EP=${STAGE2_EP:-${stage2_ep:-}}
EP_COMMON=${EP:-${ep:-}}
VAL_AND_SAVING_PER_EP=${VAL_AND_SAVING_PER_EP:-${val_and_saving_per_ep:-2}}
RECON_SAVE_INTERVAL=${RECON_SAVE_INTERVAL:-0}
RECON_MAX_SAMPLES=${RECON_MAX_SAMPLES:-4}
RECON_DIR_NAME=${RECON_DIR_NAME:-${RECONSTRUCTION_DIR_NAME:-${reconstruction_dir_name:-}}}
USE_LR_HR_ALIGNMENT=${USE_LR_HR_ALIGNMENT:-${use_lr_hr_alignment:-True}}
ALIGNMENT_LOSS_TYPE=${ALIGNMENT_LOSS_TYPE:-${alignment_loss_type:-scale0_image}}
ALIGNMENT_LOSS_TYPE_KEY=$(printf '%s' "$ALIGNMENT_LOSS_TYPE" | tr '[:upper:]' '[:lower:]' | tr '-' '_')
ALIGNMENT_LOSS_WEIGHT_OVERRIDE=${ALIGNMENT_LOSS_WEIGHT:-${alignment_loss_weight:-}}
ALIGNMENT_LOSS_WARMUP_EP=${ALIGNMENT_LOSS_WARMUP_EP:-${alignment_loss_warmup_ep:-0.0}}
STAGE2_USE_KL=${STAGE2_USE_KL:-${stage2_use_kl:-True}}
STAGE2_USE_KL_KEY=$(printf '%s' "$STAGE2_USE_KL" | tr '[:upper:]' '[:lower:]')
STAGE2_KL_DISABLED=0
if [ "$STAGE2_USE_KL_KEY" = "false" ] || [ "$STAGE2_USE_KL_KEY" = "0" ] || [ "$STAGE2_USE_KL_KEY" = "no" ]; then
  STAGE2_KL_DISABLED=1
fi
if [ -z "$ALIGNMENT_LOSS_WEIGHT_OVERRIDE" ] && [ "$STAGE2_KL_DISABLED" = "1" ]; then
  ALIGNMENT_LOSS_WEIGHT=0.25
else
  ALIGNMENT_LOSS_WEIGHT=${ALIGNMENT_LOSS_WEIGHT_OVERRIDE:-0.5}
fi

# Loss weights. Stage 1 keeps the historical defaults. Stage 2 has separate
# defaults because short deterministic-AE runs need less smoothing and earlier GAN feedback.
STAGE1_L1_DEFAULT=0.2
STAGE1_L2_DEFAULT=1.0
STAGE1_LPIPS_DEFAULT=0.5
STAGE1_LPIPS_MIN_RESO_DEFAULT=48
STAGE1_DISC_WEIGHT_DEFAULT=0.4
STAGE1_DISC_START_DEFAULT=20
STAGE1_DISC_WARMUP_DEFAULT=0

if [ "$STAGE2_KL_DISABLED" = "1" ]; then
  STAGE2_L1_DEFAULT=1.0
  STAGE2_L2_DEFAULT=0.25
  STAGE2_LPIPS_DEFAULT=0.25
  STAGE2_DISC_WEIGHT_DEFAULT=0.2
  STAGE2_DISC_START_DEFAULT=0.5
  STAGE2_DISC_WARMUP_DEFAULT=0.5
else
  STAGE2_L1_DEFAULT=0.2
  STAGE2_L2_DEFAULT=1.0
  STAGE2_LPIPS_DEFAULT=0.5
  STAGE2_DISC_WEIGHT_DEFAULT=0.4
  STAGE2_DISC_START_DEFAULT=30
  STAGE2_DISC_WARMUP_DEFAULT=0
fi
STAGE2_LPIPS_MIN_RESO_DEFAULT=48

STAGE1_L1_WEIGHT=${STAGE1_L1_WEIGHT:-${stage1_l1_weight:-${L1_WEIGHT:-${L1:-$STAGE1_L1_DEFAULT}}}}
STAGE1_L2_WEIGHT=${STAGE1_L2_WEIGHT:-${stage1_l2_weight:-${L2_WEIGHT:-${L2:-$STAGE1_L2_DEFAULT}}}}
STAGE1_LPIPS_WEIGHT=${STAGE1_LPIPS_WEIGHT:-${stage1_lpips_weight:-${LPIPS_WEIGHT:-${LP:-$STAGE1_LPIPS_DEFAULT}}}}
STAGE1_LPIPS_MIN_RESO=${STAGE1_LPIPS_MIN_RESO:-${stage1_lpips_min_reso:-${LPIPS_MIN_RESO:-${LPR:-$STAGE1_LPIPS_MIN_RESO_DEFAULT}}}}
STAGE1_DISC_WEIGHT=${STAGE1_DISC_WEIGHT:-${stage1_disc_weight:-${DISC_WEIGHT:-${LD:-$STAGE1_DISC_WEIGHT_DEFAULT}}}}
STAGE1_DISC_START_EP=${STAGE1_DISC_START_EP:-${stage1_disc_start_ep:-${DISC_START_EP:-${disc_start_ep:-$STAGE1_DISC_START_DEFAULT}}}}
STAGE1_DISC_WARMUP_EP=${STAGE1_DISC_WARMUP_EP:-${stage1_disc_warmup_ep:-${DISC_WARMUP_EP:-${disc_warmup_ep:-$STAGE1_DISC_WARMUP_DEFAULT}}}}

STAGE2_L1_WEIGHT=${STAGE2_L1_WEIGHT:-${stage2_l1_weight:-${L1_WEIGHT:-${L1:-$STAGE2_L1_DEFAULT}}}}
STAGE2_L2_WEIGHT=${STAGE2_L2_WEIGHT:-${stage2_l2_weight:-${L2_WEIGHT:-${L2:-$STAGE2_L2_DEFAULT}}}}
STAGE2_LPIPS_WEIGHT=${STAGE2_LPIPS_WEIGHT:-${stage2_lpips_weight:-${LPIPS_WEIGHT:-${LP:-$STAGE2_LPIPS_DEFAULT}}}}
STAGE2_LPIPS_MIN_RESO=${STAGE2_LPIPS_MIN_RESO:-${stage2_lpips_min_reso:-${LPIPS_MIN_RESO:-${LPR:-$STAGE2_LPIPS_MIN_RESO_DEFAULT}}}}
STAGE2_DISC_WEIGHT=${STAGE2_DISC_WEIGHT:-${stage2_disc_weight:-${DISC_WEIGHT:-${LD:-$STAGE2_DISC_WEIGHT_DEFAULT}}}}
STAGE2_DISC_START_EP=${STAGE2_DISC_START_EP:-${stage2_disc_start_ep:-${DISC_START_EP:-${disc_start_ep:-$STAGE2_DISC_START_DEFAULT}}}}
STAGE2_DISC_WARMUP_EP=${STAGE2_DISC_WARMUP_EP:-${stage2_disc_warmup_ep:-${DISC_WARMUP_EP:-${disc_warmup_ep:-$STAGE2_DISC_WARMUP_DEFAULT}}}}

# 输出目录
STAGE1_BED=${STAGE1_BED:-myvaex_stage1_lr_vae}
STAGE2_BED=${STAGE2_BED:-myvaex_stage2_hr_scale0_img_aligned}
STAGE1_CKPT=${STAGE1_CKPT:-${stage1_ckpt:-}}
if [ "$ALIGNMENT_LOSS_TYPE_KEY" = "latent" ] && [ -z "$STAGE1_CKPT" ]; then
  STAGE1_CKPT="${STAGE1_BED}/ckpt-best.pth"
fi
STAGE1_DEFAULT_EXP_NAME="stage1_lr_vae"
STAGE1_DEFAULT_EXP_NOTE="Stage 1: train LR VAE to posterior mean tokens"
if [ "$USE_LR_HR_ALIGNMENT" = "False" ] || [ "$USE_LR_HR_ALIGNMENT" = "false" ] || [ "$USE_LR_HR_ALIGNMENT" = "0" ]; then
  if [ "$STAGE2_KL_DISABLED" = "1" ]; then
    STAGE2_DEFAULT_EXP_NAME="stage2_hr_ae_no_alignment"
    STAGE2_DEFAULT_EXP_NOTE="Stage 2 control: train deterministic HR multi-scale AE on paired LR-HR loader without alignment loss"
  else
    STAGE2_DEFAULT_EXP_NAME="stage2_hr_vae_no_alignment"
    STAGE2_DEFAULT_EXP_NOTE="Stage 2 control: train HR multi-scale VAE on paired LR-HR loader without alignment loss"
  fi
elif [ "$ALIGNMENT_LOSS_TYPE_KEY" = "latent" ]; then
  if [ "$STAGE2_KL_DISABLED" = "1" ]; then
    STAGE2_DEFAULT_EXP_NAME="stage2_hr_ae_latent_aligned"
    STAGE2_DEFAULT_EXP_NOTE="Stage 2 legacy: train deterministic HR multi-scale AE with first-scale posterior mean aligned to stage1 LR latent"
  else
    STAGE2_DEFAULT_EXP_NAME="stage2_hr_vae_latent_aligned"
    STAGE2_DEFAULT_EXP_NOTE="Stage 2 legacy: train HR multi-scale VAE with first-scale posterior mean aligned to stage1 LR latent"
  fi
else
  if [ "$STAGE2_KL_DISABLED" = "1" ]; then
    STAGE2_DEFAULT_EXP_NAME="stage2_hr_ae_scale0_img_aligned"
    STAGE2_DEFAULT_EXP_NOTE="Stage 2: train deterministic HR multi-scale AE with decoded scale0 image aligned to resized LR pixels"
  else
    STAGE2_DEFAULT_EXP_NAME="stage2_hr_vae_scale0_img_aligned"
    STAGE2_DEFAULT_EXP_NOTE="Stage 2: train HR multi-scale VAE with decoded scale0 image aligned to resized LR pixels"
  fi
fi

if [ "$STAGE" = "1" ]; then
  EXP_NAME=${EXP_NAME:-${exp_name:-$STAGE1_DEFAULT_EXP_NAME}}
  EXP_NOTE=${EXP_NOTE:-${exp_note:-$STAGE1_DEFAULT_EXP_NOTE}}
  torchrun --nproc_per_node=1 --nnodes=1 --node_rank=0 --master_addr=127.0.0.1 --master_port="$PORT" train_two_stage.py \
  --exp_name="$EXP_NAME" --bed="$STAGE1_BED" \
  --exp_note="$EXP_NOTE" \
  --data="$DATA_PATH" \
  --lr_folder="$LR_FOLDER" \
  --hr_folder="$HR_FOLDER" \
  --training_stage=1 \
  --lbs=8 \
  --ep="${STAGE1_EP:-${EP_COMMON:-100}}" \
  --val_and_saving_per_ep="$VAL_AND_SAVING_PER_EP" \
  --lr_img_size="$LR_IMG_SIZE" \
  --img_channels="$IMG_CHANNELS" \
  --lr_ch="$LR_CH" \
  --lr_vocab_width="$LR_VOCAB_WIDTH" \
  --lr_vq_beta="$LR_VQ_BETA" \
  --lr_kl_warmup_ep="$LR_KL_WARMUP_EP" \
  --vq_beta="$HR_VQ_BETA" \
  --vae_lr="$VAE_LR" \
  --disc_lr="$DISC_LR" \
  --disc_norm="$DISC_NORM" \
  --disc_spec_norm="$DISC_SPEC_NORM" \
  --disc_aug_prob="$DISC_AUG_PROB" \
  --disc_grad_ckpt="$DISC_GRAD_CKPT" \
  --l1="$STAGE1_L1_WEIGHT" \
  --l2="$STAGE1_L2_WEIGHT" \
  --lp="$STAGE1_LPIPS_WEIGHT" \
  --lpr="$STAGE1_LPIPS_MIN_RESO" \
  --ld="$STAGE1_DISC_WEIGHT" \
  --disc_start_ep="$STAGE1_DISC_START_EP" \
  --disc_warmup_ep="$STAGE1_DISC_WARMUP_EP" \
  --save_reconstruction_images=True \
  --reconstruction_save_interval="$RECON_SAVE_INTERVAL" \
  --reconstruction_max_samples="$RECON_MAX_SAMPLES" \
  --reconstruction_dir_name="$RECON_DIR_NAME" \
  --train_log_points_per_epoch="$TRAIN_LOG_POINTS_PER_EPOCH" \
  --dbg_nan="$DBG_NAN" \
  --debug_loss_printed_limit=10 \
  --debug_kl_count_limit=10
elif [ "$STAGE" = "2" ]; then
  EXP_NAME=${EXP_NAME:-${exp_name:-$STAGE2_DEFAULT_EXP_NAME}}
  EXP_NOTE=${EXP_NOTE:-${exp_note:-$STAGE2_DEFAULT_EXP_NOTE}}
  LR_VAE_RESUME_ARGS=()
  if [ -n "$STAGE1_CKPT" ]; then
    LR_VAE_RESUME_ARGS+=(--lr_vae_resume="$STAGE1_CKPT")
  fi
  torchrun --nproc_per_node=1 --nnodes=1 --node_rank=0 --master_addr=127.0.0.1 --master_port="$PORT" train_two_stage.py \
  --exp_name="$EXP_NAME" --bed="$STAGE2_BED" \
  --exp_note="$EXP_NOTE" \
  --data="$DATA_PATH" \
  --lr_folder="$LR_FOLDER" \
  --hr_folder="$HR_FOLDER" \
  --training_stage=2 \
  --use_lr_hr_alignment="$USE_LR_HR_ALIGNMENT" \
  --alignment_loss_type="$ALIGNMENT_LOSS_TYPE" \
  --alignment_loss_weight="$ALIGNMENT_LOSS_WEIGHT" \
  --alignment_loss_warmup_ep="$ALIGNMENT_LOSS_WARMUP_EP" \
  --stage2_use_kl="$STAGE2_USE_KL" \
  "${LR_VAE_RESUME_ARGS[@]}" \
  --lbs=4 \
  --ep="${STAGE2_EP:-${EP_COMMON:-150}}" \
  --val_and_saving_per_ep="$VAL_AND_SAVING_PER_EP" \
  --lr_img_size="$LR_IMG_SIZE" \
  --img_channels="$IMG_CHANNELS" \
  --lr_ch="$LR_CH" \
  --lr_vocab_width="$LR_VOCAB_WIDTH" \
  --lr_vq_beta="$LR_VQ_BETA" \
  --lr_kl_warmup_ep="$LR_KL_WARMUP_EP" \
  --vocab_width="$HR_VOCAB_WIDTH" \
  --vq_beta="$HR_VQ_BETA" \
  --patch_nums "${PATCH_NUMS[@]}" \
  --vae_lr="$VAE_LR" \
  --disc_lr="$DISC_LR" \
  --disc_norm="$DISC_NORM" \
  --disc_spec_norm="$DISC_SPEC_NORM" \
  --disc_aug_prob="$DISC_AUG_PROB" \
  --disc_grad_ckpt="$DISC_GRAD_CKPT" \
  --l1="$STAGE2_L1_WEIGHT" \
  --l2="$STAGE2_L2_WEIGHT" \
  --lp="$STAGE2_LPIPS_WEIGHT" \
  --lpr="$STAGE2_LPIPS_MIN_RESO" \
  --ld="$STAGE2_DISC_WEIGHT" \
  --disc_start_ep="$STAGE2_DISC_START_EP" \
  --disc_warmup_ep="$STAGE2_DISC_WARMUP_EP" \
  --save_reconstruction_images=True \
  --reconstruction_save_interval="$RECON_SAVE_INTERVAL" \
  --reconstruction_max_samples="$RECON_MAX_SAMPLES" \
  --reconstruction_dir_name="$RECON_DIR_NAME" \
  --train_log_points_per_epoch="$TRAIN_LOG_POINTS_PER_EPOCH" \
  --dbg_nan="$DBG_NAN" \
  --debug_loss_printed_limit=10 \
  --debug_kl_count_limit=10
else
  echo "Unknown STAGE=$STAGE, expected 1 or 2"
  exit 1
fi
