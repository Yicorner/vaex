#!/bin/bash
DATA_PATH_SMALL="/home/featurize/work/brats_256_t2_2021_pair_png_with_ref_small"
DATA_PATH_LARGE="/home/featurize/data/brats_256_t2_2021_pair_png_with_ref"

export CUDA_VISIBLE_DEVICES=0
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