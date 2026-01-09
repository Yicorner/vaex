#!/bin/bash
DATA_PATH="/home/featurize/work/brats_256_t2_2021_pair_png_with_ref_small"

export CUDA_VISIBLE_DEVICES=0
torchrun --nproc_per_node=1 --nnodes=1 --node_rank=0 --master_addr=127.0.0.1 --master_port=13333 train.py \
--exp_name="brats_256_t1_2021_pair_4x" --bed="vaex" \
--lbs=1 --vae_lr=1e-4 --disc_lr=1e-4 \
--data="$DATA_PATH"  \
--val_and_saving_per_ep=1 \
--ep=2 \
--vocab_size=1024 --vocab_width=32 

# --data='./data/brats_256_t1_2021_pair_4x'  \

# --resume="local_output/ckpt_last.pth"
# --data='./data/df2k_ost/GT_sub'
# --data='./data/DIV2K_train_HR' 