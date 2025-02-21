export CUDA_VISIBLE_DEVICES=0,1,2,3
torchrun --nproc_per_node=4 --nnodes=1 --node_rank=0 --master_addr=127.0.0.1 --master_port=13333 train.py \
--exp_name="brats_256_t1_2021_pair_4x" --bed="vaex" \
--lbs=26 --vae_lr=1e-4 --disc_lr=1e-4 \
--data='./data/brats_256_t1_2021_pair_4x'  \
--val_and_saving_per_ep=1 \
--ep=400 \
--vocab_size=1024 --vocab_width=32 

# --data='./data/brats_256_t1_2021_pair_4x'  \

# --resume="local_output/ckpt_last.pth"
# --data='./data/df2k_ost/GT_sub'
# --data='./data/DIV2K_train_HR' 