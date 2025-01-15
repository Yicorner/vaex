torchrun --nproc_per_node=4 --nnodes=1 --node_rank=0 --master_addr=127.0.0.1 --master_port=12345 train.py \
--exp_name="df2k_ost" --bed="vaex" \
--lbs=24 --vae_lr=1e-4 --disc_lr=1e-4 \
--data='./data/df2k_ost/GT'  \
--ep=500 \
--resume='./ckpt_vaex/df_ost-ckpt-4096*4-250.pth' \
--vocab_size=16384
# --data='./data/df2k_ost/GT_sub'
# --data='./data/DIV2K_train_HR'