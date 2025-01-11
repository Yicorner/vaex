torchrun --nproc_per_node=4 --nnodes=1 --node_rank=0 --master_addr=127.0.0.1 --master_port=12345 train.py \
--exp_name="DVI2K" --bed="vaex" --lbs=24 --vae_lr=1e-4 --disc_lr=1e-4
