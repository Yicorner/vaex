训前步骤

1. conda activate /home/featurize/work/myenv 
2. 去网站上解压
3. 跑 train.sh脚本
# 用法:
#   DATA=large bash train.sh
#   bash train.sh                      # 默认 FEATURIZE，跑 stage 1
#   STAGE=2 bash train.sh              # FEATURIZE，跑 stage 2
#   EXP_NAME=my_exp bash train.sh
#   EXP_NOTE="my custom note" bash train.sh
#   RECONSTRUCTION_DIR_NAME=my_recon_dir bash train.sh
#   DATA=large STAGE=2 EXP_NAME=s2_try EXP_NOTE="stage2 test" RECONSTRUCTION_DIR_NAME=ep_test bash train.sh


#   TRAIN_ENV=HOME bash train.sh       # HOME，跑 stage 1
#   TRAIN_ENV=HOME STAGE=2 bash train.sh



# 如何查看torch run 命令是否被kill
ps -ef | grep torchrun

# 如何查看数据集大小
日志中搜索：LR-HR Dataset

# 重建目录
# 可用 RECONSTRUCTION_DIR_NAME（或 RECON_DIR_NAME）指定 --reconstruction_dir_name；
# 也兼容小写 reconstruction_dir_name。


# exp_name
# 可用 EXP_NAME（或小写 exp_name）覆盖默认 --exp_name。

# exp_note
# 可用 EXP_NOTE（或小写 exp_note）覆盖默认 --exp_note。


