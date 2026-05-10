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
#   L1=0.3 bash train.sh
#   LR_VQ_BETA=1e-4 LR_KL_WARMUP_EP=2.0 bash train.sh
#   DATA=large STAGE=2 EXP_NAME=s2_try EXP_NOTE="stage2 test" RECONSTRUCTION_DIR_NAME=ep_test bash train.sh

# DATA=large EXP_NAME=stage1_fix_init_issue EXP_NOTE="fix: preserve mean_logvar_conv logvar initialization" RECONSTRUCTION_DIR_NAME=stage1_fix_init_issue bash train.sh

# 
DATA=large \
EXP_NAME=stage1_fix_init_issue_L1=1.0 \
EXP_NOTE="fix: preserve mean_logvar_conv logvar initialization + L1=1.0" \
RECONSTRUCTION_DIR_NAME=stage1_fix_init_issue_L1=1.0 \
L1=1.0 \
bash train.sh

# 
DATA=large \
EXP_NAME=stage1_fix_init_issue_L1=1.0_KLweightDown \
EXP_NOTE="fix: preserve mean_logvar_conv logvar initialization + L1=1.0 +KLweightDown" \
RECONSTRUCTION_DIR_NAME=stage1_fix_init_issue_L1=1.0_KLweightDown \
L1=1.0 \
LR_VQ_BETA=1e-4 \
LR_KL_WARMUP_EP=10.0 \
bash train.sh

#
DATA_PATH=/home/featurize/data/brats_256_t2_2021_pair_png_with_ref_and_LR64 \
EXP_NAME=stage1_fix_init_issue_L1=1.0_KLweightDown_LR64DataPath \
EXP_NOTE="fix: preserve mean_logvar_conv logvar initialization + L1=1.0 +KLweightDown + LR64DataPath" \
RECONSTRUCTION_DIR_NAME=stage1_fix_init_issue_L1=1.0_KLweightDown_LR64DataPath \
L1=1.0 \
LR_VQ_BETA=1e-4 \
LR_KL_WARMUP_EP=10.0 \
LR_FOLDER=LR_64x64 \
VAL_AND_SAVING_PER_EP = 10 \
bash train.sh

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

# l1
# 可用 L1（或 L1_WEIGHT）覆盖默认 --l1（默认 0.2）。

# lr_vq_beta
# 可用 LR_VQ_BETA（或小写 lr_vq_beta）覆盖默认 --lr_vq_beta（默认 1e-3）。

# lr_kl_warmup_ep
# 可用 LR_KL_WARMUP_EP（或小写 lr_kl_warmup_ep）覆盖默认 --lr_kl_warmup_ep（默认 1.0）。

# lr_folder / hr_folder
# 可用 LR_FOLDER（或小写 lr_folder）指定数据集中 LR 子目录名（默认 LR）。
# 可用 HR_FOLDER（或小写 hr_folder）指定数据集中 HR 子目录名（默认 HR）。
# 示例：LR_FOLDER=LR_64x64 bash train.sh


