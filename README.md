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

# 
DATA=large \
EXP_NAME=stage1_fix_init_issue  \
EXP_NOTE="fix: preserve mean_logvar_conv logvar initialization" \
RECONSTRUCTION_DIR_NAME=stage1_fix_init_issue \
bash train.sh

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
VAL_AND_SAVING_PER_EP=10 \
bash train.sh

# stage2 (LR_64x64 + HR), patch_nums 与 stage1 latent 对齐
DATA_PATH=/home/featurize/data/brats_256_t2_2021_pair_png_with_ref_and_LR64 \
STAGE=2 \
EXP_NAME=stage2_from_stage1_ckpt9_lr64_patch4to16 \
EXP_NOTE="stage2 with LR_64x64, align first scale 4x4 to stage1 latent" \
RECONSTRUCTION_DIR_NAME=stage2_from_stage1_ckpt9_lr64_patch4to16 \
STAGE1_CKPT=local_output/test/test_stage1_fix_init_issue_L1=1.0_KLweightDown_LR64DataPath/ckpt-9.pth \
LR_FOLDER=LR_64x64 \
HR_FOLDER=HR \
LR_IMG_SIZE=64 \
PATCH_NUMS="4 5 6 8 10 13 16" \
VAL_AND_SAVING_PER_EP=1 \
TRAIN_LOG_POINTS_PER_EPOCH=200 \
bash train.sh

# stage2 (LR_64x64 + HR), patch_nums 不与 stage1 latent 对齐（与前一次训练对比，为什么速度这么慢？）
DATA_PATH=/home/featurize/data/brats_256_t2_2021_pair_png_with_ref_and_LR64 \
STAGE=2 \
EXP_NAME=stage2_from_stage1_ckpt9_lr64_patch4to16_wo_Alignment \
EXP_NOTE="stage2 with LR_64x64, align first scale 4x4 to stage1 latent without Alignment" \
RECONSTRUCTION_DIR_NAME=stage2_from_stage1_ckpt9_lr64_patch4to16_wo_Alignment \
STAGE1_CKPT=local_output/test/test_stage1_fix_init_issue_L1=1.0_KLweightDown_LR64DataPath/ckpt-9.pth \
LR_FOLDER=LR_64x64 \
HR_FOLDER=HR \
LR_IMG_SIZE=64 \
PATCH_NUMS="4 5 6 8 10 13 16" \
VAL_AND_SAVING_PER_EP=1 \
TRAIN_LOG_POINTS_PER_EPOCH=200 \
USE_LR_HR_ALIGNMENT=False \
STAGE2_EP=2 \
bash train.sh

# stage2 (LR_64x64 + HR), patch_nums 与 stage1 latent 对齐（after fix alignment speed problem）
DATA_PATH=/home/featurize/data/brats_256_t2_2021_pair_png_with_ref_and_LR64 \
STAGE=2 \
EXP_NAME=stage2_from_stage1_ckpt9_lr64_patch4to16_new_Alignment \
EXP_NOTE="stage2 with LR_64x64, align first scale 4x4 to stage1 latent with new Alignment" \
RECONSTRUCTION_DIR_NAME=stage2_from_stage1_ckpt9_lr64_patch4to16_news_Alignment \
STAGE1_CKPT=local_output/test/test_stage1_fix_init_issue_L1=1.0_KLweightDown_LR64DataPath/ckpt-9.pth \
LR_FOLDER=LR_64x64 \
HR_FOLDER=HR \
LR_IMG_SIZE=64 \
PATCH_NUMS="4 5 6 8 10 13 16" \
VAL_AND_SAVING_PER_EP=1 \
TRAIN_LOG_POINTS_PER_EPOCH=200 \
USE_LR_HR_ALIGNMENT=True \
STAGE2_EP=2 \
bash train.sh


# stage2 (LR_64x64 + HR), patch_nums 与 stage1 latent 对齐（after fix alignment speed problem）(第二次训练 for ckpt)
DATA_PATH=/home/featurize/data/brats_256_t2_2021_pair_png_with_ref_and_LR64 \
STAGE=2 \
EXP_NAME=stage2_from_stage1_ckpt9_lr64_patch4to16_new_Alignment_1 \
EXP_NOTE="stage2 with LR_64x64, align first scale 4x4 to stage1 latent with new Alignment_1" \
RECONSTRUCTION_DIR_NAME=stage2_from_stage1_ckpt9_lr64_patch4to16_news_Alignment_1 \
STAGE1_CKPT=local_output/test/test_stage1_fix_init_issue_L1=1.0_KLweightDown_LR64DataPath/ckpt-9.pth \
LR_FOLDER=LR_64x64 \
HR_FOLDER=HR \
LR_IMG_SIZE=64 \
PATCH_NUMS="4 5 6 8 10 13 16" \
VAL_AND_SAVING_PER_EP=1 \
TRAIN_LOG_POINTS_PER_EPOCH=0 \
USE_LR_HR_ALIGNMENT=True \
STAGE2_EP=3 \
bash train.sh

# stage2 (LR_64x64 + HR), patch_nums 不与 stage1 latent 对齐 (第二次训练看看训练3个epoch效果怎么样和上面的相比)
DATA_PATH=/home/featurize/data/brats_256_t2_2021_pair_png_with_ref_and_LR64 \
STAGE=2 \
EXP_NAME=stage2_from_stage1_ckpt9_lr64_patch4to16_no_Alignment_1 \
EXP_NOTE="stage2 with LR_64x64, align first scale 4x4 to stage1 latent without new Alignment_1" \
RECONSTRUCTION_DIR_NAME=stage2_from_stage1_ckpt9_lr64_patch4to16_no_Alignment_1 \
STAGE1_CKPT=local_output/test/test_stage1_fix_init_issue_L1=1.0_KLweightDown_LR64DataPath/ckpt-9.pth \
LR_FOLDER=LR_64x64 \
HR_FOLDER=HR \
LR_IMG_SIZE=64 \
PATCH_NUMS="1 2 3 4 5 6 8 10 13 16" \
VAL_AND_SAVING_PER_EP=1 \
TRAIN_LOG_POINTS_PER_EPOCH=0 \
USE_LR_HR_ALIGNMENT=False \
STAGE2_EP=3 \
bash train.sh

#   TRAIN_ENV=HOME bash train.sh       # HOME，跑 stage 1
#   TRAIN_ENV=HOME STAGE=2 bash train.sh



# 当前推荐：stage2 scale0 图像空间对齐（不依赖 stage1 LR latent）
# scale0 latent 会走 stage2 共享 decoder 解码成 256x256 图像，再和 LR 图像 resize 后做 L1。
DATA_PATH=/home/featurize/data/brats_256_t2_2021_pair_png_with_ref \
STAGE=2 \
EXP_NAME=stage2_scale0_img_align_lr256_patch4to16 \
EXP_NOTE="stage2 scale0 decoded image aligned to resized LR pixels; no stage1 latent dependency" \
RECONSTRUCTION_DIR_NAME=stage2_scale0_img_align_lr256_patch4to16 \
LR_FOLDER=LR \
HR_FOLDER=HR \
LR_IMG_SIZE=256 \
PATCH_NUMS="4 5 6 8 10 13 16" \
USE_LR_HR_ALIGNMENT=True \
ALIGNMENT_LOSS_TYPE=scale0_image \
ALIGNMENT_LOSS_WEIGHT=0.5 \
ALIGNMENT_LOSS_WARMUP_EP=0 \
VAL_AND_SAVING_PER_EP=1 \
TRAIN_LOG_POINTS_PER_EPOCH=200 \
STAGE2_EP=3 \
bash train.sh

# 当前建议优先试：stage2 deterministic AE（去掉 stage2 KL 和训练采样）+ scale0 图像空间对齐
# 适合“只要重建准确性，不需要 latent 多样性”的实验；预期日志里 Lkl=0。
DATA_PATH=/home/featurize/data/brats_256_t2_2021_pair_png_with_ref \
STAGE=2 \
EXP_NAME=stage2_scale0_img_align_lr256_patch4to16_no_kl \
EXP_NOTE="stage2 deterministic AE; scale0 decoded image aligned to LR pixels; no KL or sampling" \
RECONSTRUCTION_DIR_NAME=stage2_scale0_img_align_lr256_patch4to16_no_kl \
LR_FOLDER=LR \
HR_FOLDER=HR \
LR_IMG_SIZE=256 \
PATCH_NUMS="4 5 6 8 10 13 16" \
USE_LR_HR_ALIGNMENT=True \
ALIGNMENT_LOSS_TYPE=scale0_image \
ALIGNMENT_LOSS_WEIGHT=0.25 \
ALIGNMENT_LOSS_WARMUP_EP=0 \
STAGE2_USE_KL=False \
STAGE2_L1_WEIGHT=1.0 \
STAGE2_L2_WEIGHT=0.25 \
STAGE2_LPIPS_WEIGHT=0.25 \
STAGE2_DISC_WEIGHT=0.2 \
STAGE2_DISC_START_EP=0.5 \
STAGE2_DISC_WARMUP_EP=0.5 \
VAL_AND_SAVING_PER_EP=1 \
TRAIN_LOG_POINTS_PER_EPOCH=100 \
STAGE2_EP=3 \
bash train.sh

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
# 可用 L1（或 L1_WEIGHT）覆盖两阶段默认 --l1。
# 也可用 STAGE1_L1_WEIGHT / STAGE2_L1_WEIGHT 分别覆盖；当前推荐 stage2 去糊先试 1.0。

# l2
# 可用 L2（或 L2_WEIGHT）覆盖两阶段默认 --l2。
# 也可用 STAGE1_L2_WEIGHT / STAGE2_L2_WEIGHT 分别覆盖；stage2 去糊建议 0.25，避免 MSE 太强导致平滑。

# lpips
# 可用 LP（或 LPIPS_WEIGHT）覆盖两阶段默认 --lp；trainer 内部会把 --lp 乘 2。
# 也可用 STAGE1_LPIPS_WEIGHT / STAGE2_LPIPS_WEIGHT 分别覆盖；stage2 去糊建议 0.25（实际约 0.5）。

# disc loss
# 可用 LD（或 DISC_WEIGHT）覆盖两阶段默认 --ld。
# 也可用 STAGE1_DISC_WEIGHT / STAGE2_DISC_WEIGHT 分别覆盖；stage2 去糊建议 0.2。
# 可用 DISC_START_EP / DISC_WARMUP_EP 做通用覆盖，或用 STAGE2_DISC_START_EP / STAGE2_DISC_WARMUP_EP 只覆盖 stage2。
# 注意：如果 STAGE2_EP=3 但 STAGE2_DISC_START_EP=30，GAN 完全不会启动，图像偏糊是正常的。

# hr_vq_beta
# 可用 HR_VQ_BETA（或 VQ_BETA / vq_beta）覆盖 HR VAE 的 --vq_beta；STAGE2_USE_KL=False 时该项不进入 stage2 loss。

# lr_vq_beta
# 可用 LR_VQ_BETA（或小写 lr_vq_beta）覆盖默认 --lr_vq_beta（默认 1e-3）。

# lr_kl_warmup_ep
# 可用 LR_KL_WARMUP_EP（或小写 lr_kl_warmup_ep）覆盖默认 --lr_kl_warmup_ep（默认 1.0）。

# lr_folder / hr_folder
# 可用 LR_FOLDER（或小写 lr_folder）指定数据集中 LR 子目录名（默认 LR）。
# 可用 HR_FOLDER（或小写 hr_folder）指定数据集中 HR 子目录名（默认 HR）。
# 示例：LR_FOLDER=LR_64x64 bash train.sh

# stage2 alignment
# 默认 ALIGNMENT_LOSS_TYPE=scale0_image，不需要 STAGE1_CKPT。
# 可用 ALIGNMENT_LOSS_WEIGHT 调整 scale0 图像 loss 权重；默认 VAE 模式 0.5，STAGE2_USE_KL=False 时 0.25。
# 可用 ALIGNMENT_LOSS_WARMUP_EP 给 scale0 loss 做线性 warmup，默认 0。
# 如果要复现实验旧版本 latent 对齐，设置 ALIGNMENT_LOSS_TYPE=latent 并提供 STAGE1_CKPT。

# stage2 KL / AE mode
# 默认 STAGE2_USE_KL=True，保持原 stage2 VAE：训练时采样 latent，并加入 HR 多尺度 KL。
# 设置 STAGE2_USE_KL=False 时，stage2 训练和验证都走 posterior mean，Lkl=0，更像普通 autoencoder。

# test
python eval_stage1_ckpt.py \
  --ckpt_path local_output/ckpt-3.pth \
  --test_dir /home/featurize/data/brats_256_t2_2021_pair_png_with_ref/test/LR \
  --output_dir local_output/stage1_test_eval \
  --num_samples 100 \
  --seed 42 \
  --batch_size 4

# test
python eval_stage1_ckpt.py \
  --ckpt_path local_output/test/test_stage1_fix_init_issue_L1=1.0_KLweightDown_LR64DataPath/ckpt-9.pth \
  --test_dir /home/featurize/data/brats_256_t2_2021_pair_png_with_ref_and_LR64/test/LR_64x64 \
  --output_dir local_output/test/test_stage1_fix_init_issue_L1=1.0_KLweightDown_LR64DataPath \
  --num_samples 100 \
  --seed 42 \
  --batch_size 4

# test
python eval_stage2_ckpt.py \
  --ckpt_path local_output/test/test_stage2_with_alignment_epoch3/ckpt-2.pth \
  --test_dir /home/featurize/data/brats_256_t2_2021_pair_png_with_ref_and_LR64/test/HR \
  --output_dir local_output/test/test_stage2_with_alignment_epoch3 \
  --num_samples 100 \
  --seed 42 \
  --batch_size 4


