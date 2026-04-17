训前步骤

1. conda activate /home/featurize/work/myenv 
2. 去网站上解压
3. 跑 train.sh脚本
# 用法:
#   bash train.sh                      # 默认 FEATURIZE，跑 stage 1
#   STAGE=2 bash train.sh              # FEATURIZE，跑 stage 2
#   TRAIN_ENV=HOME bash train.sh       # HOME，跑 stage 1
#   TRAIN_ENV=HOME STAGE=2 bash train.sh

# 如何查看torch run 命令是否被kill
ps -ef | grep torchrun