训前步骤

1. conda activate /home/featurize/work/myenv 
2. 去网站上解压
3. 跑 train.sh脚本
# 用法:
#   DATA=large bash train.sh
#   bash train.sh                      # 默认 FEATURIZE，跑 stage 1
#   STAGE=2 bash train.sh              # FEATURIZE，跑 stage 2


#   TRAIN_ENV=HOME bash train.sh       # HOME，跑 stage 1
#   TRAIN_ENV=HOME STAGE=2 bash train.sh



# 如何查看torch run 命令是否被kill
ps -ef | grep torchrun

# 如何查看数据集大小
日志中搜索：LR-HR Dataset

观察这里的训练结果，@backup1_stdout.txt (27970-27985)。我发现它有异常，因为我每次打印日志的时候（1088iter）的时候还会跑出一张重建图片，我现在发现它非常的模糊，结果只是一个模糊的白色的球，看不出任何有效的信息，我给你截图 ，其中左边是groundtruth，右边是重建结果。

请问这是为什么？请结合调试信息和代码帮我仔细分析分析为什么会这样，并做出相应的代码修改或者调试信息补充。训练参数在这@backup1_stdout.txt (27136-27281) 
