好的，注意看@backup1_stdout.txt (32520-32561) 这个训练过程，我将这个训练过程记为x，我认为stage1的任务应该已经差不多结束了。

接下来将进行下一个任务：stage2的训练。

进行stage2训练之前，首先需要你注意以下几点：


1.stage1训练出来的ckpt路径为：local_output/test/test_stage1_fix_init_issur_L1=1.0_KLweightDown/ckpt-3.pth。我的理解stage2的训练应该会依赖到他？
2.注意到路径/home/featurize/data/brats_256_t2_2021_pair_png_with_ref,这是你的训练数据，可以简单了解一下它的结构。
3.希望你能够直接可以给我一个可用的脚本我直接来运行，就像readme中的stage1的训练命令一样，我记得stage2的训练脚本应该已经是有的，希望你帮我检查检查目前有没有什么缺陷，没有的话，直接给我一个可运行的命令。
4.所有代码以及脚本务必写的好维护，不要太乱，可扩展性也应当不差。
5.如有必要，更新skill。