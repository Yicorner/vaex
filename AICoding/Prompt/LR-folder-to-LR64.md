D:\DATA\brats_256_t2_2021_pair_png_with_ref,注意看这个路径下的文件
——train
————LR
————HR
————Ref
————LR_64x64(新增的)
——test（同理）
——val（同理）

现在我希望你能够填充LR_64x64文件夹内容，具体来说，就是模仿新写一个脚本@make_data(2d_pair2d).py ，但是最后不重新插值回去即可