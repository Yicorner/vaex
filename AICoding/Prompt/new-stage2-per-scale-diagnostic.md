@myvaex/local_output/stage2_gray_scale0_img_align_lr256_patch4to16_no_kl_3gpu_align_weight_5/diagnostic 注意看这个文件夹，目前它只输出了LR_gt，与s0_decoder。
我希望你能够输出每一个尺度的样例图像。
注意，这里的每一个尺度，应该是这样，如果patch_num = 8 10 13 16

累加 scale0后的解码图

累加 s0 + scale1（10×10） 后

累加 s0 + s1 + scale2（13×13） 后

累加 s0 + s1 + s2 + scale3（16×16） 后