# 两阶段训练指南 (Two-Stage Training Guide)

## 概述

本项目实现了用于图像超分辨率的两阶段VAE训练方案：

- **阶段1**：训练单尺度LR VAE，将低分辨率图像编码为 5×5 tokens
- **阶段2**：训练多尺度HR VAE，将高分辨率图像编码为多尺度tokens (5×5, 6×6, 8×8, ..., 16×16)，并与LR的5×5对齐

## 数据集结构

```
DATA_PATH/
├── train/
│   ├── LR/  # 低分辨率图像 (建议80x80)
│   ├── HR/  # 高分辨率图像 (建议256x256)
│   └── REF/ # 参考图像 (可选)
└── val/
    ├── LR/
    ├── HR/
    └── REF/
```

## 训练流程

### 阶段1：训练LR VAE

目标：学习将LR图像编码为5×5的潜在表示

```bash
torchrun --nproc_per_node=1 train.py \
  --exp_name="stage1_lr_vae" \
  --bed="output/stage1" \
  --exp_note="阶段1：训练LR VAE" \
  --data="path/to/data" \
  --training_stage=1 \
  --lr_img_size=80 \
  --lr_ch=128 \
  --lr_vocab_width=32 \
  --lr_vq_beta=1.0 \
  --lbs=8 \
  --vae_lr=1e-4 \
  --disc_lr=1e-4 \
  --ep=100 \
  --val_and_saving_per_ep=5
```

**关键参数**：
- `training_stage=1`：设置为阶段1
- `lr_img_size=80`：LR图像大小（80×80 → 16倍下采样 → 5×5）
- `lr_ch=128`：LR VAE的通道数
- `lr_vocab_width=32`：LR VAE的潜在维度
- `lr_vq_beta=1.0`：KL loss权重（简单版本，不压缩）

### 阶段2：训练HR VAE with Alignment

目标：训练多尺度HR VAE，并使其第一层(5×5)与LR VAE对齐

```bash
torchrun --nproc_per_node=1 train.py \
  --exp_name="stage2_hr_vae" \
  --bed="output/stage2" \
  --exp_note="阶段2：训练HR VAE with LR-HR对齐" \
  --data="path/to/data" \
  --training_stage=2 \
  --use_lr_hr_alignment=True \
  --alignment_loss_weight=1.0 \
  --patch_nums 5 6 8 10 13 16 \
  --vocab_width=32 \
  --lbs=4 \
  --vae_lr=1e-4 \
  --disc_lr=1e-4 \
  --ep=150 \
  --val_and_saving_per_ep=5 \
  --resume="output/stage1/ckpt-best.pth"
```

**关键参数**：
- `training_stage=2`：设置为阶段2
- `use_lr_hr_alignment=True`：启用LR-HR对齐loss
- `alignment_loss_weight=1.0`：对齐loss的权重
- `patch_nums 5 6 8 10 13 16`：多尺度配置（注意第一个必须是5）
- `resume="..."`：加载阶段1训练好的LR VAE checkpoint

## 模型架构

### LR VAE (单尺度)
- 输入：LR图像 [B, 3, 80, 80]
- 编码器：5层下采样 → [B, C, 5, 5]
- 量化：Gaussian采样 (mean + logvar)
- 解码器：5层上采样 → [B, 3, 80, 80]
- Loss：KL loss + 重建loss (L1 + LPIPS) + 对抗loss

### HR VAE (多尺度)
- 输入：HR图像 [B, 3, 256, 256]
- 编码器：5层下采样 → [B, C, 16, 16]
- 多尺度量化：5×5, 6×6, 8×8, 10×10, 13×13, 16×16
- 解码器：5层上采样 → [B, 3, 256, 256]
- Loss：KL loss + 重建loss + 对抗loss + **对齐loss**

### 对齐Loss

在阶段2中，HR VAE的第一个尺度(5×5)会与LR VAE的5×5对齐：

```python
L_align = MSE(HR_5x5_mean, LR_5x5)
```

这确保了HR VAE学到的第一层表示与LR VAE一致，为后续超分辨率任务打下基础。

## 重要提示

1. **数据配对**：LR和HR图像应该一一对应（文件名匹配）
2. **分辨率**：LR建议80×80，HR建议256×256（可根据需求调整）
3. **阶段顺序**：必须先完成阶段1再进行阶段2
4. **Checkpoint**：阶段2需要加载阶段1的checkpoint
5. **冻结策略**：
   - 阶段1：LR VAE训练，HR VAE冻结
   - 阶段2：HR VAE训练，LR VAE冻结

## 监控训练

关键指标：
- **阶段1**：`L_rec`（重建损失），`PSNR`，`SSIM`，`Lkl`（KL散度）
- **阶段2**：上述指标 + `L_align`（对齐损失）

可视化：
- TensorBoard日志：`local_output/tb-*/`
- 重建样本图像：
  - 阶段1：`local_output/reconstruction_samples_lr/`
  - 阶段2：`local_output/reconstruction_samples_hr/`

## 代码结构

- `models/lr_vae.py`：单尺度LR VAE模型
- `models/vqvae.py`：多尺度HR VAE模型
- `trainer_two_stage.py`：两阶段训练器
- `utils/data_lr_hr.py`：LR-HR配对数据加载器
- `train.sh`：训练脚本示例

## 常见问题

**Q: 为什么LR图像是80×80？**  
A: 80×80经过16倍下采样（2^4）刚好得到5×5。如果要改变尺度，需要保持 `lr_img_size / 16 = 第一个patch_num`。

**Q: 可以跳过阶段1直接训练阶段2吗？**  
A: 不建议。阶段1为阶段2提供了预训练的LR表示，有助于对齐loss的收敛。

**Q: 对齐loss权重怎么选择？**  
A: 建议从1.0开始，根据L_align和重建loss的相对大小调整。如果L_align过大，降低权重；过小则增加。

