---
name: two-stage-training
description: Explain the stage 1 and stage 2 training flow, LR-HR alignment logic, and the constraints that must hold when editing trainer_two_stage.py.
---

# Two-Stage Training

> 面向修改 `trainer_two_stage.py`、训练脚本和对齐损失的人。目标是说明 stage 1 / stage 2 各自训练什么、如何衔接、哪些约束最容易被破坏。

---

## 1. 数据组织

```text
DATA_PATH/
├── train/
│   ├── LR/   # 低分辨率图像，默认 80x80
│   ├── HR/   # 高分辨率图像，默认 256x256
│   └── REF/  # 预留参考图像，目前未在主流程使用
└── val/
    ├── LR/
    └── HR/
```

---

## 2. 阶段划分

通过 `--training_stage` 控制训练阶段：

| 参数 | 含义 |
|------|------|
| `training_stage=1` | 只训练 LR VAE，HR VAE 冻结 |
| `training_stage=2` | 只训练 HR VAE，LR VAE 冻结 |
| `use_lr_hr_alignment=True` | 在阶段 2 启用 5x5 latent 对齐损失 |
| `use_lr_hr_alignment=False` | 阶段 2 仍使用 LR-HR paired loader，但只训练 HR 重建，不加入对齐损失 |
| `alignment_loss_weight=1.0` | 对齐损失权重 |

### 2.1 Stage 1

目标：训练 LR VAE，得到稳定的 `lr_f_5x5`。

损失形式：

```text
L_total = L1_rec * wei_l1 + LPIPS * wei_lpips + KL * lr_vq_beta + L_adv * wei_disc
```

### 2.2 Stage 2

目标：训练 HR 多尺度 VAE，同时让 HR 最粗尺度语义与 LR `5x5` latent 对齐。

损失形式：

```text
L_total = L1_rec * wei_l1 + LPIPS * wei_lpips + KL + L_adv * wei_disc + MSE(HR_5x5, LR_5x5) * alignment_loss_weight
```

---

## 3. 对齐机制

文件：`trainer_two_stage.py`

当前实现的关键点：

1. 先用冻结的 LR VAE 对 `inp_lr` 编码，得到 `lr_f_5x5`。
2. 对 HR encoder 输出做 `area` 下采样，得到 `hr_f_5x5`。
3. 再通过 HR 侧的 `mean_logvar_conv` 取均值分支，得到 `hr_f_5x5_mean`。
4. 使用 MSE 做对齐：

```python
L_align = F.mse_loss(hr_f_5x5_mean, lr_f_5x5)
```

这一约束的作用是让 HR 模型最粗粒度的 latent 语义对齐到 LR 模型已经学到的低分辨率语义底座。

性能约束：stage2 训练时必须复用 HR 主 forward 已经算出的 encoder feature 来取得 `hr_f_5x5_mean`，不要在 loss 中再次调用 `img_to_scale_posterior_stats(inp_hr)` 之类会二次执行 HR encoder 的路径。当前推荐入口是 `VQVAE.forward(..., ret_scale_posterior_stats=True, scale_index=0)` 或 `forward_with_scale_posterior_stats()`。

关闭 alignment 的 stage2 对照实验仍会加载 `(LR, HR)` 配对数据，但 loss 退化为 HR VAE 重建训练：

```bash
STAGE=2 USE_LR_HR_ALIGNMENT=False bash train.sh
```

这个实验用于判断问题来自 stage2 入口/paired 数据流程，还是来自 `L_align` 约束本身。

---

## 4. 关键约束

修改两阶段训练逻辑时，必须优先检查：

1. `patch_nums` 的最后一个值必须等于 encoder 输出分辨率，通常为 `img_size / 16`。
2. 阶段 2 不是普通的 HR 重建训练，而是“HR 多尺度训练 + 5x5 latent 对齐”。
3. `lr_f_5x5` 是阶段 2 的监督目标之一，不能在训练过程中被误更新。
4. `disc_opt`、`lr_vae_opt`、`vae_opt` 的职责边界明确，不应混用参数组。
5. 判别器更新和生成器对抗损失不能共用同一份 `detached` fake logits：
   - `Ld` 可以使用 `rec.detach()`
   - `Lg_adv` 必须重新对未 `detach` 的重建结果做一次判别器前向
   - 否则 `Lg_adv` 不会回传到 VAE/LR_VAE，`_compute_adaptive_weight()` 会在最后一层权重上报 `Tensor appears to not have been used in the graph`
6. 当 `warmup_disc_schedule == 0`（例如 `disc_start_ep` 之前）时，必须**完全跳过** GAN 分支（`disc forward`、`Lg_adv`、`wei_g`、`Ld backward`）：
   - 不能只写 `Lg += wei_g * Lg_adv * 0`
   - 因为 `NaN * 0` 仍是 `NaN`，会把生成器 loss 污染为 `NaN`
   - 这条约束同时适用于 stage 1 和 stage 2
7. 如果修改 encoder 下采样倍率，必须同步检查：
   - `patch_nums`
   - `lr_img_size`
   - HR/LR 对齐分辨率
   - checkpoint 兼容性
8. LR VAE 的 `Lkl` 是对 `C×H×W = 32×5×5 = 800` 维全部求和再 batch 平均：
   - `lr_vq_beta` 直接乘这个求和后的值，1.0 在这个配置下会导致早期 **posterior collapse**（decoder 只能输出数据集均值 → 看起来像“模糊的白球”）
   - 建议范围 `1e-4 ~ 1e-3`，并搭配 `lr_kl_warmup_ep >= 1.0` 做线性 warmup
   - 判定塌缩的信号：`Lkl` 在前几百个 iter 保持在 10+ 量级；保存的重建图无结构只有低频成分
9. 保存训练期对比图必须走 **eval 模式 / `posterior.mode()`**（`_deterministic_reconstruction`）：
   - 训练期前向使用 `posterior.sample()`，latent 上会再叠一层高斯噪声
   - 如果保存图用训练期 forward 的输出，早期会看到比 `eval_ep` PSNR 更糟糕的图像——这属于可视化 bug，不是模型 bug
10. `LR_VAE._init_mean_logvar_conv()` 的初始化结果不能被二次通用初始化覆盖：
   - 该函数会把 `mean_logvar_conv` 的 logvar 偏置设为 `-2.0`，用于让 stage 1 起步时 `posterior std` 低于 1，避免一开始就贴近先验
   - 在 `build_two_stage_models()` 中做 `init_weights()` 时，不能再次对 `lr_vae.mean_logvar_conv` 执行通用 conv 初始化（通用逻辑会把 bias 清零）
   - 若必须重置 LR VAE 的其余模块，应在最后重新调用一次 `_init_mean_logvar_conv()`，并在训练日志核对 `[LR Posterior][it0]` 的 `std` 不是约 `1.0`

---

## 5. 常见修改入口

### 5.1 修改多尺度配置

只需调整 `patch_nums`，但必须保证最后一个尺度与 encoder 输出分辨率一致。

```bash
--patch_nums 5 6 8 10 13 16
--patch_nums 8 10 13 16
```

### 5.2 切换训练阶段

```bash
--training_stage=1
--training_stage=2 --use_lr_hr_alignment=True
```

### 5.3 关闭判别器

```bash
--ld=0 --disc_start_ep=99999
```

---

## 6. 修改建议

当需要改动对齐机制时，优先核对：

- LR 侧目标是否仍为 `5x5`
- HR 侧取的是采样值、均值，还是其他形式的 latent
- 验证流程是否同步统计了 alignment loss
- checkpoint 恢复后 `training_stage` 是否正确回填

如果问题只涉及 GAN loss 调用差异，去看：`AICoding/skills/discriminator-loss-compatibility/`