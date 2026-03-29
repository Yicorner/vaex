# Project Knowledge Base

> 面向 AI 助手和维护者的仓库知识库。目标是快速回答三个问题：项目在做什么、关键代码在哪里、修改时有哪些硬约束。

---

## 1. 文档定位

### 1.1 适用范围

本文件用于描述当前仓库的核心架构、训练流程、重要参数和常见修改入口，适合作为进入项目时的第一份背景资料。

### 1.2 使用原则

- 这里记录的是“高价值上下文”，不是逐行代码说明。
- 如果文档描述和代码实现冲突，以代码为准，并优先核对以下文件：
  - `utils/arg_util.py`
  - `models/vqvae.py`
  - `models/lr_vae.py`
  - `models/quant.py`
  - `trainer_two_stage.py`
- 修改项目结构、训练阶段逻辑或关键参数默认值后，应同步更新本文件。

---

## 2. 项目概览

这是一个多尺度连续 VAE 项目，目标是将图像编码为多个尺度的连续 latent 表示，用于：

- 图像重建与压缩
- 为后续图像超分辨率任务提供分层 latent 表示

当前演化路径如下：

- 早期版本：离散 VQ-VAE
- 当前版本：连续 VAE
  - 使用 Gaussian reparameterization
  - 使用 KL divergence 约束 latent 分布

项目的核心设计不再依赖离散 codebook，而是通过多尺度连续 latent 来表达从粗到细的图像信息。

---

## 3. 关键目录

```text
myvaex/
├── models/
│   ├── __init__.py       # build_vae_disc()，负责构建模型
│   ├── vqvae.py          # HR 多尺度 VAE 主模型
│   ├── lr_vae.py         # LR 单尺度 VAE，输出 5x5 latent
│   ├── quant.py          # ContinuousMultiScaleQuantizer 与 Gaussian posterior
│   ├── basic_vae.py      # Encoder / Decoder 基础 CNN 组件
│   └── dino.py           # DinoDisc 判别器
├── utils/
│   ├── arg_util.py       # 训练参数定义
│   ├── data.py           # 原始 HR 数据流程
│   ├── data_lr_hr.py     # LR-HR 配对数据流程
│   ├── data_loader.py    # Dataset 实现
│   ├── data_sampler.py   # 采样器
│   ├── amp_opt.py        # AmpOptimizer 封装器
│   ├── lpips.py          # LPIPS 感知损失
│   ├── loss.py           # 对抗损失函数
│   ├── image_saver.py    # 重建图保存
│   └── misc.py           # logger 与训练辅助工具
├── trainer.py            # 原始单阶段 HR trainer
├── trainer_two_stage.py  # 两阶段 LR+HR trainer
├── train.py              # 训练入口
├── dist.py               # 分布式工具与统一打印
└── train.sh              # 训练脚本示例
```

---

## 4. 架构总览

### 4.1 HR 多尺度 VAE

文件：`models/vqvae.py`

职责：对 HR 图像进行多尺度连续 latent 建模。

结构要点：

- Encoder 采用 5 层下采样，`ch_mult=(1, 1, 2, 2, 4)`。
- 当输入为 `256x256` 时，encoder 输出特征分辨率为 `16x16`。
- 中间 latent 由 `ContinuousMultiScaleQuantizer` 处理。
- Decoder 对称上采样，输出重建图像。

适用场景：

- HR 重建训练
- 两阶段训练中的第二阶段主模型

### 4.2 连续多尺度量化器

文件：`models/quant.py`

职责：将 encoder 特征分解为多个尺度的连续 latent，并计算 KL loss。

核心逻辑：

- 对 encoder 输出做多尺度残差分解。
- 每个尺度都执行以下流程：
  1. 从当前 residual 中提取该尺度特征。
  2. 通过 `mean_logvar_conv` 生成 Gaussian posterior 参数。
  3. 采样得到当前尺度 latent。
  4. 上采样并经 `quant_resi` 精炼。
  5. 累加到 `f_hat`，继续拟合更细尺度。
- 默认尺度配置来自 `patch_nums=(5, 6, 8, 10, 13, 16)`。
- KL loss 使用 `log1p` 压缩，以降低大尺度项对总 loss 的支配效应。

### 4.3 LR 单尺度 VAE

文件：`models/lr_vae.py`

职责：将 LR 图像编码为稳定的 `5x5` latent，作为阶段 2 的对齐目标。

结构要点：

- 输入默认是 `80x80`。
- 经过 encoder 后得到 `5x5` 特征。
- 使用 `mean_logvar_conv + DiagonalGaussianDistribution` 生成 posterior。
- KL loss 采用简单均值形式，不做 `log1p` 压缩。

关键接口：

- `forward(inp)` -> `(rec_B3HW, f_5x5, kl_loss)`
- `encode_to_5x5(inp)` -> `f_5x5`

### 4.4 判别器

文件：`models/dino.py`

职责：为 VAE 训练提供对抗信号。

特点：

- 基于冻结的 DINO ViT-Small 特征。
- 仅后续判别头参与训练。
- 预训练权重路径为 `./ckpt_vaex/vit_small_patch16_224_dino.pth`。

---

## 5. 训练流程

### 5.1 数据组织

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

### 5.2 两阶段训练模式

通过 `--training_stage` 控制训练阶段：

| 参数 | 含义 |
|------|------|
| `training_stage=1` | 只训练 LR VAE，HR VAE 冻结 |
| `training_stage=2` | 只训练 HR VAE，LR VAE 冻结 |
| `use_lr_hr_alignment=True` | 在阶段 2 启用 5x5 latent 对齐损失 |
| `alignment_loss_weight=1.0` | 对齐损失权重 |

### 5.3 阶段 1

目标：训练 LR VAE，得到稳定的 `lr_f_5x5`。

损失形式：

```text
L_total = L1_rec * wei_l1 + LPIPS * wei_lpips + KL * lr_vq_beta + L_adv * wei_disc
```

### 5.4 阶段 2

目标：训练 HR 多尺度 VAE，同时让 HR 最粗尺度语义与 LR `5x5` latent 对齐。

损失形式：

```text
L_total = L1_rec * wei_l1 + LPIPS * wei_lpips + KL + L_adv * wei_disc + MSE(HR_5x5, LR_5x5) * alignment_loss_weight
```

### 5.5 对齐机制

文件：`trainer_two_stage.py`

当前实现的关键点：

- 先用冻结的 LR VAE 对 `inp_lr` 编码，得到 `lr_f_5x5`。
- 对 HR encoder 输出做 `area` 下采样，得到 `hr_f_5x5`。
- 再通过 HR 侧的 `mean_logvar_conv` 取均值分支，得到 `hr_f_5x5_mean`。
- 使用 MSE 做对齐：

```python
L_align = F.mse_loss(hr_f_5x5_mean, lr_f_5x5)
```

这一约束的作用是让 HR 模型最粗粒度的 latent 语义对齐到 LR 模型已经学到的低分辨率语义底座，从而为后续超分辨率条件建模提供一致的 latent 空间。

---

## 6. 关键约束

以下约束在改代码时必须优先检查：

1. `patch_nums` 的最后一个值必须等于 encoder 输出分辨率，通常为 `img_size / 16`。
2. 阶段 2 不是普通的 HR 重建训练，而是“HR 多尺度训练 + 5x5 latent 对齐”。
3. `lr_f_5x5` 是阶段 2 的监督目标之一，不能在训练过程中被误更新。
4. `disc_opt`、`lr_vae_opt`、`vae_opt` 的职责边界明确，不应混用参数组。
5. 如果修改 encoder 下采样倍率，必须同步检查：
   - `patch_nums`
   - `lr_img_size`
   - HR/LR 对齐分辨率
   - checkpoint 兼容性

---

## 7. 参数速查

参数定义文件：`utils/arg_util.py`

### 7.1 HR VAE

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `patch_nums` | `(5, 6, 8, 10, 13, 16)` | 多尺度配置 |
| `vocab_width` | `32` | HR latent 通道数 |
| `ch` | `160` | HR 主干基础通道数 |
| `share_quant_resi` | `4` | `quant_resi` 共享策略 |
| `vq_beta` | `0.25` | HR KL loss 权重 |

### 7.2 LR VAE

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `lr_img_size` | `80` | LR 输入分辨率，默认映射到 `5x5` |
| `lr_ch` | `128` | LR VAE 通道数 |
| `lr_vocab_width` | `32` | LR latent 通道数 |
| `lr_vq_beta` | `1.0` | LR KL loss 权重 |

### 7.3 阶段控制

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `training_stage` | `1` | `1` 表示 LR 阶段，`2` 表示 HR 阶段 |
| `use_lr_hr_alignment` | `False` | 是否启用 LR-HR 对齐损失 |
| `alignment_loss_weight` | `1.0` | 对齐损失权重 |

### 7.4 实验与训练

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `exp_name` | 必填 | 实验名 |
| `exp_note` | `''` | 日志首部展示的实验标题 |
| `bed` | 必填 | checkpoint 保存目录 |
| `ep` | `250` | 总 epoch 数 |
| `lbs` | `8` | local batch size |
| `vae_lr` / `disc_lr` | `3e-4` | 学习率 |
| `ld` | `0.4` | 判别器损失权重；`0` 表示关闭判别器 |
| `disc_start_ep` | `0` | 判别器启动 epoch；`0` 表示自动设为 `0.2 * ep` |
| `img_size` | `256` | HR 输入图像大小 |
| `val_and_saving_per_ep` | `5` | 每隔多少个 epoch 验证并保存 |

---

## 8. Optimizer 分工

项目中的 `*_opt` 不是裸 `torch.optim.Optimizer`，而是 `utils/amp_opt.py` 中的 `AmpOptimizer` 封装器，统一负责：

- optimizer 调度
- AMP 上下文
- 梯度裁剪
- 梯度累计
- `backward + step + zero_grad`
- loss scale 相关日志

### 8.1 `lr_vae_opt`

负责更新 LR VAE。

主要用于：

- LR encoder
- LR posterior 参数层
- LR decoder

### 8.2 `vae_opt`

负责更新 HR 多尺度 VAE。

主要用于：

- HR encoder
- 多尺度 quantizer
- HR decoder

### 8.3 `disc_opt`

负责更新 `DinoDisc`。

主要用于：

- 学习区分真实图像与重建图像
- 为生成器提供有效对抗梯度

### 8.4 拆分原因

将三类 optimizer 分开，是为了：

- 明确参数边界
- 支持阶段性冻结与解冻
- 支持不同学习率与训练策略
- 降低误更新风险

---

## 9. 常见修改入口

### 9.1 修改多尺度配置

只需调整 `patch_nums`，但必须保证最后一个尺度与 encoder 输出分辨率一致。

```bash
--patch_nums 5 6 8 10 13 16
--patch_nums 8 10 13 16
```

### 9.2 设置实验标题

```bash
--exp_note="测试更小的 KL 权重"
```

### 9.3 切换训练阶段

```bash
--training_stage=1
--training_stage=2 --use_lr_hr_alignment=True
```

### 9.4 关闭判别器

```bash
--ld=0 --disc_start_ep=99999
```

---

## 10. 日志与产物

### 10.1 日志

- `dist.py` 对 `print()` 做了统一封装，输出默认带时间戳和文件位置信息。
- stdout 备份文件：`local_output/backup1_stdout.txt`
- TensorBoard 目录：`local_output/tb-{exp_name}__{config}/`

### 10.2 重建图输出

- 阶段 1：`local_output/reconstruction_samples_lr/`
- 阶段 2：`local_output/reconstruction_samples_hr/`
- 通用重建目录：`local_output/reconstruction_samples/`

### 10.3 Checkpoint

每隔 `val_and_saving_per_ep` 个 epoch 保存一次，通常包括：

- `ckpt-last.pth`
- `ckpt-best.pth`
- `ckpt-{ep}.pth`

checkpoint 内容包含：

```text
{ epoch, iter, trainer, args }
```

恢复训练示例：

```bash
--resume="path/to/ckpt.pth"
```

---

## 11. 修改建议

当需要改动训练逻辑时，建议按以下顺序排查：

1. 先看 `utils/arg_util.py`，确认入口参数与默认值。
2. 再看对应模型文件，确认 tensor 形状和模块边界。
3. 最后看 `trainer_two_stage.py`，确认 loss 拼接、冻结逻辑和日志输出。

当需要改动对齐机制时，优先核对：

- LR 侧目标是否仍为 `5x5`
- HR 侧取的是采样值、均值，还是其他形式的 latent
- 验证流程是否同步统计了 alignment loss
- checkpoint 恢复后 `training_stage` 是否正确回填

---

## 12. 近期重要演化

1. 离散 VQ-VAE 已迁移为连续 KL-VAE，详见 `CONTINUOUS_VAE_CHANGES.md`。
2. 默认 `patch_nums` 已调整为 `(5, 6, 8, 10, 13, 16)`。
3. 新增 `exp_note` 参数，用于实验标题展示。
4. 新增 LR VAE 与两阶段训练框架，详见 `TWO_STAGE_TRAINING_README.md`。

---

## 13. 一句话总结

可以把这个项目理解为：先用 LR VAE 学一个稳定的 `5x5` 低分辨率语义底座，再让 HR 多尺度 VAE 在这个底座上继续学习更细粒度的高分辨率 latent 表示。
