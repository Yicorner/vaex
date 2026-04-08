---
name: project-overview
description: Explain what this repository builds, where the core modules live, and how the major model components are divided.
---

# Project Overview

> 面向第一次进入仓库的人。目标是快速建立对“项目在做什么、关键代码在哪里、模块如何分工”的整体认识。

---

## 1. 项目在做什么

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

## 2. 关键目录

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

## 3. 核心模型

### 3.1 HR 多尺度 VAE

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

### 3.2 连续多尺度量化器

文件：`models/quant.py`

职责：将 encoder 特征分解为多个尺度的连续 latent，并计算 KL loss。

核心逻辑：

1. 对 encoder 输出做多尺度残差分解。
2. 每个尺度从当前 residual 中提取该尺度特征。
3. 通过 `mean_logvar_conv` 生成 Gaussian posterior 参数。
4. 采样得到当前尺度 latent。
5. 上采样并经 `quant_resi` 精炼。
6. 累加到 `f_hat`，继续拟合更细尺度。

默认尺度配置来自 `patch_nums=(5, 6, 8, 10, 13, 16)`。

### 3.3 LR 单尺度 VAE

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

### 3.4 判别器

文件：`models/dino.py`

职责：为 VAE 训练提供对抗信号。

特点：

- 基于冻结的 DINO ViT-Small 特征。
- 仅后续判别头参与训练。
- 预训练权重路径为 `./ckpt_vaex/vit_small_patch16_224_dino.pth`。

---

## 4. 进入代码时的优先顺序

如果你刚接手一个问题，建议先按这个顺序看：

1. `utils/arg_util.py`：确认入口参数与默认值
2. `models/vqvae.py` / `models/lr_vae.py`：确认模型职责和张量尺度
3. `trainer_two_stage.py`：确认训练阶段、loss 拼接和冻结逻辑
4. 对应专题 skill：补足某个局部主题的背景

---

## 5. 相关 skill

- 两阶段训练与对齐：`AICoding/skills/two-stage-training/`
- 判别器损失兼容性：`AICoding/skills/discriminator-loss-compatibility/`
- 训练运维：`AICoding/skills/training-operations/`