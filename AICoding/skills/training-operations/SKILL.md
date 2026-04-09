---
name: training-operations
description: Collect training parameters, optimizer responsibilities, logging outputs, checkpoint behavior, and experiment runbook guidance.
---

# Training Operations

> 面向跑实验、恢复训练、看日志和调整训练参数的人。目标是把训练运维相关的信息单独收拢，避免混在架构说明里。

---

## 1. 参数速查

参数定义文件：`utils/arg_util.py`

### 1.1 HR VAE

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `patch_nums` | `(5, 6, 8, 10, 13, 16)` | 多尺度配置 |
| `vocab_width` | `32` | HR latent 通道数 |
| `ch` | `160` | HR 主干基础通道数 |
| `share_quant_resi` | `4` | `quant_resi` 共享策略 |
| `vq_beta` | `0.25` | HR KL loss 权重 |

### 1.2 LR VAE

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `lr_img_size` | `80` | LR 输入分辨率，默认映射到 `5x5` |
| `lr_ch` | `128` | LR VAE 通道数 |
| `lr_vocab_width` | `32` | LR latent 通道数 |
| `lr_vq_beta` | `1.0` | LR KL loss 权重 |

### 1.3 阶段控制

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `training_stage` | `1` | `1` 表示 LR 阶段，`2` 表示 HR 阶段 |
| `use_lr_hr_alignment` | `False` | 是否启用 LR-HR 对齐损失 |
| `alignment_loss_weight` | `1.0` | 对齐损失权重 |

### 1.4 实验与训练

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

## 2. Optimizer 分工

项目中的 `*_opt` 不是裸 `torch.optim.Optimizer`，而是 `utils/amp_opt.py` 中的 `AmpOptimizer` 封装器，统一负责：

- optimizer 调度
- AMP 上下文
- 梯度裁剪
- 梯度累计
- `backward + step + zero_grad`
- loss scale 相关日志

### 2.1 `lr_vae_opt`

负责更新 LR VAE。

主要用于：

- LR encoder
- LR posterior 参数层
- LR decoder

### 2.2 `vae_opt`

负责更新 HR 多尺度 VAE。

主要用于：

- HR encoder
- 多尺度 quantizer
- HR decoder

### 2.3 `disc_opt`

负责更新 `DinoDisc`。

主要用于：

- 学习区分真实图像与重建图像
- 为生成器提供有效对抗梯度

### 2.4 拆分原因

将三类 optimizer 分开，是为了：

- 明确参数边界
- 支持阶段性冻结与解冻
- 支持不同学习率与训练策略
- 降低误更新风险

---

## 3. 常用训练入口

### 3.1 设置实验标题

```bash
--exp_note="测试更小的 KL 权重"
```

### 3.2 恢复训练

```bash
--resume="path/to/ckpt.pth"
```

---

## 4. 日志与产物

### 4.1 日志

- `dist.py` 对 `print()` 做了统一封装，输出默认带时间戳和文件位置信息。
- stdout 备份文件：`local_output/backup1_stdout.txt`
- TensorBoard 目录：`local_output/tb-{exp_name}__{config}/`

### 4.2 重建图输出

- 阶段 1：`local_output/reconstruction_samples_lr/`
- 阶段 2：`local_output/reconstruction_samples_hr/`
- 通用重建目录：`local_output/reconstruction_samples/`
- 两阶段训练里若未显式覆盖目录名，stage 1 / stage 2 默认分别写入上述两个固定目录；可通过 `--reconstruction_dir_name` 覆盖。
- `epXXXX_itYYYYYY_comparison.png` 表示第 `ep` 个 epoch、第 `it` 个 iteration 保存的一张对比图。
- 若 `--reconstruction_save_interval=0`，保存时机跟随每个 epoch 内的日志迭代点；若 `>0`，则改为“每 N 个 iteration 保存一张”。
- 每个重建目录现在会写入 `run_metadata.json`，用于记录本次训练参数、保存频率和后处理说明，避免只能回查 stdout 或 checkpoint。

### 4.3 Checkpoint

每隔 `val_and_saving_per_ep` 个 epoch 保存一次，通常包括：

- `ckpt-last.pth`
- `ckpt-best.pth`
- `ckpt-{ep}.pth`

checkpoint 内容包含：

```text
{ epoch, iter, trainer, args }
```

---

## 5. 排查顺序

当你要改训练脚本、跑实验或恢复训练时，建议按这个顺序检查：

1. 先看 `utils/arg_util.py`，确认入口参数与默认值。
2. 再看 `train.sh` 或具体运行命令，确认阶段、输出目录和 resume 参数。
3. 再看对应 trainer，确认 loss、冻结逻辑和 optimizer 是否匹配。
4. 最后检查 TensorBoard、stdout 备份和 checkpoint 路径是否与预期一致。

如果问题涉及 stage 逻辑或对齐损失，去看：`AICoding/skills/two-stage-training/`