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

### 1.3.5 数据子目录

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `lr_folder` | `'LR'` | 每个 split 下存放 LR 图像的子目录名，可自定义（如 `LR_64x64`） |
| `hr_folder` | `'HR'` | 每个 split 下存放 HR 图像的子目录名 |

在 `train.sh` 中通过环境变量 `LR_FOLDER` / `HR_FOLDER` 覆盖，支持小写别名 `lr_folder` / `hr_folder`。

### 1.3.6 `train.sh` 关键覆盖项

除 `arg_util.py` 的 CLI 参数外，`train.sh` 还支持以下常用环境变量覆盖：

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `PATCH_NUMS` | `"5 6 8 10 13 16"` | 多尺度配置字符串，脚本内会拆成数组传给 `--patch_nums` |
| `LR_IMG_SIZE` | `80` | 传给 `--lr_img_size`，需与 `patch_nums[0]` 对齐（`lr_img_size/16`） |
| `STAGE1_CKPT` | `${STAGE1_BED}/ckpt-best.pth` | stage2 的 `--lr_vae_resume` 路径，可直接指向指定 stage1 checkpoint |
| `TRAIN_LOG_POINTS_PER_EPOCH` | `40` | 控制每个 epoch 内 `[Ep]: [...]` 进度日志打印点数量；值越大打印越频繁 |
| `USE_LR_HR_ALIGNMENT` | `True` | stage2 是否启用 LR-HR latent 对齐；设为 `False` 可做“paired loader + HR 重建”对照实验 |
| `ALIGNMENT_LOSS_WEIGHT` | `1.0` | stage2 对齐损失权重，仅在 `USE_LR_HR_ALIGNMENT=True` 时影响训练 loss |

stage2 关闭 alignment 的对照入口：

```bash
STAGE=2 USE_LR_HR_ALIGNMENT=False bash train.sh
```

该对照仍使用 paired loader，即每个 batch 返回 `(inp_lr, inp_hr)`，但 `trainer_two_stage.py` 不计算或加入 `L_align`，可用于隔离“LR-HR 数据加载/训练入口变化”和“alignment loss 本身”的影响。

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
- 可通过 `TRAIN_LOG_POINTS_PER_EPOCH`（映射到 `--train_log_points_per_epoch`）调节每 epoch 进度日志密度；`<=0` 使用旧的自动策略。

### 4.2 重建图输出

- 阶段 1：`local_output/reconstruction_samples_lr/`
- 阶段 2：`local_output/reconstruction_samples_hr/`
- 分两阶段之前的重建目录：`local_output/reconstruction_samples/`
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

---

## 6. Stage 1 Checkpoint 测试脚本

脚本：`eval_stage1_ckpt.py`

用途：对某个 stage 1 的 `ckpt` 在单一测试目录上做随机抽样评估（输入图像自重建），并导出可复盘产物。

### 6.1 功能

- 在 `--test_dir` 下随机抽样（默认 `100` 张）做推理
- 逐张输出 `PSNR`、`SSIM` 到 `metrics_per_image.csv`
- 输出统计摘要到 `metrics_summary.json` 和 `metrics.log`
- 输出重建可视化到 `comparisons/`（`4x2`，左 `Input` 右 `Pred`）
- 同步输出 `100` 张单图预测到 `predictions/`

### 6.2 用法示例

```bash
python eval_stage1_ckpt.py \
  --ckpt_path local_output/ckpt-3.pth \
  --test_dir /home/featurize/data/brats_256_t2_2021_pair_png_with_ref/test/LR \
  --output_dir local_output/stage1_test_eval \
  --num_samples 100 \
  --seed 42 \
  --batch_size 4
```

说明：

- 脚本只接收一个测试目录参数 `--test_dir`，直接读取该目录下的图片文件（不会再拼接 `LR/HR` 子目录）
- 模型结构参数默认从 checkpoint 的 `args` 读取（可用 CLI 覆盖）