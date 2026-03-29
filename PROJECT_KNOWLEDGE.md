# Project Knowledge Base

> 本文档是AI助手的项目背景速查手册，记录了项目结构、关键设计和常用修改点，避免每次对话都重新熟悉上下文。

---

## 1. 项目定位

这是一个**多尺度连续VAE**项目，核心目标是将图像编码为多个尺度的token，用于图像压缩和未来的**图像超分辨率**任务。

- 原始版本：VQ-VAE（离散codebook）
- 当前版本：已改造为连续VAE（KL divergence + Gaussian reparameterization）

---

## 2. 目录结构速览

```
myvaex/
├── models/
│   ├── __init__.py          # build_vae_disc()，构建所有模型
│   ├── vqvae.py             # HR 多尺度VAE（主模型）
│   ├── lr_vae.py            # LR 单尺度VAE（5x5输出，新增）
│   ├── quant.py             # ContinuousMultiScaleQuantizer + DiagonalGaussianDistribution
│   ├── basic_vae.py         # Encoder / Decoder（基础CNN模块）
│   └── dino.py              # DinoDisc（判别器）
├── utils/
│   ├── arg_util.py          # 所有超参数（Args类，基于Tap）
│   ├── data.py              # 原始数据加载（HR only）
│   ├── data_lr_hr.py        # LR-HR配对数据加载（新增）
│   ├── data_loader.py       # DIV2KData Dataset类
│   ├── data_sampler.py      # DistInfiniteBatchSampler
│   ├── amp_opt.py           # AmpOptimizer（混精度优化器）
│   ├── lpips.py             # LPIPS感知损失
│   ├── loss.py              # hinge/softplus/linear判别器损失
│   ├── image_saver.py       # save_reconstruction_comparison()
│   └── misc.py              # MetricLogger / TensorboardLogger
├── trainer.py               # 原始VAETrainer（HR单阶段）
├── trainer_two_stage.py     # 两阶段Trainer（LR+HR，新增）
├── train.py                 # 主训练入口
├── dist.py                  # 分布式工具，print()被改写为带时间戳版本
└── train.sh                 # 训练脚本
```

---

## 3. 核心模型架构

### 3.1 HR多尺度VAE（`models/vqvae.py`）

- **Encoder**：5层下采样，`ch_mult=(1,1,2,2,4)`，16倍下采样
  - 输入256×256 → 输出16×16特征图
- **ContinuousMultiScaleQuantizer**（`models/quant.py`）：
  - 对encoder输出做多尺度残差分解
  - 每个尺度：下采样residual → `mean_logvar_conv` → 采样 → 上采样 → `quant_resi` 精炼 → 累加到 `f_hat`
  - 默认尺度：`patch_nums = (5, 6, 8, 10, 13, 16)`（可通过参数修改）
  - KL loss使用 `log1p` 压缩，防止大尺度主导
- **Decoder**：对称的5层上采样，输出重建图像

### 3.2 LR单尺度VAE（`models/lr_vae.py`，新增）

- **作用**：将LR图像（80×80）编码为 5×5 tokens
- **结构**：标准Encoder + `mean_logvar_conv` + `DiagonalGaussianDistribution` + Decoder
- **KL loss**：简单版本，不使用 log1p 压缩（`torch.mean(kl) * beta`）
- **关键方法**：
  - `forward(inp)` → `(rec_B3HW, f_5x5, kl_loss)`
  - `encode_to_5x5(inp)` → `f_5x5`（无梯度，用于推理）

### 3.3 DinoDisc（判别器）

- 基于冻结的DINO ViT-Small特征
- 只有后面的判别头是可训练的
- 权重文件：`./ckpt_vaex/vit_small_patch16_224_dino.pth`

---

## 4. 训练流程

### 4.1 数据集结构

```
DATA_PATH/
├── train/
│   ├── LR/   # 低分辨率图像（80×80）
│   ├── HR/   # 高分辨率图像（256×256）
│   └── REF/  # 参考图像（暂未使用）
└── val/
    ├── LR/
    └── HR/
```

### 4.2 两阶段训练开关

通过 `--training_stage` 控制：

| 参数 | 说明 |
|------|------|
| `training_stage=1` | 只训练LR VAE，HR VAE冻结 |
| `training_stage=2` | 只训练HR VAE，LR VAE冻结 |
| `use_lr_hr_alignment=True` | 阶段2额外加5×5对齐loss |
| `alignment_loss_weight=1.0` | 对齐loss权重 |

### 4.3 Loss构成

**阶段1（LR VAE）**：
```
L_total = L1_rec * wei_l1 + LPIPS * wei_lpips + KL * lr_vq_beta + L_adv * wei_disc
```

**阶段2（HR VAE）**：
```
L_total = L1_rec * wei_l1 + LPIPS * wei_lpips + KL + L_adv * wei_disc + MSE(HR_5x5, LR_5x5) * alignment_loss_weight
```

### 4.4 阶段2的核心对齐逻辑

- 阶段2训练时，**HR多尺度VAE的第一个尺度 5×5 必须和 LR 单尺度VAE输出的 5×5 latent 对齐**
- 训练流程是：
  - 先用**冻结的LR VAE**把 `inp_lr` 编码成 `lr_f_5x5`
  - 再从HR VAE里取出对应的 `hr_f_5x5`
  - 最后加入对齐loss：

```python
L_align = F.mse_loss(hr_f_5x5, lr_f_5x5)
```

- 这个约束的意义是：让HR VAE在最粗粒度的语义表示上和LR VAE共享同一套5×5空间，方便后续做超分辨率条件建模
- 可以把它理解成：**LR VAE先学一个稳定的低分辨率语义底座，HR VAE再在这个底座上继续学更细的高分辨率尺度**

---

## 5. 关键参数速查（`utils/arg_util.py`）

### HR VAE相关
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `patch_nums` | `(5,6,8,10,13,16)` | **多尺度配置**，直接修改即可 |
| `vocab_width` | 32 | HR VAE潜在维度（z_channels） |
| `ch` | 160 | HR VAE基础通道数 |
| `share_quant_resi` | 4 | phi层共享策略 |
| `vq_beta` | 0.25 | KL loss权重（HR） |

### LR VAE相关（新增）
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `lr_img_size` | 80 | LR图像大小（80→5×5） |
| `lr_ch` | 128 | LR VAE通道数 |
| `lr_vocab_width` | 32 | LR潜在维度 |
| `lr_vq_beta` | 1.0 | LR KL loss权重 |

### 训练阶段控制（新增）
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `training_stage` | 1 | 1=LR阶段, 2=HR阶段 |
| `use_lr_hr_alignment` | False | 启用对齐loss |
| `alignment_loss_weight` | 1.0 | 对齐loss权重 |

### 实验标注
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `exp_name` | 必填 | 实验名，影响bed目录和tb命名 |
| `exp_note` | `''` | **训练标题**，打印到日志首部醒目横幅 |
| `bed` | 必填 | checkpoint保存目录 |

### 常用训练参数
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `ep` | 250 | 总epoch数 |
| `lbs` | 8 | local batch size |
| `vae_lr` / `disc_lr` | 3e-4 | 学习率 |
| `ld` | 0.4 | 判别器loss权重；0=不使用判别器 |
| `disc_start_ep` | 0 | 判别器启动epoch；0=自动设为0.2*ep |
| `img_size` | 256 | 输入图像大小 |
| `val_and_saving_per_ep` | 5 | 每N个epoch验证并保存 |

---

## 6. Optimizer 角色说明

项目里这几个 `*_opt` 不是普通的 `torch.optim.Optimizer`，而是 `utils/amp_opt.py` 里的 **`AmpOptimizer` 封装器**。它的作用是统一处理：

- 底层optimizer（如 AdamW / Lion / Lamb）
- AMP混合精度上下文（`amp_ctx`）
- 梯度裁剪
- 梯度累计
- `backward + step + zero_grad` 的统一调度
- loss scale 相关日志

### 6.1 `disc_opt`

- 全称：**discriminator optimizer**
- 负责更新判别器 `DinoDisc` 的参数
- 用在对抗训练里，让判别器学会区分：
  - 真实图像 `inp`
  - VAE重建图像 `rec`

如果没有 `disc_opt`：
- 判别器就不会更新
- 对抗loss只会变成“空转”
- 生成器/VAE拿不到有效的对抗训练信号

### 6.2 `lr_vae_opt`

- 全称：**low-resolution VAE optimizer**
- 负责更新 **LR单尺度VAE** 的参数
- 主要在**阶段1**使用，用来训练：
  - LR encoder
  - LR `mean_logvar_conv`
  - LR decoder

如果没有 `lr_vae_opt`：
- LR VAE就没法单独训练
- 也就得不到稳定的 `lr_f_5x5`
- 阶段2的HR 5×5 对齐目标就不存在了

### 6.3 `vae_opt`

- 负责更新 **HR多尺度VAE** 的参数
- 主要在**阶段2**使用
- 训练目标是：
  - 重建HR图像
  - 学好多尺度latent
  - 同时让第一个5×5尺度对齐到 `lr_f_5x5`

### 6.4 为什么要拆成三个 optimizer？

因为三类模块的训练职责不同：

- `lr_vae_opt`：只管LR VAE
- `vae_opt`：只管HR VAE
- `disc_opt`：只管判别器

拆开的好处：

- 参数边界清楚，不会误更新
- 可以分别冻结/解冻
- 可以单独设置学习率、权重衰减、梯度裁剪
- 更适合两阶段训练切换

---

## 7. 常见修改点

### Q: 如何修改多尺度配置？
只需修改 `patch_nums` 参数。最后一个值必须等于encoder输出分辨率（`img_size / 16`）：
```bash
--patch_nums 5 6 8 10 13 16  # 默认
--patch_nums 8 10 13 16       # 从8x8开始
```

### Q: 如何给训练起标题？
```bash
--exp_note="这次测试更小的KL权重，lr_vq_beta=0.5"
```
训练开始时会打印醒目横幅，并保存到checkpoint的args中。

### Q: 如何控制两阶段训练？
```bash
--training_stage=1   # 训练LR VAE
--training_stage=2 --use_lr_hr_alignment=True  # 训练HR VAE with对齐
```

补充说明：
- 阶段2不是“普通HR重建训练”
- 而是“**HR多尺度VAE训练 + HR的5×5 latent对齐到冻结LR VAE的5×5 latent**”

### Q: 判别器怎么关闭？
```bash
--ld=0 --disc_start_ep=99999
```

---

## 8. 日志系统

- **控制台/文件**：`dist.py` 改写了 `print()`，所有输出自动带时间戳和文件行号
- **日志文件**：`local_output/backup1_stdout.txt`（stdout备份）
- **TensorBoard**：`local_output/tb-{exp_name}__{config}/`
- **重建图像**：
  - HR训练：`local_output/reconstruction_samples/`
  - LR训练（阶段1）：`local_output/reconstruction_samples_lr/`
  - HR训练（阶段2）：`local_output/reconstruction_samples_hr/`

---

## 9. Checkpoint机制

- 每 `val_and_saving_per_ep` 个epoch保存一次，同时保存：
  - `ckpt-last.pth`：最新checkpoint
  - `ckpt-best.pth`：验证loss最低的checkpoint
  - `ckpt-{ep}.pth`：按epoch编号保存

- Checkpoint内容：`{ epoch, iter, trainer（含所有模型state_dict）, args }`

- 恢复训练：`--resume="path/to/ckpt.pth"`

---

## 10. 最近做过的改动

1. **2026-01** 离散VQ-VAE → 连续KL-VAE（见`CONTINUOUS_VAE_CHANGES.md`）
2. **2026-01** 添加 `exp_note` 参数（训练标题）
3. **2026-01** 修改 `patch_nums` 默认值为 `(5,6,8,10,13,16)`（去掉了1,2,3,4）
4. **2026-01** 新增LR VAE + 两阶段训练框架（见`TWO_STAGE_TRAINING_README.md`）
