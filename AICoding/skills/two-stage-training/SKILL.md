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
| `use_lr_hr_alignment=True` | 在阶段 2 启用 scale0 辅助对齐损失 |
| `use_lr_hr_alignment=False` | 阶段 2 仍使用 LR-HR paired loader，但只训练 HR 重建，不加入对齐损失 |
| `alignment_loss_type=scale0_image` | 默认新路径：不依赖 stage1 latent，把 HR scale0 latent 解码成图像后对齐 LR 图像 |
| `alignment_loss_type=latent` | 旧路径：对齐 HR scale0 posterior mean 与 stage1 LR VAE latent |
| `alignment_loss_weight=0.5` | 对齐损失权重 |
| `alignment_loss_warmup_ep=0.0` | 对齐损失线性 warmup epoch 数，0 表示不启用 |
| `stage2_use_kl=True` | 兼容旧路径：stage2 训练时采样 latent，并加入 HR 多尺度 KL |
| `stage2_use_kl=False` | 新 AE 路径：stage2 训练时也使用 posterior mean，`Lkl=0`，不做采样 |
| `use_stage2_mid_scale_loss=True` | 启用 stage2 中间累计尺度监督 |
| `stage2_mid_scale_loss_type=pixel_l1` | 默认旧行为：中间尺度 decode 与 band-limited HR target 做 L1 |
| `stage2_mid_scale_loss_type=haar_dwt` | 新行为：先做一级 Haar DWT，再按 `LL/LH/HL/HH` 子带加权监督 |
| `stage2_mid_scale_dwt_*_weights` | DWT 子带权重；长度为 1 时广播到所有中间尺度，或与 `stage2_mid_scale_indices` 对齐 |
| `l1/l2/lp/ld` | 主重建、LPIPS、GAN loss 权重；`lp` 在 trainer 内会再乘 2 |

### 2.1 Stage 1

目标：训练 LR VAE，得到稳定的 `lr_f_5x5`。

损失形式：

```text
L_total = L1_rec * wei_l1 + LPIPS * wei_lpips + KL * lr_vq_beta + L_adv * wei_disc
```

### 2.2 Stage 2

目标：训练 HR 多尺度 VAE，同时让 HR 最粗尺度能直接重建 LR 视觉结构。默认不再依赖 stage1 LR VAE latent。

损失形式：

```text
L_total = L_hr_rec + stage2_kl_term + L_adv * wei_disc + L_scale0_img * alignment_loss_weight + L_mid_scale
L_hr_rec = l1 * L1(rec, HR) + l2 * MSE(rec, HR) + (2 * lp) * LPIPS(rec, HR)
L_scale0_img = L1(DecodeStage2Scale0(HR_scale0_mean), Resize(LR_image, decoded_scale0_size))
stage2_kl_term = KL when stage2_use_kl=True, otherwise 0
```

当目标是最高重建准确性而不需要 latent 多样性时，推荐实验开关为 `stage2_use_kl=False`。这时 stage2 更像一个 deterministic multi-scale autoencoder：训练和推理都走 posterior mean，`mean_logvar_conv` 的 logvar 分支仍保留在结构里用于 checkpoint 兼容，但不会贡献 KL loss 或采样噪声。

---

## 2.3 Stage2 中间尺度 Haar DWT 监督

`use_stage2_mid_scale_loss=True` 会让 HR VAE 的同一次 forward 返回指定 scale 的累计 decode 图像，不会额外跑第二次 encoder。每个 scale 的 target 仍然由 HR 图像先按 `patch_nums[si] / patch_nums[-1]` 做 band-limited 下采样，再上采样回原图大小。

默认 `stage2_mid_scale_loss_type=pixel_l1` 保持旧行为：

```text
L_mid(si) = L1(mid_scale_rec[si], bandlimited_hr_target[si])
```

`stage2_mid_scale_loss_type=haar_dwt` 时，先对 prediction 和 target 做一级 Haar DWT，再对四个子带做加权 loss：

```text
LL = low-frequency approximation
LH / HL / HH = high-frequency details
L_mid(si) = w_ll[si] * loss(LL_pred, LL_tgt)
          + w_lh[si] * loss(LH_pred, LH_tgt)
          + w_hl[si] * loss(HL_pred, HL_tgt)
          + w_hh[si] * loss(HH_pred, HH_tgt)
L_mid_scale = sum_i stage2_mid_scale_weights[i] * L_mid(si)
```

Diffusion-4K 的训练脚本使用 `pytorch_wavelets.DWTForward(J=1, mode='zero', wave='haar')`，将 `xll/xlh/xhl/xhh` 沿 channel 拼接后计算 flow-matching MSE。`myvaex` 中为了不新增依赖，trainer 内部直接用等价的 2x2 Haar 切片公式实现。`stage2_mid_scale_dwt_loss_type` 默认为 `l1`，也支持 `mse` 来更贴近 Diffusion-4K 的形式。

推荐实验思路：低尺度更多约束 `LL`，高尺度逐步提高 `LH/HL/HH`。例如对 `stage2_mid_scale_indices 1 2`：

```bash
--stage2_mid_scale_loss_type=haar_dwt
--stage2_mid_scale_dwt_ll_weights 1.0 0.25
--stage2_mid_scale_dwt_lh_weights 0.15 1.0
--stage2_mid_scale_dwt_hl_weights 0.15 1.0
--stage2_mid_scale_dwt_hh_weights 0.05 0.75
```

日志中 `L_mid_pn*` 表示该尺度子带加权后的未乘 `stage2_mid_scale_weights` loss，`L_midw_pn*` 表示乘上尺度权重后的贡献；DWT 模式会额外打印 `L_mid_ll_pn*`、`L_mid_lh_pn*`、`L_mid_hl_pn*`、`L_mid_hh_pn*`。

---

## 3. 对齐机制

文件：`trainer_two_stage.py`

当前默认实现（`alignment_loss_type=scale0_image`）的关键点：

1. HR VAE 主 forward 仍然一次性完成全尺度重建；若 `stage2_use_kl=True`，同时计算多尺度 KL。
2. 同一次 forward 暴露 `scale_index=0` 的 posterior mean，记作 `hr_scale0_mean`。
3. 调用 HR VAE 共享 decoder：先把 `hr_scale0_mean` 按正常 scale 累积路径上采样、过 `quant_resi` 和 `post_quant_conv`，再 decode 成 scale0-only 图像。
4. 将 `inp_lr` resize 到 scale0-only 图像大小，默认是 `256x256`。
5. 使用 L1 做图像空间对齐：

```python
L_align = F.l1_loss(scale0_img, resized_lr_img)
```

这一约束的作用是让 scale0 直接承担低频/LR 结构，而不是被强行拉到 stage1 LR VAE 的 latent 分布里。这样可以减少 stage1 decoder latent 空间与 stage2 共享 decoder latent 空间不一致带来的干扰。

旧实现仍可作为 ablation 保留：

```bash
STAGE=2 ALIGNMENT_LOSS_TYPE=latent STAGE1_CKPT=/path/to/stage1/ckpt-best.pth bash train.sh
```

性能约束：stage2 训练时必须复用 HR 主 forward 已经算出的 encoder feature 来取得 scale0 posterior mean，不要在 loss 中再次调用 `img_to_scale_posterior_stats(inp_hr)` 之类会二次执行 HR encoder 的路径。当前推荐入口是 `VQVAE.forward(..., ret_scale_posterior_stats=True, scale_index=0)` 或 `forward_with_scale_posterior_stats()`。

关闭 alignment 的 stage2 对照实验仍会加载 `(LR, HR)` 配对数据，但 loss 退化为 HR VAE 重建训练：

```bash
STAGE=2 USE_LR_HR_ALIGNMENT=False bash train.sh
```

这个实验用于判断问题来自 stage2 入口/paired 数据流程，还是来自 `L_align` 约束本身。

关闭 stage2 KL 的实验入口：

```bash
STAGE=2 STAGE2_USE_KL=False ALIGNMENT_LOSS_TYPE=scale0_image bash train.sh
```

预期现象：进度日志中的 `Lkl` 为 `0.00e+00`；保存重建图和验证 forward 都是 deterministic mean 路径。

如果只跑 `STAGE2_EP=2~3` 的短实验，不要沿用 `disc_start_ep=30`，否则 GAN 分支完全不启动，重建偏糊是预期的。短实验可以从下面这组较稳的去糊配比开始：

```bash
STAGE2_USE_KL=False
STAGE2_L1_WEIGHT=1.0
STAGE2_L2_WEIGHT=0.25
STAGE2_LPIPS_WEIGHT=0.25
ALIGNMENT_LOSS_WEIGHT=0.25
STAGE2_DISC_WEIGHT=0.2
STAGE2_DISC_START_EP=0.5
STAGE2_DISC_WARMUP_EP=0.5

# If NaN appears exactly when dlr becomes positive, suspect the DINO discriminator path.
# Start a diagnostic retry with:
#   DBG_NAN=True
#   STAGE2_DISC_WEIGHT=-0.05   # fixed 0.05 GAN weight, adaptive Wg disabled
#   DISC_AUG_PROB=0.5
#   STAGE2_DISC_START_EP=1.0
```

这组配比的意图：降低 MSE 平滑倾向，用 L1 保结构，用适度 LPIPS/GAN 提高清晰度，同时避免 GAN 和 scale0 alignment 过强导致伪细节或低频约束压过 HR 重建。

---

## 4. 关键约束

修改两阶段训练逻辑时，必须优先检查：

1. `patch_nums` 的最后一个值必须等于 encoder 输出分辨率，通常为 `img_size / 16`。
2. 阶段 2 不是普通的 HR 重建训练，而是“HR 多尺度训练 + scale0 辅助对齐”。
3. 默认 `scale0_image` 对齐不需要 stage1 checkpoint；只有 `alignment_loss_type=latent` 才需要加载并冻结 LR VAE。
4. `disc_opt`、`lr_vae_opt`、`vae_opt` 的职责边界明确，不应混用参数组。
5. 判别器更新和生成器对抗损失不能共用同一份 `detached` fake logits：
   - `Ld` 可以使用 `rec.detach()`
   - `Lg_adv` 必须重新对未 `detach` 的重建结果做一次判别器前向
   - 否则 `Lg_adv` 不会回传到 VAE/LR_VAE，`_compute_adaptive_weight()` 会在最后一层权重上报 `Tensor appears to not have been used in the graph`
6. 当 `warmup_disc_schedule == 0`（例如 `disc_start_ep` 之前）时，必须**完全跳过** GAN 分支（`disc forward`、`Lg_adv`、`wei_g`、`Ld backward`）：
   - 不能只写 `Lg += wei_g * Lg_adv * 0`
   - 因为 `NaN * 0` 仍是 `NaN`，会把生成器 loss 污染为 `NaN`
   - 这条约束同时适用于 stage 1 和 stage 2
   - 如果 NaN 第一次出现在 `dlr` 变成正数之后，优先排查 DINO discriminator path，而不是先怀疑重建 loss 或 scale0 alignment
   - DINO 判别器初始化必须保留 frozen DINO checkpoint，只初始化 `disc.heads`；手动初始化 spectral-norm Conv1d 后要刷新 `weight_u/weight_v`
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
   - 当 `stage2_use_kl=True` 时，训练期前向使用 `posterior.sample()`，latent 上会再叠一层高斯噪声
   - 当 `stage2_use_kl=False` 时，训练期和保存图都使用 `posterior.mode()`，这是预期行为
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
--alignment_loss_type=scale0_image --alignment_loss_weight=0.5
--stage2_use_kl=False
```

### 5.3 关闭判别器

```bash
--ld=0 --disc_start_ep=99999
```

---

## 6. 修改建议

当需要改动对齐机制时，优先核对：

- LR 侧目标是图像像素还是 legacy `5x5` latent
- 当前使用的是 `scale0_image` 还是 legacy `latent`
- HR 侧取的是采样值、均值，还是其他形式的 latent；默认取 posterior mean 来降低辅助 loss 噪声
- Stage2 主重建路径是否需要 KL/采样；追求准确重建时优先试 `stage2_use_kl=False`
- 验证流程是否同步统计了 alignment loss
- checkpoint 恢复后 `training_stage` 是否正确回填

如果问题只涉及 GAN loss 调用差异，去看：`AICoding/skills/discriminator-loss-compatibility/`

## Stage3 LR -> Stage2 Scale0 Latent

Stage3 is a separate `training_stage=3` path implemented by `train_stage3.py`.
It trains only the LR encoder and quant conv; decoder, discriminator, and KL are
not used.

Flow:

```text
LR -> Stage3Scale0Encoder.encoder -> quant_conv -> adaptive_pool(n,n)
   -> frozen copied stage2 mean_logvar_conv mean -> s0_pred

HR -> frozen stage2 teacher encoder -> quant_conv
   -> quantize.get_scale_posterior_stats(scale_index=0).mean -> s0_target
```

Required asserts:

- `teacher.quantize.v_patch_nums[0] == args.stage3_latent_size`.
- `s0_pred.shape == s0_target.shape == [B, Cvae, n, n]`.
- `n == STAGE3_LATENT_SIZE`, default `4`.

Default loss:

```text
1.00 * MSE(s0_pred, s0_target)
0.10 * SmoothL1(s0_pred, s0_target)
0.25 * L1(decode_stage2_s0(s0_pred), resize(LR))
0.10 * L1(decode_stage2_s0(s0_pred), decode_stage2_s0(s0_target))
```

Run:

```bash
STAGE=3 STAGE2_CKPT=/path/to/stage2/ckpt-best.pth STAGE3_LATENT_SIZE=4 bash train.sh
```

Eval:

```bash
python eval_stage3_ckpt.py \
  --ckpt_path /path/to/stage3/ckpt-best.pth \
  --stage2_ckpt /path/to/stage2/ckpt-best.pth \
  --test_dir /path/to/dataset/test \
  --lr_folder LR_64x64 \
  --hr_folder HR \
  --output_dir /path/to/stage3_eval
```

## 7. 单通道灰度注意事项

- `IMG_CHANNELS=1` 时 stage2 的 HR 重建和 scale0 image alignment 都在 `[B,1,H,W]` 上计算，LR target 也由 paired dataset 以灰度读取。
- LPIPS / DINO discriminator 分支内部会把灰度 repeat 到 RGB，避免预训练网络输入通道不匹配；这不改变 decoder 的真实输出通道。
- PSNR / SSIM 使用 `utils.image_saver.compute_psnr_ssim()`：灰度图不再转换 Y 通道，而是直接按单通道 `[0,1]` 图比较。
- 灰度和 RGB checkpoint 的首尾卷积形状不同；切换 `IMG_CHANNELS` 时应视为一个新模型配置。
