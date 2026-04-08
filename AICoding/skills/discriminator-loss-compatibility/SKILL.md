---
name: discriminator-loss-compatibility
description: Ensure discriminator loss functions support both legacy signed-logits calls and explicit is_real_pred or for_g calling styles.
---

# Discriminator Loss Compatibility

> 面向修改 `utils/loss.py`、`trainer.py`、`trainer_two_stage.py` 中 GAN loss 调用的人。目标是解释为什么这个仓库必须兼容两种判别器损失接口，以及改动时该怎么避免回归。

---

## 1. 相关文件

- `utils/loss.py`
- `trainer.py`
- `trainer_two_stage.py`

---

## 2. 两种调用风格

### 2.1 旧风格：signed logits

来自 `trainer.py`：

```python
Ld = self.d_criterion(logits_real) + self.d_criterion(-logits_fake)
```

这里的约定是：

- 调用方自己先把真假样本的符号处理好
- 损失函数只接收一个“已经处理过符号的 logits”
- 因此旧版 `hinge_loss(logits)`、`softplus_loss(logits)`、`linear_loss(logits)` 只需要一个参数

### 2.2 显式风格：传入语义

来自 `trainer_two_stage.py`：

```python
Ld_real = self.d_criterion(is_real_pred=True, logits=inp_d_logits)
Ld_fake = self.d_criterion(is_real_pred=False, logits=rec_d_logits)
Lg_adv = self.d_criterion(is_real_pred=True, logits=rec_d_logits, for_g=True)
```

这里的约定是：

- 调用方不再提前翻转符号
- 调用方直接把“真实样本 / 假样本 / 生成器目标”的语义传给损失函数
- 损失函数内部统一决定如何处理 logits

---

## 3. 为什么必须扩展原接口

之所以扩展 `utils/loss.py`，而不是只修当前报错点，原因有三点：

1. 两个 trainer 已经稳定采用了不同的调用习惯，直接只保留其中一种会破坏另一条训练路径。
2. 这三种对抗损失本质上都只是围绕“符号后的 logits”计算，扩展同一个函数比复制两套实现更不容易漂移。
3. 兼容旧接口可以避免把历史训练逻辑、调参经验和潜在 checkpoint 工作流一起打断。

换句话说，这里修的是“接口边界不一致”，不是单纯给 `hinge_loss` 多塞两个参数。

---

## 4. 推荐实现原则

- 保留旧接口：`loss(logits)`
- 新增显式接口：`loss(logits, is_real_pred=..., for_g=...)`
- 当 `is_real_pred is None` 时，严格按旧逻辑执行
- 当传入 `is_real_pred` 时，由损失函数内部统一决定符号翻转与生成器目标

当前 `utils/loss.py` 的实现就是按这个原则写的。

直接收益：

- `trainer.py` 无需改动，旧训练流程保持可用
- `trainer_two_stage.py` 可以继续使用更清晰的显式语义
- 后续如果统一 trainer 风格，也只需要收敛调用方式，不需要再改损失定义本身

---

## 5. 维护时的检查顺序

1. 先检查 `trainer.py` 是否仍在传入“已带符号”的 logits。
2. 再检查 `trainer_two_stage.py` 是否仍在传入 `is_real_pred` / `for_g`。
3. 修改 `utils/loss.py` 后，必须同时验证这两条路径，而不是只测当前报错的那一条。

---

## 6. 错误做法

- 只为了修 two-stage 报错，把 `loss(logits)` 旧签名直接删掉。
- 在 `trainer_two_stage.py` 里临时手写一份 `hinge` / `softplus` / `linear` 分支，导致两边实现分叉。
- 把“真实样本 / 假样本 / 生成器目标”的语义混在调用方和损失函数两边同时处理，造成重复翻转符号。

---

## 7. 一句话判断标准

只要这个仓库还同时保留 `trainer.py` 和 `trainer_two_stage.py` 两条训练路径，`utils/loss.py` 就不应该只支持单一调用签名。