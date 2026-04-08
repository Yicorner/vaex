---
name: repo-navigation
description: Route readers to the correct repository skill before they dive into detailed project, training, or compatibility documentation.
---

# Repo Navigation

> 面向 AI 助手和维护者的仓库导航页。这里不负责讲项目细节，只负责把你带到正确的 skill。

---

## 1. 使用方式

- 先读本文件，判断当前问题属于哪一类。
- 再进入对应专题 skill，避免在一个超长文档里来回搜索。
- 如果某条知识已经能形成独立主题，应优先写入专题 skill，而不是继续堆在这里。
- 如果你想理解“项目本身是什么”，应去 `project-overview`，而不是留在这里。

---

## 2. Skill 导航

### 2.1 项目整体结构

文件夹：`AICoding/skills/project-overview/`

适合回答：

- 这个项目整体在做什么
- 核心模型分别在哪
- `models/`、`utils/`、`trainer*.py` 的职责边界是什么

---

### 2.2 两阶段训练与对齐逻辑

文件夹：`AICoding/skills/two-stage-training/`

适合回答：

- stage 1 和 stage 2 分别训练什么
- LR `5x5` latent 为什么重要
- HR-LR 对齐损失是怎么接进训练流程的
- 修改训练阶段逻辑时要同步检查什么

---

### 2.3 判别器损失接口兼容性

文件夹：`AICoding/skills/discriminator-loss-compatibility/`

适合回答：

- 为什么 `utils/loss.py` 要支持两种调用风格
- `trainer.py` 和 `trainer_two_stage.py` 的判别器 loss 调用差异是什么
- 修改 GAN loss 时为什么不能只修一边

---

### 2.4 训练运维与产物

文件夹：`AICoding/skills/training-operations/`

适合回答：

- 常用训练参数在哪里看
- `lr_vae_opt`、`vae_opt`、`disc_opt` 各管什么
- 日志、TensorBoard、checkpoint 输出到哪里
- 改训练脚本或恢复训练时该先检查什么

---

## 3. 维护规则

- 这里保留索引、入口和最小摘要，不再堆叠实现细节。
- 新增专题时，优先保持“一个 skill 解决一类问题”。
- 如果某一节已经长到需要滚屏搜索，就应考虑拆出去。
- 与 Agent memory 使用规范相关的内容，参见：`AICoding/skills/memory-and-skill-practice/SKILL.md`

---

## 4. 命名约定

- `repo-navigation`：导航型 skill，只负责分流，不负责展开内容。
- `project-overview`：项目整体介绍，负责回答“这个项目是什么”。
- 其他专题 skill：分别承载训练、兼容性、运维等局部主题。