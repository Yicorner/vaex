---
name: memory-and-skill-practice
description: Define how this repository stores knowledge across memory scopes and SKILL.md files, including when to write facts versus reusable guidance.
---

# Memory And Skill Practice

> 面向在本仓库中工作的 Agent。目标是把“临时上下文”“仓库事实”“长期规范”分层管理，避免知识漂移、重复记录和错误继承。

---

## 1. 知识存放位置

Agent 可使用三类 memory，以及仓库内的 skill 文档。

### 1.1 User Memory

位置：`/memories/`

说明：这是 Agent memory 工具暴露的逻辑路径，不是当前仓库工作区里的实际目录，因此本地执行 `ls` 时通常看不到同名文件夹。

用途：记录跨仓库、跨会话长期有效的用户偏好与工作习惯。

适合记录：

- 用户偏好的输出格式
- 用户常用命令或环境习惯
- 长期稳定的协作约束

不适合记录：

- 当前仓库的实现细节
- 某一轮任务的临时结论

### 1.2 Session Memory

位置：`/memories/session/`

说明：同样属于工具侧 memory 命名空间，不对应当前仓库中的物理目录。

用途：记录当前会话内的计划、阶段结论、待确认事项。

适合记录：

- 多步骤任务的阶段状态
- 尚未落盘但后续会继续使用的中间判断
- 需要在同一会话中反复引用的临时上下文

不适合记录：

- 已经过时的执行日志
- 应该直接固化到仓库文档中的稳定规则

### 1.3 Repository Memory

位置：`/memories/repo/`

说明：这是 repository-scoped memory 的逻辑路径。它面向当前仓库生效，但不会以普通工作区文件夹的形式直接出现在项目根目录。

用途：记录当前仓库范围内、经过验证、但未必适合直接写入代码注释或主文档的仓库事实。

适合记录：

- 已验证的接口兼容性约束
- 构建、训练、调试中的关键前提
- 某些容易被后续修改破坏的隐含规则

当前已记录事实：

- `utils/loss.py` 中的判别器损失函数必须同时兼容：
  - `trainer.py` 的“signed logits”旧调用方式
  - `trainer_two_stage.py` 的 `is_real_pred` / `for_g` 显式调用方式

### 1.4 Repository Skill

源目录位置：`AICoding/skills/<skill-name>/SKILL.md`

对外入口：

- `.agents/skills/`
- `.cursor/skills/`

用途：沉淀“可复用的方法论、稳定约束、关键入口、修改策略”。

适合记录：

- 新 Agent 进入仓库后需要先知道的内容
- 某一类任务的最佳实践
- 多文件联动修改时必须遵守的规则

skill 和 repository memory 的区别：

- memory 更轻量，适合短事实、提醒、约束
- skill 更完整，适合形成规范、流程、决策准则

skill 文件格式约定：

- 每个 `SKILL.md` 顶部都应包含 YAML frontmatter
- 最少包含：`name`、`description`
- `name` 应与 skill 目录语义一致，使用稳定、可检索的英文短横线命名
- `description` 应直接说明“什么时候该读这份 skill”

推荐模板：

```md
---
name: example-skill
description: Explain when this skill should be used and what problem it solves.
---
```

---

## 2. 什么时候写 Memory，什么时候写 Skill

优先级判断如下：

### 2.1 写 Memory 的场景

- 事实刚被验证，但还不值得扩写成完整文档
- 需要简短提醒后续 Agent 避免回归
- 内容强依赖当前仓库或当前会话，不具备广泛复用性

### 2.2 写 Skill 的场景

- 已经能总结出稳定最佳实践
- 同类问题未来大概率还会反复出现
- 需要讲清“为什么这样做”以及“错误做法会造成什么后果”
- 需要给出执行顺序、入口文件、边界条件

### 2.3 同时写两者的场景

当一次任务既产出“短事实”，又能沉淀“长期方法论”时：

- 在 `repository memory` 中保留一句话事实
- 在 `skill` 中展开背景、约束和最佳实践

这是最佳做法，因为它同时兼顾检索效率和长期可维护性。

---

## 3. 本仓库的最佳实践

### 3.1 优先把稳定知识写入 Skill

以下类型优先写入 `AICoding/skills/<skill-name>/SKILL.md`：

- 训练流程分阶段逻辑
- 模型之间的职责边界
- 数据输入输出约定
- 修改某个关键模块时必须同步检查的文件清单

原因：这些内容不是“偶然发现”，而是仓库长期结构的一部分。

### 3.2 把容易回归的实现约束写入 Repository Memory

例如：

- 同一个损失函数被两个 trainer 以不同接口调用
- 某个参数改动会破坏 checkpoint 兼容性
- 某个训练脚本依赖特定目录布局

原因：这类知识通常短小，但极易被后来修改破坏。

### 3.3 不把一次性排障日志写进 Skill

以下内容不应进入 skill：

- 某次报错的完整 traceback
- 某台机器上的临时环境状态
- 一次性调试命令的流水账

这些内容应留在会话上下文，或在必要时精炼为结论后再写入 memory / skill。

### 3.4 Skill 中记录意图、边界和检查顺序

不要只写“改哪里”，还要写清：

- 为什么需要这样改
- 改动会影响哪些模块
- 修改后必须验证什么

这比堆砌过程描述更有复用价值。

---

## 4. 推荐维护流程

每完成一类深入任务后，按以下顺序处理知识沉淀：

1. 先判断这次结论是临时结论、仓库事实，还是可复用规范。
2. 临时结论留在 session memory，不做长期沉淀。
3. 短而稳定的仓库事实写入 repository memory。
4. 可复用的方法论、边界条件、检查清单写入 skill。
5. 如果 skill 已存在相近主题，优先增量更新，不新建重复文件。

---

## 5. 当前仓库示例

### 5.1 已验证约束

`utils/loss.py` 中的 `hinge_loss`、`softplus_loss`、`linear_loss` 不能只支持单一签名。

必须同时兼容：

- `trainer.py` 的旧式调用：直接传入已带符号的 `logits`
- `trainer_two_stage.py` 的新式调用：显式传入 `is_real_pred` 和可选 `for_g`

### 5.2 推荐写法原则

- 保持旧接口可用，避免破坏现有训练流程
- 对新接口做最小扩展，而不是复制一套并行损失实现
- 将这种“跨调用风格兼容性”记录为仓库级知识，而不是仅停留在一次修 bug 的上下文里

---

## 6. 维护要求

- 只记录已经验证的事实，不写猜测
- 尽量短句、强约束、可执行
- 与代码冲突时，以代码为准，并尽快更新文档
- 新增 skill 前先检查 `AICoding/skills/` 下是否已有相近主题目录
- 新增或重写 `SKILL.md` 时，先写 frontmatter，再写正文
- 若一次任务只产生一句稳定结论，优先写 memory；若能抽象成规范，再写 skill