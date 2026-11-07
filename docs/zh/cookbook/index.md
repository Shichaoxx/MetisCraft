# Cookbook

从一道业务问题开始，做出能评测、能接入应用的判断模型。这里先看真实记录的效果，再跑通本地 CPU 流程，最后接入自己的标注数据。

<div class="cookbook-guide" markdown="1">

<p><img src="../../visuals/figures/avatar.png" alt="Metis，Cookbook 阅读向导" width="96"></p>

<div markdown="1">

## Metis · 选一条起步路线

- **先看能做什么**：[认识一次判断](decision.md)。
- **先把流程跑起来**：[在 CPU 上训练第一个模型](first-model.md)。
- **接入自己的标签**：[把自己的任务接进来](own-task.md)。
- **进一步理解与调优**：[原理](../../cookbook/principles.md)与[最佳实践](../../cookbook/best-practices.md)。

</div>
</div>

## 从你想得到的结果开始

| 你的目标 | 阅读路线 | 读完或完成后得到什么 |
|---|---|---|
| 理解输入、输出和业务用途 | [认识一次判断](decision.md) | 一份完整客服请求、真实模型响应，以及理解评测证据的方法 |
| 检查安装和完整软件流程 | [在 CPU 上训练第一个模型](first-model.md) | 随机小模型的本地产物、评测报告和 HTTP 响应 |
| 为已有标注的任务训练模型 | [把自己的任务接进来](own-task.md) | 一个有版本的数据集、连接预训练基座的配方，以及评测结果的步骤 |

客服示例来自已训练模型和有记录的评测；CPU 教程使用构造数据与随机权重，验证软件是否正常。前者帮助你理解工具包能产出怎样的结果，后者帮助你掌握操作流程。

## 从一道题到一个应用

先定义答案空间，收集带标签的样本，并分开训练、校准和评测数据。配方会产出模型文件，供 Python、批量预测或 HTTP 服务加载。应用再根据判断结果决定下一步动作。

例如，客服系统可以把消息交给安全团队，并标记需要优先处理的工单。[真实记录示例](decision.md)展示了这两道题的输出，以及模型相对“不读消息”的先验基线有多少提升。

## 需要时查阅

- [安装](../installation.md)与[支持范围](../limitations.md)。
- [Record JSONL 与数据清单](../data.md)。
- [基座准备与配方设置](../training.md)。
- [预测、HTTP 与任务映射](../serving.md)。
- [配方字段参考](../../cookbook/recipe-reference.md)与[读出图解](../../readouts/README.md)。

[English](../../en/cookbook/index.md)
