# Cookbook

从一条真实响应开始，逐步构建自己的判断模型。

<div class="cookbook-guide" markdown="1">

<figure class="cookbook-avatar"><img src="../../visuals/figures/metis-guide.png" alt="抱着星图书的 Metis，Cookbook 阅读向导" width="160" height="240"></figure>

<div class="cookbook-guide-copy" markdown="1">

## Metis · 阅读向导

先看一次判断，再跑通示例，最后接入你的数据。

</div>
</div>

## 选一条起步路线

<div class="start-grid cookbook-paths" markdown="1">

<div markdown="1">

### 01 · 认识一次判断

从一条客服消息到两个业务判断，查看完整请求、真实响应与评测依据。

[看真实案例](decision.md) →

</div>
<div markdown="1">

### 02 · 跑通第一个模型

无需 GPU 或下载权重。从随机小模型开始，走完训练、校准、评测和 HTTP 调用。

[开始 CPU 教程](first-model.md) →

</div>
<div markdown="1">

### 03 · 接入自己的任务

替换标注数据与真实基座，用同一份配方产出可加载、可评测的模型。

[接入业务数据](own-task.md) →

</div>
</div>

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
