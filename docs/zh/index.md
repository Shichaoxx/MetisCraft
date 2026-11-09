# JevLike 模型工具包

<div class="home-lead" markdown="1">

训练、复现与部署 JevLike 判断模型。一份配方，连接训练、校准、评测和服务。

[开始使用](cookbook/first-model.md){ .md-button .md-button--primary }
[浏览 Cookbook](cookbook/index.md){ .md-button }

</div>

## Tev1 4B 复现结果

<div class="reproduction-results" markdown="1">

<div markdown="1">

<span class="result-label">主决策集 · 1,000 题</span>

**87.3%**

873/1,000 答对 · 原版 Tev1 **88.0%**

</div>
<div markdown="1">

<span class="result-label">政策迁移 · 300 题</span>

**99.33%**

298/300 答对 · 原版 Tev1 **100%**

</div>
</div>

两项预定准确率验收均达标。单种子、同一批公开开发题；原始训练历史未恢复，校准可能恶化 NLL。

[复现记录与限制](../reproductions/tev1.md) · [训练配方](../../recipes/tev1/README.md)

## 从一条真实响应开始

> My card was stolen this morning and there are two payments on it that I did not make.

<div class="decision-result" markdown="1">

**security** · 转交安全团队

`confidence: 0.9999995054983101` · `urgent: 0.9999314182034214`

来自已记录的客服分流模型：人工构造消息、真实模型响应。概率不是准确率。

[查看完整案例](cookbook/decision.md) →

</div>

## 从这里开始

<div class="start-grid" markdown="1">

<div markdown="1">

### 01 · 看见效果

输入如何变成带概率的判断，结果依据在哪里。

[认识一次判断](cookbook/decision.md)

</div>
<div markdown="1">

### 02 · 跑通流程

无需 GPU 或下载权重，完成第一个 CPU 示例。

[跑通第一个模型](cookbook/first-model.md)

</div>
<div markdown="1">

### 03 · 接入任务

替换数据和基座，沿用同一条训练与部署路径。

[接入自己的任务](cookbook/own-task.md)

</div>
</div>

## 判断有类型，过程有依据

是非、选择、等级和多标签通过 Python／批量预测使用。HTTP `/v1/systemone` 支持前三种。

[配方与真实结果](limitations.md) · [数据格式](data.md) · [训练与校准](training.md) · [预测与服务](serving.md)

<picture>
  <source media="(max-width: 600px)" srcset="../visuals/figures/pipeline-mobile.svg">
  <img src="../visuals/figures/pipeline.svg" alt="Data, train, calibrate, evaluate, serve">
</picture>

## 理解计算机制

[原理与张量解析](tensor-analysis/index.md)从 System One 契约展开答案 token 和标量头计算；[五分钟技术导览](tensor-analysis/walkthrough.md)串联矢量图、数值例与源码依据。
