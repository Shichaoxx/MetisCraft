---
hide:
  - toc
---

<div class="home-hero" markdown="1">
<div class="home-intro" id="jevlike-模型工具包" markdown="1">

# *Metis*

同一条客服消息，模型要判断该交给哪个团队，也要判断是否需要紧急处理。从这条消息出发，可以读模型原理、查工具用法，也可以亲手跑一次实验。

<nav class="home-actions" aria-label="主要栏目">
  <a href="jevlike/index.md"><strong lang="en">JevLike</strong><span>模型研究</span></a>
  <a href="metiscraft/index.md"><strong lang="en">MetisCraft</strong><span>开发手册</span></a>
  <a href="cookbook/index.md"><strong lang="en">Cookbook</strong><span>实验与复现</span></a>
</nav>

[0.1.0 正式版](https://github.com/Shichaoxx/MetisCraft/releases/tag/v0.1.0) · [安装说明](installation.md)

</div>
<div class="home-specimen" markdown="1">

## 从一条真实响应开始

<blockquote lang="en">My card was stolen this morning and there are two payments on it that I did not make.</blockquote>

<svg class="choice-readout" viewBox="0 0 335 156" role="img" aria-label="模型从六个客服队列中选出了安全团队 security。"><use href="../visuals/figures/choice-readout.svg#readout"></use></svg>

<dl class="specimen-decisions">
  <dt>交给谁处理？</dt><dd>安全团队</dd>
  <dt>是否紧急？</dt><dd>模型判为紧急</dd>
</dl>

<div class="specimen-caption" markdown="1">

消息为演示编写。两个判断均来自客服分流模型对这条消息的实测响应。[查看请求与响应](cookbook/decision.md)

</div>

<details class="specimen-probabilities"><summary>查看模型给出的概率</summary>
<dl><dt>queue.confidence</dt><dd>0.9999995054983101</dd><dt>urgent.noul</dt><dd>0.9999314182034214</dd></dl>
</details>

</div>
</div>

## Tev1 4B 复现结果

[同做 1,000 道题，答对的题数只差 7 道，为什么有 54 道题选了不同答案？](jevlike/tev1.md#相差七道正确题哪些选择变了)

<div class="home-results" markdown="1">
<div markdown="1">

### 主决策集

**87.3%**

1,000 题中答对 873 题 · 原版 Tev1 **88.0%**

</div>
<div markdown="1">

### 政策迁移

**99.33%**

300 题中答对 298 题 · 原版 Tev1 **100%**

</div>
</div>

两项预先设定的准确率指标均达标。复现实验采用单个随机种子，在同一批公开开发题上评测。我们未取得原始训练记录，校准可能使负对数似然变差。

[复现记录与限制](../reproductions/tev1.md) · [训练配方](../../recipes/tev1/README.md)

## Cookbook 教程 { #从这里开始 }

<div class="home-paths" markdown="1">

- [认识一次判断](cookbook/decision.md) · 读懂六个客服队列的预测概率和独立的紧急程度判断。
- [跑通第一个模型](cookbook/first-model.md) · 在 CPU 上完成示例，无需下载权重。
- [接入自己的任务](cookbook/own-task.md) · 替换数据和基座模型，沿用同一条训练与部署路径。

</div>

## MetisCraft 支持的任务 { #判断有类型过程有依据 }

Python 和批量预测支持是非、选择、等级和多标签判断。HTTP `/v1/systemone` 支持前三种。

[配方与真实结果](limitations.md) · [数据格式](data.md) · [训练与校准](training.md) · [预测与服务](serving.md)

<picture>
  <source media="(max-width: 600px)" srcset="../visuals/figures/pipeline-mobile-zh.svg">
  <img src="../visuals/figures/pipeline-zh.svg" alt="数据、训练、校准、评测与服务">
</picture>

## JevLike 的模型原理 { #理解计算机制 }

客服模型怎样从六个队列中选出一个？[任务与答案空间](tensor-analysis/introduction.md)给出训练样本与实测响应。[候选打分与概率](jevlike/candidate-scores.md)用两道小题解释单选和多标签的差别。[模型架构图谱](tensor-analysis/architectures.md)按实现对照取分位置与计算过程。
