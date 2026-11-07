# 把业务问题，变成模型的判断。

<div class="home-lead" markdown="1">

用自己的数据训练类型化判断模型。一份配方，连接训练、校准、评测和部署。

[开始使用](cookbook/first-model.md){ .md-button .md-button--primary }
[浏览 Cookbook](cookbook/index.md){ .md-button }

</div>

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
