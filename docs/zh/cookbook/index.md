# Cookbook

查看一条客服消息的模型输出，在 CPU 上运行自己的训练，或检查已发表的复现实验。运行训练教程前，请按[安装说明](../installation.md)准备环境。

## 实践教程 { #选一条起步路线 }

<div class="start-grid cookbook-paths" markdown="1">

<div markdown="1">

### 客服分流示例 { #01--认识一次判断 }

从一条客服消息到两个业务判断，查看完整请求、真实响应与评测依据。

[看真实案例](decision.md) →

</div>
<div markdown="1">

### CPU 训练 { #02--跑通第一个模型 }

无需 GPU 或下载权重。从随机小模型开始，走完训练、校准、评测和 HTTP 调用。

[开始 CPU 教程](first-model.md) →

</div>
<div markdown="1">

### 自定义任务 { #03--接入自己的任务 }

替换标注数据与真实基座，用同一份配方产出可加载、可评测的模型。

[接入业务数据](own-task.md) →

</div>
</div>

## 实验配方与结果 { #从一道题到一个应用 }

[Banking77 客服分流](../../../recipes/banking77_support/README.md)从 77 个原始意图构造队列和紧急程度两类标签，记录了数据划分、训练命令和留出集结果。[GoEmotions](../../../recipes/go_emotions/README.md)提供多标签任务的训练配方，同一段文本可对应多个情绪标签。

[Tev1 4B 复现](../../reproductions/tev1.md)给出两套评测题、与原作者结果的对照，以及训练和校准条件的差异。[配套训练配方](../../../recipes/tev1/README.md)列出重新运行的步骤。

<div class="cookbook-guide" data-metis-theme="light" data-metis-light-src="../../visuals/figures/metis-guide-chibi.webp" data-metis-light-full-src="../../visuals/figures/metis-guide-chibi.png" data-metis-dark-src="../../visuals/figures/metis-guide-night.webp" data-metis-dark-full-src="../../visuals/figures/metis-guide-night.png" data-metis-light-alt="穿着学院装、抱着星图书的小墨" data-metis-dark-alt="穿着夜间衣装、抱着星图书的小墨" markdown="1">

<figure class="cookbook-avatar"><a data-metis-full href="../../visuals/figures/metis-guide-chibi.png" target="_blank" rel="noopener" aria-label="在新标签页查看小墨插图"><img data-metis-portrait loading="lazy" decoding="async" src="../../visuals/figures/metis-guide-chibi.webp" alt="穿着学院装、抱着星图书的小墨" width="160" height="160"></a></figure>

<div class="cookbook-guide-copy" markdown="1">

## 小墨便签 · 实验记录 { #metis--阅读向导 }

训练结束后，打开运行目录里的 `report.md`。同一目录下的 `eval/metrics.json` 保存可供程序读取的指标。

</div>
</div>

## 手册与参考 { #需要时查阅 }

- [安装](../installation.md)与[支持范围](../limitations.md)。
- [运行环境、接口与产物兼容范围](../compatibility.md)。
- [Record JSONL 与数据清单](../data.md)。
- [基座准备与配方设置](../training.md)。
- [预测、HTTP 与任务映射](../serving.md)。
- [配方字段参考](../../cookbook/recipe-reference.md)与[原理与张量解析](../tensor-analysis/index.md)。

[English](../../en/cookbook/index.md)
