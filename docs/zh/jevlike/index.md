# JevLike 决策模型研究

给定一条客服消息，模型可以选择负责的客服组，也可以估计这条消息是否紧急。这里用 JevLike 指一类根据输入材料、问题和给定答案空间做出判断的模型。

## 任务与答案空间

从[决策模型引言](../tensor-analysis/introduction.md)了解一条输入如何对应多个具名问题，再用[符号与维度](../tensor-analysis/notation.md)对照图中的问题数、候选数和序列长度。完整请求与模型实际返回的两个答案见[客服分流示例](../cookbook/decision.md)。

## 方法与开源实现

同一道单选题可以有不同的取分方式。下表中的三篇文章分别沿一种实现追踪到原始分数。

| 实现 | 读出从哪里取分 | 可核对的证据 |
| --- | --- | --- |
| [Tev1-4B](tev1.md) | 在完整提示的末尾，读取各个答案字母的词表分数 | 上游逐题输出与 MetisCraft 留档复现 |
| [JevForge](jevforge.md) | 每个候选单独成行，共享标量头读取各行末端 | 固定版本的编码、模型与发布训练脚本 |
| [Kev-4B](kev.md) | 将同一题的决策向量与各候选结束位置的向量配对 | 固定版本的指针头、推理路径与训练配置 |

[模型架构图谱](../tensor-analysis/architectures.md)从输入序列追踪到候选分数。按感兴趣的计算步骤，可以直接读以下实现。

- [答案 token 与固定槽位](../tensor-analysis/architectures-token.md)比较 jebadiah、d3 等实现从哪些位置取分。
- [标量、指针与联合 schema](../tensor-analysis/architectures-structured.md)对照 JevForge、Kev、Clef 和 JevAny 如何给候选或字段打分。
- [编码器分类与逐对重排](../tensor-analysis/architectures-encoders.md)比较 GLiClass 等模型如何编码文本、标签或候选对。

## 机制比较与复现证据

用两道候选数不同的选择题手算概率，并检查批次填充和多标签输出。[候选打分与概率](candidate-scores.md)

只把第二个选项里的 `green` 换成 `red`，第一个选项的分数会变吗？[切换三种读出方式](../tensor-analysis/readout-comparison.md#candidate-edit)，看模型读到哪个位置时已经给第一项打了分。

核对某个模型时，[支持矩阵](../../support-matrix.md)列出查过的版本、工具支持范围和已做的验证。[Tev1 复现记录](../../reproductions/tev1.md)另列相同评测题上的结果和这次实验的限制。
