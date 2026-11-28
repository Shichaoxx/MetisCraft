# 候选打分与概率

同一道题的两个选项各有一个分数。要把它们读作概率，需要先知道分数属于哪道题，以及这些选项能否同时成立。本页先对照两种读出，再用两道候选数不同的题检查实现。

## 每个候选的分数从哪里来

**MetisCraft scalar-list。** 对选择题，MetisCraft 构建提示时记录每段候选文本的最后一个 token 位置，即使提示中的候选重排，也把位置还原到原候选顺序。模型取出这些位置的隐藏向量，用同一个小型神经网络（MLP）逐个生成标量。候选各有一个读取位置，打分头共享的是参数。[候选位置](https://github.com/Shichaoxx/MetisCraft/blob/e9a100bc6cb781270c9935a225e087f1f9167d7a/src/metiscraft/modeling/prompts.py#L220-L235) · [读取隐藏向量](https://github.com/Shichaoxx/MetisCraft/blob/e9a100bc6cb781270c9935a225e087f1f9167d7a/src/metiscraft/modeling/model.py#L79-L81) · [标量头](https://github.com/Shichaoxx/MetisCraft/blob/e9a100bc6cb781270c9935a225e087f1f9167d7a/src/metiscraft/modeling/model.py#L14-L29)

**Kev pointer。** 每个候选以 `</opt>` 结尾，每道题另有一个 `<decide>`。读出把后者的隐藏向量投影为该题的查询向量 \(q\)，把各个候选结束位置投影为键向量 \(k_i\)。温度为 1 时，候选分数为 \(q^\top k_i / \sqrt{d_p}\)，其中 \(d_p\) 为投影维度。同一题的候选共享一个查询向量。合并多道题计算时，源码用 `owner` 为每个候选选取所属题目的查询向量。[候选和决策标记](https://github.com/jaredpalmer/kev/blob/2074cc44256200bb37540103fa4572901faa10f3/kev/model.py#L107-L126) · [指针头与批量打分](https://github.com/jaredpalmer/kev/blob/2074cc44256200bb37540103fa4572901faa10f3/kev/model.py#L205-L223)

## 两道题的手算

取温度为 1。表中的 logit 是为演算指定的分数。第一题只有 A、B 两个选项，第三列用于补齐批次，故意填入一个很大的数。第二题有三个有效选项。

| 题目 | 三列分数 | 有效候选 | 单选概率 |
| --- | --- | --- | --- |
| 第一题 | ln 3，0，90 | A、B | 0.75，0.25 |
| 第二题 | 0，ln 2，0 | A、B、C | 0.25，0.50，0.25 |

单选题的 softmax 把**同一题所有有效候选**的指数加起来作分母。第一题的权重为 3 和 1，A 的概率为 3/4。第二题用自己的分母 1 + 2 + 1。两行各自加总为 1。

若将第一行的两个有效分数用于允许标签同时成立的 `multi_label` 任务，逐项 sigmoid 得到 3/(1+3) = 0.75 和 1/(1+1) = 0.50。此时总和为 1.25，每一项都回答相应标签是否成立。这里借用了同一组教学分数解释任务语义；Kev 上述选择题路径实际使用逐题 softmax。[MetisCraft 概率转换](https://github.com/Shichaoxx/MetisCraft/blob/e9a100bc6cb781270c9935a225e087f1f9167d7a/src/metiscraft/judgments/__init__.py#L112-L118) · [Kev 分组与归一化](https://github.com/jaredpalmer/kev/blob/2074cc44256200bb37540103fa4572901faa10f3/kev/model.py#L494-L505)

<figure class="score-experiment" hidden>
  <figcaption>只改变 B 的分数，观察 A 的概率</figcaption>
  <p>固定 A 的 logit 为 ln 3，改变 B 的 logit。两个读出使用相同的输入分数。</p>
  <label for="score-b">B 的 logit <output for="score-b">0.0</output></label>
  <input id="score-b" type="range" min="-3" max="3" step="0.1" value="0">
  <dl>
    <div><dt>单选 · P(A)</dt><dd><span class="score-track" aria-hidden="true"><i></i></span><span data-probability="choice">75.0%</span></dd></div>
    <div><dt>多标签 · P(A)</dt><dd><span class="score-track" aria-hidden="true"><i></i></span><span>75.0%</span></dd></div>
  </dl>
</figure>
<script src="/MetisCraft/assets/candidate-scores.js" defer></script>

## 用 Python 核对

下面的例子只用 Python 标准库，不加载模型权重。计算指数前减去本题最大有效分数，可以避免数值溢出，同时保持各候选的相对概率。

```python
from math import exp, log

scores = [[log(3), 0, 90], [0, log(2), 0]]
valid = [[True, True, False], [True, True, True]]

for row, mask in zip(scores, valid):
    logits = [score for score, keep in zip(row, mask) if keep]
    largest = max(logits)
    weights = [exp(score - largest) for score in logits]
    total = sum(weights)
    print([round(weight / total, 4) for weight in weights])

print([round(1 / (1 + exp(-score)), 4) for score in scores[0][:2]])
```

三行依次输出 `[0.75, 0.25]`、`[0.25, 0.5, 0.25]` 和 `[0.75, 0.5]`。

1. **核对候选 ID 与有效列。** 第一题应始终只返回 A、B。把补齐分数 90 改成 −90，第一题仍应得到 0.75 和 0.25。MetisCraft 用每题目标数构造候选掩码，训练时先屏蔽无效列，推理时截取有效分数。这个掩码作用于输出候选轴，输入 token 的注意力掩码管的是另一条轴。[候选掩码](https://github.com/Shichaoxx/MetisCraft/blob/e9a100bc6cb781270c9935a225e087f1f9167d7a/src/metiscraft/train/collate.py#L76-L87) · [训练损失](https://github.com/Shichaoxx/MetisCraft/blob/e9a100bc6cb781270c9935a225e087f1f9167d7a/src/metiscraft/judgments/__init__.py#L83-L97) · [推理有效区间](https://github.com/Shichaoxx/MetisCraft/blob/e9a100bc6cb781270c9935a225e087f1f9167d7a/src/metiscraft/evaluate/infer.py#L41-L50)
2. **核对分组。** 保持两行分数固定，单独计算第一题或与第二题一起计算，第一题的结果应相同。若 softmax 后整批概率加总为 1，却有某一题的有效概率之和小于 1，检查分母是否混入了其他题的候选。
3. **核对任务。** 只有题目要求从候选中选一个时，才用该题的总和为 1 检查单选概率。对独立标签逐项核对 sigmoid；保持 A 的分数固定，仅改变 B 的分数，A 的概率应保持不变。

本例验证的是给定分数后的计算。两种读出的模型权重均未用于本例，教学数字不能用来比较预测性能。
