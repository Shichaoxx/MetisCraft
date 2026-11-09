# 认识一次判断

客服团队需要两个答案：**这条消息交给哪个组，是否需要优先处理？** 在 MetisCraft 中，你可以把它们写成两道有明确类型的题，训练一个模型根据同一条消息回答。

**目标：**读懂一份完整请求和响应，并区分“一个看起来合理的案例”与模型效果的评测证据。本篇无需安装或运行程序。

## 1. 定义问题和答案空间

可下载完整的[请求 JSON](../../examples/support-request.json)和[真实响应 JSON](../../examples/support-response.json)。

`state` 是需要判断的材料。每道具名问题包含类型与说明；选择题 `choice` 为每个答案提供 ID 和描述，是非题 `noul` 则询问回答“是”的概率。

下面是客服分流示例记录的完整请求。消息是**为演示人工构造的**；六个客服组的定义完整保留，便于对照原始答案空间理解响应。

```json
{
  "model": "banking77_support",
  "state": "My card was stolen this morning and there are two payments on it that I did not make.",
  "questions": {
    "queue": {
      "type": "choice",
      "instructions": "Which support queue should handle this customer message?",
      "criteria": {
        "cards": "Ordering, delivering, activating, linking or replacing cards; PINs, card limits and card support.",
        "payments": "Card payments, cash withdrawals, refunds, fees, exchange rates and currencies.",
        "transfers": "Sending and receiving money, transfer status, cancellations and beneficiaries.",
        "top_up": "Topping up the account and problems with top-ups or deposits.",
        "account": "Identity verification, personal details, passcodes, age limits and closing the account.",
        "security": "Lost, stolen or compromised cards or phones, and transactions the customer does not recognise."
      }
    },
    "urgent": {
      "type": "noul",
      "instructions": "Could the customer lose money or access to their account if this message waits until tomorrow?"
    }
  }
}
```

`queue` 和 `urgent` 用于把响应对应到各自问题。选项 ID 及其含义应与模型训练时的数据保持一致。

## 2. 读懂真实模型响应

以下响应来自 Banking77 衍生场景的已训练产物，运行编号为 `20261105-101057-30757bf7`。这些概率是实际记录的模型输出，既不是编造的示意数字，也不是 CPU 教程的结果。

```json
{
  "model": "metiscraft/banking77_support",
  "answers": {
    "queue": {
      "type": "choice",
      "choice": "security",
      "confidence": 0.9999995054983101,
      "probabilities": {
        "cards": 2.0465875752668956e-7,
        "payments": 2.8552732808090524e-7,
        "transfers": 1.3682393373337923e-9,
        "top_up": 1.1094860412756372e-9,
        "account": 1.8378789176297659e-9,
        "security": 0.9999995054983101
      }
    },
    "urgent": {
      "type": "noul",
      "noul": 0.9999314182034214
    }
  },
  "usage": {
    "input_tokens": 199,
    "output_tokens": 0
  }
}
```

| 字段 | 如何理解 |
|---|---|
| `answers.queue.choice` | 选中的客服组是 `security`。 |
| `answers.queue.probabilities` | 请求中六个选项 ID 对应的概率分布。 |
| `answers.queue.confidence` | 被选中客服组的概率。 |
| `answers.urgent.noul` | 模型对“是否紧急”回答“是”的概率。 |
| `usage.output_tokens` | 这份记录没有生成答案 token；结构化响应由模型分数构建。 |

高概率描述的是这一次预测，本身不能证明系统可靠。应用可以据此分流或安排复核，但实际的升级处理规则仍需在应用会遇到的数据上评测。

<aside class="metis-reading-note" data-metis-theme="light" data-metis-light-src="../../visuals/figures/metis-guide-chibi.png" data-metis-dark-src="../../visuals/figures/metis-guide-night.png" data-metis-light-alt="穿着学院装、抱着星图书的小墨" data-metis-dark-alt="穿着夜间衣装、抱着星图书的小墨" aria-label="小墨阅读提示" markdown="1">

<figure class="metis-reading-avatar"><a data-metis-full href="../../visuals/figures/metis-guide-chibi.png" target="_blank" rel="noopener" aria-label="在新标签页查看小墨插图"><img data-metis-portrait src="../../visuals/figures/metis-guide-chibi.png" alt="穿着学院装、抱着星图书的小墨" width="120" height="120"></a></figure>

<div class="metis-reading-copy" markdown="1">

<p class="metis-reading-title"><strong>小墨提示 · 先核对，再比较</strong></p>

先核对响应与请求中的选项 ID，再把下方留出评测结果与先验基线比较。单条消息上的高置信回答不能证明系统可靠。

</div>
</aside>

## 3. 先和基线比较，再看校准

同一次运行在 **3,080 条留出消息**上评测，每条消息各有一个客服组标签与紧急程度标签。这些标签由 Banking77 意图通过固定映射得到。

| 问题 | 模型准确率 | 先验基线准确率 | 温度校准前 → 后的 NLL |
|---|---:|---:|---:|
| 客服组 | 97.8% | 24.7% | 0.102 → 0.087 |
| 是否紧急 | 98.6% | 87.0% | 0.065 → 0.049 |

先验基线只使用训练标签的频率，不读取消息。紧急程度的先验准确率已有 87.0%，这也是不能只看模型准确率的原因。温度参数使用独立的校准集拟合，表中的 NLL 则在测试集上测量。

这些结果说明工具包在一个有标注的公开数据场景中跑通了有效模型，不代表生产部署效果。完整题目、配方、指标和运行来源见[客服分流证据](../../../recipes/banking77_support/README.md)。

## 4. 按业务需要选择类型

| 你想问的问题 | 类型 | 输出 | HTTP 支持 |
|---|---|---|---|
| 是否紧急？ | `noul` | 回答“是”的概率 | 支持 |
| 该交给哪个组？ | `choice` | 一个选项及各选项概率 | 支持 |
| 在有序等级中，情绪有多强？ | `score` | 各有序等级的概率 | 支持 |
| 消息中提到了哪些产品？ | `multi_label` | 每个候选独立的概率 | 使用 Record 预测 |

## 看不懂结果时，先检查什么

- **选项 ID 与界面不一致：**核对请求的完整答案空间与训练标签，将 ID 当作稳定的键使用。
- **概率很高，但不知道是否可信：**查看留出数据上的错误与校准指标；单条响应不等于评测。
- **一个 HTTP 请求包含多个任务：**按[任务映射说明](../serving.md)把问题名映射到训练时的任务 ID。

## 下一步

用 [CPU 流程](first-model.md)熟悉命令；如果已经有标注数据和准备好的基座，继续[接入自己的任务](own-task.md)。

[English](../../en/cookbook/decision.md)
