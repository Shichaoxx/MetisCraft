# 银行客服分流（Banking77 衍生的垂直场景）

垂直场景示例：把 Banking77 的 77 个意图按一张映射表改写成两道业务题，演示怎样用一份配方把自己的业务判断做成 Jev。同一条消息的两道题一起训练、一起回答。

| 项目 | 内容 |
|---|---|
| 注册名 | `banking77_support` |
| 上游 | [`PolyAI/banking77`](https://huggingface.co/datasets/PolyAI/banking77) @ `18469841` |
| 许可证 | CC-BY-4.0；条款：需署名；用它训练的产物：按上游许可证 |
| 判断类型 | 选择 choice、是非 noul |
| 划分（按题计，每条消息两题） | train 17,934（上游 train）；calib 1,054（从上游 train 按哈希切出 5%）；dev 1,002（从上游 train 按哈希切出 5%）；test 6,160（上游 test） |

## 场景定义

- `queue`（choice）：Which support queue should handle this customer message?
  - `cards`：22 个原始意图，如 `activate_my_card`、`card_about_to_expire`
  - `payments`：19 个原始意图，如 `card_payment_fee_charged`、`card_payment_wrong_exchange_rate`
  - `transfers`：11 个原始意图，如 `balance_not_updated_after_bank_transfer`、`beneficiary_not_allowed`
  - `top_up`：11 个原始意图，如 `automatic_top_up`、`balance_not_updated_after_cheque_or_cash_deposit`
  - `account`：8 个原始意图，如 `age_limit`、`edit_personal_details`
  - `security`：6 个原始意图，如 `compromised_card`、`lost_or_stolen_card`
- `urgent`（noul）：Could the customer lose money or access to their account if this message waits until tomorrow? 有 10 个原始意图答“是”，如 `compromised_card`、`lost_or_stolen_card`

映射表在 [`registry.yaml`](../../src/metiscraft/data/registry.yaml) 的 `banking77_support` 条目里。每个原始意图必须恰好落到一个答案上，漏写或重复都会在 `metiscraft data prepare` 时报错。接入自己的任务使用[本地 manifest](../../docs/zh/cookbook/own-task.md)，无需修改注册表。

## 部署

```bash
metiscraft serve <run-directory>/artifact
```

请求（`POST /v1/systemone`）：

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

验收产物 `20261105-101057-30757bf7` 的实际响应：

```json
{
  "model": "metiscraft/banking77_support",
  "answers": {
    "queue": {
      "type": "choice",
      "choice": "security",
      "confidence": 0.9999995054983101,
      "probabilities": {
        "cards": 2.0465875752668956e-07,
        "payments": 2.8552732808090524e-07,
        "transfers": 1.3682393373337923e-09,
        "top_up": 1.1094860412756372e-09,
        "account": 1.8378789176297659e-09,
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

## 配方

| 配方 | 读出 | 微调 | 有效批量 | 轮数 | 学习率（LoRA / 头） |
|---|---|---|---|---|---|
| [`qwen3-0.6b-base_candidate-scalar.yaml`](qwen3-0.6b-base_candidate-scalar.yaml) | ③ 逐候选标量头（list 布局） | LoRA r=16 | 16 | 2 | 2e-4 / 1e-3 |

## 运行

```bash
metiscraft data prepare banking77_support
metiscraft train recipes/banking77_support/qwen3-0.6b-base_candidate-scalar.yaml
```

## 验收结果

### ③ 逐候选标量头：`qwen3-0.6b-base_candidate-scalar.yaml`

运行 `20261105-101057-30757bf7`；提交 `85fd8ac`（工作区有未跟踪文件，已跟踪的代码与提交一致）；数据 `rev-18469841`；1× RTX 3090；2,242 步，训练 43 分钟；温度：choice 1.50，noul 1.55。

**test**，是非 noul（n=3,080）

| 方法 | 准确率 | AUC | NLL | Brier | ECE |
|---|---|---|---|---|---|
| 模型 | 0.986 | 0.995 | 0.065 | 0.012 | 0.012 |
| 模型 + 温度 | 0.986 | 0.995 | 0.049 | 0.011 | 0.008 |
| 先验 | 0.870 | 0.500 | 0.386 | 0.113 | 0.002 |

**test**，选择 choice（n=3,080）

| 方法 | 准确率 | macro-F1 | NLL | Brier | ECE |
|---|---|---|---|---|---|
| 模型 | 0.978 | 0.975 | 0.102 | 0.037 | 0.010 |
| 模型 + 温度 | 0.978 | 0.975 | 0.087 | 0.037 | 0.006 |
| 先验 | 0.247 | 0.066 | 1.704 | 0.805 | 0.047 |

**dev**，是非 noul（n=501）

| 方法 | 准确率 | AUC | NLL | Brier | ECE |
|---|---|---|---|---|---|
| 模型 | 0.996 | 0.998 | 0.039 | 0.004 | 0.006 |
| 模型 + 温度 | 0.996 | 0.998 | 0.031 | 0.005 | 0.006 |
| 先验 | 0.886 | 0.500 | 0.356 | 0.101 | 0.018 |

**dev**，选择 choice（n=501）

| 方法 | 准确率 | macro-F1 | NLL | Brier | ECE |
|---|---|---|---|---|---|
| 模型 | 0.984 | 0.984 | 0.066 | 0.028 | 0.013 |
| 模型 + 温度 | 0.984 | 0.984 | 0.063 | 0.028 | 0.010 |
| 先验 | 0.287 | 0.074 | 1.690 | 0.801 | 0.006 |

ECE 用 15 个等宽分箱：choice 和 score 取 top-1 概率，noul 和 multi_label 的每个候选取正类概率。“模型 + 温度”只用 calib 划分拟合温度；“先验”按训练集的类别频率给概率，是不看输入的基线。

## 参照

下面的数字来自公开报告，评测子集和协议与这里不同，只用来定位量级。

- 这是自定义场景，没有公开参照；可以和同一数据上的 [Banking77](../banking77/README.md) 意图识别对照。
