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

映射表在 [`registry.yaml`](../../src/metiscraft/data/registry.yaml) 的 `banking77_support` 条目里。每个原始意图必须恰好落到一个答案上，漏写或重复都会在 `metiscraft data prepare` 时报错。换成自己的业务，只要改这张表和题目说明。

## 部署

```bash
metiscraft serve <运行目录>/artifact
```

请求格式与官方 `/v1/systemone` 一致；示例响应待验收产物生成后补上。

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

待验收。

ECE 用 15 个等宽分箱：choice 和 score 取 top-1 概率，noul 和 multi_label 的每个候选取正类概率。“模型 + 温度”只用 calib 划分拟合温度；“先验”按训练集的类别频率给概率，是不看输入的基线。

## 参照

下面的数字来自公开报告，评测子集和协议与这里不同，只用来定位量级。

- 这是自定义场景，没有公开参照；可以和同一数据上的 [Banking77](../banking77/README.md) 意图识别对照。
