# Understand a decision

A support team needs two answers: **which queue should receive this message, and is it urgent?** MetisCraft lets you express both as typed questions and train a model to answer them from the same message.

**Goal:** read a complete request and response, then distinguish a convincing example from evidence of model quality. You do not need to install or run anything for this guide.

## 1. Define the question and its answers

Download the complete [request JSON](../../examples/support-request.json) and [recorded response JSON](../../examples/support-response.json).

The `state` is the material to judge. Each named question defines its type and instructions. A `choice` question supplies an option ID and description for every possible answer; `noul` asks for the probability of yes.

This is the complete request recorded with the customer-routing example. The message was **written for the demonstration**. The six queue definitions are kept intact so that the output can be interpreted against the original answer space.

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

The IDs `queue` and `urgent` let the caller match each answer to its question. Keep the option IDs and their meanings consistent with the data used to train the model.

## 2. Read the recorded response

The following response was produced by the trained Banking77-derived artifact from run `20261105-101057-30757bf7`. It is a recorded model output, not invented probabilities and not the output of the CPU tutorial.

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

| Field | How to read it |
|---|---|
| `answers.queue.choice` | The selected queue is `security`. |
| `answers.queue.probabilities` | A distribution over the same six IDs in the request. |
| `answers.queue.confidence` | The probability assigned to the selected queue. |
| `answers.urgent.noul` | The model's probability of answering yes to the urgency question. |
| `usage.output_tokens` | This recorded response reports no generated answer tokens; the structured result is built from model scores. |

The high probability describes this prediction. It does not by itself establish that the system is reliable. An application can use the two answers for routing and review, but a real escalation rule needs evaluation on the cases that application receives.

## 3. Check the result against a baseline

The same recorded run was evaluated on **3,080 held-out messages**, with one queue label and one urgency label per message. Labels were derived from Banking77 intents using a fixed mapping.

| Question | Model accuracy | Prior baseline accuracy | NLL before → after temperature fitting |
|---|---:|---:|---:|
| Queue | 97.8% | 24.7% | 0.102 → 0.087 |
| Urgency | 98.6% | 87.0% | 0.065 → 0.049 |

The prior baseline uses training-label frequencies and does not read the message. Its 87.0% urgency accuracy shows why model accuracy needs context. Temperature fitting used a separate calibration split; the NLL values above were measured on test data.

These results demonstrate a labeled public-data scenario, not a production deployment. Full criteria, recipe settings, metrics and run provenance are in the [customer-routing evidence](../../../recipes/banking77_support/README_en.md).

## 4. Choose the type your application needs

| Question you want to ask | Type | Result | HTTP support |
|---|---|---|---|
| Is this urgent? | `noul` | Probability of yes | Yes |
| Which queue should handle it? | `choice` | One choice and option probabilities | Yes |
| How strong is the sentiment, on ordered levels? | `score` | Probabilities over ordered levels | Yes |
| Which products are mentioned? | `multi_label` | Independent candidate probabilities | Use Record prediction |

## If an answer is hard to interpret

- **The option IDs do not match your UI:** check the request's full answer space and the labels used during training. Treat IDs as stable keys.
- **Confidence is high, but you do not know whether to trust it:** inspect held-out errors and calibration metrics; one response is not an evaluation.
- **Several tasks share one HTTP request:** map question names to the training task IDs as described in [task routing](../serving.md#route-tasks-consistently).

## Next step

Run the [CPU workflow](first-model.md) to learn the commands, or [bring your own task](own-task.md) if you already have labeled data and a prepared base model.

[中文](../../zh/cookbook/decision.md)
