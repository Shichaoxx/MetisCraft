# Customer routing from Banking77

A teaching scenario maps Banking77's 77 intents to two business questions. One recipe trains and answers both questions for each customer message.

| Item | Details |
|---|---|
| Registry name | `banking77_support` |
| Upstream | [`PolyAI/banking77`](https://huggingface.co/datasets/PolyAI/banking77) @ `18469841` |
| License | CC-BY-4.0; attribution required; trained artifacts retain applicable upstream terms |
| Judgment type | Choice、Yes/no noul |
| Splits (questions; two per message) | train 17,934 (upstream train); calib 1,054 (5% hash split from upstream train); dev 1,002 (5% hash split from upstream train); test 6,160 (upstream test) |

## Scenario definition

- `queue` (choice): Which support queue should handle this customer message?
  - `cards`: 22 source intents, including `activate_my_card`, `card_about_to_expire`
  - `payments`: 19 source intents, including `card_payment_fee_charged`, `card_payment_wrong_exchange_rate`
  - `transfers`: 11 source intents, including `balance_not_updated_after_bank_transfer`, `beneficiary_not_allowed`
  - `top_up`: 11 source intents, including `automatic_top_up`, `balance_not_updated_after_cheque_or_cash_deposit`
  - `account`: 8 source intents, including `age_limit`, `edit_personal_details`
  - `security`: 6 source intents, including `compromised_card`, `lost_or_stolen_card`
- `urgent` (noul): Could the customer lose money or access to their account if this message waits until tomorrow? Ten source intents map to yes, including `compromised_card`, `lost_or_stolen_card`.

The `banking77_support` entry in [registry.yaml](../../src/metiscraft/data/registry.yaml) contains the built-in mapping. Each intent must map to exactly one answer; missing or duplicate mappings fail data preparation. For your own tasks, use a [local manifest](../../docs/en/cookbook/own-task.md) without modifying the registry.

## Serving

```bash
metiscraft serve <run-directory>/artifact
```

Request (`POST /v1/systemone`):

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

Actual response from artifact `20261105-101057-30757bf7`:

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

## Recipe

| Recipe | Readout | Tuning | Effective batch | Epochs | Learning rate (LoRA / head) |
|---|---|---|---|---|---|
| [`qwen3-0.6b-base_candidate-scalar.yaml`](qwen3-0.6b-base_candidate-scalar.yaml) | ③ Scalar head (list layout) | LoRA r=16 | 16 | 2 | 2e-4 / 1e-3 |

## Run

```bash
metiscraft data prepare banking77_support
metiscraft train recipes/banking77_support/qwen3-0.6b-base_candidate-scalar.yaml
```

## Recorded results

### ③ Scalar head: `qwen3-0.6b-base_candidate-scalar.yaml`

Run `20261105-101057-30757bf7`; commit `85fd8ac` (untracked files present; tracked code matched the commit); data `rev-18469841`; 1× RTX 3090; 2,242 steps, 43 minutes; temperatures: choice 1.50, noul 1.55.

**test**, Yes/no noul (n=3,080)

| Method | Accuracy | AUC | NLL | Brier | ECE |
|---|---|---|---|---|---|
| Model | 0.986 | 0.995 | 0.065 | 0.012 | 0.012 |
| Model + temperature | 0.986 | 0.995 | 0.049 | 0.011 | 0.008 |
| Prior | 0.870 | 0.500 | 0.386 | 0.113 | 0.002 |

**test**, Choice (n=3,080)

| Method | Accuracy | macro-F1 | NLL | Brier | ECE |
|---|---|---|---|---|---|
| Model | 0.978 | 0.975 | 0.102 | 0.037 | 0.010 |
| Model + temperature | 0.978 | 0.975 | 0.087 | 0.037 | 0.006 |
| Prior | 0.247 | 0.066 | 1.704 | 0.805 | 0.047 |

**dev**, Yes/no noul (n=501)

| Method | Accuracy | AUC | NLL | Brier | ECE |
|---|---|---|---|---|---|
| Model | 0.996 | 0.998 | 0.039 | 0.004 | 0.006 |
| Model + temperature | 0.996 | 0.998 | 0.031 | 0.005 | 0.006 |
| Prior | 0.886 | 0.500 | 0.356 | 0.101 | 0.018 |

**dev**, Choice (n=501)

| Method | Accuracy | macro-F1 | NLL | Brier | ECE |
|---|---|---|---|---|---|
| Model | 0.984 | 0.984 | 0.066 | 0.028 | 0.013 |
| Model + temperature | 0.984 | 0.984 | 0.063 | 0.028 | 0.010 |
| Prior | 0.287 | 0.074 | 1.690 | 0.801 | 0.006 |

ECE uses 15 equal-width bins: top-1 probability for choice/score and affirmative probability for each noul/multi_label candidate. Temperatures are fitted only on calib. The prior baseline uses training-set label frequencies without reading the input.

## External comparisons

External numbers use different subsets and protocols; they provide context rather than a matched comparison.

- This constructed scenario has no external matched baseline; compare the separate [Banking77](../banking77/README_en.md) intent task with its different labels.
