# Reproducing Tev1-4B

MetisCraft retrains a Tev1-style model using the public data builder and recommended recipe, checking data, prompts, per-question outputs and predeclared accuracy criteria. **The 4B run meets those accuracy criteria; it does not recover the original weights or hosted training job.**

The source is pinned to [togethercomputer/tev1@1dde778](https://github.com/togethercomputer/tev1/tree/1dde7782382c9f49d627153759b8d1deab426ce0), reviewed 2026-11-05. Original readout and per-question checks were completed 2026-11-07.

## Model architecture

**Tev1 fine-tunes Qwen3.5-4B for decisions, predicting answer letters with its existing LM head.** The [upstream description](https://github.com/togethercomputer/tev1/blob/1dde7782382c9f49d627153759b8d1deab426ce0/README.md) documents LoRA supervised fine-tuning; the [pinned model card](https://huggingface.co/togethercomputer/Tev1-4B-experimental/blob/0b7becf017daa0e5eb222f8ce7483c8c8259c52f/README.md) states that it retains the standard next-token output head. MetisCraft's 4B reproduction uses the same readout family.

| Component | Published architecture and this reproduction |
|---|---|
| Base | Upstream uses Qwen3.5-4B; this run pins `Qwen/Qwen3.5-4B@851bf6e8` and uses its text causal-language-model branch. |
| Hybrid backbone | 32 layers, hidden width 2,560; groups of 3 Gated DeltaNet linear-attention layers and 1 gated full-attention layer, totaling 24 + 8; each layer includes a dense FFN. |
| Output projection | The existing LM head, tied to token embeddings, has vocabulary width 248,320; the readout is `answer_token`. |
| Tuning and objective | `all-linear` LoRA: r=8, alpha=16, dropout=0; non-thinking `chat_json` prompts; teacher-forced `completion` training applies full-vocabulary cross-entropy only to the answer letter and EOS. |
| Inference protocol | MetisCraft reads candidate-letter logits and normalizes over candidates; upstream saved results use temperature-0 generation constrained by an option regex. |

The [upstream configuration](https://huggingface.co/togethercomputer/Tev1-4B-experimental/blob/0b7becf017daa0e5eb222f8ce7483c8c8259c52f/config.json) and [this run's base configuration](https://huggingface.co/Qwen/Qwen3.5-4B/blob/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a/config.json) are byte-identical; see the [pinned Qwen model description](https://huggingface.co/Qwen/Qwen3.5-4B/blob/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a/README.md) for the layer layout. Matching configurations do not establish identical trained weights or recovery of the historical hosted job. LoRA settings follow the [public recommendation](https://github.com/togethercomputer/tev1/blob/1dde7782382c9f49d627153759b8d1deab426ce0/docs/TRAINING.md); this run's [recipe](../../../recipes/tev1/qwen3.5-4b_chat-completion.yaml) does not pack samples. The public configuration includes vision modules, but this experiment and MetisCraft's interfaces validate text tasks only.

### Tensor objects and indices

The following describes our 4B inference implementation. \(B\) counts questions, \(T\) is the padded sequence length, \(K_b\) counts valid options for question \(b\), \(K_{\max}\) is the batch maximum, and \(V=248320\).

| Object | Type and shape | Meaning of one entry |
|---|---|---|
| `input_ids` \(X\) | int64, \([B,T]\) | A vocabulary ID in \(0\ldots V-1\), distinct from an option index. |
| `attention_mask` \(M\) | int64, \([B,T]\) | 1 marks a valid input token; 0 marks left padding. This is not a complete attention-probability matrix. |
| Final-position index \(t_b\) | Integer, conceptual \([B]\) index | \(\max\{t:M_{b,t}=1\}\); current left padding gives \(t_b=T-1\) for every row. |
| Hidden states \(H\), decision representation \(h\) | Floating point, \([B,T,2560]\), \([B,2560]\) | \(h_b=H_{b,t_b,:}\) is the final prompt position's contextual representation. |
| LM weights \(W_{\mathrm{lm}}\) | Floating point, \([V,2560]\) | Each row projects to one vocabulary token. |
| Vocabulary logits \(Z\) | float32, \([B,V]\) | \(Z=hW_{\mathrm{lm}}^{\mathsf T}\), before normalization. |
| `codes` \(\ell\) | int64, \([B,K_{\max}]\) | Vocabulary IDs for option letters; \(\ell_{b,k}\) maps a candidate column to a vocabulary column. |
| Candidate `mask` \(C\) | bool, \([B,K_{\max}]\) | True marks a valid candidate; its axes differ from the token-padding mask. |
| Candidate scores \(S\), probabilities \(p_b\) | Floating point, \([B,K_{\max}]\), per-question \([K_b]\) | Scores come from code columns; probabilities normalize only over the question's valid candidates. |

### Operator chain and probabilities

`State/question/all options → non-thinking chat template → left padding → hybrid causal backbone → final representation → full LM projection → gather letter columns → select valid candidates → softmax`

\[
h_b=H_{b,t_b,:},\qquad
Z=hW_{\mathrm{lm}}^{\mathsf T},\qquad
S_{b,k}=Z_{b,\ell_{b,k}},\qquad
p_{b,k}=\frac{\exp(S_{b,k}/\tau_b)}{\sum_{j=0}^{K_b-1}\exp(S_{b,j}/\tau_b)}.
\]

The matrix product contracts the hidden axis: \((B\times2560)(2560\times V)\rightarrow B\times V\). Current [`DecisionModel.forward`](../../../src/metiscraft/modeling/model.py#L62) computes the complete vocabulary logits before `gather`; `H[:, -1]` implements the final-valid-position selection above. The [collator](../../../src/metiscraft/train/collate.py#L21) derives `position_ids` from cumulative valid-mask counts, so left padding does not advance valid-token positions. Prediction [removes padded candidate columns by valid count](../../../src/metiscraft/evaluate/infer.py#L49) before [normalization](../../../src/metiscraft/judgments/__init__.py#L112). The candidate-space denominator includes only valid answer codes and generally differs from the letter's full-vocabulary softmax probability.

All candidates share one causal prompt: earlier options cannot read later options, while the final answer position can read the preceding prompt. Reordering options can change contextual representations and logits; the shared softmax denominator also couples candidate probabilities.

**Teaching example:** valid logits \([0,1,2]\) give \([0.090031,0.244728,0.665241]\) at \(\tau=1\), and \([0.186324,0.307196,0.506480]\) at \(\tau=2\). Positive temperature preserves the highest-scoring candidate while changing the distribution. These constructed numbers are not measured Tev1 results.

### Training objective and inference readout

`completion` training appends the correct letter as input and applies the LM head to \(H[:, -2:]\), producing \([B,2,V]\) logits for the answer letter and EOS. Loss averages the **full-vocabulary** cross-entropy at those two positions. Inference appends no target: it reads the final prompt position and normalizes over the candidate subset. See [`Collator`](../../../src/metiscraft/train/collate.py#L105) and [`completion_loss`](../../../src/metiscraft/judgments/__init__.py#L63). Candidate-space CE and this recipe's completion objective are therefore distinct.

See the existing diagrams and derivations in [answer-token tensor analysis](../tensor-analysis/answer-token.md) and [training, calibration and output](../tensor-analysis/training-calibration.md). The [upstream request code](https://github.com/togethercomputer/tev1/blob/1dde7782382c9f49d627153759b8d1deab426ce0/examples/decide.py) additionally specifies constrained generation, the non-thinking template and logprobs parameters. A shared readout family does not establish identical serving protocols, calibration or latency, or reveal Jev's undisclosed internal architecture.

## Results against the original

| Set | MetisCraft 2B | MetisCraft 4B | Original Tev1 4B | 4B difference |
|---|---:|---:|---:|---:|
| Main, 1,000 questions | 864/1,000 · 86.4% | **873/1,000 · 87.3%** | 880/1,000 · 88.0% | −0.70 pp |
| Policy transfer, 300 questions | 276/300 · 92.0% | **298/300 · 99.33%** | 300/300 · 100% | −0.67 pp |

Criteria were fixed on 2026-11-06 before observing results: 2B main accuracy within 5 percentage points of the original, transfer close to 100%; 4B accuracy within 2 percentage points. Only the 2B main check passed; both 4B checks passed without changing thresholds.

These public questions were used during development; they are not untouched final tests. The upstream separately labeled v2 final evaluation was not used here.

### Per-question agreement

All 1,300 IDs, inputs, gold labels, candidate contents and order matched. Main answers agree on 946/1,000 (94.6%); transfer answers agree on 298/300 (99.33%). On main: both correct 852, only MetisCraft correct 21, only original correct 28, both wrong 99. No additional agreement threshold was introduced after observing results.

Main choice accuracy is 822/900 (91.33%), versus 828/900 (92.0%) upstream; SST-5 is 51/100 (51.0%), versus 52/100 (52.0%). The 2B SST-5 result is 57.0%, so larger size did not improve every component.

### Probabilities and calibration

| 4B metric | Raw | Temperature calibrated |
|---|---:|---:|
| Main question-weighted NLL | 0.3019 | 0.3035 |
| Policy-transfer NLL | 0.0522 | 0.0990 |

Policy temperature is 0.476. Positive temperatures preserve these argmax decisions, while NLL worsens; this is not evidence of universal calibration improvement. For 2B, main-choice ECE is 0.027→0.019; transfer NLL is 0.20→0.44 with temperature 0.27. Its cause remains undetermined.

## Conditions and training differences

The 4B run `20261106-183017-f4cb8155` uses the [4B recipe](../../../recipes/tev1/qwen3.5-4b_chat-completion.yaml), `tev1/rev-1dde7782`, and `Qwen/Qwen3.5-4B@851bf6e8` on one RTX 3090. Its 1,052 optimizer steps took 22,620.72 seconds (about 6 hours 17 minutes). The sealed source manifest SHA-256 is `00db1319eec8a5e936da53dc55f1ef57c23dfae9c1e8e196bacce28227d16d1b`; artifact/calibration checksums and parent-artifact binding passed.

The 2B run uses `Qwen/Qwen3.5-2B@15852e8c`, completing 1,052 steps in about 7.2 hours with approximately 6 GB peak memory. Resource measurements describe that configuration, not arbitrary hardware or sequence lengths.

| Setting | Public upstream recommendation | MetisCraft |
|---|---|---|
| Base | Qwen3.5-4B | Pinned 2B control and 4B |
| Tuning | All-linear LoRA, r=8, alpha=16, dropout=0 | Same |
| Optimization | 1 epoch, 5e-5, cosine, warmup 0.03, gradient clip 1 | Same recommended values |
| Length | 2,048, packing | 2,048, no sample packing |
| Batch | 8 packed sequences, approximately 36 questions per step | batch 4 × grad_accum 9, target 36 questions per step |
| Objective | Answer letter and EOS only | Full-vocabulary `completion` |
| Prompt | Non-thinking chat template, system prompt and JSON | `chat_json`, all 43,708 prompts token-identical |

The hosted Together job ID, upload hashes and actual hyperparameters were not retained; historical-job retrieval returned 403. The repository explicitly supplies a recommendation. Agreement above is with that recommendation.

The original [evaluation.json](https://github.com/togethercomputer/tev1/blob/1dde7782382c9f49d627153759b8d1deab426ce0/runs/new-v1/evaluation.json) reports 88.0% / 100% using temperature 0 with option-regex-constrained generation. MetisCraft chooses the largest candidate-letter logit. These are not the same unrestricted full-vocabulary first-token protocol; unrestricted diagnostics were not performed.

### Historical sample consumption and configuration identity

37,840 records at batch 4 produce 9,460 microbatches. The old loop filled 1,052 complete optimizer steps by reading 8 microbatches from the next epoch: **32 additional sample presentations**, not 32 new independent examples. This agrees with historical `epochs_seen=2`. The current loop submits the remaining tail within the configured budget. This documentation work does not rerun the experiment.

Historical `config.sha256` / `recipe_sha256` identify the recipe before path resolution, while saved `config.yaml` is resolved. The original input hash can be reconstructed from the frozen recipe and recorded overrides. Current software records both identities; historical files and evaluation numbers remain unchanged.

## Rebuilding data and prompts

| Source | Training records | Pinned version |
|---|---:|---|
| MultiNLI | 5,000 | `nyu-mll/multi_nli@da70db2a` |
| BoolQ | 3,000 | `google/boolq@35b264d0` |
| Banking77 | 3,000 | `legacy-datasets/banking77@f5412156` |
| AG News | 1,500 | `fancyzhx/ag_news@eb185aad` |
| SST-5 | 2,000 | `SetFit/sst5@e51bdcd8` |
| Synthetic policy decisions | 13,500 | Deterministic generator |
| Priority routing | 6,000 | Deterministic generator |
| Synthetic research classification | 3,840 | Generator and taxonomy |

Training contains 37,840 records, 16,929,529 tokens and maximum length 1,526 tokens; new-v1 dev contains 4,568 records. Synthetic data comes from rules/templates, without teacher models or Jev outputs. Source rows retain individual terms; artifacts trained on the mixture are not marked commercially usable.

```bash
pip install -e ".[reproduce]" flash-linear-attention
metiscraft data prepare tev1
```

The external builder pins the source tree and five upstream data versions. A wrapper provides local inputs with Hub access disabled; four build scripts and three verification scripts run unchanged before Record conversion. State-key and option order are preserved. **All 21 public hashes match**: 7 new-v1 final files, 12 intermediate files and 2 evaluation sets. All 43,708 records reconstruct their original messages.

| MetisCraft split | Source | Count |
|---|---|---:|
| train | new-v1 train | 37,840 |
| calib, dev | new-v1 dev, split in half by group hash | 2,448 + 2,120 |
| test | v1 test | 1,000 |
| eval_only | v1 policy_transfer | 300 |

Fitting and evaluation inputs/groups do not overlap; deduplication removed no rows. SST-5 becomes score; other tasks become choice. BoolQ preserves its two lettered upstream options, with its original type in provenance. `task_id=tev1/<source>` separates calibration and reporting by source.

The builder uses Python 3.11, datasets 5.1.0 and transformers 5.22.1, versus upstream 3.12 / 4.8.5 / 5.17.0. Explicit `Qwen2Tokenizer` restores upstream counts: newer tokenization counts 2 / 8 fewer tokens in two BoolQ passages containing Devanagari / Tamil combining marks. This changes count hashes in three train files, not prompt text. Training still tokenizes using the base model's own tokenizer.

## Conclusion and limits

The evidence supports predeclared accuracy reproduction on the same public questions and similar per-question behavior. It does not establish identical weights, recovered historical training or generalization to untouched tasks. Single-seed, readout-protocol and calibration limitations remain. See the [Tev1 recipe](../../../recipes/tev1/README_en.md) for setup and [training, calibration and output](../tensor-analysis/training-calibration.md) for the mechanism.
