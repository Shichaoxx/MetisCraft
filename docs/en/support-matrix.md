# Ecosystem and support matrix

[Per-project model architecture atlas](tensor-analysis/architectures.md): token layouts, backbone configurations, readout matrices and evidence at pinned revisions.

This page separates external capabilities, MetisCraft implementation scope and actual reproduction evidence. Most pinned materials were reviewed 2026-11-03 through 05; JevAny and framework positioning were additionally checked 2026-11-07. This is a dated research snapshot, not a statement of every project's current features. Third-party toolkits were not run; reading documentation is not reproduction.

Unified decision-model toolkits already exist, including JevAny and ms-swift's decision functionality. MetisCraft is a JevLike toolkit connecting custom data, training, calibration, evaluation and serving. Its value is demonstrated through inspectable recipes and usable workflows, without claiming to be first or unique.

## Architectures of recorded recipe runs

An architecture combines a **base model, readout and training objective**. The table links recipes with saved training and evaluation records: Tev1 compares against published third-party results; the remaining rows validate dataset tasks, rather than reproducing third-party models with those names.

| Recipe and record | Base and tuning | Readout / training objective | Evidence scope |
|---|---|---|---|
| [Tev1 4B](reproductions/tev1.md) | Qwen3.5-4B `851bf6e8`; LoRA r=8 | Answer token; full-vocabulary `completion`, letter + EOS | Both main and transfer sets meet predeclared accuracy criteria; historical hosted training not recovered. |
| [Tev1 2B control](../../recipes/tev1/README_en.md) | Qwen3.5-2B `15852e8c`; LoRA r=8 | Answer token; same objective | Only the main criterion passed; this does not replace the 4B result. |
| [Banking77](../../recipes/banking77/README_en.md) | Qwen3-0.6B-Base; LoRA r=16 | Answer token / scalar-list; categorical CE | Recorded runs for both readouts; different length limits prevent a strict readout-only ablation. |
| [Customer routing](../../recipes/banking77_support/README_en.md) | Qwen3-0.6B-Base; LoRA r=16 | Scalar-list; choice CE + noul BCE | Banking77-derived teaching task with a recorded request and response. |
| [BoolQ](../../recipes/boolq/README_en.md) | Qwen3-0.6B-Base; LoRA r=16 | Answer token; BCE on the yes/no logit difference | Yes/no task validation. |
| [SST-5](../../recipes/sst5/README_en.md) | Qwen3-0.6B-Base; LoRA r=16 | Answer token; level-code categorical CE | Five-level `score` task validation. |
| [GoEmotions](../../recipes/go_emotions/README_en.md) | Qwen3-0.6B-Base; LoRA r=16 | Scalar-list; per-label sigmoid, weighted BCE | Recorded training predates the prior correction; only offline diagnostics apply the fix, without retraining results. |
| [CLINC150](../../recipes/clinc150/README_en.md) | Qwen3-0.6B-Base; LoRA r=16 | Scalar-list; 151-candidate categorical CE | Intent-task validation including an out-of-scope null option. |

These 0.6B recipes share the [pinned Qwen3-0.6B-Base configuration](https://huggingface.co/Qwen/Qwen3-0.6B-Base/blob/da87bfb608c14b7cf20ba1ce41287e8de496c0cd/config.json): a 28-layer causal backbone with hidden width 1,024. Their recipes specify `all-linear` LoRA with r=16, alpha=32 and dropout=0.05; scalar-list adds a shared `1024 → 256 → 1` MLP to score candidate-end representations. Candidates share a causal sequence and are not independent pairs. See [answer tokens](tensor-analysis/answer-token.md), [scalar heads](tensor-analysis/scalar-heads.md), and [Tev1's architecture section](reproductions/tev1.md) for its 32-layer hybrid backbone and tensor chain.

## Third-party projects at pinned revisions

This table records external sources and package adaptation status. Apart from the explicitly reported Tev1 accuracy reproduction, mechanism inspection does not establish training, importing or reproducing these checkpoints.

| Project | Mechanism and license evidence | MetisCraft status |
|---|---|---|
| [Tev1][gh-tev1] / [4B weights][hf-tev1] | Qwen3.5-4B + existing LM head, LoRA SFT; code MIT, weight-card license undeclared | [Architecture and reproduction](reproductions/tev1.md): 4B predeclared accuracy criteria met; per-question comparison complete, hosted training not recovered |
| [jebadiah][gh-jebadiah] / [4b-v2][hf-jebadiah] | Answer token; code/weights Apache-2.0 | Same-family retraining is possible; not reproduced; synthetic-pool availability unverified |
| [Hmm][gh-hmm] / [weights][hf-hmm] | Answer token; code MIT, weights Apache-2.0 | Not reproduced; synthetic data and per-question results absent from the reviewed repository |
| [JevForge][gh-jevforge] / [weights][hf-jevforge] | Independent-candidate scalar head; code has one-line MIT declaration, weights Apache-2.0 | Pairs mechanism corresponds; no checkpoint import or reproduction of its RL/benchmark protocol |
| [Kev][gh-kev] / [weights][hf-kev] | Pointer; code/weights Apache-2.0 | Mechanism comparison; head and third-party checkpoint loading not implemented |
| [Clef][hf-clef] | Joint schema; weights Apache-2.0 | Mechanism comparison; not implemented, training data/history unverified |
| [d3][hf-d3] / [d3-ultra][hf-d3ultra] | Fixed slots; Apache-2.0, with non-causal attention requirements | Not directly supported by the two current readouts; weights cannot be loaded directly |
| [mammoth-1][hf-mammoth] | readout.json declares slot_lm, 256 slots; MIT | Pinned source verifies slot_lm and block-bidirectional full-attention; actual training history, 6.21M records and 72×GB300 remain author reports, not reproduced |
| [ms-swift][gh-msswift] | Decision training/inference/temperature calibration; Apache-2.0 | Existing framework capabilities; MetisCraft has no head-format/backend integration |
| [Unsloth][gh-unsloth] | Decision-model interface; repository Apache-2.0, reviewed decision.py AGPL-3.0-only | Source comparison only; decision implementation neither run nor copied |
| [tasksource-jev][ds-tasksource] | Data; per-row license_use, 440 of 754 sources marked commercial | Registered as tasksource_jev; training filters license_use=commercial, without assuming one license for all sources |
| [Open-Jev][ds-openjev] | Data; CC0-1.0 covers author-generated content only | Registered as open_jev; metadata_json removed to prevent leakage; inherited prompts require separate license review |
| [typed-decisions][ds-typed] | Benchmark; Apache-2.0, labels are sampled teacher means | Registered as typed_decisions, evaluation only; undisclosed teacher, soft labels are not noise-free truth |
| [JevAny][gh-jevany] | Independent toolkit, direct-token/pointer; code Apache-2.0 | README, pyproject and train.py reviewed; data/training/evaluation/serving entrances exist, recipes not run and checkpoints not adapted |

## What implementation and reproduction mean

The package implements answer tokens and default scalar-list / optional scalar-pairs. Other mechanisms in the [seven-readout comparison](tensor-analysis/readout-comparison.md) do not imply checkpoint compatibility. Predictor loads MetisCraft artifacts only; there is no third-party checkpoint conversion or vLLM, SGLang or llama.cpp export.

“Train the same family” means that a readout/type can express the task, not that it recovers the same model. Model size, data, answer spaces, prompts and optimization differ; such results are not directly comparable.

The completed [Tev1 4B reproduction](reproductions/tev1.md) reports 873/1,000 (87.3%, original 88.0%) on main and 298/300 (99.33%, original 100%) on policy transfer, with 94.6% / 99.33% answer agreement. All 21 data hashes and per-question inputs match. Single-seed, public-development-set, missing historical hyperparameter and worsened-NLL limitations remain.

## Evidence levels and limits

- `source-verified`: pinned code/configuration/metadata inspected; `doc-verified`: model card/README/official docs inspected; `author-reported`: author or leaderboard report; `unverified`: not checked. Inspecting mechanisms does not reproduce accuracy.
- jebadiah's subset reports include BoolQ 0.847 on 300 questions, MNLI 0.880 on 299, and Banking77 0.700 / ECE 0.103 on 300. Hmm reports typed-decisions 0.709 versus base 0.596. These are different subsets, not MetisCraft experiments.
- Undisclosed training data or actual hyperparameters limit independent reproduction. This does not establish ecosystem-wide untraceability; insufficient single-machine resources do not establish untraceability either.
- An undeclared weight-card license does not grant permission. Review mixed-data terms per row; respect restrictions on training from Jev/TypeSafe outputs. This snapshot provides no unified commercial authorization.
- typed-decisions has an undisclosed teacher and remaining provenance uncertainty, so the package treats it as evaluation-only. Teacher probabilities are not independent hard truth.

## Frameworks and standardization proposals

The [official ms-swift Decision documentation](https://swift.readthedocs.io/en/latest/BestPractices/Decision.html) supports its listed functionality; external capabilities and package adaptation are separate. In the 2026-11-07 review, [TRL #8040](https://github.com/huggingface/trl/pull/8040), [Transformers #50054](https://github.com/huggingface/transformers/pull/50054) and [#50203](https://github.com/huggingface/transformers/pull/50203) were unmerged. That dated proposal state is not a released API and makes no assertion about subsequent changes.

See [interfaces and artifacts](interfaces.md) for field/probability semantics and [Cookbook](cookbook/index.md) for workflows.

[gh-tev1]: https://github.com/togethercomputer/tev1/tree/1dde7782382c9f49d627153759b8d1deab426ce0
[hf-tev1]: https://huggingface.co/togethercomputer/Tev1-4B-experimental/tree/0b7becf017daa0e5eb222f8ce7483c8c8259c52f
[gh-jebadiah]: https://github.com/getainode/jebadiah/tree/a7e8108fbedaa08faa016bf4aeaa7c281b068639
[hf-jebadiah]: https://huggingface.co/frontier-infra/jebadiah-4b-v2/tree/872ae9509607761d4484ab49270311d9308bba18
[gh-hmm]: https://github.com/n4ze3m/hmm/tree/78d8d4a62a41857ce67d903aadf504580feea832
[hf-hmm]: https://huggingface.co/n4ze3m/Qwen3.5-4B-Hmm/tree/adcef8b4701704a37c33053a8a262a7bd076a511
[gh-jevforge]: https://github.com/zwliJay/jev-forge/tree/039badbafc242b4f8105a59e27f789320efe6e9c
[hf-jevforge]: https://huggingface.co/AndeyTait/JevForge-0.8B/tree/5ca7c0861a0d9b72629a7bc40130a8f154c4840f
[gh-kev]: https://github.com/jaredpalmer/kev/tree/2074cc44256200bb37540103fa4572901faa10f3
[hf-kev]: https://huggingface.co/jaredpalmer/kev-4b/tree/6cfce5c2fa4b4bd64026336ab649c5ca78857d52
[hf-clef]: https://huggingface.co/Cloudflare/clef/tree/cfddda043dd190ef3b01cff449d5ee8cff1e8ec6
[hf-d3]: https://huggingface.co/vllm-sr/d3/tree/62f07e54c6c3a5ff5f695107377edc43e1a86ea8
[hf-d3ultra]: https://huggingface.co/vllm-sr/d3-ultra/tree/ccdcfa611649a59e095e0b5e1732a15cedafadd6
[hf-mammoth]: https://huggingface.co/baselabs/mammoth-1/tree/61cec44c281c693992e3d6ebeb44dd3402dfa186
[gh-msswift]: https://github.com/modelscope/ms-swift/tree/df0bc0fba64dcacd2e9250df4e924ed9e32e290f
[gh-unsloth]: https://github.com/unslothai/unsloth/tree/8d2d7b644a54372576a8e7a022854dca4b8d8c95
[ds-tasksource]: https://huggingface.co/datasets/tasksource/tasksource-jev-typed-decisions/tree/8df2e5c9a33f19a2c2b8febf291c3acb0a91a8a8
[ds-openjev]: https://huggingface.co/datasets/ZefanCai/Open-Jev/tree/c67699e13d0ae25e35b77165a4b6b079bedc8aba
[ds-typed]: https://huggingface.co/datasets/LocalLLaMA/typed-decisions/tree/c9e8c4167bf9b47edfcba3b9580e69bddd8d9e5a
[gh-jevany]: https://github.com/SimpleJev/JevAny/tree/7832fd7ba3b0ba4205bf9f6d3457a7256e11e24e
