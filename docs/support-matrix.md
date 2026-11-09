# 生态与支持矩阵

本页区分外部项目能力、MetisCraft 实现范围和实际复现证据。固定版本材料主要核对于 2026-11-03 至 05；JevAny 与框架定位补充核对于 2026-11-07。它是有日期的研究快照，不代表外部项目当前全部能力。未运行第三方工具包，不把文档阅读视为复现。

统一决策工具包已经存在，包括 JevAny 和 ms-swift 的决策能力。MetisCraft 定位是连接自有数据、训练、校准、评测与服务的 JevLike 工具包；价值通过可核查配方与实际使用路径体现，不宣称首个或唯一。

## 固定版本对照

| 项目 | 机制与许可依据 | MetisCraft 状态 |
|---|---|---|
| [Tev1][gh-tev1] / [4B weights][hf-tev1] | 答案 token；代码 MIT，权重卡未声明许可 | 4B 预定准确率达标；同题逐题核对完成，历史托管训练未恢复 |
| [jebadiah][gh-jebadiah] / [4b-v2][hf-jebadiah] | 答案 token；代码／权重 Apache-2.0 | 同类读出可重训，未复现；合成训练池是否完整公开未核实 |
| [Hmm][gh-hmm] / [weights][hf-hmm] | 答案 token；代码 MIT，权重 Apache-2.0 | 未复现；仓库不含合成数据与逐题结果 |
| [JevForge][gh-jevforge] / [weights][hf-jevforge] | 独立候选标量头；代码仅一行 MIT 声明，权重 Apache-2.0 | 对应 pairs 机制；未导入检查点，未复现其 RL 或榜单协议 |
| [Kev][gh-kev] / [weights][hf-kev] | pointer；代码／权重 Apache-2.0 | 机制对照，未实现该头或第三方检查点加载 |
| [Clef][hf-clef] | 联合 schema；权重 Apache-2.0 | 机制对照；未实现，训练数据／历史配置未核实 |
| [d3][hf-d3] / [d3-ultra][hf-d3ultra] | 固定槽位；Apache-2.0，含非因果注意力要求 | 不由当前两个读出直接支持；不能直接加载其权重 |
| [mammoth-1][hf-mammoth] | readout.json 声明 slot_lm、256 槽；MIT | 初始化／mask 未核实；27B 全参、6.21M 数据、72×GB300 为作者自报条件，未复现 |
| [ms-swift][gh-msswift] | 决策训练／推理／温度校准；Apache-2.0 | 已有框架能力；MetisCraft 未接入其头格式或后端 |
| [Unsloth][gh-unsloth] | 决策模型接口；仓库 Apache-2.0，所核 decision.py 为 AGPL-3.0-only | 只作来源对照，未运行或复制决策实现 |
| [tasksource-jev][ds-tasksource] | 数据；逐行 license_use，754 源中 440 标记可商用 | 注册名 tasksource_jev；训练筛选 license_use=commercial，不把数据集整体视为统一授权 |
| [Open-Jev][ds-openjev] | 数据；CC0-1.0 仅覆盖作者生成内容 | 注册名 open_jev；移除 metadata_json 防泄漏，上游引入题面须单独核许可 |
| [typed-decisions][ds-typed] | 基准；Apache-2.0，教师采样均值标签 | 注册名 typed_decisions，仅评测；教师未公开，不能把软标签当作无噪声真值 |
| [JevAny][gh-jevany] | 独立工具包，direct-token／pointer；代码 Apache-2.0 | 已核 README、pyproject、train.py；有数据／训练／评测／服务入口，未运行其配方或适配检查点 |

## 实现与复现的含义

当前包实现答案 token，以及默认 scalar-list／可选 scalar-pairs。其余机制见[七类读出对照](zh/tensor-analysis/readout-comparison.md)，不能据此推断检查点兼容。Predictor 只加载 MetisCraft 产物；没有第三方检查点转换，也不导出 vLLM、SGLang 或 llama.cpp 格式。

“可训练同类模型”表示读出和题型能表达该任务，不表示恢复相同模型。原版规模、数据、答案空间、提示与优化协议不同，数字不能直接横比。

已完成的 [Tev1 4B 复现](reproductions/tev1.md)为主集 873/1,000（87.3%，原版 88.0%）、政策迁移 298/300（99.33%，原版 100%）；一致率 94.6%／99.33%。数据 21 个哈希与逐题输入一致。单种子、公开开发题、历史超参缺失和 NLL 恶化限制均保留。

## 证据等级与限制

- `source-verified`：核查固定版本源码、配置或元数据；`doc-verified`：核查模型卡／README／官方文档；`author-reported`：作者或榜单自报；`unverified`：未核查。核查机制不等于复跑其准确率。
- jebadiah 的 300 题报告包括 BoolQ 0.847、MNLI 0.880（299 题）、Banking77 0.700／ECE 0.103。Hmm 自报 typed-decisions 0.709，基座 0.596。这些不同子集数字不是 MetisCraft 实验。
- 未披露训练数据或实际超参会限制独立复现，但不能推广为整个生态不可追溯；单机资源不足也不等于项目不可追溯。
- 权重卡未声明许可不表示自由可用。混合数据按行核查条款；禁止以 Jev／TypeSafe 输出作训练标签的条款须遵守。研究快照不提供统一商业授权。
- typed-decisions 的教师未公开、来源尚有不确定性，因此本包按评测用途处理。教师软概率不是独立的硬标签。

## 框架与标准提案

[ms-swift 官方 Decision 文档](https://swift.readthedocs.io/en/latest/BestPractices/Decision.html)是其功能来源；适配状态与外部能力分别列出。2026-11-07 的核查中，[TRL #8040](https://github.com/huggingface/trl/pull/8040)、[Transformers #50054](https://github.com/huggingface/transformers/pull/50054)与 [#50203](https://github.com/huggingface/transformers/pull/50203)尚未合并；该日期的提案状态不等同于已发布 API，也不推断之后状态。

字段和概率语义见[接口与产物](zh/interfaces.md)，配方操作见 [Cookbook](zh/cookbook/index.md)。

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
