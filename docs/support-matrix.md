# 支持矩阵（首版）

本矩阵收录 14 个 Jev 类项目（8 个开放权重或训练代码项目、3 个工具包或框架功能、3 个相关数据集或基准），逐项记录 MetisCraft 的适配与复现状态。原始核对日期为 2026-11-05；2026-11-07 补入 JevAny 并更新框架定位。未运行第三方工具包，不把阅读文档等同于复现。

**生态定位**：统一的决策模型工具包已经存在。JevAny 提供数据、训练、评测与服务入口；ms-swift 已有 decision 训练、推理和温度校准。MetisCraft 的工作重点是业务配方、可核查运行记录与同协议比较，这些是需要持续验证的目标，不是已证明的独有能力。

**结论**：v0.1 的读出只有 ① 答案 token 与 ③ 逐候选标量头（见 [`readouts/README.md`](readouts/README.md)）。入选项目中，Tev1、jebadiah、Hmm（①）和 JevForge（③）可以按配方直接训练；mammoth-1、ms-swift 部分覆盖；Kev（④）、Clef（⑤）、d3（②）、Unsloth（⑤）的读出不在 v0.1，只作参照。三个数据集都已在注册表中。状态依据 v0.1 设计范围（[`design.md`](design.md) §2）判断，不代表这些项目的配方已在 MetisCraft 内跑通。

## 首批（2026-11-05 定）

先做下面 5 项：读出落在 ①/③，数据与许可证相对干净，有公开数字可对照。

1. **Tev1**（①）：训练源是 MNLI、BoolQ、Banking77、AG News、SST-5，其中 BoolQ、Banking77、SST-5 就是 v0.1 的验收集；仓库带锁定的源 revision 与构建脚本，并声明未用 Jev 输出与教师标签；代码 MIT。公开数字（BoolQ 0.905、MNLI 0.916、Banking77 0.905 等）来自开发集抽样且无校准，MetisCraft 可在全量 test 上补出 NLL、ECE。注意：AG News、SST-5 条款受限，用它们训练的产物不标可商用；权重卡未声明许可证。
2. **jebadiah**（①）：配方与 v0.1 设计几乎一一对应（fp32 字母 logits、soft CE、score 的序数软目标、分题型温度、按 family 切校准集）；代码与权重均 Apache-2.0。4B-v2 的逐题评测记录随权重公开，可对拍 BoolQ 84.7（300 题）、MNLI 88.0（299 题）、Banking77 70.0（300 题、77 选项，ECE 0.103）。注意：训练池含合成数据（开源教师）与 typed-decisions 训练集，合成部分是否公开未核实。
3. **Hmm**（①）：最小的训练单元，LoRA r=32，一张 3090 约 2 小时、约 3 万题；模型卡给出 typed-decisions 0.709（基座 0.596）；代码 MIT，权重 Apache-2.0。注意：合成数据由 DeepSeek 接口生成，仓库不含数据与结果文件，只能近似重建。
4. **JevForge**（③）：入选项目中唯一同时有训练代码与权重的纯 ③，对应 MetisCraft 的 pairs 布局，是 v0.1「① 与 ③ 对照」的外部参照；权重 Apache-2.0。注意：S1MB 分数不高（23.53，第 158/212），代码 LICENSE 只有一行 MIT 声明，复用代码前需向作者确认；价值在读出对照，不在追分。
5. **LocalLLaMA typed-decisions**（共同标尺，仅评测）：Apache-2.0，2,000 个决策、一键评分，已有 Hmm、Clef、d3、jebadiah 等数字（零样本表与拟合表分开看；jebadiah 的 80.0 属训练池内）。标签是教师采样均值，准确率超过教师自一致率 0.735 不是好事，应以 KL、Brier 为主。

暂不先做：Kev、Clef、d3（读出 ④⑤②）、mammoth-1（② 加 27B 全参）、Unsloth（⑤ 加 AGPL）。

## 支持矩阵

| # | 项目（固定链接） | 类别 | 读出 | 基座与规模 | 判断类型 | 许可证 | 证据 | 训练 | 复现 | 部署 |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | [Tev1][gh-tev1]<br>[Tev1-4B-experimental][hf-tev1] | 代码+权重<br>★263 · 09-24 | ① 单字母生成，无概率 | Qwen3.5-4B，LoRA SFT | noul / choice / score | 代码 MIT；权重卡未声明 | doc-verified | **能**：同读出同类型，训练源与验收集重合 | **进行中**：训练集已用 MetisCraft 按锁定脚本重建，公开的 21 个哈希全部一致；Qwen3.5-2B 在 Tev1 的 1,000 道 test 上 86.4%（Tev1 4B 为 88.0%），300 道 policy_transfer 上 92.0%（Tev1 为 100%），4B 待跑（见[复现记录](reproductions/tev1.md)）；实际超参未留存；数字取自 100–250 条开发集抽样，Banking77 仅 4/6/8 个候选，只作量级对照 | **能**（重训产物） |
| 2 | [jebadiah][gh-jebadiah]<br>[4b-v2][hf-jebadiah] | 代码+权重<br>★10 · 10-22 | ① fp32 字母 logits | Qwen3.5-4B / 9B、Qwen3.8-27B，LoRA | noul / choice / score | Apache-2.0（代码与权重） | source-verified | **能**：soft CE、序数软目标、分题型温度，与 v0.1 配方一致 | **计划**：BoolQ、MNLI、Banking77 有逐题记录可对拍；合成训练池是否公开未核实 | **能**（重训产物） |
| 3 | [Hmm][gh-hmm]<br>[Qwen3.5-4B-Hmm][hf-hmm] | 代码+权重<br>★5 · 09-20 | ① 首 token 选项字母（≤26 个） | Qwen3.5-4B，LoRA r=32 | noul / choice / score | 代码 MIT；权重 Apache-2.0 | doc-verified | **能**：字母 CE，硬软标签各半 | **计划**：typed-decisions 可同协议评测；合成数据需重新生成 | **能**（重训产物） |
| 4 | [JevForge][gh-jevforge]<br>[JevForge-0.8B][hf-jevforge] | 代码+权重<br>★164 · 10-16 | ③ 每候选一行，两层 GELU 头 | Qwen3.5-0.8B | noul / choice / score | 代码 LICENSE 仅一行 MIT 声明；权重 Apache-2.0 | source-verified | **能**：同为 ③，对应 pairs 布局；其 RL 部分不在 v0.1 | **计划**：S1MB 数字含自报同源暴露，须按其协议重做 | **能**（重训产物） |
| 5 | [Kev][gh-kev]<br>[kev-4b][hf-kev] | 代码+权重<br>★9.8k · 10-22 | ④ pointer | Qwen3.5-4B-Base，LoRA r16 加 pointer 头（共 3380 万可训练参数）；0.8B–27B 多档 | noul / choice / score | Apache-2.0（代码与权重） | source-verified | **不能**：④ 不在 v0.1；评测协议（锁定 test、manifest、单温度）可借鉴 | **不能**：读出不同 | **不能** |
| 6 | [Clef][hf-clef] | 权重<br>♥2339 · 10-30 | ⑤ 联合 schema 头 | Qwen3.8-27B（Clef-Flash 为 9B） | noul / choice / score，含图像 | Apache-2.0 | source-verified | **不能**：⑤ 不在 v0.1；训练代码未发布 | **不能**：读出不同 | **不能**：由 SGLang 等引擎服务（卡片附命令） |
| 7 | [d3][hf-d3]<br>[d3-ultra][hf-d3ultra] | 权重<br>♥287 · 11-02 | ② 255 码槽位头，非因果注意力 | Qwen3.8-27B（26.1B；0.6B–27B 多档） | noul / choice / score | Apache-2.0 | source-verified | **不能**：② 不在 v0.1，训练法未披露 | **不能** | **不能**：依赖非因果注意力与其检查点布局 |
| 8 | [mammoth-1][hf-mammoth] | 权重<br>♥75 · 10-22 | ② readout.json 声明 slot_lm、256 槽 | Qwen3.5-27B 全参 | noul / choice / score，含“不知道”选项 | MIT | doc-verified | **部分**：配方（CE 加 Brier，无 RL）可用 ① 做小规模版；② 与 27B 全参不在 v0.1 | **不能**：72×GB300 与 6.21M 条训练数据不可得 | **不能** |
| 9 | [ms-swift 决策任务][gh-msswift]<br>`--task_type decision` | 框架功能<br>★16k · 11-04 | ②④⑤⑥ 等 7 种头，另有 generic_decision | 头绑定模型族；generic_decision 可接任意 CausalLM | noul / choice / score | Apache-2.0 | source-verified | **部分**：generic_decision 的字母头与 ① 相近；其余头不在 v0.1 | **不能**：未见可对照的公开数字 | **不能**：头产物格式不同，适配层在 v0.2 |
| 10 | [Unsloth FastDecisionModel][gh-unsloth] | 框架功能<br>★79k · 11-05 | ⑤ Clef 头接任意 LM；⑥ Laya 编码器 | 任意文本或视觉 LM | noul / choice / score | 仓库 Apache-2.0；决策文件 AGPL-3.0-only | source-verified | **不能**：⑤ 不在 v0.1；决策代码为 AGPL，只读不拷 | **不能** | **不能** |
| 11 | [tasksource-jev-typed-decisions][ds-tasksource] | 数据集<br>♥30 · 10-24 | — | — | noul / choice / score；default 配置 999,832 / 15,000 / 15,000 行 | other；逐行 license_use，754 源中 440 个可商用 | source-verified | **能**：已注册 `tasksource_jev`，训练只取 license_use=commercial | — | — |
| 12 | [Open-Jev][ds-openjev] | 数据集<br>♥115 · 09-20 | — | — | noul / choice / score；release-v2-redistributable 共 113,568 行 | CC0-1.0，仅覆盖作者生成内容 | source-verified | **能**：已注册 `open_jev`；test、ood 标签公开，入模前剔除 metadata_json | — | — |
| 13 | [typed-decisions][ds-typed] | 基准<br>♥270 · 11-03 | — | — | noul / choice / score；1,200 / 400 case，test 共 2,000 个决策 | Apache-2.0 | source-verified | **仅评测**：已注册 `typed_decisions`，标签是未公开教师的采样均值 | **能**（作评测集）：可与 Hmm、Clef、d3 等榜单数字对拍 | — |
| 14 | [JevAny][gh-jevany] | 独立工具包 | direct-token / pointer；另有免训练 choice-token | 多基座，具体兼容范围以其文档为准 | 类型化决策 | Apache-2.0（代码） | source-verified：README、pyproject 与 train.py | **未适配**：已有数据准备、自有数据、训练 CLI/API | **未验证**：未运行其配方 | **未适配**：其检查点尚不能由 MetisCraft 直接加载 |

## 工具包补充核对（2026-11-07）

- **JevAny**：已查看固定提交的 README、包配置和训练代码，包含独立 Python 包、CLI/API、数据准备、训练、评测与服务，以及 manifest/数据用途验证相关实现。不能根据旧矩阵漏收它而断言市场没有统一包，也没有审计其全部训练结果。
- **ms-swift**：[官方 Decision 文档](https://swift.readthedocs.io/en/latest/BestPractices/Decision.html)已有训练、推理与温度校准；其已实现能力与 MetisCraft 适配状态分别记录。
- **标准化进展**：[TRL #8040](https://github.com/huggingface/trl/pull/8040)仍为未合并草稿；[Transformers #50054](https://github.com/huggingface/transformers/pull/50054)与 [#50203](https://github.com/huggingface/transformers/pull/50203)仍未合并。提案不代表已发布 API。

## 训练过程公开程度（2026-11-05 样本，11-07 修正口径）

类 Jev 模型很少把训练过程完整开源。下表只记已核实的内容，"—"表示未核实或未披露。

| 项目 | 训练代码 | 训练数据 | 配方与超参 | 评测记录 | 能否复现训练 |
|---|---|---|---|---|---|
| Tev1 | 有，MIT | 公开数据集（MNLI、BoolQ、Banking77、AG News、SST-5），构建脚本锁定源 revision | 实际超参未留存 | 100–250 条开发集抽样，不输出概率 | 基本可以，超参要补 |
| jebadiah | 有，Apache-2.0 | 开源教师生成的合成数据加 typed-decisions 训练集；合成部分是否公开未核实 | 清楚（soft CE、序数软目标、分题型温度） | 逐题记录随权重公开 | 部分可以 |
| Hmm | 有，MIT | 仓库不含（合成数据由 DeepSeek 接口生成） | LoRA r=32，单张 3090 约 2 小时，约 3 万题 | 仓库不含结果文件 | 只能近似重建数据 |
| JevForge | 有（LICENSE 仅一行 MIT 声明），含 RL | — | — | S1MB 自报，含同源暴露 | 需按其协议重做评测 |
| Kev | 有，Apache-2.0 | 规模与算力远超单机 | pointer 头加 LoRA | 锁定 test、manifest、Brier/ECE | 单机不可行 |
| mammoth-1 | 无 | 6.21M 条，不可得 | 仅模型卡自述（CE 加 Brier、72×GB300、无 RL），未复核 | 榜单分数 | 不能 |
| Clef | 未发布 | 来源未披露 | — | — | 不能 |
| d3 | 无 | 来源未披露 | 训练法未披露 | — | 不能 |
| KnowLine、GLiDE、Sieve（榜单前列） | 未披露；GLiDE 仓库只有推理代码 | 未披露 | 未披露 | 榜单分数 | 不能 |
| 官方 Jev | 闭源，只有 API | — | — | — | 不能；TypeSafe 条款禁止用其输出训练 |

在已核对样本中，部分项目缺少可取得的数据、实际超参或逐题评测，限制了独立复现；本表的未核实项不能推广为整个生态都不存在。单机算力不足也不等于项目不可追溯。MetisCraft 将通过完整记录配置、数据版本、提交、环境和逐题评测，以及实际复现工作来积累证据，见 [`status.md`](status.md)。

## 各列口径

- **收录标准**：近两个月仍在更新，有可核对的证据，对自训 Jev 有参考价值；不收用 Jev / TypeSafe API 输出蒸馏的数据或模型（TypeSafe 条款禁止用 Jev 输出训练）。
- **固定链接**：GitHub 为默认分支的 40 位提交号，Hugging Face 为 40 位 revision。revision、许可证、星标（★ GitHub stars）、点赞（♥ HF likes）取自 Hub 与 GitHub 的只读接口（2026-11-05）；读出、配方与数字来自 2026-11-03 至 05 对固定版本的代码、模型卡或 README 的阅读。未下载权重或数据正文，未运行第三方代码。日期为默认分支最近提交（HF 为最近修改），均为 2026 年。
- **读出**：①–⑦ 同 [`readouts/README.md`](readouts/README.md)；v0.1 只实现 ① 与 ③。
- **基座**：多数条目基于 Qwen3.5 / 3.8，v0.1 验证范围是 Qwen3 0.6B–4B，表中「能」均指换用 Qwen3 基座、按同配方训练，数字不可与原项目直接比较。
- **判断类型**：入选项目都只有 noul、choice、score。multi_label 与 ranking 在本表没有可对照的项目（open-jev 的 Subset、bekko 的 reranker 任务未收）。
- **许可证**：代码与权重分写，取自接口返回；写「未声明」不等于可商用。
- **证据级别**：行内级别指读出、基座、许可证三项事实的核对级别。source-verified：读过固定提交的代码、配置或文件，或接口按文件算出的元数据；doc-verified：读过模型卡、README、官方文档；author-reported：仅作者或榜单自报；unverified：未核对。**所有公开数字（准确率、ECE、榜单分）都是作者或榜单自报，未复跑。**
- **训练**：能＝读出与判断类型都在 v0.1 范围内，按配方即可训练同类模型；部分＝只有部分读出、类型或规模在范围内；不能＝读出在 v0.1 之外（②④⑤⑥⑦ 推后或不做），或许可证不允许借用代码。数据集行写注册表中的用法。
- **复现**：能＝在 MetisCraft 内用同数据同协议得到可与其公开数字对拍的结果；计划＝数据与读出具备，但协议或数据需对齐，v0.1 验收之后再做；不能＝读出、规模或数据不可得。其公开数字多为开发集抽样或作者自测，即便「能」「计划」也只作量级对照。
- **部署**：指 MetisCraft 的 Predictor 与兼容 `/v1/systemone` 请求格式的 HTTP 服务能否服务该类模型。v0.1 只加载 MetisCraft 自己的产物，不直接导入第三方检查点，也不导出到 vLLM、SGLang、llama.cpp；「能（重训产物）」指按同配方在 MetisCraft 内重训后可部署。

## 未核实项

1. 公开数字均为自报，未复跑；不同协议（候选数、抽样规模、训练是否含同源数据）的数字不能横比。榜单（Decision Index、S1MB、JevBench）含私有或封存题，不可审计。
2. Clef、d3、mammoth-1 的训练数据来源未披露，无法核对是否使用 Jev / TypeSafe API 输出。本表只列权重，不使用它们的输出做训练。
3. mammoth-1 的读出分类仅依据 `readout.json`（slot_lm、256 槽），初始化方式与注意力掩码未核实；训练配方（CE 加 Brier、6.21M 条、72×GB300、无 RL）取自模型卡，属 author-reported。d3-ultra 的读出沿用 d3 的分类（文件布局相同，未读配置）。
4. 许可证空白：Tev1 与 bekko-400m 的权重卡未声明；JevForge 代码 LICENSE 仅一行；Open-Jev 的 CC0 只覆盖作者生成内容，部分题面取自 TypeSafe 文档，其许可证未核；tasksource 约四成源未确认可商用。
5. typed-decisions 的标签来自未公开的教师（约 4B 级）三次采样均值，无法排除 Jev 输出。是否可作训练标签：待定，推荐只用于评测与校准评估。
6. ms-swift、Unsloth 的功能核对停在 2026-11-03 的提交，此后默认分支已前进（ms-swift 9853e46、Unsloth 09fd0d3）；只复核了 Unsloth 的 `decision.py` 在新提交上仍是 AGPL-3.0-only。JevForge-0.8B、d3-ultra、mammoth-1 的权重 revision 在此前记录之后有更新，所列为 11-05 读数，读出与配方沿用此前的核对。
7. transformers 的决策接口提案 #50054（open，未合并）与 TRL 的 DecisionTrainer #8040（draft，未合并）在 2026-11-05 状态未变，未收入主表；合并后再评估适配层。
8. 暂缓收录的候选（取自 2026-11-03 的检索结果，未复核）：OneJev（①，训练数据带 teacher 字段，教师未核实）、bekko（⑥，MIT）、NanoJev（③，Qwen3-0.6B，MIT，数字多在游戏任务上）、decider（①，Apache-2.0）、jeff（②，MIT，0.8B）。⑥⑦ 暂无入选；是否在下一版收入：待定，推荐随 v0.2 的 encoder 基线一并评估。

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
