# 决策模型（System One / Jev 类）开源 SOTA 现状

> 设计前调研材料（2026-11-04）。设计结论以 [`docs/design.md`](../design.md) 为准；文中引用的 coverage-map、`notes/`、`partial/`、`landscape/` 等是维护者本机的调研材料，不随仓库发布。

> 材料来源：本机调研目录 `partial/` 下 sota-scan-1 至 7（官方、任务强基线、开源项目两组、统一基准与读出/校准论文、两份对抗核查）。日期 2026-11-03。**本报告是对已有检索结果的汇总，未重新检索**；表中数字全部来自上述材料，核查判 refuted 的说法一条也没有采用，判 weakened 的已降级措辞。

## 1. 结论先行

1. **没有公认的开源 SOTA 方案**。原因：Jev 闭源且无论文；社区榜有三套（Decision Index、JevBench、typed-decisions），指标、版本、私有题各不相同，且榜首模型（27–31B 全参微调）几乎都只放权重、不放训练代码和数据。
2. **榜单分数上开源已追平或超过 Jev**：Decision Index 0.4.1 中 Jev 1.14 为 60.24（第 49），高于它的 48 个模型几乎都是 25–36B；开放权重里 vllm-sr/d3-ultra（Apache-2.0）70.18、baselabs/mammoth-1（MIT）68.44。但 4B 以下与 Jev 仍有明显差距（≤5B 最佳 55.99，≤1B 最佳 31.21），而 MetisCraft 的本地算力正落在这一档。
3. **证据最强、且配方可读的一条路线是 mammoth-1 一族：LM 骨干 + 选项概率读出 + CE + Brier 的监督训练，无 RL**（榜单分数 source-verified，训练配方 author-reported）。另一条是 **Kev**：pointer head + 选项 CE + 单温度校准，有锁定 test、manifest 和 Brier/ECE，是公开仓库里评测协议最完整的（source-verified，但 test 为自建、个别数据集仅 80 条）。
4. **RLCD/策略梯度 RL 没有任何开源证据支持其收益**：JevAny 的 RLCR 比 SFT 低 0.10pp（区间跨 0），eve-rlcd 比同量 SFT 低 0.9pp，JevForge 与 NanoJev 的 RL 结果均为单种子且混有反向指标。
5. 对工具包的含义：**MetisCraft 值得做"同协议对照基准 + CE/Brier 监督配方"，不值得把 RLCD 当卖点**。

## 2. 可比结果表

图例：证据级别 SV = source-verified，AR = author-reported，UV = unverified。核查栏为 scan-6/7 的判定。"任务 SOTA"取自 scan-2，均为作者在论文/模型卡报告值，未复跑。

### 2.1 Banking77（77 类，官方 test 3,080）

| 系统 | 数字 | 协议 | 级别 | 核查 | 与任务 SOTA 可比？ |
|---|---|---|---|---|---|
| SPACE-3（任务 SOTA） | acc 94.94 | 全量微调，预训练语料含该集 train | AR | confirmed | 可比，但非纯微调 |
| ICDA RoBERTa-large + OPT-66B 合成 | acc 94.42 | 全量 CE | AR | — | 可比，训练成本不对等 |
| DeepOpen（Laya encoder + SAM） | acc 94.04（3 种子均值） | 全量；标题 94.12 为看过 test 后挑的种子 | AR | confirmed，区间跨 0 | 可比 |
| BERT-large / RoBERTa-large | 93.66 / 93.70 | 全量 CE | AR | — | 可比 |
| Decision Index：d3-ultra / mammoth-1 / GLiDE | macro-F1 0.9416 / 0.9391 / 0.9201 | 官方 test 全量，榜单统一 | SV | confirmed | 指标是 macro-F1，非 accuracy |
| Decision Index：Kev-27B / Jev 1.14 | macro-F1 0.8606 / 0.7994 | 同上；Kev 训练含 1,000 条官方 train | SV | confirmed | 同上，非零样本 |
| Decision Index：Hopper 12B / Strands 2B / Kev-4B | 0.7254 / 0.8036 / 0.8287 | 同上 | SV | confirmed | 同上 |
| Jev 1.13 零样本（IIA 预注册研究） | acc 0.82 | 抽 1,000 条，只给标签名，并列按 1/k 计分 | AR | confirmed | 不可比，仅说明量级 |
| Kev-4B | acc 0.8375 | 80 条，77 类全集，训练源 | SV | confirmed | 不可比（n=80，±8pp） |
| Tev1-4B | acc 0.905 | 200 条，仅 4/6/8 个候选 | SV | 数字属实 | 不可比（候选缩减） |
| Hopper 1.1.1 | acc 0.673 | 300 条，锦标赛 shortlist | AR | weakened：Banking77 在其训练集中 | 不可比 |
| Strands v14 | acc 0.983 | 训练混合内抽样，多数行训练时见过 | AR | confirmed | 不可比（非 test） |
| Laya | acc 0.425，ECE 0.540 | 400 条，77 类，零样本 | SV | confirmed | 不可比；预测塌缩（macro-F1 0.11） |
| LLM-as-Jev（Qwen3-0.6B + LoRA，选项似然） | acc 59.0（未训练 22.2） | 官方 test 3,080 | AR | — | 可比，说明小模型长菜单的难度 |
| brier（PyPI，免训练读出 + 300 标签头） | acc 0.807，ECE 0.035 | banking20（20 类子集，取自 train 划分） | SV | weakened：三个数是点估计与区间，仅一次运行 | 不可比 |

结论：全量 77 类 + 官方 test 上，监督强基线约 94–95；决策模型里只有榜单上的 27B 级模型（macro-F1 0.92–0.94）接近这一水平，4B 以下尚无可靠同协议数字。

### 2.2 BoolQ（validation 3,270）

| 系统 | 数字 | 协议 | 级别 | 核查 | 可比？ |
|---|---|---|---|---|---|
| 多数类 | 62.17 | — | AR | — | 下限 |
| DeBERTa-v3-large（HF 社区卡） | 88.35，NLL 0.4601 | 单任务 CE，单种子 | AR | 疑似按验证 loss 选 checkpoint，偏乐观 | 可比，唯一带 NLL 的参照 |
| RoBERTa-large（HF 社区卡） | 85.69，NLL 0.6057 | 同上 | AR | — | 可比 |
| T5-11B / PaLM-540B / ST-MoE-32B | 90.8 / 92.2 / 93.1 | SuperGLUE 微调 | AR | weakened：取逐任务峰值验证分；test 为 92.4 | 部分可比，应以 92.4 为准 |
| Kev-4B | 0.875，NLL 0.437，ECE 0.092 | 官方 validation 抽 80 条，训练源 | SV | confirmed | 不可比（n=80） |
| Tev1-4B | 0.905 | validation 抽 200 条，开发中反复使用 | SV | 数字属实 | 不可比 |
| Hmm 4B（读自 Bespoke 套件） | 0.870（基座 0.813；Jev 1.13 0.897） | validation 抽 300 条 | AR | confirmed | 不可比（n=300，±4–5pp） |
| LLaMA3-8B LoRA/DoRA | 70.8 / 74.6 | commonsense170k 多任务、生成读出 | AR | — | 不可当上限 |

### 2.3 MNLI（matched/mismatched validation）

| 系统 | 数字 | 协议 | 级别 | 核查 | 可比？ |
|---|---|---|---|---|---|
| DeBERTa-v3-large | 91.8 / 91.9 | 单任务全参 | AR | confirmed | 任务强基线 |
| Ettin-Enc-1B vs Ettin-Dec-1B | 91.8 vs 89.9 | 同数据同配方 | AR | — | 受控对照：同规模 decoder 低 1.9 |
| RoBERTa-large | 90.2 / 90.2 | 5 次取中位 | AR | confirmed | 可比 |
| GPT-3 175B + LoRA | 91.7 | — | AR | — | 超大 decoder 仅略高 |
| Kev-4B | 0.925，NLL 0.512，ECE 0.072 | matched 抽 80 条，训练源 | SV | confirmed | 不可比（±6pp） |
| Tev1-4B | 0.916 | matched+mismatched 抽 250 条 | SV | 数字属实 | 不可比 |
| Hmm 4B | 0.850（Jev 1.13 0.829） | 抽 299 条 | AR | confirmed；matched/mismatched 未标明 | 不可比 |
| RoBERTa-base 域外 ECE | 3.62 → 温度缩放 1.46 | SNLI 训练、MNLI 评估 | AR | confirmed | 少数带 ECE 的 NLI 参照；温度在目标域拟合，偏乐观 |

### 2.4 AG News（官方 test 7,600）

| 系统 | 数字 | 协议 | 级别 | 核查 | 可比？ |
|---|---|---|---|---|---|
| XLNet-large | 95.55 | 全量全参 | AR | confirmed | 任务 SOTA 级 |
| LS-unLLaMA-2-7B | 95.68 | LoRA + 线性头 | AR | weakened：单次运行，选模方式不明 | 部分可比 |
| LS-LLaMA-2-7B / RoBERTa-large | 95.38 / 94.78 | 同上 | AR | — | 可比 |
| SetFit 每类 64 条 | 88.0 | 少样本 | AR | — | 少样本参照 |
| Claude Sonnet 4.5 零样本 | 91.36 | 自划分半个 test | AR | — | 不可比 |
| Jev 1.13 零样本 | 0.87 | 抽 500 条 | AR | confirmed | 不可比，差约 8.5 点 |
| Kev-4B | 0.8625 | 抽 80 条，训练源 | SV | confirmed | 不可比 |
| Laya | 0.950，ECE 0.032 | 抽 400 条，in_training=true | SV | — | 不可比 |
| Tev1-4B | 0.87 | 抽 100 条 | SV | 数字属实 | 不可比 |

### 2.5 SST-5（官方 test 2,210）

| 系统 | 数字 | 协议 | 级别 | 核查 | 可比？ |
|---|---|---|---|---|---|
| LS-LLaMA-2-7B | 62.31 | LoRA + 线性头；13B 为 62.17 | AR | weakened：无验证集，选择方式不明 | 部分可比 |
| RoBERTa-large | 59.6 至 59.8 | 全参 | AR | — | 任务强基线（不同论文基线相差约 2 点） |
| Heinsen routing（冻结 RoBERTa + routing 头） | 59.8 | 只训头 | AR | — | 冻结骨干 + 多层读出参照 |
| Kev-4B（score 题） | 0.6125，ECE 0.264 | 抽 80 条，训练源 | SV | confirmed | 不可比；等级题校准明显更差 |
| Tev1-4B | 0.52 | 抽 100 条 | SV | 数字属实 | 不可比 |

### 2.6 NFCorpus（BEIR，nDCG@10，323 个 test query）

| 系统 | 数字 | 协议 | 级别 | 核查 | 可比？ |
|---|---|---|---|---|---|
| BM25 flat / multifield | 0.3218 / 0.3254 | Anserini 回归配置 | SV | confirmed | 下限；R@100 仅 0.25，限制重排上限 |
| BM25 + MiniLM CE | 0.350 | 重排 top-100 | AR | — | 可比 |
| monoT5-3B / GPT-4 listwise | 0.3897 / 0.3847 | 重排 | AR | — | 可比（重排深度不同） |
| BGE-base dense / SPLADE-v3 | 0.3735 / 0.3629 | 单阶段 | SV | — | 参照（BGE 训练数据是否含 NFCorpus 未核实） |
| RankLLaMA-7B | 0.303 | 重排 | AR | — | 低于 BM25，说明大不等于迁移好 |
| Qwen3-8B 重排器（Tevatron 3.0） | 0.410 | 重排 BM25 top-200，单 token 打分 | AR | RLHN 训练集是否含 NFCorpus 未核对 | 重排深度不同 |
| NV-Embed-v2 | 0.4517 | 域内训练 | AR | confirmed | 不可比 |
| Metis 旧版 ScoreHead / 原始 Qwen3-Reranker | 0.2910 / 0.3548 | 本地旧结果，本次未复核 | UV | — | 前者低于 BM25 |

注：开源决策模型项目里没有找到任何一个报告 NFCorpus 的数字。

### 2.7 决策模型自身的统一协议数字（榜内可比，榜间不可比）

| 榜 / 协议 | 系统 | 数字 | 级别 | 核查 | 备注 |
|---|---|---|---|---|---|
| Decision Index 0.4.1 Full | KnowLine 31B / GLiDE 1.2 / d3-ultra / mammoth-1 | 71.86 / 71.52 / 70.18 / 68.44 | SV | confirmed | 私有题，无法审计；跨版本不可比（mammoth-1 在 0.4 为 71.24） |
| 同上 | Jev 1.14 / Jev 1.13 | 60.24（第 49）/ 59.22 | SV | confirmed | 另有 Kev-27B 59.56、Hopper 12B 45.76、Kev-4B 34.93、Strands 2B 20.72、Qwen3-0.6B-Base 最佳 16.43 |
| S1MB English v1（137 benchmark） | vllm-sr d3 / Jev 1.13 / Jev 1.14 preview | 71.45 / 59.59 / 60.43 | SV（我方复算） | confirmed，未与页面核对 | 训练暴露由作者自报，0 只表示未申报 |
| S1MB | d3-lite 0.85B / bekko-400m / JevForge-0.8B | 38.69 / 50.60 / 23.53 | SV | bekko 自报 98 项同源暴露 | 亚 1B 最佳零暴露参照为 d3-lite |
| JevBench v1.8.6 capability | mammoth-1 / Kev-27B / Jev 1.13 / Jev 1.14 | 85.23 / 78.94 / 77.13 / 75.01 | SV | confirmed，粗略 | 各行测于不同版本；Kev 的领先主要来自校准轴 |
| typed-decisions 零样本（与 4B 级老师的一致率） | pplx-decider / d3 / d3-mini / Jev 1.13 | 0.753 / 0.749 / 0.7425 / 0.727 | SV（日志重算） | confirmed | 老师自一致上限 0.735，超过上限无意义 |
| 同上 | Cloudflare clef | acc 0.7275，ECE 0.0253，KL 0.191 | SV | confirmed | 准确率低于 d3 但 KL/Brier 明显更好 |
| 同上 | Quyet-1.0-Large | 0.791 | AR | weakened：托管端打分，无逐题输出 | 高于老师上限 |
| 同上（fitted 表） | Decider-4B / AgentJev-0.6B / Hmm 4B | 0.826 / 0.7925 / 0.709 | AR | weakened（Decider-4B：训练数据未披露、ECE 0.160） | 与零样本表不可比；Phocinae 污染更正 0.906 → 0.791 |
| S1-Arena 非 agentic | Vega-27B / Jev 1.13 / Opus 5.5 | acc 73.03 / 72.57 / 81.14，Brier 0.343 / 0.349 / 0.232 | SV | weakened：来源取自公开 test、未做污染审计 | 闭源 LLM 先推理再口头报概率仍是上限 |
| Artificial Analysis Decision Index | Jev 1.13 | 58（GLiDE 54） | AR | confirmed（原页已核） | 私有混合指标 |

### 2.8 Jev 官方数字（TypeSafe，参照系）

| 项 | 数字 | 级别 | 核查 | 说明 |
|---|---|---|---|---|
| WorkflowEvals 4 工作流平均一致率 | 67.76%（GPT-5.6 Sol 74.11，Opus 5 73.08，Sonnet 5 67.75） | SV（复算） | confirmed | 参照是 GPT-6 Astra + Claude Fable 5.1 的 consensus，非真值；每配置取最优运行 |
| EvalSafe O*NET 综合分 | 0.9339（第 3） | SV | confirmed | 合成文档，数据卡自称非 held-out |
| O*NET Noul 1−JSD | 0.9756（称最高） | SV | weakened：LLM 失败题按 0 计，剔除后可能反超 | 衡量与 LLM 概率的接近度，非校准 |
| 193.6x 更快 / 444.6x 更便宜 | 可复算（444.5x） | SV | weakened：对比对象各挑最慢/最贵者；对 Luna 仅 8.4x 便宜、32x 更快 | — |
| 70–500 ms、40–200x、"自然校准" | — | AR | unverifiable | 无测量数据 |
| ECE / Brier / NLL | 官方未在任何公开基准上报告 | — | — | — |
| 第三方对 Jev 的校准证据 | 放射报告 ECE 0.1192→0.0077、Medium spam ECE 0.053 | UV | 原文均未读到 | 不作证据 |
| 二手误归属 | "SST-2 acc 0.807 / ECE 0.032" 是 TF-IDF 基线 | SV | confirmed 为误归属 | jev-calibration-lab 的 Jev 组是 dry-run 合成输出 |

### 2.9 项目自报数字中被核查降级的（不要引用为"击败 Jev"）

| 项目 | 说法 | 核查结论 |
|---|---|---|
| JevAny 27B | Transfer-v9 86.04% > Jev 85.37% | weakened：开发集、差 0.67pp 约 0.6 个标准误；冻结基座 80.69% |
| JevAny 27B | JevBench 公开题 90.04% | weakened：冻结基座 87.88%，公开题已饱和 |
| Luce | PhishNChips 97.4% vs Jev 62.6% | weakened：两行正则即 91.6–91.8，Jev 数字非配对 |
| Luce | 组织规则工单 ECE 0.003 vs Jev 0.107 | weakened：样本内温度缩放，且对方未缩放 |
| JevForge | Mind2Web choice top-1 0.579 > Jev 0.543 | weakened：候选集缩小、指标挑选，同表 Jev 在 noul 上更好 |
| NanoJev | 四游戏加权 66.85% vs 65.39% | weakened：差值取决于加权，Maze 4/10 vs 7/10 更差，OOD 复现失败 |
| decido-26b-a4b | 六组均值 85.38 > Jev 83.85 | weakened：去掉同域训练 split 后 82.04，低于 Jev |
| Mangaba brier v5 | B2W 75.8% vs Jev 66.8% | weakened：同域监督对零样本；作者自设标准下未胜出 |
| OpenJev-RLCD | MMLU-Pro 47.7% vs GRPO 42.4% | weakened：stage-1 校准单独即 47.03%，RL 只加约 0.6pp |
| GLiNER2.5-Decide | 自建集 60.2% | weakened：私有 test，Decision Index 上仅 10.14（第 234） |
| Hopper | JevBench 公开 82.3% / 封存 34.1% | 公开题过拟合的直接证据 |

## 3. 统一基准与排行榜

| 基准 | 内容 | 采用度 | 主要问题 |
|---|---|---|---|
| typed-decisions（LocalLLaMA） | 400 case / 2,000 决策，标签是 4B 级老师 3 次采样均值；acc/KL/Brier/ECE；有一键评分脚本 | 最高：卡片约 90 行，≥36 个 GitHub 仓库引用，Unsloth/Axolotl 有适配 | 标签是老师输出，上限 0.735；出现过 test 污染（Phocinae） |
| Decision Index 0.4.1 | 公开 37 项不计分；Full = 60% 私有同技能 + 40% 私有新领域；附 ECE/Brier | 很高：247 行，Space 861 likes，≥35 个仓库 | 私有题不可审计；版本更新快，跨版本不可比；全部 lab_run=False |
| JevBench（Benchmark Heaven） | 231 公开 + 封存题；capability = (Intelligence + Calibration)/2 | 高：v1.8.6 约 337 个系统，EvalScope 内置 | 公开题已饱和（冻结 4B 达 80.5%）；各行版本不同 |
| S1MB | 137 个 typed-task benchmark，要求申报训练暴露 | 中高：212 个完整提交 | 暴露为自报；我方复算未与页面核对 |
| S1-Arena（arXiv 2610.21244） | 43,226 请求 + 138 个闭环环境 | 低（3 stars） | 未做污染审计，只给汇总 |
| EvalScope 内置 systemone_api | open_decisions、jevbench 适配 | 唯一内置决策基准的主流评测框架 | lm-eval-harness、lighteval、opencompass 未查到 |
| TypeSafe WorkflowEvals / EvalSafe O*NET | 附 Jev 1.13 逐题概率与 consensus 目标 | 仅 TypeSafe 自用 | 参照由 LLM 生成；**可不调 API 直接与 Jev 已记录输出对比** |
| 其余 | AA Decision Index、Bespoke Nimble、OpenDecisions、XDecideBench、TypedBench | 低到中 | 多为私有或新发布 |

**判断**：社区有事实标准榜，但没有一个同时满足"公开题、人工真值、统一读出、报校准、可复现"。更重要的是：**这些榜测的都是成品模型，没有一个把"配方/读出/目标"当变量做同骨干同数据的受控对照**。因此 MetisCraft 做"同协议对照基准"有空位，建议做成薄的一层，不新造数据集：
- 题集：采用 Banking77 全量 test、BoolQ val、MNLI-m val、AG News test、SST-5 test（2.1–2.5 的强基线直接对齐）+ typed-decisions test（采用度最高）；
- 报告：accuracy/macro-F1 与 NLL、Brier、ECE 同表；固定 bin 定义；同时放 encoder 微调基线与多数类/先验行；
- 对 Decision Index、S1MB 只做"跑公开部分 + 提交"，不自建私有题；
- 不再自建 PhishNChips 一类合成集（有构造捷径）。

## 4. 技术配方层面

### 4.1 读出

| 读出 | 证据 | 判断 |
|---|---|---|
| 答案字母 / 答案词（LM 头） | Hmm（LM 头字母，ECE 0.02–0.03，单张 3090 训练 2 小时）；Jeff 1.0 消融：新学的头不增益，头冻结在字母初始化 84.9 ≈ 联合训练 84.4；Visual Jev 的 typed head 无一致优势；mammoth-1、d3、Hopper、decido 均为选项字母/概率读出 | **现有证据下最站得住**，实现最简、可复用预训练 LM 头；受限于选项数（字母读出无法处理 77 选项，Tev1 在榜单 Banking77 得 0；Hopper 需 shortlist） |
| 选项似然（候选后缀对数似然） | LLM-as-Jev：81.4 vs 字母 80.5；能处理 Banking77；Qwen3-0.6B + LoRA 达 59.0 | 长菜单可行，但是作者自报、同一论文内的对比 |
| 逐候选标量头 / pointer / 集合交互 | Kev（pointer）、JevAny（residual pointer，对 MLP 只高 0.22pp，单种子短程）、AgentJev（集合 Transformer）、NanoJev（集合 attention）、JevForge | 能支持任意选项数，但**没有受控论文证明优于 LM 头读出**；Decision Index 里"LoRA + head"类最高仅 61.12（混杂观察） |
| encoder + 选项打分 | GLiClass v4 edge 0.44B Full 31.21（亚 1B 榜单最佳）；GLiNER2.5-Decide 10.14；Laya 9.21；Ettin 受控对照中同规模 decoder 在判别任务上低于 encoder 1.9 点 | 小模型/低延迟可行，但新域泛化弱（GLiClass new-domain 15.64）；Laya 严重过度自信 |
| 生成后读答案 | OpenJev-RLCD（推理后读 "Answer:" 分布）；Scored vs Generated 论文：打分读出在 12/13 单元 AUC 更优；S1-Arena 中 Opus 5.5 口头概率准确率最高但慢约 10 倍 | 与一次前向的 System One 不是同一类；排序/校准上打分读出占优 |
| 冻结骨干 + 免训练读出 | Gemma-4-31B 冻结加字母读出 Full 61.63（覆盖率 0.76）；brier 包 L2 头 0.807（banking20） | 作为"零成本基线行"有价值 |

骨干后训练才是主要差距来源：XDecideBench 中后训练模型比自身骨干准确率高 7.6–18.2pp，ECE 低 0.06–0.19（author-reported）；Hmm 微调基座只加约 3 点（Bespoke 套件）。

### 4.2 训练目标

| 目标 | 证据 | 判断 |
|---|---|---|
| 选项 CE / soft-CE | Kev、JevAny、Hmm（硬软标签各半）、Strands、decido；全部社区榜前列均为 CE/SFT 类 | **证据最广** |
| CE + Brier | mammoth-1（λ=1，6.21M 条，1 epoch，72×GB300，无 RL）；AgentJev（soft CE + 0.1×Brier）；JevForge | 榜单第 9 名且开放权重（MIT），**配方最明确**；但 Brier 项相对纯 CE 的增益本报告材料里没有消融 |
| 教师蒸馏 | Strands（gemma-4-31B 分布蒸馏 + KL anchor）、Hmm（DeepSeek 软标签） | 对小模型有用的证据是间接的，Strands 2B 在公开题上仍低于冻结 4B 基座 |
| 策略梯度 RL / RLCD | JevAny RLCR −0.10pp（区间 [−0.58, +0.39]）；eve-rlcd 低于 SFT 0.9pp（CI [−1.7, −0.2]）；JevForge +0.9pp 但 score MAE 变差且单次；NanoJev 单种子、OOD 反转；OpenJev-RLCD 的增益大部分来自 stage-1 校准；Calibration-aware RL 只核到摘要 | **证据不足，不能作为默认**；"RLCD"在 TypeSafe 处只有名称 |

### 4.3 校准方式

| 方式 | 证据 | 判断 |
|---|---|---|
| 训练后单温度缩放（按题型各一） | Kev（T=2.41，Brier 0.224、ECE 0.017，但锁定读出口径下 ECE 0.085，两份文件口径差未解释）、Hopper（三类温度）、JevForge、Strands、Jeff 1.0（按题型温度，ECE 0.047 vs Jev 0.066）、Desai & Durrett（ECE 3.62→1.46） | **最站得住且最便宜**；要在独立 dev 上拟合，并报告 test 口径 |
| 训练目标内的 proper score（Brier/soft-CE） | mammoth-1；Kev-27B 榜单 ECE 0.0218、GLiDE 0.0204（榜单自测） | 有效，与温度缩放可叠加；增益未被隔离 |
| 外加头 + 少量标签重校准 | brier 包（300 标签头）；警方事故论文（样本内重校准降 3.3 倍） | 部分为样本内，证据弱 |
| 评测口径 | 论文 "Same Answer, Different Confidence"：8/20 设置结论随协议翻转；Jev API 概率取整到两位小数会抬高 KL/NLL；ECE 的 bin 数各家不同 | **须固定一套 ECE 定义并同时报 NLL、Brier** |

### 4.4 证据不足或被削弱的说法

- "RL 带来校准或精度收益"、"RLCD 是关键"：无证据（见 4.2）。
- "pointer/独立头优于 LM 头"：无受控证据；Jeff 1.0 的消融方向相反。
- "小模型击败 Jev"：核查后均与协议挑选、同域训练或开发集有关（见 2.9）。
- "Jev 自然校准"：官方无数据；第三方两条相关数字未能核实；AA 页面显示决策模型整体过度自信。
- "Jev 比 LLM 便宜/快 193.6x/444.6x"：对比对象被挑选。
- Jev 在 TypeSafe 工作流上准确率并非最高（67.76% 对 Sol 74.11%）；优势在成本与延迟。

## 5. 对 MetisCraft 的建议

**默认配方（第一版）**：LM 骨干（Qwen3.5/3 系，先 0.6B–4B）+ LoRA；选项字母读出（选项数超限时用选项似然或 shortlist 作为后备）；目标 = 选项 soft-CE + 小权重 Brier（可开关做消融）；校准 = 按题型各一温度，在独立 dev 上拟合；评测固定 ECE 定义并同表报 acc/NLL/Brier/ECE。pointer/标量头、encoder 打分、RL 作为可选插件，**默认关闭**，直到在同骨干同数据下自己跑出消融。

**第一版复现目标（按本地 4×RTX 3090 24GB 可行性排序）**

| 排序 | 目标 | 要复现/超过的数字 | 可行性 | 价值 |
|---|---|---|---|---|
| 1 | 五个公开分类集上，决策模型对 encoder 微调强基线的"同协议对照" | Banking77 全量 test acc 约 93.7–94.4；BoolQ val 88.35；MNLI-m 91.8；AG News 94.8–95.4；SST-5 59.6–62.3；每项同时报 NLL/ECE | 高：DeBERTa-v3-large 与 4B LoRA 都能在单卡完成；Hmm 单张 3090 训练约 2 小时 | 最能证明工具包可信：数字来自公开强基线，且补上"几乎没有公开参照"的校准列 |
| 2 | typed-decisions 零样本表复现 Hmm 路线 | Hmm 0.709（Jev 1.13 0.727、老师上限 0.735）；校准 ECE 0.02–0.03（作者自报，仓库无结果文件） | 高：Qwen3.5-4B LoRA 字母读出；评测脚本一键 | 采用度最高的榜；有 HF Jobs 日志可对表 |
| 3 | 亚 1B 档对齐 S1MB / Decision Index 公开部分 | S1MB d3-lite 0.85B 38.69（零暴露）；Decision Index Qwen3-0.6B-Base 最佳 16.43，≤1.2B 最佳 31.21 | 中：0.6B–1B 全参/LoRA 可行，但需要大规模混合训练数据（榜单项目用 25 万至 620 万条），数据构建是主要成本 | 对比同档公开行，不必追 Jev |
| 4（暂缓） | Kev-4B / JevAny-4B 级 | Kev-4B transfer-v4 test 0.838；JevAny-4B transfer-v9 78.68%（330.6 H200 小时） | 低：训练数据与算力远超本地；JevAny 是开发集 | 不作为第一版目标 |
| 不做 | 27B 级 mammoth-1 / d3-ultra 复现；RLCD 复现 | — | 不可行（72×GB300；配方未公开）；无收益证据 | — |

补充：O*NET / WorkflowEvals 可在不调 API 的情况下，用已发布的 Jev 1.13 逐题概率做对照，作为第二阶段的"与 Jev 同题对比"，但其参照由 LLM 生成，只能称"与 LLM consensus 的一致度"。

## 6. 已知空白与不确定性

- **榜首模型不透明**：KnowLine、GLiDE、Sieve 等训练方法未披露；GLiDE 仓库只有推理代码；KnowLine 31B 在 HF 上为 Apache-2.0 开放权重（scan-6 更正了 scan-4"不能证明开源"的说法），但榜单未固定被测 revision。mammoth-1 的配方来自模型卡自述，未独立复核。
- **私有题不可审计**：Decision Index 的 Full 分、AA、JevBench 封存题均无法复现；S1MB 复算值未与页面显示核对（页面无法加载）。
- **Brier、ECE 口径不统一**：二元/多类 Brier、10 等宽/等质量/15 bin ECE、对老师 argmax 与对人工标签，数字不能互比。Kev 同一批 656 题在两份文件中 ECE 为 0.017 与 0.085，未解释。
- **训练集包含公开数据集的官方 train**：Kev、Tev1、Hopper、Hmm、Strands、Laya 等相应数字是"训练源留出"，不是零样本迁移；亚样本仅 80–400 条。
- **没有受控消融**：读出（LM 头 vs pointer）、目标（CE vs CE+Brier vs RL）、校准在 0.6B–4B 同骨干同数据下的对照材料里没有；Brier 项增益、蒸馏增益均未隔离。
- **任务 SOTA 数字多为作者报告**：只有 Anserini 的 5 个 NFCorpus 数字为 source-verified；Banking77 的 Llama-2-7B 94.35 找不到原始出处（unverified）；不少 dev 数字有选择偏差（见 scan-2 caveats）。单任务 7B decoder 加分类头的 BoolQ 数字没有找到可靠来源。
- **Jev 是移动靶**：托管模型会漂移，jev-1.14 无任何评测数字；各榜用的 Jev 版本不同；二手资料称 9/19 至 10/10 间有 20 个答案变化（unverified）。
- **校准真值的缺位**：官方没有公开 ECE/Brier；O*NET 的 1−JSD 是与 LLM 概率的接近度。
- **未读到全文的论文**：OpenJev-RLCD 的 v2（PRDO）、Calibration-aware RL（2601.13284）仅摘要；Medium 测评 403；放射论文的 ECE 数字未核。
- **时效**：Decision Index、XDecideBench、JevBench 在数周内会变化，引用须固定 revision 或 sha256。
- 本报告生成过程中材料内部没有出现必须上网才能裁决的自相矛盾，因此**未做任何网络检索**。
