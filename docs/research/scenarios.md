# MetisCraft 业务场景总表与"能力 × 场景"矩阵（2026-11-04）

> 设计前调研材料（2026-11-04）。设计结论以 [`docs/design.md`](../design.md) 为准；文中引用的 coverage-map、`notes/`、`partial/`、`landscape/` 等是维护者本机的调研材料，不随仓库发布。

> 状态：完成（2026-11-04）。证据级别：source-verified / doc-verified / author-reported / unverified，未标注者按 unverified 对待。
> 范围：只读调研；未安装、未下载权重或数据、未训练。

## 0. 结论先行

1. **18 个场景**（5 个必选 + 13 个补充），每个场景有原语表述、数据（许可证/规模）、SOTA/基线、现有方案取舍、零样本余量、需求证据和 6 项评分。
2. **首批配方排序**：Schema Linking > 检索重排/RAG 筛选 > 意图识别/路由 > 内容审核 > 工具选择（见 §5）。前两项共用"逐候选标量头 + 共享前缀 + 分块"，是官方四原语和 ms-swift 单选读出都覆盖不到的部分，也就是差异化所在。实现顺序建议先做意图识别（最简单，用来跑通管线并产出同协议对照），再做 1、2。
3. **v0.1 范围建议**（§4.3）：noul / choice / score 的完整语义 + **multi_label + pointwise/listwise ranking** + 校准；**官方 Box（图像）不做**，只有 GUI 场景需要；span 与 pointer 放 v0.2。需要 multi_label 的场景 5 个，需要 ranking 的 3 个（检索类集中）。
4. **诚实的警告**：有标签时 encoder 通常胜出（CLINC：BGE-small + LR 0.948 > Jev 0.913；Banking77 零样本 Jev 约 0.82 对监督约 0.94），信息抽取、事实核查、重排等场景已有强专用 encoder。自训练的理由只能落在私有化、校准、多标签/排序和菜单可变，不能写成"更准"。
5. **许可证雷区**：Spider/BIRD/NFCorpus 为 CC-BY-SA；ToxicChat/BeaverTails/PKU-SafeRLHF/工单集为非商用；LLM-AggreFact 为禁止演绎；不能用 Jev 输出训练。详见 §6.1。
6. **要用户决定**：①v0.1 是否纳入 ranking（建议纳入 pointwise + ListNet）；②Schema Linking 的公开数据只能自己转换，工作量是否接受；③是否接受"意图识别"作为对照基线而非卖点。

## 1. 场景清单（18 个）

| # | 场景 | 必选 | 主要判断类型 | 是否需要扩展原语 |
|---|---|---|---|---|
| 1 | Schema Linking（表/列链接） | 是 | multi_label（逐候选 noul） | 多标签 |
| 2 | 检索重排 / RAG 段落筛选 | 是 | score + ranking | 排序 |
| 3 | 意图识别 / 模型路由 | 是 | choice（+ 置信度门控） | 否（+ 开放集拒识） |
| 4 | 内容审核 / 安全护栏 | 是 | noul / multi_label | 多标签 |
| 5 | 工具选择 / Agent 步骤决策 | 是 | choice / multi_label / ranking | 多标签、排序 |
| 6 | 实体匹配 / 实体消歧 | 否 | noul（成对）/ choice（候选） | 否 |
| 7 | 信息抽取（NER / 关系 / 属性） | 否 | 文本跨度（官方 Box 只管图像，需 span/pointer 扩展）/ choice | span |
| 8 | LLM-as-judge / 奖励打分 | 否 | score / choice（成对偏好） | 否 |
| 9 | 数据质量过滤 / 数据选择 | 否 | score / noul | 否 |
| 10 | 工单分流 / 客服分类 | 否 | choice / multi_label | 多标签 |
| 11 | 风控 / 欺诈 / 钓鱼检测 | 否 | noul + score | 否 |
| 12 | 推荐 / 电商搜索相关性 | 否 | score（序数）/ ranking | 排序 |
| 13 | Agent 继续或停止判断 | 否 | noul / choice | 否 |
| 14 | 事实核查 / NLI / 幻觉检测 | 否 | choice / noul（+ box 定位） | box |
| 15 | 医疗 / 法律编码（ICD、条款分类） | 否 | multi_label（大标签集） | 多标签、分层 |
| 16 | GUI / 网页元素定位 | 否 | choice（候选元素，pointer）/ box（截图坐标，官方 Box） | pointer / box |
| 17 | 简历 / 岗位匹配（招聘筛选） | 否 | score / ranking | 排序 |
| 18 | 行业 / 文档分类（SEC、新闻、合同） | 否 | choice / multi_label | 多标签 |

## 2. 逐场景卡片

通用约定：
- 证据级别写在括号里。`doc-verified` = 本轮读了 HF API 卡片元数据或 arXiv 摘要（链接固定到 revision/版本号）；`source-verified` = 来自上一轮已读源码或复算；`author-reported` = 作者自报；`unverified` = 记忆或二手，未核。
- HF 卡片信息抓取日期 2026-11-04，revision 取 API 返回的 sha 前 7 位。
- 官方原语（TypeSafe OpenAPI 0.2.0，source-verified）：Noul = 是/否概率；Choice = 在给定选项里选一个并给每个选项的概率；Score = 在给定等级上给期望分；Box = **图像**里的框（四条边占宽高的比例，需 Jev 1.14）。所以文本跨度抽取不属于官方 Box，要另做扩展（span/pointer）。
- 评分 6 项 1–5 分，含义统一为"越高越适合 MetisCraft 做"：F = 一次前向是否合适；M = 多标签/排序需求强度（5 = 官方原语填不了，正是差异化点）；D = 公开数据（可用、许可证干净、规模够）；R = 需求证据；E = 能否评测（有现成指标和 gold）；G = 4×3090 可行性。
- "为什么不直接用现有方案"一栏只接受 5 种自训练理由：私有化、领域精度、延迟、多标签/排序、按自家分布校准。

### 场景 1　Schema Linking（必选）

| 项 | 内容 |
|---|---|
| 用原语怎么表述 | 每个候选表/列一个 Noul（"回答此问题是否需要该列"），整库 schema 共享前缀；输出应为**多标签子集**（官方无此原语）。扩展：`multi_label`（逐候选独立 Bernoulli + 校准阈值）、可选"表→列"两段式。 |
| 关键公开数据 | Spider（xlangai/spider，train 7,000 / dev 1,034，CC-BY-SA-4.0，doc-verified，rev 0c35091）；BIRD mini-dev（birdsql/bird_mini_dev，CC-BY-SA-4.0，doc-verified，rev 5b256b5；完整 BIRD 的 train/dev 规模为记忆值 9,428/1,534，unverified）；Spider 2.0-lite 派生集 griffith-bigdata/GRAST-SQL-Spider2.0-lite（CC-BY-4.0，doc-verified）；现成的 BIRD 链接标注 griffith-bigdata/schema_linking_dataset（MIT，doc-verified，来源与质量未核）。tasksource/spider 的 `column` 配置只是单选 SELECT 列（train 2,911 行，source-verified，见 VL §1.1），不是完整链接。**没有 Jev 格式、带多标签 gold 的公开集**，gold 要从 gold SQL 解析（自己写转换器）。 |
| SOTA / 强基线 | RSL-SQL：双向 schema linking，BIRD 上严格召回 94%、输入列减少 83%，BIRD 执行准确率 67.2%、Spider 87.9%（GPT-4o；arXiv 2411.00073v2 摘要，doc-verified）；CHESS 严格召回 89.7%、MCS-SQL 89.8%（RSL-SQL 论文对比，经检索摘录转述，unverified）。小模型端：LitE-SQL（向量式链接 + 7B 微调，BIRD 72.10% / Spider 88.45%，arXiv 2510.09014v2，author-reported，检索摘要）；CodeS 1B–15B 开源（arXiv 2402.16347，doc-verified）。"Death of Schema Linking?"（arXiv 2408.07702）认为强推理模型下链接可省略——这是需求侧的风险，见下。 |
| 现有方案与不直接用的理由 | 向量检索/embedding 召回（LitE-SQL 类）、encoder 打分（RESDSQL 类，记忆，unverified）、prompt 大模型（RSL-SQL）、直接调 Jev API。自训练理由：**私有化**（库结构是机密）、**多标签**（Jev 只能逐列调用，成本随列数线性增长）、**按自家库分布校准**（召回优先，阈值要可调）、**延迟**（几百列的大库）。 |
| 零样本余量 | 无公开 Jev 零样本数字（unverified）。间接信号：Jev 在 Mind2Web 候选选择上 0.543 < 监督 0.8B 的 0.579（author-reported）。 |
| 需求证据 | 用户本人的业务场景；RSL-SQL 把"遗漏必需列"列为核心风险（arXiv 2411.00073 摘要）；LitE-SQL、CHS-SQL（arXiv 2607.22624，检索结果日期与编号对不上，需复核）都在做"小模型 + 链接"。反证：2408.07702 主张可省略链接。 |
| 评分 | F4 · M5 · D3 · R3 · E5 · G5 |

### 场景 2　检索重排 / RAG 段落筛选（必选）

| 项 | 内容 |
|---|---|
| 用原语怎么表述 | (a) 逐 (query, 段落) 一个 Noul/Score，按概率排序（TypeSafe 的 CLERC 重排做法）；(b) 列表式：一次前向对 top-k 候选出排序（需 `ranking` 扩展，Plackett–Luce/ListNet 损失）；(c) RAG 段落筛选 = 多标签子集（"哪些段落足以回答"）。 |
| 关键公开数据 | BEIR/NFCorpus（corpus 3,633 / queries 3,237，CC-BY-SA-4.0，doc-verified，rev b5026a0）；BeIR/msmarco（corpus 8,841,823 / queries 509,962，CC-BY-SA-4.0，doc-verified）；microsoft/ms_marco（HF 未声明许可证；官方条款限研究用途，记忆，unverified）；BRIGHT（xlangai/BRIGHT，CC-BY-4.0，各域约 100 条查询，doc-verified）；CLERC（jhu-clsp/CLERC，HF 未声明许可证，doc-verified）；Tevatron/msmarco-passage（Apache-2.0，doc-verified）；DI 公开套件含 BRIGHT（nDCG@10，source-verified）。 |
| SOTA / 强基线 | NFCorpus nDCG@10：BGE-base 0.3735、SPLADE-v3 0.3629（稠密/稀疏单阶段），Qwen3-8B 重排器（Tevatron 3.0）0.410（SOTA §2.6，author-reported；Anserini 的 5 个数字为 source-verified）。Qwen3 Embedding/Reranker 系列（arXiv 2506.05176v3，doc-verified，具体榜单数字未核）。TypeSafe cookbook：CLERC 40 查询、BM25 top-30，top-1 由 5% 到 18%（author-reported，自选子集）。开源决策模型项目没有一个报告 NFCorpus（SOTA §2.6 已查）。 |
| 现有方案与不直接用的理由 | cross-encoder/reranker（BGE-reranker、Qwen3-Reranker）、LLM listwise（RankZephyr 等，记忆）、Jev API。自训练理由：**排序**（reranker 只给标量，没有校准概率，无法做"够不够回答"的截断）、**校准**（把分数变成可设阈值的概率，用于 RAG 段落过滤）、**领域精度**、**延迟**（共享 query 前缀 + 段落分块）。诚实评估：通用相关性上 Qwen3-Reranker 级别的专用模型很强，MetisCraft 不该以"击败它们"为卖点，而是以"校准 + 段落集合选择"为卖点。 |
| 零样本余量 | CLERC 上 Jev 18%，远低于微调系统（具体上限未核）。余量大但样本小。 |
| 需求证据 | TypeSafe 官方 CLERC 重排与 RAG passages cookbook（source-verified，见 coverage-map (a)）；DI 公开套件列入 ToolRet/BRIGHT/ESCI。 |
| 评分 | F4 · M5 · D5 · R5 · E5 · G5 |

### 场景 3　意图识别 / 模型路由（必选）

| 项 | 内容 |
|---|---|
| 用原语怎么表述 | 意图 = 一个 Choice（几十到上百选项，需"以上都不是"/OOS 选项与置信度门控）；模型路由 = Choice（选模型/档位）或 Noul（"是否需要强模型"）+ 阈值；Arch-Router 式"域 + 动作"可写成两个 Choice。无需扩展原语，需要扩展的是**开放集拒识**与**大选项集**（选项 > 26 时字母 token 读出不够，见读出类 ②④）。 |
| 关键公开数据 | Banking77（PolyAI/banking77，train 10,003 / test 3,080，CC-BY-4.0，doc-verified，rev 1846984）；CLINC150（clinc/clinc_oos，plus 配置 train 15,250 / val 3,100 / test 5,500，CC-BY-3.0，doc-verified）；MASSIVE intent（mteb/amazon_massive_intent，Apache-2.0，doc-verified）；路由：RouterBench（withmartian/routerbench，HF 未声明许可证，10K–100K 行）、RouteLLM gpt4_dataset（Apache-2.0）、lmsys-arena-human-preference-55k（Apache-2.0，底层对话条款需另核）均 doc-verified。 |
| SOTA / 强基线 | Banking77 test 准确率：DeepOpen（Laya encoder 396M + 77 类头）94.04（arXiv 2610.31104v2，author-reported）；Llama-2-7B 94.35 无原始出处（unverified）；Strands Decider v14 0.983 但训练混合含该集（不可比，author-reported）。CLINC150：微调 BERT-base 0.940、BGE-small + 逻辑回归 0.948、Jev 1.13 0.913（arXiv 2610.00346v1，author-reported）。路由：RouteLLM（arXiv 2406.18665v4）、RouterBench（2403.12031v2）、Arch-Router 1.5B（arXiv 2506.16655v1）均 doc-verified（摘要级）；一篇 Jev 路由实测：Jev 中位延迟约 0.64–0.67 秒、约 $0.000025 每次，4 档复杂度 10/10 正确，对比 LLM 分类器 2.1/7.2 秒（dev.classmethod.jp，样本极小，author-reported）；另有 arXiv 2505.14524（Guarded Query Routing）称经典分类器可达 LLM 精度且延迟可低于 1 ms（摘要级转述，unverified）。 |
| 现有方案与不直接用的理由 | encoder 微调 / SetFit / embedding + 逻辑回归（已被证明在有标签时胜出：CLINC 上 BGE-small + LR 0.948 > Jev 0.913）、semantic-router、RouteLLM、Jev API。**诚实结论**：只要有几百条标签且选项固定，encoder 更便宜且更准。自训练理由只剩：**选项集经常变**（决策模型零样本跟随新菜单，encoder 要重训）、**校准概率 + 拒识**、**同一模型兼做多种判断**（路由 + 护栏 + 分流共用一个前向）。 |
| 零样本余量 | 大：Banking77 上 Jev 1.13 零样本 0.82（1,000 条，arXiv 2610.16655v1）对监督约 0.94（author-reported）；CLINC 0.913 对 0.94–0.948。 |
| 需求证据 | Jev 路由实测博客（上）；LangChain skills 的 decision-models 转换手册把路由作为首要替换对象（raw.githubusercontent.com/langchain-ai/langchain-skills/main/config/skills/langgraph-decision-models/references/conversion-playbook.md，未固定 commit，抓取摘录有乱码，unverified）；Banking77/CLINC 是 DI 与 typed-decisions 的核心集。 |
| 评分 | F5 · M2 · D5 · R5 · E5 · G5 |

### 场景 4　内容审核 / 安全护栏（必选）

| 项 | 内容 |
|---|---|
| 用原语怎么表述 | 每个风险类别一个 Noul（多标签，类别间不互斥）；或 Choice（safe/controversial/unsafe 三档，Qwen3Guard 式）；提示与回复各一次判断；流式护栏需要"逐 token/逐块"读出（扩展）。 |
| 关键公开数据 | WildGuardMix（allenai/wildguardmix，92K 例，ODC-By，gated=auto，doc-verified，rev d29c47f；规模见 arXiv 2406.18495v3 摘要）；Aegis 2.0（nvidia/Aegis-AI-Content-Safety-Dataset-2.0，CC-BY-4.0，doc-verified）；**含非商用限制**：ToxicChat（lmsys/toxic-chat，CC-BY-NC-4.0）、BeaverTails（CC-BY-NC-4.0）、PKU-SafeRLHF（CC-BY-NC-4.0），均 doc-verified；JailbreakBench JBB-Behaviors（MIT，doc-verified）；When2Call（nvidia，CC-BY-4.0）。 |
| SOTA / 强基线 | WildGuard（开源 7B，13 类风险，arXiv 2406.18495v3，doc-verified）；Qwen3Guard（三档严重度 + 流式，arXiv 2510.14276v1，doc-verified，具体数字未核）；Llama Guard 系列、ShieldGemma、Granite Guardian（记忆，unverified，本轮未查）。评测：AA Decision Index 含 WildGuardTest 占 25%，Jev 1.13 得 58（unverified，原页未开）。 |
| 现有方案与不直接用的理由 | 专用 guard 模型（WildGuard/Qwen3Guard/Llama Guard）、OpenAI Moderation、Jev API。自训练理由：**自家政策**（类别与阈值是业务定义，guard 模型的分类体系固定）、**多标签 + 校准**（按类别设拒绝阈值）、**私有化 / 延迟**（每请求必过，前向要小）。guard 模型已是"单次前向 + 类别"，竞争最直接——MetisCraft 的价值是"按自家政策配置 + 校准"，而不是再训一个通用 guard。 |
| 零样本余量 | 无直接 Jev 数字（AA 页未开）。PhishNChips 类相邻任务显示余量大（场景 11）。 |
| 需求证据 | TypeSafe 自一致性 cookbook、AA Decision Index 把 WildGuardTest 列为四项之一（unverified）；LangChain 转换手册把"串行 yes/no 护栏合并成一次多问请求"当典型用法（unverified）。 |
| 评分 | F5 · M4 · D4 · R5 · E5 · G5 |

### 场景 5　工具选择 / Agent 步骤决策（必选）

| 项 | 内容 |
|---|---|
| 用原语怎么表述 | 从 N 个工具里选 1 个 = Choice；选"一组并行调用的工具" = 多标签；大工具库 = 先召回 top-k 再 Choice（两段式）；"是否该调用工具/是否该澄清/是否拒答" = Choice（When2Call 四类）。 |
| 关键公开数据 | BFCL（gorilla-llm/Berkeley-Function-Calling-Leaderboard，Apache-2.0，doc-verified，rev 61fc060）；ToolRet（7.6k 检索任务、43k 工具，arXiv 2503.01763v2 摘要；mangopy/ToolRet-Tools 无许可证声明，doc-verified）；xLAM function-calling-60k（Salesforce，CC-BY-4.0，gated=auto，doc-verified）；When2Call（nvidia，CC-BY-4.0）；Mind2Web（osunlp，CC-BY-4.0，1K–10K 条，doc-verified，rev 17ece8e）；ToolBench/MetaTool 本轮 HF 接口 401，未核（unverified）。 |
| SOTA / 强基线 | ToolRet 的结论是现有 IR 模型"不擅长选工具"（标题即结论，doc-verified）。Mind2Web 候选选择 top-1：JevForge 0.8B 0.579（RLCD 风格 0.5875）> Jev-1.13 0.543（JevForge README，author-reported，候选集经缩小，source-verified 的是"数字出自 README"）。TypeSafe skill 推荐 cookbook 452 条（author-reported）。BFCL/When2Call 在 Unsloth 中被列为只评不训（SOTA/ANT 材料，source-verified）。 |
| 现有方案与不直接用的理由 | 函数调用微调（xLAM 类）直接生成调用、embedding 召回工具（ToolRet 说明召回弱）、Jev API、semantic-router。自训练理由：**多标签/排序**（并行工具、召回 + 精选）、**延迟**（每个 agent 步骤都要判断，生成式函数调用太慢）、**校准**（低置信度时转人工/澄清）。 |
| 零样本余量 | 有：JevForge 0.8B 监督版已超 Jev-1.13（同一划分，author-reported）。 |
| 需求证据 | JevForge、WebJev、GUI-Decisions、cua-s1 等仓库（coverage-map (c)(h)）；DI 公开套件含 ToolRet 和 BFCL；LangChain 转换手册（上）。 |
| 评分 | F4 · M5 · D4 · R5 · E4 · G4 |

### 场景 6　实体匹配 / 实体消歧

| 项 | 内容 |
|---|---|
| 用原语怎么表述 | 成对匹配 = 一个 Noul；"从候选里选出同一实体" = Choice（带"都不是"）；一条记录对多个候选 = 多标签。ComEM 论文的结论是 selecting（在候选中选）优于逐对 matching（arXiv 2405.16884v3，doc-verified，8 个 ER 数据集 × 10 个 LLM），正好对应 Choice + 共享前缀。 |
| 关键公开数据 | Abt-Buy（matchbench/Abt-Buy，许可证未声明）、WDC Products（wdc/products-2017，license: unknown）、GLAMI-1M 实体匹配集（zidcenek/GLAMI-Entity-Matching-Dataset，Apache-2.0）均 doc-verified，规模未取；Magellan/DeepMatcher 经典集（记忆，unverified）。TypeSafe 有 450 对实体对齐 cookbook（author-reported）。 |
| SOTA / 强基线 | Ditto（预训练 LM 序列对分类，比此前 SOTA 高至多 29% F1，两个公司数据集 789K/412K 条上 F1 96.5%，arXiv 2004.00584v3，doc-verified）；AnyMatch（小模型零样本，平均质量距 GPT-4 的 MatchGPT 4.4% 以内，推理成本低 3,899 倍，arXiv 2409.04073v2，doc-verified）；ComEM（上）。 |
| 现有方案与不直接用的理由 | Ditto/AnyMatch/Jellyfish 类专用模型、字符串相似度 + 阻塞（blocking）规则、Jev API。自训练理由：**私有化**（客户主数据不能外发）、**领域精度**（SKU/地址等领域行话）、**延迟/成本**（匹配对数巨大）。注意 AnyMatch 已经把"小模型 + 零样本"做到了，差异化要靠"校准概率 + 阻塞后候选集 Choice"。 |
| 零样本余量 | 中。零样本小模型已接近 GPT-4 4.4% 以内（AnyMatch，author-reported）；无 Jev 数字。 |
| 需求证据 | TypeSafe 实体对齐 cookbook；GLAMI 集的存在；数据整合是长期需求。Jev 专有需求证据偏弱。 |
| 评分 | F5 · M2 · D4 · R3 · E5 · G5 |

### 场景 7　信息抽取（NER / 关系 / 属性）

| 项 | 内容 |
|---|---|
| 用原语怎么表述 | 封闭类型的属性抽取 = Choice / Noul（"该字段是否出现"）；**跨度定位不属于官方 Box**（官方 Box 只在图像上，source-verified），需自定义 span 读出（起止指针，类似场景 16 的 pointer）。一次前向做到逐 token BIO 是编码器 NER 的强项，decoder 做它并不自然。 |
| 关键公开数据 | Few-NERD（DFKI-SLT/few-nerd，CC-BY-SA-4.0，supervised 配置 train 130,112 / val 18,817 / test 14,007，doc-verified，rev 205f3e9）；pile-mistral-v0.1（urchade，Apache-2.0）、NuNER（numind，MIT）均 doc-verified；CoNLL-2003（tner/conll2003，license: other，路透原始语料条款限制，doc-verified 到"other"）。 |
| SOTA / 强基线 | GLiNER（紧凑双向 encoder，任意实体类型并行抽取，arXiv 2311.08526v1，doc-verified）与 GLiNER2.x（coverage-map 提及 GLiNER2、GLiNER2.5-Decide，未细读）。 |
| 现有方案与不直接用的理由 | GLiNER 类 encoder 在此场景体量小、速度快、已开源，**MetisCraft 不应主打**。仅当任务是"文档级字段是否抽到/属于哪类"才与 Choice/Noul 重合。 |
| 零样本余量 | 本轮无数据。 |
| 需求证据 | 需求巨大但已被 GLiNER 覆盖；TypeSafe 的 extraction 配置与警方事故叙述编码（coverage-map (h)，浅）。 |
| 评分 | F2 · M3 · D5 · R3 · E5 · G3 |

### 场景 8　LLM-as-judge / 奖励打分

| 项 | 内容 |
|---|---|
| 用原语怎么表述 | 单条打分 = Score（1–10 级，官方上限 10）；成对偏好 = Choice(A/B/tie) 或两次 Score 相减（需处理位置偏差，PriDe 类，coverage-map 缺口）；多维 rubric = 多个 Score 共享前缀。 |
| 关键公开数据 | RewardBench（allenai/reward-bench，ODC-By，raw 5,123 / filtered 2,985，doc-verified，rev 168d848）；arena-human-preference-140k（lmarena-ai，CC-BY-4.0，135,634 条，doc-verified）；lmsys-arena-human-preference-55k（Apache-2.0）；HelpSteer3（nvidia，CC-BY-4.0）；Skywork-Reward-Preference-80K-v0.2（77,016 条，HF 未声明许可证，doc-verified）；JevJudge-Public v0.3（3,220 条，22 族，author-reported）。 |
| SOTA / 强基线 | Skywork-Reward-V2：8 个模型 0.6B–8B，26M 偏好对（取自 SynPref-40M，arXiv 2507.01352v3，doc-verified）；RewardBench 论文（arXiv 2403.13787v2，doc-verified）。具体榜单分数本轮未核。 |
| 现有方案与不直接用的理由 | 开源 reward model（Skywork-Reward-V2 等）、Prometheus/JudgeLM（记忆，unverified）、LLM-as-judge 提示、Jev。自训练理由：**按自家评分标准校准**（rubric 是业务定义）、**延迟/成本**（RL 训练在线打分要百万级调用）、**私有化**。reward model 竞争强，且它们输出标量，缺校准概率。 |
| 零样本余量 | 无 Jev 数字（JevJudge-Public 只被 JevAny 报告，低采用）。 |
| 需求证据 | TypeSafe 提供 judge 用途（coverage-map (h) bandits/JevJudge）；RLHF 生态对廉价打分的需求公认存在（定性）。 |
| 评分 | F5 · M2 · D5 · R5 · E4 · G4 |

### 场景 9　数据质量过滤 / 数据选择

| 项 | 内容 |
|---|---|
| 用原语怎么表述 | 教育价值/质量 = Score（0–5 序数，期望分作排序键）；"保留与否" = Noul + 阈值；多维（质量、毒性、语言）= 多个 Score/Noul 共享前缀。吞吐比精度更重要，**一次前向 + 长文档截断/分块**是关键。 |
| 关键公开数据 | HuggingFaceFW/fineweb-edu-llama3-annotations（467,424 条 LLM 标注，ODC-By，doc-verified，rev 72df4c9）；FineWeb 论文含教育质量过滤消融（arXiv 2406.17557v2，doc-verified）；DCLM（记忆，unverified，未查）。 |
| SOTA / 强基线 | FineWeb-Edu 分类器（embedding 加回归头，阈值 3，记忆，unverified）与 DCLM fastText 过滤器，均是极廉价基线。 |
| 现有方案与不直接用的理由 | fastText/embedding 回归头已足够便宜；LLM 标注质量更高但贵。自训练理由：**延迟**（万亿 token 吞吐）、**自家质量标准**。小模型决策头相对 embedding 回归的收益需实测（unverified）。 |
| 零样本余量 | 无公开数字。 |
| 需求证据 | FineWeb-Edu 是数据整理的标准做法（doc-verified 论文）；与 MetisCraft 自己的"数据处理"关联弱。 |
| 评分 | F5 · M2 · D4 · R4 · E4 · G5 |

### 场景 10　工单分流 / 客服分类

| 项 | 内容 |
|---|---|
| 用原语怎么表述 | 队列/优先级/情绪各一个 Choice 或 Score，共享前缀同一前向；一单多标签（多个问题类型）= multi_label；"是否升级人工" = Noul + 校准阈值。 |
| 关键公开数据 | Bitext 客服集（bitext/Bitext-Customer-Support-LLM-Chatbot-Training-Dataset，CDLA-Sharing-1.0，合成，10K–100K 条，doc-verified）；Tobi-Bueck/customer-support-tickets（CC-BY-NC-4.0，非商用，doc-verified）；Banking77 / CLINC（场景 3）；TypeSafe WorkflowEvals customer_service 204 例与 security_incidents 240 例（source-verified，rev fbe1ea5c 对应后者）。公开真实工单数据稀缺，且多为合成。 |
| SOTA / 强基线 | WorkflowEvals：Jev-1.13 在 4 个工作流上与 LLM 共识的动作一致率平均 67.76%，security_incidents 61.67%（source-verified，rev fbe1ea5c）。分类部分的强基线就是 SetFit / encoder 微调（见场景 3）。 |
| 现有方案与不直接用的理由 | 工单系统自带规则、SetFit/encoder、LLM 路由。自训练理由：**多标签**、**自家分布校准**、**私有化**（工单含客户数据）。 |
| 零样本余量 | 大：WorkflowEvals 一致率仅 61.67–67.76%（但参照是 LLM 共识，不是人工金标）。 |
| 需求证据 | TypeSafe 把客服工单作为 WorkflowEvals 四个工作流之一；eve-rlcd 工单集（coverage-map (d)）。 |
| 评分 | F5 · M4 · D2 · R4 · E3 · G5 |

### 场景 11　风控 / 欺诈 / 钓鱼检测

| 项 | 内容 |
|---|---|
| 用原语怎么表述 | 一个 Noul（钓鱼/欺诈概率）+ 一个 Score（严重度）+ 若干特征类 Noul（紧迫感、冒充、链接异常）共享前缀；校准概率直接进风控阈值。 |
| 关键公开数据 | PhishNChips（AreLit/PhishNChips，2,000 封合成邮件，license: other，doc-verified，rev 89afcc3）；ealvaradob/phishing-dataset（Apache-2.0，10K–100K，doc-verified）；表格欺诈数据（信用卡等）不适合语言模型，不收。 |
| SOTA / 强基线 | PhishNChips 上：Jev 1.14 0.8375（DI 公开分项，source-verified，rev c130faf）；Jev 1.13 0.6295（Tev1 配对重测，source-verified）；同一集不同协议差约 20pp；Luce（Qwen3-4B-Base + LoRA + 决策头，1,000 条标签）97.4%，ECE 0.010 对 Jev 0.154（自划分 500 条测试集，author-reported）；Tev1-4B 50.6%（source-verified）。 |
| 现有方案与不直接用的理由 | 传统分类器 + 规则 + 黑名单、LLM 提示、Jev。自训练理由：**按自家分布校准**（Luce 的 ECE 对比是教科书式例证）、**私有化**、**延迟**。 |
| 零样本余量 | 大且已有证据：Luce 用 1,000 标签把准确率推到 97.4%，而 Jev 零样本 0.63–0.84，协议不同（需用统一协议复测，见风险）。 |
| 需求证据 | PhishNChips 被 Hmm、Tev1、DI 采用（70 个条目已评，author-reported）；Luce 仓库。 |
| 评分 | F5 · M2 · D3 · R3 · E4 · G5 |

### 场景 12　推荐 / 电商搜索相关性

| 项 | 内容 |
|---|---|
| 用原语怎么表述 | 查询-商品相关性是 4 级序数（Exact/Substitute/Complement/Irrelevant）= Score 或 Choice；整页重排 = ranking；"是否同款" = Noul。 |
| 关键公开数据 | Amazon ESCI（tasksource/esci，Apache-2.0，train 2,027,874 / test 652,490，doc-verified，rev 8113b17）；DI 公开套件含 Amazon ESCI（source-verified）。 |
| SOTA / 强基线 | KDD Cup 2022 的获奖方案为 cross-encoder 集成（记忆，unverified）；本轮没有核到可引用的数字，不填。 |
| 现有方案与不直接用的理由 | cross-encoder / 排序学习（LambdaMART）、embedding 召回。自训练理由：**排序**、**按自家分布校准**（点击偏差）、**延迟**（每次搜索 top-k 打分）。 |
| 零样本余量 | 无数据。 |
| 需求证据 | DI 采用 ESCI（source-verified）；AutoTrust 有推荐与搜索用途（coverage-map (h)）。 |
| 评分 | F4 · M4 · D5 · R4 · E5 · G5 |

### 场景 13　Agent 继续或停止判断

| 项 | 内容 |
|---|---|
| 用原语怎么表述 | "任务是否已完成/是否继续" = Noul + 阈值；"下一步：继续/停止/求助/澄清" = Choice；序列级监控 = 每步一个 Noul，再接序贯检验把分数变成可保证的停止规则。 |
| 关键公开数据 | StepShield（9,213 条代码 agent 轨迹，1,278 训练对 + 7,935 测试，arXiv 2601.22136，检索摘要，author-reported，是否公开未核）；AgentProcessBench（1,000 轨迹、8,509 步人工标注，检索摘要，unverified）；EarlyEval（21,000+ 带结果标签的轨迹，LightGBM 成败分类器，检索摘要，unverified）；rmems/agent-honest-stop-episodes（Apache-2.0，63 次下载，质量未核，doc-verified 仅许可证）；TypeSafe WorkflowEvals agent_trace 111 例（source-verified）。**没有一个数据集直接给"每步是否该停"的标签**（检索结论），需自己从轨迹的最终成败反推。 |
| SOTA / 强基线 | E-valuator：把验证器分数转成序贯检验，在每步都有效，可提前终止失败轨迹（arXiv 2512.03109v1，检索摘要，unverified）；EarlyEval：LightGBM 成败分类器逐步运行，置信度过阈值即停（同上，unverified）。 |
| 现有方案与不直接用的理由 | 最大步数/规则、LightGBM 特征分类器、LLM 自评、Jev API。自训练理由：**延迟**（每个 agent 步骤都要判断）、**校准**（停止规则要概率保证）、**私有化**（轨迹含内部工具输出）。 |
| 零样本余量 | WorkflowEvals agent_trace 上 Jev-1.13 只是与 LLM 共识的一致率（4 个工作流平均 67.76%，source-verified），不是停止判断的准确率。 |
| 需求证据 | LangChain 转换手册（场景 3）；E-valuator、EarlyEval、StepShield 说明研究界在做（均为检索摘要）。 |
| 评分 | F5 · M1 · D2 · R4 · E3 · G5 |

### 场景 14　事实核查 / NLI / 幻觉检测

| 项 | 内容 |
|---|---|
| 用原语怎么表述 | (声明, 证据) 一个 Noul（有据可依）或三分类 Choice（支持/反驳/不足）；RAG 幻觉**定位**需要文本跨度（span 扩展，非官方 Box）；长上下文要证据分块。 |
| 关键公开数据 | HaluEval（pminervini/HaluEval，Apache-2.0，两配置各 10,000，doc-verified）；LLM-AggreFact（lytang，**CC-BY-ND-4.0**，gated=auto，dev 30,420 / test 29,320，doc-verified，禁止演绎，不宜转换后再发布）；RAGTruth（HF 上 wandb 副本未声明许可证，原仓库 401 未核；许可证记忆为 MIT，unverified）；VitaminC、MNLI、ANLI（Nimble/DI 公开套件，coverage-map (d)）。 |
| SOTA / 强基线 | LettuceDetect（ModernBERT，8k 上下文，比最佳模型小约 30 倍；large 在 RAGTruth 样本级整体 F1 79.22%，对照 gpt-4-turbo 63.4%，arXiv 2502.17125v1，作者自报，检索转述，author-reported）；MiniCheck（GPT-4 级事实核查，成本低 400 倍，arXiv 2404.10774v2，doc-verified 摘要；LLM-AggreFact 分数本轮未核）；Vectara HHEM（记忆，unverified）。 |
| 现有方案与不直接用的理由 | encoder 类（LettuceDetect、HHEM、MiniCheck 770M）已小而准。自训练理由：**领域精度**（企业文档）、**校准**（阈值）、**与同一前向的其他判断合并**。同样不该主打"击败 MiniCheck"。 |
| 零样本余量 | 无 Jev 数字；Nimble 公开套件 13 子集 Jev 1.13 macro 76.0%，含 VitaminC/MNLI/PAWS（author-reported）。 |
| 需求证据 | RAG 幻觉检测是共识需求；RAGTruth 为 ECE/校准研究常用。 |
| 评分 | F4 · M3 · D4 · R4 · E5 · G4 |

### 场景 15　医疗 / 法律编码（ICD、条款分类）

| 项 | 内容 |
|---|---|
| 用原语怎么表述 | 大标签集（ICD 约 8–9 万码，MIMIC-III full 约 8.9K 码，记忆，unverified）的多标签；需要**两段式召回**（先召回候选码再逐码 Noul）和长文本分块；法律用 multi_label（LexGLUE 的 EUR-LEX、ECtHR）。 |
| 关键公开数据 | LexGLUE（coastalcph/lex_glue，CC-BY-4.0，7 个任务，doc-verified，rev c23fdff；单任务规模本轮未逐个核）；MIMIC-III/IV（PhysioNet 需认证后使用，记忆，unverified，不能随开源仓库分发）；ContractNLI/CUAD（coverage-map (d)，未核许可证）。 |
| SOTA / 强基线 | PLM-ICD：预训练 LM 在 ICD 编码上的不足归因于大标签空间、长输入、领域错配三点，并提出对策（arXiv 2207.05289v1，doc-verified 摘要，具体 micro-F1 未核）。LexGLUE 基线（arXiv 2110.00976v4，doc-verified，数字未核）。 |
| 现有方案与不直接用的理由 | PLM-ICD 类标签注意力 encoder 是专用解法；decoder 一次前向读 8K+ 候选并不现实。自训练理由：**多标签**、**私有化**（病历）。但**一次前向与 4×3090 可行性都差**，建议后置。 |
| 零样本余量 | 无数据。 |
| 需求证据 | 医疗编码是长期产业需求；Jev 相关材料里有放射、医疗、保险垂直应用（coverage-map (h)，浅）。 |
| 评分 | F2 · M5 · D3 · R3 · E5 · G3 |

### 场景 16　GUI / 网页元素定位

| 项 | 内容 |
|---|---|
| 用原语怎么表述 | 给定 DOM 候选元素列表选目标 = Choice（pointer 读出，Kev/JevAny/Strands 路线）；截图上定位 = 官方 Box（需 Jev 1.14，多模态）。 |
| 关键公开数据 | Mind2Web（osunlp，CC-BY-4.0，doc-verified）；ScreenSpot（rootsautomation/ScreenSpot，Apache-2.0，test 1,272，doc-verified，rev 0be0878）；WebJev、GUI-Decisions、cua-s1（coverage-map，只读过 README）。 |
| SOTA / 强基线 | JevForge Mind2Web choice top-1 0.579（RLCD 风格 0.5875）> Jev-1.13 0.543（author-reported，见场景 5）。截图端 SOTA（OS-Atlas、SeeClick 等）本轮未查。 |
| 现有方案与不直接用的理由 | 专用 GUI grounding VLM；DOM 元素排序的 cross-encoder。自训练理由：**延迟**与**候选多**（一页数百元素）。需要 VLM 与长上下文，4×3090 上偏紧。 |
| 零样本余量 | 有（JevForge 0.8B 已超 Jev-1.13）。 |
| 需求证据 | 仓库层面的活跃度（coverage-map (c)）；商业需求偏向 computer-use agent。 |
| 评分 | F3 · M2 · D4 · R3 · E4 · G3 |

### 场景 17　简历 / 岗位匹配（招聘筛选）

| 项 | 内容 |
|---|---|
| 用原语怎么表述 | (简历, 岗位) 一个 Score（适配度）；候选人池排序 = ranking；硬性条件 = Noul。 |
| 关键公开数据 | cnamuangtoun/resume-job-description-fit（许可证未声明，doc-verified）；nonameee12233/job-resume-matching（MIT，小，多为合成）；batuhanmtl/job_resume_fit（license: other），均 doc-verified。公开集小、合成多、标注质量未核。 |
| SOTA / 强基线 | 无权威基准（本轮未找到，不编造）。 |
| 现有方案与不直接用的理由 | 关键词/ATS 规则、embedding 相似度、reranker。自训练理由：**私有化**（简历含个人信息）、**排序**。**合规风险高**：招聘筛选在多个法域属于高风险自动决策（欧盟 AI 法案等，记忆，unverified），建议只放在文档里提示，不做首批。 |
| 零样本余量 | 无数据。 |
| 需求证据 | coverage-map (h) 列招聘筛选为 Jev 垂直应用（浅）。 |
| 评分 | F5 · M3 · D2 · R3 · E2 · G5 |

### 场景 18　行业 / 文档分类（SEC、新闻、合同）

| 项 | 内容 |
|---|---|
| 用原语怎么表述 | 单标签 = Choice（可达 75 类，TypeSafe SEC cookbook）；多标签（合同条款、主题）= multi_label；长文档需截断/分块。 |
| 关键公开数据 | EDGAR-CORPUS（eloukas/edgar-corpus，Apache-2.0，约 22 万份 10-K，doc-verified，rev 7e90f0f）；LexGLUE LEDGAR（上）；AG News、DBpedia、TREC 等（Decision_Model_Benchmark 使用，source-verified）。 |
| SOTA / 强基线 | TypeSafe SEC cookbook：60 份 10-K、75 个 SIC 行业组，65.0%（39/60）；置信度 ≥ 0.9 的 30 条 90.0%（27/30）（author-reported，自选样本，jev-1.12）。AG News 94.8–95.4 一类强基线见 SOTA §2.4。 |
| 现有方案与不直接用的理由 | TF-IDF/encoder 微调（有标签时稳赢）、SetFit。自训练理由与场景 3 同：**选项集变化频繁**、**校准**。 |
| 零样本余量 | SEC 75 类零样本 65%，高置信子集 90%：说明**校准 + 弃权**比整体准确率更有价值。 |
| 需求证据 | TypeSafe 官方 cookbook；Decision_Model_Benchmark（arXiv 2610.03387，缺答案检测与误拒率）。 |
| 评分 | F5 · M3 · D4 · R4 · E5 · G5 |

<!-- cards-end -->


## 3. 评分总表（6 项 1–5 分）

F = 一次前向是否合适；M = 多标签/排序需求强度（越高越是官方原语填不了的差异化点）；D = 公开数据；R = 需求证据；E = 能否评测；G = 4×3090 可行性。分数是调研者判断，依据见各卡片。

| # | 场景 | F | M | D | R | E | G | 合计 |
|---|---|---|---|---|---|---|---|---|
| 1 | Schema Linking | 4 | 5 | 3 | 3 | 5 | 5 | 25 |
| 2 | 检索重排 / RAG 段落筛选 | 4 | 5 | 5 | 5 | 5 | 5 | 29 |
| 3 | 意图识别 / 模型路由 | 5 | 2 | 5 | 5 | 5 | 5 | 27 |
| 4 | 内容审核 / 安全护栏 | 5 | 4 | 4 | 5 | 5 | 5 | 28 |
| 5 | 工具选择 / Agent 步骤决策 | 4 | 5 | 4 | 5 | 4 | 4 | 26 |
| 6 | 实体匹配 / 消歧 | 5 | 2 | 4 | 3 | 5 | 5 | 24 |
| 7 | 信息抽取 | 2 | 3 | 5 | 3 | 5 | 3 | 21 |
| 8 | LLM-as-judge / 奖励打分 | 5 | 2 | 5 | 5 | 4 | 4 | 25 |
| 9 | 数据质量过滤 | 5 | 2 | 4 | 4 | 4 | 5 | 24 |
| 10 | 工单分流 | 5 | 4 | 2 | 4 | 3 | 5 | 23 |
| 11 | 风控 / 钓鱼 | 5 | 2 | 3 | 3 | 4 | 5 | 22 |
| 12 | 推荐 / 搜索相关性 | 4 | 4 | 5 | 4 | 5 | 5 | 27 |
| 13 | Agent 继续或停止 | 5 | 1 | 2 | 4 | 3 | 5 | 20 |
| 14 | 事实核查 / 幻觉检测 | 4 | 3 | 4 | 4 | 5 | 4 | 24 |
| 15 | 医疗 / 法律编码 | 2 | 5 | 3 | 3 | 5 | 3 | 21 |
| 16 | GUI / 网页元素定位 | 3 | 2 | 4 | 3 | 4 | 3 | 19 |
| 17 | 简历 / 岗位匹配 | 5 | 3 | 2 | 3 | 2 | 5 | 20 |
| 18 | 行业 / 文档分类 | 5 | 3 | 4 | 4 | 5 | 5 | 26 |

读表：
- 合计只作粗排，不替代判断。1 号 Schema Linking 合计只有 25，被"公开数据 3、需求证据 3"拖低；它仍列首位是因为它是用户的锚点业务，并且最能展示"逐候选多标签 + 自写转换器"。
- 需求证据列大多来自 TypeSafe 自家 cookbook、论文摘要或博客，**不是独立的付费需求证据**，R=5 不等于有人愿意为此买单。
- M 分最高（5）的是 1、2、5、15：恰好是官方四原语填不了的场景，也是 ms-swift 的 `generic_decision`（单选读出）覆盖不到的地方。

## 4. "能力 × 场景"矩阵

### 4.1 判断类型（行 = 场景，列 = 类型）

● = 主要形式，○ = 可选/辅助，— = 不需要。span 是文本跨度扩展，官方 Box 仅指图像框。

| # | 场景 | noul | choice | score | box | multi_label | ranking |
|---|---|---|---|---|---|---|---|
| 1 | Schema Linking | ○（逐候选） | — | — | — | ● | ○（列相关度排序，截断用） |
| 2 | 检索重排 / RAG 筛选 | ● | — | ○ | — | ○（段落集合） | ● |
| 3 | 意图识别 / 路由 | ○（需不需要强模型） | ● | — | — | — | — |
| 4 | 内容审核 / 护栏 | ● | ○（三档严重度） | ○ | — | ● | — |
| 5 | 工具选择 / Agent 步骤 | ○ | ● | — | — | ● | ○（召回后排序） |
| 6 | 实体匹配 | ● | ● | — | — | ○ | — |
| 7 | 信息抽取 | ○ | ○ | — | —（span 扩展） | ○ | — |
| 8 | LLM-as-judge | — | ●（成对） | ● | — | — | ○ |
| 9 | 数据质量过滤 | ○ | — | ● | — | — | — |
| 10 | 工单分流 | ○ | ● | ○ | — | ● | — |
| 11 | 风控 / 钓鱼 | ● | — | ○ | — | — | — |
| 12 | 推荐 / 搜索相关性 | ○ | ○ | ● | — | — | ● |
| 13 | Agent 继续或停止 | ● | ○ | — | — | — | — |
| 14 | 事实核查 / 幻觉 | ● | ● | — | —（span 扩展） | — | — |
| 15 | 医疗 / 法律编码 | ○ | — | — | — | ● | ○ |
| 16 | GUI 元素定位 | — | ●（pointer） | — | ●（截图） | — | — |
| 17 | 简历 / 岗位匹配 | ○ | — | ● | — | — | ● |
| 18 | 行业 / 文档分类 | — | ● | — | — | ○ | — |

按类型统计需求场景数（● 计 1，○ 不计）：noul 6（2、4、6、11、13、14）· choice 9 · score 5 · box 1 · multi_label 5（1、4、5、10、15）· ranking 3（2、12、17）。结论：**box 只有 GUI 一个场景，且要多模态，不进 v0.1；multi_label 的 ● 数与 score 并列，ranking 与 choice/noul 相比较少但集中在检索类。**

### 4.2 读出、损失、转换器、指标、效率特性

读出编号沿用任务背景的 7 类：①答案 token；②lm_head 行初始化的槽位头；③逐候选标量头；④pointer；⑤集合交互；⑥encoder 打分；⑦先生成再读。"建议读出"是首选，括号里是对照组。

| # | 场景 | 建议读出 | 损失 | 数据转换器 | 指标 | 效率特性 |
|---|---|---|---|---|---|---|
| 1 | Schema Linking | ③（对照 ⑥ encoder、①） | 逐候选 BCE（正例加权 / ASL），召回优先；事后温度或逐候选 Platt 校准 | Spider/BIRD：用 SQL 解析器（如 sqlglot，未试）从 gold SQL 取出所用表和列 → 逐候选 noul 或 multi_label 行 | strict recall（gold 列全中的样本占比）、precision、F-β(β=2)、输入列缩减率、逐候选 ECE、下游执行准确率（可选） | 共享前缀（整库 schema 作前缀）；大候选集分块（数百列）；两段式（先表后列） |
| 2 | 检索重排 / RAG 筛选 | ③（listwise 用 ⑤；对照 Qwen3-Reranker、cross-encoder） | pointwise BCE + listwise softmax CE / ListNet；足够性头用 BCE | BEIR qrels → (query, 段落, 相关度)；RAG 筛选需另造"集合是否足够回答"标签 | nDCG@10、MRR、Recall@k；段落筛选的 F1 与 ECE | 共享 query 前缀；候选分块；两段式召回（稠密召回 top-k → 决策模型重排） |
| 3 | 意图识别 / 路由 | ①（≤26 选项）；②（≤255 选项）；对照 ⑥ encoder + LR | CE + 温度缩放；OOS 选项或阈值拒识 | Banking77/CLINC/MASSIVE → choice，菜单随机重排以防位置偏差；路由：RouteLLM/RouterBench 偏好 → noul（"强模型更好"） | accuracy、macro-F1、ECE、OOS 召回与误拒率；路由用成本-质量曲线 | 共享菜单前缀；选项 >26 走 ② 或 ④ |
| 4 | 内容审核 / 护栏 | ②（多槽位一次读出）或 ③；对照 WildGuard / Qwen3Guard | 逐类别 BCE；三档用序数 CE | WildGuardMix/Aegis 2.0 → 逐类别 noul；提示与回复分别建行 | 逐类 F1、PR-AUC、固定召回下的误报率、ECE | 一次前向读多个槽位（共享前缀）；流式场景需逐块读出（后置） |
| 5 | 工具选择 / Agent 步骤 | ④ 或 ③ + 两段式 | 单选 CE；并行调用用逐候选 BCE | BFCL/xLAM → (任务, 工具库, gold 调用)；When2Call → 四类 choice；ToolRet → 召回 | top-1、集合 F1 / exact match、When2Call macro-F1、ECE | 工具库作前缀并缓存；大库分块；两段式召回 |
| 6 | 实体匹配 | ①（noul）；选择式用 ① choice | BCE / CE | 带标签的对 → noul；阻塞后候选 → choice（含"都不是"） | F1、PR-AUC | 左记录共享前缀；候选 choice 一次出结果 |
| 7 | 信息抽取 | ④（span 扩展） | 起止 CE | Few-NERD → 类型 + span | 实体 F1 | 无特别；不建议首批 |
| 8 | LLM-as-judge | ②（Score 槽位，1–10）；成对用 ① | CE + 期望分 MSE / EMD；成对用 Bradley–Terry | RewardBench/HelpSteer3/arena → 成对 choice，交换 A/B 做增广 | 偏好准确率、Spearman/Kendall、位置一致性、ECE | 共享 prompt 前缀；A/B 交换两次平均 |
| 9 | 数据质量过滤 | ③ 回归头或 ② Score | 序数 CE（CORAL 类）+ MSE | fineweb-edu 标注 → score 行 | 阈值处 F1、Spearman、吞吐（token/s） | 长文截断；大批量吞吐优化 |
| 10 | 工单分流 | ① choice + ② 多槽位 | CE + BCE | 工单 → (队列, 优先级, 情绪, 升级) | macro-F1、升级的 precision@recall、ECE | 同一前向读多字段 |
| 11 | 风控 / 钓鱼 | ① noul + ② Score | BCE + 校准 | PhishNChips → noul | PR-AUC、固定 TPR 的 FPR、ECE/Brier | 无特别 |
| 12 | 推荐 / 搜索相关性 | ③ 序数头或 ⑤ | 序数 CE + listwise | ESCI → 4 级相关度 | nDCG、4 类 macro-F1 | 共享 query 前缀；候选分块 |
| 13 | Agent 继续或停止 | ① noul | BCE + 序贯校准（共形/序贯检验） | 轨迹 + 终局成败 → 逐步行（无直接 gold，需反推） | AUROC、误停率、节省步数 | **轨迹前缀 KV 复用**（逐步增长的前缀） |
| 14 | 事实核查 / 幻觉 | ① noul / 3 类 choice；定位用 ④ | CE / BCE | RAGTruth/HaluEval/NLI → (证据, 声明) | BAcc、F1、ECE | 证据分块 + 聚合 |
| 15 | 医疗 / 法律编码 | ③ + 两段式（标签注意力 encoder 为对照） | 逐码 BCE | LexGLUE → multi_label；ICD 需受限数据 | micro/macro-F1、P@k | **两段式召回必需**、长文分块 |
| 16 | GUI 元素定位 | ④ pointer（截图用 Box 回归） | CE | Mind2Web → 候选元素 choice | 元素准确率、步骤成功率 | 候选分块 |
| 17 | 简历 / 岗位匹配 | ② Score | 序数 CE | 公开集小，需自建 | 与人工评分的相关性 | 无特别 |
| 18 | 行业 / 文档分类 | ① / ②（choice）+ ③（多标签） | CE / BCE | EDGAR-CORPUS、LEDGAR → choice 或多标签 | accuracy、macro-F1、ECE、选择性准确率（风险-覆盖曲线） | 长文截断/分块 |

### 4.3 对 v0.1 范围的含义

按场景覆盖面反推，需要的能力模块和支撑它们的场景数（只算前 5 个必选场景与得分 ≥ 26 的场景）：

| 能力模块 | 需要它的场景 | 建议 |
|---|---|---|
| noul / choice / score 的答案 token 或槽位读出（①②） | 3、4、6、8、9、10、11、13、14、18 | v0.1 必做；与 ms-swift 重叠，作为对照基线和校准管线的载体 |
| 逐候选标量头 + multi_label 损失（③） | 1、2、4、5、10、15 | **v0.1 必做**，这是差异化主体 |
| ranking 损失（listwise）与 nDCG 评测 | 2、12、17（5 的召回后排序） | v0.1 做 pointwise + ListNet；Plackett–Luce 视预算 |
| 共享前缀 + 大候选集分块 | 1、2、5、6、15 | v0.1 必做（pairs 布局；tree 后置，见决策点 5） |
| 两段式召回 | 5、15，也可用于 1、2 的大库 | v0.1 提供接口，不内置召回器 |
| pointer（④） | 5、16、7 | v0.2 |
| 官方 Box（图像） | 16 | 不做（需多模态与 Jev 1.14 对齐） |
| 校准（温度/Platt/逐候选阈值）与 ECE/Brier | 全部 | v0.1 必做，贯穿所有配方 |

## 5. 推荐首批场景配方

### 5.1 排序与理由

| 优先级 | 场景 | 读出 × 损失 | 理由 | 主要风险 |
|---|---|---|---|---|
| 1 | Schema Linking | ③ 逐候选 BCE，共享整库前缀，校准召回阈值 | 用户的锚点业务；多标签是官方原语和 ms-swift 单选读出都覆盖不了的差异点；没有现成 Jev 格式数据，转换器本身是可展示的成果；评测（strict recall/precision）可全自动，gold 来自 SQL | 公开数据只能 Spider/BIRD 自己转；2408.07702 主张强推理模型下可省略链接，需用"召回/成本"论证；CC-BY-SA 需处理再分发 |
| 2 | 检索重排 / RAG 段落筛选 | ③ pointwise + listwise，query 前缀共享，足够性头 | 与场景 1 共用同一读出和前缀机制，边际成本最低；数据最丰富（NFCorpus、MS MARCO、BRIGHT、ESCI）；nDCG 现成；校准与"段落集合筛选"是专用 reranker 缺的东西 | Qwen3-Reranker 等专用模型很强，不能以绝对 nDCG 取胜；MS MARCO 条款未核 |
| 3 | 意图识别 / 模型路由 | ① 答案 token（≤26）与 ② 槽位（≤255），CE + 温度缩放，OOS 拒识 | 实现最简单，适合做**管线冒烟测试和"同协议对照"**（Banking77/CLINC 上对比 encoder 与 Jev）；数据许可证干净；诚实结果可能是 encoder 更好——这本身是对用户有价值的结论 | 与 ms-swift 重叠；encoder 在有标签时胜出（CLINC：BGE-small + LR 0.948 > Jev 0.913） |
| 4 | 内容审核 / 安全护栏 | ② 多槽位一次读出，逐类别 BCE，逐类阈值校准 | 多标签 + 校准 + 自家政策，需求证据强；WildGuardMix（ODC-By）+ Aegis 2.0（CC-BY-4.0）可商用 | ToxicChat/BeaverTails/PKU-SafeRLHF 非商用，不能进正式权重训练；与专用 guard 模型正面竞争 |
| 5 | 工具选择 / Agent 步骤决策 | ④ pointer 或 ③ + 两段式 | 需求证据强，JevForge 已显示小模型可超 Jev；多标签（并行工具）和两段式是增量 | 需要 pointer 读出（v0.1 之外）和长工具库；ToolBench/MetaTool 数据未核 |

### 5.2 建议的实现顺序（与优先级不同）

1. 意图识别（3）：用最少代码跑通 数据 → 训练 → 校准 → 评测 全流程，并产出第一份"同协议对照"数字。
2. Schema Linking（1）：加入逐候选头、多标签损失、共享前缀和转换器。
3. 检索重排（2）：在同一读出上加 ranking 损失与 nDCG；复用分块。
4. 内容审核（4）：在槽位读出上加逐类别阈值校准。
5. 工具选择（5）：v0.2，随 pointer 一起。

### 5.3 紧随其后（v0.2 候选）

- 推荐/搜索相关性（12）：ESCI 数据干净，是 ranking 的第二个落点。
- LLM-as-judge（8）与文档分类（18）：数据多、读出已有，主要是转换器工作。
- 实体匹配（6）：与 AnyMatch 竞争，需要先看对照。

### 5.4 暂不做

- 信息抽取（7）：GLiNER 类 encoder 已覆盖，且需 span 扩展。
- 医疗/法律编码（15）：大标签集需要两段式与长文，数据受限（MIMIC 需认证）。
- GUI 定位（16）：多模态，4×3090 偏紧。
- 简历匹配（17）：数据弱、合规风险高。
- Agent 继续/停止（13）与风控（11）、数据过滤（9）、工单（10）：可以在文档里作为"同一读出的用法示例"，不单独做数据转换器。

## 6. 风险、许可证与未核实项

### 6.1 许可证与条款

| 项 | 问题 |
|---|---|
| Spider、BIRD mini-dev、NFCorpus、BeIR/msmarco、Few-NERD | CC-BY-SA-4.0：转换后的衍生数据若再分发需同许可（共享相同方式）；建议转换器只发布代码，数据由用户本地生成 |
| ToxicChat、BeaverTails、PKU-SafeRLHF、Tobi-Bueck 工单集 | CC-BY-NC-4.0：非商用，不能用于面向商用的正式权重 |
| LLM-AggreFact | CC-BY-ND-4.0：禁止演绎，不宜做格式转换后再发布 |
| MS MARCO、CLERC、RouterBench、RAGTruth（HF 副本）、Skywork 偏好集、ToolRet-Tools | HF 卡片未声明许可证，需读上游条款后才能纳入"开箱即用" |
| MIMIC | 需 PhysioNet 认证，不能随仓库提供（记忆，unverified） |
| 用 Jev 输出作训练标签 | 违反 TypeSafe 条款（coverage-map §2.3 已记 jimothy、Jevstiller）；所有配方的教师标签只能来自人工 gold 或许可允许的开源教师 |
| AGPL | Unsloth 的 decision*.py 是 AGPL-3.0-only，不能抄代码，只能对照接口 |

### 6.2 方法与证据风险

- **协议漂移**：同一个 PhishNChips，Jev 1.13 在 Tev1 协议下 62.95%，Jev 1.14 在 DI 协议下 83.75%，Luce 的 97.4% 是自划分 500 条；不同数字不可比，首批配方必须自己统一协议复测，不转引。
- **监督基线通常更强**：CLINC（0.948 对 0.913）、Banking77（~0.94 对 0.82）都是有标签的 encoder 胜过零样本 Jev。自训练的理由要落在私有化/校准/多标签/菜单可变，而不是"准确率更高"。
- **"需求证据"偏自报**：多数来自 TypeSafe 自家 cookbook、论文摘要、单篇博客。没有找到付费意愿数据。
- **Schema Linking 本身的前提被质疑**（arXiv 2408.07702）：强推理模型不做链接也能拿高分；MetisCraft 的论证要放在大库、低延迟、私有化场景。

### 6.3 本轮没有查、结论里不能当作"不存在"的项

- Llama Guard / ShieldGemma / Granite Guardian 的数字；BIRD / Spider 2.0 官方榜；CSpider、DuSQL 等中文 text-to-SQL 集；ToolBench、MetaTool（HF 接口 401，未取到）；RAGTruth 原仓库；Cohere/Jina/Voyage 的闭源 rerank；KDD Cup 2022 ESCI 获奖数字；DCLM；OS-Atlas/SeeClick。
- 本轮仅用 HF API 与 arXiv 摘要核实，所有"论文数字"都是摘要或检索转述（author-reported 或 unverified），没有读 PDF 表格。
- 检索结果里的 CHS-SQL（arXiv 2607.22624）编号与日期对不上，已降级为 unverified。

### 6.4 渠道记录

HF API（datasets 元数据，8 路并行，2 批）；arXiv export API（摘要，2 批）；WebSearch 4 次（Schema Linking、Jev 用例、LettuceDetect/MiniCheck、agent 停止）。未触碰 GitHub API（避免共用 token 限流）。"不存在"类结论（如"没有 Jev 格式的 Schema Linking 多标签数据集"）继承 VL §1.1，本轮 HF 搜索 `schema linking` 的前 5 条（habapchan/*、griffith-bigdata/schema_linking_dataset）为 BIRD 派生的链接标注，格式非 Jev，未改变结论。

<!-- complete -->
