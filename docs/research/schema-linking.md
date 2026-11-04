# Schema Linking 专题报告（MetisCraft）

> 设计前调研材料（2026-11-04）。设计结论以 [`docs/design.md`](../design.md) 为准；文中引用的 coverage-map、`notes/`、`partial/`、`landscape/` 等是维护者本机的调研材料，不随仓库发布。

> 作者：小克（schema-linking 检索代理）；日期 2026-11-04；状态：完成。条目见本机调研目录的 `landscape/scenario-schema-linking.jsonl`。
> 证据级别：source-verified / doc-verified / author-reported / unverified。数字未注明协议的一律视为 author-reported。

## 1. 结论先行

1. **schema linking 天然是"逐候选多标签加校准"**，正好落在 Jev 官方原语（Noul/Choice/Score/Box）之外，也是 ms-swift 和 Unsloth 的决策训练没有覆盖的场景。用它做 MetisCraft 的示范场景，定位成立。
2. **最接近的先例是 IBM 的 ExSL**（2501.17174）。它用 decoder-only LLM（DeepSeek-Coder-6.7B），候选列依次排在一条因果序列末尾，在标记 token 处读出，损失是 BCE。BIRD-dev 列级 PR AUC 93.77，生成式只有 69.18；速度是生成式的 20 倍以上。这就是我们的 ③ 逐候选标量头，只是换了一种更便宜的布局。代码没找到。
3. **v0.1 建议用 ③-list 布局**（ExSL 式，普通因果注意力），**tree 不进 v0.1**。在 schema linking 上，tree 和 list 计算量同量级；list 不需要 FlexAttention，还能直接用推理引擎的按库 prefix cache。顺序敏感度要实验确认（§5.8）。
4. **"Death of schema linking" 的争论有明确边界**：强闭源 LLM 且 schema 放得进上下文时，过滤是净负收益（Distyl 和 E-SQL）；大库、小模型、低延迟、Spider 2.0 级别（超过 1,000 列）时仍然必需。共识是**高召回筛选加兜底**，所以 MetisCraft 应输出**校准过的逐列概率**，让业务按召回预算选阈值。业内没有人报告逐列校准，这是差异点。
5. **数据**：没有现成的逐列多标签、按库划分的 schema linking 数据集（HF API 加 verify-landscape 两个渠道确认）。要自己写 Spider、BIRD、CSpider（以及 DuSQL）→ multi_label 的 converter：用 sqlglot `qualify` 从 gold SQL 推导表和列。Spider 和 BIRD 是 **CC-BY-SA-4.0**，派生数据要沿用 SA。
6. **指标口径要固定**：Recall@k、SRR（RSL-SQL 的 strict recall，BIRD 上 94%）、F6、列级 ROC/PR AUC、逐列 ECE 和 Brier，再加下游 EX 对照（全 schema、预测子集、gold 子集）。各家口径不同，外部数字不能直接放在一起比。
7. **数据契约**只加可选字段（level、parent、meta、group.split_key、roles、provenance），旧 JSONL 仍然合法。同一配方换掉 converter 就能用于**工具选择**。
8. **要用户决定**：①实际业务约束（§4.1）；②v0.1 是否接受"list 主推、tree 推迟"；③DuSQL 和 Chase 许可证没核实，是否纳入。

## 2. 方法与 SOTA

核对方式：GitHub 元数据和 HEAD 用 `gh api` GET 拿（as_of 2026-11-04）；论文摘要用 arXiv API；ExSL 和 LinkAlign 的表格用 WebFetch 读 arXiv HTML，由小模型抽取，所以记为 doc-verified，没有逐格人工核对。没有读原文表格的数字一律记 author-reported（只来自摘要）或 unverified（凭记忆）。

### 2.1 一张表

| 方法 | 问题形式（读出类） | 模型 | schema linking 指标与数字 | 端到端 | 证据 | 开源 / 许可证 |
|---|---|---|---|---|---|---|
| **RESDSQL**（AAAI'23，arXiv 2302.05965） | 逐表、逐列二分类后按分数排序，取 top-k 表和列送给 T5 生成（③/⑥，cross-encoder） | RoBERTa-large / XLM-R：列 token 先 BiLSTM 池化，再做表-列 cross-attention，接 2 类 MLP 头。损失是 focal loss（γ、α 可配） | ExSL 论文把它当基线复测，Spider-dev 列级 ROC AUC 99.39、PR AUC 96.51、F6 97.61；BIRD 上没有数 | Spider-dev EX 80.9（ExSL 表 5，用的是同一个下游生成器） | 头结构 source-verified（`utils/classifier_model.py`、`classifier_loss.py`）；数字 doc-verified | [RESDSQL@7472f7a](https://github.com/RUCKBReasoning/RESDSQL/tree/7472f7a51fdd054d8139b1bc2627d955aff855e4)，MIT，282★，2024-05 后不再更新 |
| **CodeS**（SIGMOD'24，2402.16347） | 沿用 RESDSQL 式的 schema item classifier，筛 top-k 表和列；值检索用 BM25 加 LCS；生成用 1B–15B 的 SQL 预训练 LM | StarCoder 系增量预训练 | 摘要没有给 schema linking 数字；"top-k 表 6、列 10"是记忆，未核实 | 摘要只说在 Spider、BIRD 上达到 SOTA | 有 `schema_item_filter.py` 和 `train_schema_item_filter.py`，source-verified；数字 unverified | [codes@11a9f1c](https://github.com/RUCKBReasoning/codes/tree/11a9f1ceb292d6e9887990e7b108204d94c82cb0)，Apache-2.0；权重许可证按 StarCoder（OpenRAIL-M），未核实 |
| **DIN-SQL**（NeurIPS'23，2304.11015） | 用 few-shot prompt 让 LLM 生成 schema links（①/⑦，生成式） | GPT-4 / Codex | Spider-dev 列级 ROC AUC 96.54、PR AUC 92.33（ExSL 表 3）；LinkAlign 表 3 的 BIRD-dev P/R/EM 为 79.9/55.7/13.1 | Spider test EX 85.3（摘要） | doc-verified / author-reported | [repo@0474801](https://github.com/MohammadrezaPourreza/Few-shot-NL2SQL-with-prompting/tree/0474801616130413b024fd008b016a190684649e)，MIT |
| **CHESS**（2405.16755） | Schema Selector 智能体分三步：逐列过滤、选表、选列。前面有 Information Retriever 做 LSH 加向量的值检索（⑦，LLM 逐项判断） | GPT-4 / 开源 LLM | 摘要：在工业规模 schema 上 token 减少约 5 倍、准确率提高约 2%；没有 recall 数 | BIRD test EX 71.10（摘要） | author-reported | [CHESS@3d6e835](https://github.com/ShayanTalaei/CHESS/tree/3d6e835f858d26885d21d4bc0215aeecf855efbe)，Apache-2.0 |
| **RSL-SQL**（2411.00073） | 双向链接：前向先用 LLM 在全 schema 上剪枝，后向从初步 SQL 里解析出用到的列，两路取并集；再在完整模式和精简模式之间二选一投票（⑦） | GPT-4o / DeepSeek | 摘要：BIRD 上 **strict recall 94%**，输入列数减少 83%。strict recall（文中叫 SRR）= 全部 gold 元素都被召回的样本比例。LinkAlign 复测 BIRD-dev P/R/EM 为 78.1/77.5/27.7（口径不同，见 §3.4） | BIRD-dev EX 67.2（GPT-4，LinkAlign 表 5） | author-reported / doc-verified | [RSL-SQL@fb1dcf0](https://github.com/Laqcce-cao/RSL-SQL/tree/fb1dcf02923e5fd89171cc6cfe1ae3e4e5bc3b94)，Apache-2.0 |
| **E-SQL**（2409.16751） | 不做过滤，把相关表、列、值和条件写回问题（question enrichment），再加候选谓词增强 | GPT-4o；小模型用 DeepSeek-Coder-7B | 摘要：用最强的闭源 LLM 时，**schema filtering 没有正收益** | BIRD test EX 66.29；DeepSeek-Coder-7B 不微调在 BIRD-dev 上 56.45 | author-reported | [E-SQL@72ec6a3](https://github.com/HasanAlpCaferoglu/E-SQL/tree/72ec6a38945b6f8ce85e373e159fe86f0a862d2f)，**没有 LICENSE**（GitHub 返回空 spdx） |
| **LinkAlign**（EMNLP'25，2503.18596） | 两步：先从多个库里选库（多轮语义增强检索、隔离无关库），再在库内定位表和列。有 Agent 和 Pipeline 两种模式（检索 + ⑦） | 链接用 GLM-4-air；向量用 bge-large-en-v1.5；生成用 DeepSeek-V3/R1、Qwen-72B | 多库设置（LA/EM/Recall）：BIRD 83.4/22.1/64.9；单库 BIRD-dev P/R/EM 77.1/79.4/29.0（Agent） | Spider 2.0-Lite 33.09（DeepSeek-R1，投稿时排第一） | doc-verified | [LinkAlign@c036f38](https://github.com/Satissss/LinkAlign/tree/c036f38e0930843de9f000bf69b0bf79df79733a)，MIT；AmbiDB 的许可证没写 |
| **XiYan-SQL**（2411.08599） | 用 M-Schema 半结构化表示 schema；多个生成器集成，再用微调的选择模型挑 SQL。schema linking 由检索加 LLM 选列完成（细节没核） | Qwen2.5-Coder 系 SFT（XiYanSQL-QwenCoder） | 没有给 schema linking 数字 | BIRD test 75.63、Spider test 89.65（摘要） | author-reported | [XiYan-SQL@603deda](https://github.com/XGenerationLab/XiYan-SQL/tree/603dedac706d57ece47bc30d02f90744a537b6a0) Apache-2.0，1019★；[M-Schema](https://github.com/XGenerationLab/M-Schema) Apache-2.0；`XiYanSQL-QwenCoder` 仓库**没有 LICENSE** |
| **ExSL 抽取式 schema linking**（IBM，2501.17174） | **最接近 MetisCraft ③**：完整 schema 和问题放在前面，后面再把候选列各自用 « » 包起来列一遍；取每列开、闭标记处的末层向量拼接，经线性头得到逻辑值。ExSLf 对 5 种 SQL 角色（select/join/condition/order/group）做多标签。损失是 BCE；gold 由 mo-sql-parsing 解析 SQL 得到 | DeepSeek-Coder-6.7B，全参数微调。LR 5e-6，bs 16，2 epoch，输入最长 3000 token，1×A100-80G 训 9–10 小时。schema 超长时按表分块，结果取并集 | Spider-dev 列级 ROC AUC 99.79、PR AUC 98.52、F6 98.45；BIRD-dev 列级 ROC 99.35、PR 93.77、F6 96.21；BIRD-dev 表级 P/R 95.86/93.94。速度是生成式的 20 倍以上（只有这一句定性说法） | Spider-dev EX 82.4（ExSLf）；BIRD-dev 63.23。对照：gold 列 73.40，全 schema 未单独报 | doc-verified | 文中说会开源代码，**没找到仓库**（GitHub 搜索没有，未用第二渠道确认），记 unverified-existence |
| **CRUSH4SQL**（2311.01173） | 先让 LLM "幻觉"出一个最小 schema，再用它做多路稠密检索，按**集合**覆盖率选子集（检索，⑤ 的思想） | GPT-3 加稠密检索 | 自建三个基准：SpiderUnion（4,502 个元素）、BirdUnion（798）、真实数据仓库 SocialDB（17,844）；摘要只说 recall 显著更高 | — | author-reported | 代码没核 |
| **The Death of Schema Linking?**（Distyl，2408.07702） | 观点：只要 schema 放得进上下文，就**完全不做 schema linking**，改做增强、选择和纠错 | GPT-4o 等 | — | BIRD test 71.83（当时第一） | author-reported | 没核开源 |
| **Is Long Context All You Need?**（Google，2501.12372） | 用 gemini-1.5-pro 长上下文直接塞全 schema、示例值、文档，研究准确率和时延的权衡 | Gemini | — | — | author-reported | — |
| **AutoLink**（AAAI'26） | 智能体自主探索并扩展 schema，面向可扩展的 schema linking | — | 没核 | — | unverified | [wzy416/AutoLink](https://github.com/wzy416/AutoLink)，Apache-2.0，73★ |

### 2.2 "还要不要 schema linking"的争论

- **否定派**：Death of Schema Linking（BIRD 71.83）和 E-SQL 都认为，用最强的闭源 LLM、schema 又放得进上下文时，过滤是净负收益：漏掉一列就整题失败，多放无关列模型能忍。Google 的长上下文研究给出同一方向的证据。
- **肯定派**（两派其实不矛盾）：
  1. schema 放不进上下文时。Spider 2.0 的库常有 1,000 列以上；CRUSH4SQL 的 SocialDB 有 17,844 个元素；工业数据仓库更大。
  2. 小模型或本地模型。E-SQL 自己就说 7B 模型从相关信息中获益明显；ExSL 的 BIRD-dev 数据显示，gold 列 73.40 对比 ExSLf 63.23，链接质量对下游影响大。
  3. 成本和时延。CHESS 报告 token 减少 5 倍。
  4. 下游需要一个"可解释的候选子集"，例如做权限控制或给用户看依据。
- **共识**：从"精确过滤"转向**高召回筛选**。指标看 recall、SRR、F6（β=6 重召回）；同时保留兜底，比如 RSL-SQL 在完整和精简两种模式之间投票、ExSL 用阈值控制 P/R 权衡。**这正是 MetisCraft 能做的：输出校准过的逐列概率，阈值和 top-k 由业务按召回预算选，不要输出一个硬集合。**

### 2.3 Spider 2.0 时代的做法

Spider 2.0（632 个工作流题，BigQuery、Snowflake 等，库常超过 1,000 列）的主流解法已经是**智能体式探索**：读元数据、查文档、试跑 SQL。代表有 Spider-Agent、ReFoRCE、LinkAlign Agent、AutoLink。LinkAlign 表 1 中，Spider 2.0-Lite 最高 33.09（DeepSeek-R1）、ReFoRCE+o1-preview 30.35，传统的 CHESS 只有 3.84、DIN-SQL 1.46（doc-verified）。在这种设定下，schema linking 退化成两个子问题：**库或表的检索**（多库选库），以及**智能体每一步该看哪些元素**。一个快速、校准过的逐元素打分器，在这两处都可以当"第一道筛子"，也可以当智能体的工具。MetisCraft 不该去做端到端 text-to-SQL。


## 3. 数据集与指标

### 3.1 公开基准

"每库表/列数"一列凭记忆，标 unverified。converter 写好后应由我们自己统计，写进 `dataset.yaml`。

| 基准 | 规模 | 每库表 / 列（量级） | 语言 | 许可证 | 划分 | 证据 |
|---|---|---|---|---|---|---|
| **Spider 1.0**（1809.08887） | 10,181 问 / 5,693 条 SQL / 200 库 / 138 领域 | 平均约 5 表、约 27 列（unverified） | 英 | CC-BY-SA-4.0（HF `xlangai/spider`，rev `0c350918`） | **按库划分**：train 7,000（140 库）、dev 1,034（20 库）、test 2,147（40 库）；鲁棒性变体有 DK 535、Syn 1,034、Realistic 508（以上数字来自 ExSL 表 1） | doc-verified |
| **BIRD**（2305.03111） | 12,751 对 / 95 库 / 33.4 GB / 37 领域；带 evidence（外部知识） | 平均约 7 表、约 50 列（unverified）；有脏值 | 英 | CC-BY-SA-4.0 | train 约 9.4k、dev 1,534、test 隐藏（unverified）；按库划分。另有 `birdsql/bird_mini_dev`（rev `5b256b5a`，CC-BY-SA-4.0） | 规模 author-reported；划分 unverified |
| **Spider 2.0**（2411.07763） | Lite 547 题（BigQuery 214、Snowflake 198、SQLite 135）；Snow 547；DBT 68 | 常超过 1,000 列，README 写"超过 3,000 列" | 英 | 仓库 MIT（`cafb867`）；HF `xlangai/spider2-lite` 标 MIT | 只有评测集，没有训练集。**官方提供 gold tables**（`methods/gold-tables`，用了要注明 oracle tables）和 gold SQL | source-verified（README L58–L88、L30） |
| **CSpider**（1909.13293） | 把 Spider 的问题翻成中文，**schema 仍是英文** | 同 Spider | 中问英表 | 沿用 Spider 的许可证（unverified）；`taolusi/chisp` 仓库没有 LICENSE | 与 Spider 一致（unverified） | author-reported |
| **DuSQL**（EMNLP 2020，百度） | 23,797 对 / 200 库 / 813 表；大量行列计算类问题 | 平均约 4 表 | **中问中表** | 没找到许可证（千言 LUGE 发布，可能仅限研究，unverified） | 按库划分（unverified） | doc-verified（ACL Anthology 摘要，经搜索交叉核对） |
| **Chase**（ACL 2021，西交大） | 多轮上下文相关的中文 text-to-SQL；"约 5.5k 段对话、17.9k 问、280 库"是记忆 | — | 中 | 未核 | 未核；仓库 `xjtu-intsoft/chase` 用 GET 返回 404 | unverified |
| **BEAVER**（2409.02038 v3） | 9,128 对，来自真实企业查询日志加专家合成；812 张表，19 个领域；有 5 个子任务的人工标注 | 企业级 | 英 | 仓库 `peterbaile/beaver-may-2025` MIT（数据条款没核） | — | author-reported。摘要 v3 称"GPT-5.2 智能体只有 10.8%，给 oracle 子任务提示后到 30.1%" |
| CRUSH4SQL 的三个基准 | SpiderUnion 4,502、BirdUnion 798、SocialDB 17,844 个 schema 元素 | 把多个库并起来当一个大库 | 英 | 没核 | — | author-reported |
| AmbiDB（LinkAlign） | 合成的歧义多库数据 | — | 英 | 没写 | — | doc-verified |
| **SynSQL-2.5M**（OmniSQL，2503.02240） | 250 万条样本，16,000 多个合成库 | — | 英 | Apache-2.0（HF `seeklhy/SynSQL-2.5M`，rev `cca2c84c`） | 只有训练集 | doc-verified |

### 3.2 HF 上的 schema linking 数据集（11-04 查 HF API，`search=schema-linking / schema_linking`）

- `habapchan/bird_sql_schema_linking`（rev `a3adf578`，**没有许可证标签**，下载 54）：字段有 `db_id, schema(DDL), question, output, evidence, linking_prompt, reasoning, solution`。这是 SFT 用的生成格式，不是逐列标签。同作者还有 5 个变体（`bird_schema_linking*`、`bird-schema-linking-train-sft`），都是生成或推理格式。
- `griffith-bigdata/schema_linking_dataset`（rev `2b818c3f`，MIT，下载 26）：datasets-server 解析 `train.json` 失败，结构不明。
- `tasksource/spider` 的 `column` 配置（rev `0bbb0290`，CC-BY-SA-4.0）：train 2,911 行，**单选** SELECT 列（verify-landscape §1.1）。
- 结论与 verify-landscape 一致：**没有逐列多标签、按库划分、带校准评测的 schema linking 数据集**。两个渠道（HF API 搜索 3 个关键词 + 先前 verify-landscape 的 HF 加 TS 检索）都没找到，可以下"不存在"的结论。

### 3.3 从 SQL 推导表 / 列 gold（sqlglot）

sqlglot（MIT，`f7a3672`，9.7k★）的 `optimizer/qualify.py`、`qualify_columns.py`、`scope.py`、`lineage.py` 都在（source-verified）。转换流程：

1. 解析：`sqlglot.parse_one(sql, read=dialect)`。Spider 和 BIRD 用 `sqlite`；Spider 2.0 用 `bigquery` 或 `snowflake`。
2. 限定：用库的 `MappingSchema`（表 → 列 → 类型）调用 `qualify(expr, schema=..., expand_stars=True)`。这一步把裸列名解析到具体表，展开 `*`，处理别名和 CTE。
3. 收集：遍历 scope，取 `exp.Table` 得到真实表名（排除 CTE 和派生表），取 `exp.Column` 得到 `table.column`，统一小写。可选：按所在子句给每列标**角色**（select / join / where / group / order / having），对应 ExSL 的 ExSLf 多标签。
4. 边界情况，转换时要显式记录：
   - `COUNT(*)` 只有表、没有列，表算正例，列都是负例；
   - `SELECT *` 展开后会让全部列变成正例，建议单独打标记，或在训练时剔除这类样本；
   - 大小写和带引号的标识符；
   - BIRD 有带空格的反引号列名；
   - 解析失败的样本要计数，并写进 `dataset.yaml`。
5. **gold 有偏差**：从 SQL 推出的 gold 只是"一种写法用到的列"，等价写法可能用别的列，比如两张表上都有的外键。评测要报"表级"和"列级（含外键等价）"两个口径。可以参考 Spider 2.0 的官方 gold tables。

ExSL 用的是 mo-sql-parsing；RESDSQL 和 CodeS 的预处理脚本也能抽出 gold，可以用来交叉核对。

### 3.4 指标约定（建议 MetisCraft 固定以下口径）

| 指标 | 定义 | 出处 |
|---|---|---|
| 表 / 列 Recall@k | 每题取 top-k，gold 中被召回的比例，再对题求平均 | 通用 |
| **SRR（Strict Recall Rate）** | 每题 gold 全部被召回则记 1，再对题求平均。按 top-k 或阈值报 | RSL-SQL（94%，BIRD） |
| 选中列数 / 压缩率 | 平均保留多少列；RSL-SQL 报的是"输入列减少 83%" | RSL-SQL |
| Fβ（β=6） | 重召回的 F | ExSL |
| ROC AUC / PR AUC（列级，微平均） | 与阈值无关 | ExSL、RESDSQL |
| EM（集合完全匹配）/ LA（选库准确率） | 多库设置 | LinkAlign |
| **逐列校准**：ECE / Brier / reliability 图 | 把每个 (题, 列) 当成一次二元预测；另报"每题至少一个 gold 漏掉"的概率是否校准 | 文献里没人报，**MetisCraft 的差异点** |
| 下游 EX | 固定生成器（例如 Qwen2.5-Coder-7B），对比全 schema、预测子集、gold 子集三种输入 | ExSL 表 5、6 的做法 |

注意 LinkAlign 的 Recall 是逐题平均的元素召回率，RSL-SQL 的 strict recall 是逐题全中率，两者口径不同，**不能放在一起比**。


## 4. 工业规模

公开方法在"上千张表、中文字段、低延迟"场景下的共同做法如下。没有标证据的是小克的工程判断，未验证。

1. **分层漏斗**。先选库或主题域（LinkAlign 的 LA），再选表，最后选列。每一层都设召回预算，比如表层要求 SRR 不低于 98%，再交给下一层。CRUSH4SQL 和 LinkAlign 都是这个思路。第一层用 BM25 加向量检索（LinkAlign 用 bge-large-en-v1.5；中文可以换 bge-m3），保证候选数不超过几百，后面两层用决策模型打分。
2. **值索引**。问题里出现的实体值（例如"北京"、"2024 年 Q3"）往往是定位列的最强信号。CHESS 用 LSH 加向量检索数据库里的值，CodeS 用 BM25 加 LCS。匹配到的值要作为候选列的附加文本（`examples` 或 `matched_values`）放进输入。
3. **schema 文本的丰富化**。中文业务库的列名常常是拼音缩写或英文代码（如 `khmc`、`cust_lvl_cd`），真实语义在注释里。M-Schema（Apache-2.0）的做法是把注释、类型、主外键和示例值都序列化进去；XiYan-DBDescGen 可以自动生成缺失的描述。所以候选文本必须带 `comment`、`type` 和示例值，契约要留出这些字段。
4. **超长 schema 分块**。ExSL 按表贪心地装进上下文窗口，各块结果取并集（doc-verified）。决策模型的分块打分天然可以并行。
5. **低延迟**。schema 对每个库是静态的，按"schema → 问题 → 候选标记"的顺序排布时，**schema 部分的 KV 可以按库缓存**（vLLM / SGLang 都有 prefix caching）。这样每次查询只需算问题和候选列表的几百到几千个 token，一次前向就能得到全部列的概率。比生成式 linking 快 20 倍以上（ExSL 的定性说法）。这一点要靠 §5 的布局设计来保证。
6. **业务兜底**。阈值按召回预算设定，置信度低时回退到"整张表全部列"或"全 schema"，这是 RSL-SQL 双模式投票的简化版。这里需要**校准过的概率**，而不是排序分数。

### 4.1 待用户补充：实际业务约束

- 库、表、列的规模：单库最多多少表，单表最多多少列，有多少个库或主题域。
- 字段命名：中文名、拼音缩写还是英文代码？注释覆盖率多少？有没有数据字典或指标口径文档？
- 有没有历史"问题 → SQL"日志可以做 gold？量有多大？能不能出机器？
- 时延和吞吐预算：例如 P95 不超过多少毫秒、QPS 多少；部署在什么 GPU 或 CPU 上。
- 下游是谁：自研 LLM 生成 SQL，还是交给人看？召回和精度哪个更贵？
- 多轮上下文（类似 Chase 的追问）占比多少？
- 方言：MySQL、Hive、ClickHouse 等。另外，权限和脱敏是否限制示例值进入模型。


## 5. MetisCraft 配方

### 5.1 数据契约扩展（向后兼容旧 JSONL）

旧契约的 `{id, task_id, input:{query, context?}, candidates:[{id,text}], supervision:{kind:"candidate_labels", labels}}` 保留，只增加**可选**字段：

```jsonc
{
  "id": "bird-dev-0042", "task_id": "schema_linking",
  "input": {"query": "...", "context": {"evidence": "...", "dialect": "sqlite", "history": []}},
  "group": {"kind": "database", "id": "bird/california_schools", "split_key": "bird/california_schools"},   // 按库划分的键
  "structure": {                          // 库级上下文，按 group.id 去重存放，行内只放引用也可以
    "tables":  [{"id": "t:schools", "name": "schools", "comment": "..."}],
    "foreign_keys": [["c:frpm.CDSCode", "c:schools.CDSCode"]]
  },
  "candidates": [
    {"id": "t:schools", "level": "table", "text": "schools: ..."},
    {"id": "c:schools.City", "level": "column", "parent": "t:schools",
     "text": "City", "meta": {"type": "TEXT", "comment": "...", "pk": false, "examples": ["Fresno"], "matched_values": []}}
  ],
  "supervision": {"kind": "multi_label", "labels": {"t:schools": 1, "c:schools.City": 1},
                  "roles": {"c:schools.City": ["where"]}, "default": 0},
  "provenance": {"source": "bird@<rev>", "gold_from": "sqlglot==<ver> qualify", "parse_ok": true, "star_expanded": false}
}
```

要点：

- `level` 和 `parent` 表达表/列层级。
- `meta` 放类型、注释、主外键和示例值；序列化模板由配方决定，参考 M-Schema。
- `group.split_key` 让划分器**强制按库划分**，校准集和测试集都必须是没见过的库。
- `supervision.kind` 新增 `multi_label`；`roles` 可选，对应 ExSLf。
- `provenance` 记录 gold 是怎么推出来的，满足宪法 §4.1 的可追溯要求。

### 5.2 候选生成

- 小库（不超过 200 列，Spider、BIRD 都属于这种）：**全量候选**，表和列同时打分。
- 大库：第一阶段用 BM25、bge-m3 和值索引，召回 top-T 表；第二阶段对这 T 张表的全部列打分。第一阶段的 SRR 要单独报告，它是整个系统的召回上限。
- 训练时的负例就是全部非 gold 列，不需要采样。BIRD 一个库约 50 列，正例约占 5–10%，量级未核。大库可以加入同表的难负例和外键对侧列。

### 5.3 读出对比（针对 schema linking）

| 方案 | 布局与计算量（每题 N 列，schema 前缀 P token） | 优点 | 缺点 | 文献对应 | 建议 |
|---|---|---|---|---|---|
| ③-pairs：每个候选一行 | N 条序列，每条约 P 个 token，总量约 N·P。BIRD 上约为 50×2k = 100k token | 实现最简单，与旧版一致 | 太贵，且每列看不到其他候选 | 旧 Metis pairs | 只作为正确性参照 |
| ③-tree：共享前缀树加逐列标量头 | 约 P + N·c 个 token，c 为每列文本长度；需要块对角掩码（FlexAttention） | 与 pairs 逐位等价，候选之间互不可见，**结果与候选顺序无关** | 依赖 FlexAttention，SDPA 没有原生支持；导出到 vLLM/SGLang 要自定义 | 旧 Metis dense tree | 可以作为 v0.2 的加速和等价布局 |
| **③-list（ExSL 式）：候选依次排在因果序列末尾，在标记 token 处读出** | 约 P + N·c 个 token，**普通因果注意力即可** | 和 tree 一样便宜；不需要特殊掩码；推理引擎的 prefix caching 直接可用；有 ExSL 的强结果（BIRD 列级 PR AUC 93.77） | 后面的候选能看到前面的候选，**结果依赖顺序**。缓解办法：训练时打乱顺序，推理时做 K 次排列取平均（可选） | ExSL | **v0.1 主推** |
| ④ pointer | 在候选上做 softmax | 适合"恰好一个"的判断，如选库、选主表 | schema linking 是多标签，softmax 不合适；改成 sigmoid pointer 就等于 ③-list | Kev、JevAny | 只用于选库或选表的子任务 |
| ⑤ 联合集合头 | 在 ③ 的候选向量上叠加 1–2 层候选间 self-attention，加入表-列 cross-attention 和外键边 | 能建模共现关系（外键成对出现）和层级一致性 | 多一个模块，校准更难 | RESDSQL 的表-列 cross-attn、Clef、CRUSH4SQL 的集合覆盖 | v0.1 做成 ③-list 上的可选插件；收益要靠消融证明 |
| ⑥ cross-encoder | 每对 (问题, 列) 独立编码，如 bge-reranker-v2-m3 或 RoBERTa | 小而快，支持中文，是强基线 | 每列看不到完整 schema，只能靠拼接表名和注释 | RESDSQL、GLiClass | 作为**基线**，不作为主线 |

### 5.4 目标

- **多标签 BCE**：表和列两级，按正例率设 `pos_weight`，或用 focal（RESDSQL 用 focal γ）。
- **排序项**：同一题内，正列对负列做 pairwise logistic 或 listwise softmax（正例对全体），直接优化 recall@k。旧版的 BCE + pairwise 可以直接沿用。
- **层级一致性**（可选）：惩罚 `max(0, p(col) − p(table))`。推理时也可以直接用 `p(col) ← min(p(col), p(table))`。
- **角色多标签**（可选）：做 ExSLf 式的 5 个角色头。
- **校准**：训练之后在**没见过的库**组成的校准集上，按 level 分组拟合温度。可选的升级是按库大小分桶。

### 5.5 评测

- 必报：表级和列级的 Recall@k（k=5/10/20）、SRR@k、SRR@τ，以及在目标 SRR（如 95%）下的平均保留列数；列级 ROC AUC 和 PR AUC；F6；逐列 ECE 和 Brier，附 reliability 图；"整题漏召回"的概率校准。
- 选报：下游 EX。固定一个本地生成器（如 Qwen2.5-Coder-7B-Instruct），比较全 schema、预测子集和 gold 子集，复现 ExSL 表 5、6 的对照。
- 数据：Spider dev 和 test、BIRD dev（用 train 训练）；CSpider 测跨语言；DuSQL 测中文字段（许可证待核实）；Spider 2.0-Lite 的 gold tables 只做表级零样本评测，**注明 oracle 口径**。

### 5.6 基线

1. BM25 和 bge-m3 余弦相似度（不训练）。
2. bge-reranker-v2-m3 零样本，以及在 BIRD 和 Spider 上微调后的版本（⑥）。
3. 用 RESDSQL 的分类器在同一数据上重训（MIT，RoBERTa-large）。
4. 生成式 linking：用 Qwen2.5-Coder-7B 提示它"列出相关列"。只有硬集合，没有概率，所以只能报 P/R。
5. 外部报告的数字，例如 ExSL 的 PR AUC、RSL-SQL 的 SRR。只能**标注口径后引用**，不能当作同协议对照。

### 5.7 效率与 4×3090 算力（估算，未测）

- 序列长度：BIRD 用 M-Schema 序列化，加问题和候选列表，约 2–4k token；ExSL 的上限是 3000。Spider 约 1–2k。
- 训练量：BIRD train 约 9.4k 题 × 3k ≈ 28M token 每 epoch，Spider 7k × 1.5k ≈ 10M。Qwen3-1.7B/4B 加 LoRA、bf16、梯度检查点，按每卡每秒 2–3k token 粗算，4 张卡一个 epoch 不到 1 小时。0.6B 可以全参数训练。**这些是估算，需要实测**；正式跑之前按宪法红线 3 先问用户。
- 推理：③-list 每题一次前向。按库缓存 schema 前缀后，每次查询只算问题和候选，几百到几千 token。
- 对比：pairs 布局要多算约 N 倍（BIRD 约 50 倍），是不可行的。

### 5.8 tree 布局该不该进 v0.1

**结论：不进。v0.1 用 ③-list，tree 作为 v0.2 的等价加速布局。** 理由：

1. 在 schema linking 上，③-list 的计算量与 tree 同量级，又不需要 FlexAttention 和自定义导出；
2. 有 ExSL 的公开强结果背书；
3. tree 唯一的独有优势是"与候选顺序无关、与 pairs 逐位等价"，用打乱顺序训练加多排列推理基本能补上。

这条要靠实验确认：在 BIRD dev 上比较 ③-list 单次推理、③-list 4 次排列平均、③-pairs（抽样子集）三者的列级 PR AUC 和 ECE。如果顺序敏感度（同一列在不同排列下的概率标准差）超过 0.05，就提前做 tree。保留旧版的 `shared_token_prefix` 工具函数和等价性测试（salvage 文件里标为 keep）。


## 6. 旧版映射与推广到工具选择

### 6.1 旧版 → 新工具包

| 旧版 | 新工具包 | 处理 |
|---|---|---|
| 合成的 Schema Linking 示例 | `scenarios/schema_linking/`：Spider、BIRD、CSpider 的 converter（sqlglot），加上 `dataset.yaml` 和 `SHA256SUMS` | 重写；合成示例只留作单元测试 |
| JSONL 契约 `candidate_labels` | 同一契约，增加 `level/parent/meta/group/roles/provenance`（§5.1）；`kind` 新增 `multi_label` | 扩展，旧行仍然合法 |
| Qwen3 Base + MLP ScoreHead | Readout `candidate-scalar`（③），基座可替换 | 保留 |
| pairs 布局 | Layout `pairs`，作为正确性参照 | 保留 |
| dense tree 布局 | Layout `tree`（v0.2，FlexAttention） | 推迟；等价性测试保留 |
| （无） | Layout `list`（ExSL 式，标记 token 读出） | **新增，v0.1 主推** |
| 读候选的末 token | 读候选标记 token；可选开、闭标记拼接（ExSL） | 改动 |
| BCE + pairwise | `multi_label_bce` + `pairwise` / `listwise`，可选 `hierarchy` | 保留并扩展 |
| （无） | 按 level 做温度校准；评测 recall@k、SRR、F6、AUC、ECE | 新增 |

### 6.2 同一配方推广到工具选择

结构完全同构：

- 库 → 工具集或 MCP 服务器；表 → 工具或服务器；列 → 工具或其参数；
- gold SQL → 调用轨迹里实际用到的工具集合；
- 指标：SRR 对应"所需工具全部召回"，recall@k 不变，逐工具校准不变；
- 大工具库同样先检索再打分，工具描述可以按服务器做前缀缓存。

数据来源可以参考 ToolBench、MetaTool、ToolRet（DI 的评测口径里有，见 verify-landscape）。这些数据的许可证这次没有核。配方里只有 converter 和候选文本模板需要换。这是"一个配方服务多个业务场景"的有力示范，适合写进 v0.1 的文档。

## 7. 风险与未知

1. **竞品**：ms-swift `--task_type decision` 和 Unsloth 已有决策训练，但它们的读出头绑定在 model_type 上，没有多标签和逐候选的 schema linking 场景（verify-landscape）。transformers #50054 和 TRL #8040 合并之后，MetisCraft 应该兼容它们的导出格式。差异点要落在场景、评测和校准上，**不要重复造训练器**。
2. **gold 噪声**：从 SQL 推出的 gold 不唯一（等价写法、外键两侧都可以）。`SELECT *` 和 `COUNT(*)` 的处理会直接影响指标。BIRD 的 gold SQL 本身也有已知错误（社区多次报告，未核）。
3. **许可证**：Spider 和 BIRD 是 CC-BY-SA-4.0，由它们转出的数据集要**沿用 SA**。DuSQL、Chase、CSpider 的许可证没有核实。E-SQL 和 XiYanSQL-QwenCoder 的仓库没有 LICENSE。HF 上 habapchan 系列没有许可证标签。不能用 Jev 的输出做标注，否则违反 TypeSafe 条款。
4. **ExSL 代码没有找到**：只查了 GitHub 搜索，arXiv HTML 里也没有链接，复现要自己实现。数字是小模型从 HTML 抽取的，写进正式材料前要人工核对原文表格。
5. **顺序敏感度未知**：③-list 对候选顺序的敏感程度没有公开数据，§5.8 的实验要先做。
6. **Spider 2.0 的现实**：端到端的瓶颈在智能体探索和方言，单靠 schema linking 提升有限。MetisCraft 只能宣称自己是"第一道筛子"或"工具"。
7. **算力估算未实测**；正式训练要先过宪法红线 3。
8. **没覆盖到的**：CodeS 的具体 schema linking 数字、Chase 的规模和许可证、BIRD 的划分数字、AutoLink 的方法和数字、各数据集的平均表列数。这些都标了 unverified，下一轮用 2 个渠道补。


## 附：本轮补充

- Chase 论文页已核：https://aclanthology.org/2021.acl-long.180/（标题匹配）；规模和许可证仍未核。
- DuSQL 规模经搜索交叉核对：https://aclanthology.org/2020.emnlp-main.562
- 条目：本机调研目录的 `landscape/scenario-schema-linking.jsonl`，共 26 条。
- 过程说明：更新文首状态行时误用了一次 `sed -i`，只改了本报告自己的一行，没有碰其他文件。

<!-- complete -->
