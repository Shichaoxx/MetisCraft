# 数据注册表补充调研（v0.1）

> 设计调研材料（2026-11-04）。设计结论以 [`docs/design.md`](../design.md) 为准。

核对日期 2026-11-04。只读核对：Hub API、数据卡、datasets-server（info / first-rows）、GitHub 原文件、官方 OpenAPI；未下载数据集正文，未运行第三方代码。原始响应留在维护者本机，不随仓库发布。

证据级别：**source-verified** 读自仓库文件，或服务端按文件算出的元数据（Hub API 的 sha 与文件列表、datasets-server、`sources.yaml`、manifest、源码）；**doc-verified** 读自数据卡、官方文档、OpenAPI；**author-reported** 作者、论文、排行榜自报，未复现；**unverified** 未核对，附 URL。未另注时，revision、split 规模、字段为 source-verified，许可证与条款为 doc-verified。revision 取前 8 位，全文见 §6。缩写：TS = tasksource-jev，OJ = Open-Jev，LL = LocalLLaMA typed-decisions，DI = Decision Index。

## 1. 结论

**官方格式属实**：`questions:{题名:{type,instructions,criteria}}`，`model` 必填（doc-verified）。依据 OpenAPI 0.2.0，2026-11-04 重取自 `api.typesafe.ai/openapi.json`，与 11-03 保存的副本逐字节相同，sha256 `3ca93b6e6b9aa8d7820531b78c66ed7a9798b11e6e163ad96b140fb3d75fbcf4`。

- `model`、`state`、`questions` 必填；`questions` 是以自定题名为键的对象（至少 1 题），题名不发给模型（官方文档）。各类型字段见 §4.1。
- “数组 + `options` / `levels`”是 OpenAI Decisions、SGLang 的写法，不是 TypeSafe 的（这两种本次未重新核对）。

**多标签基准**：推荐 **GoEmotions `simplified`**（Apache-2.0，库内 parquet，28 标签，54,263 条，BERT-base macro-F1 0.46）；次选 LexGLUE `eurlex`（CC-BY-4.0，100 标签，65,000 篇，有榜单，但文档长、标签需另配描述）；备选 Civil Comments（CC0，软属性，需下采样）。MultiEURLEX、SemEval-2018 E-c、Reuters-21578、Jigsaw 不选（§3）。干净的工具选择多标签公开集没有找到。

**首批名单建议**：

- 必收：BoolQ（noul）、Banking77 与 CLINC150（choice）、SST-5（score）、NFCorpus（ranking）、GoEmotions（multi_label）；Open-Jev `release-v2-redistributable`、LocalLLaMA typed-decisions、tasksource-jev `default`。
- 可选：MNLI、AG News、MS MARCO v1.1、LexGLUE `eurlex`、Civil Comments。
- 只评测：Decision Index（DI）公开套件（本地重建）。JevBench 输入封存，无法注册。
- 条款：AG News、MS MARCO 仅限非商业研究；NFCorpus 上游只明示学术用途免费（Hub 标签写 CC-BY-SA-4.0，二者冲突）；SST-5 未声明许可证。用这四者训练的权重不标“可商用”。

**对设计稿 §3.4 的更正**：

1. Open-Jev 没有原生多标签题：12 个配置的 `kind` 只有 choice、noul、score（manifest 统计）；“独立标签”只能由同组多个 noul 合成。
2. LocalLLaMA 的 `state` + `questions` 缺必填 `model`，不是完整请求体。
3. tasksource 与 DI 同源不止 11 个：DI 表 44 项（计分 37）中 15 项有对应源（计分 12 项；4 项仅按名称匹配），full 训练行合计 77,069，占 3.0%（§5.1）。
4. tasksource 含 ChaosNLI（MNLI-matched 开发集 1,599 对），与 `multi_nli` 的 validation_matched 直接重叠（§5.2）。
5. 三份 Jev 类数据集的既有数字全部复核一致；仅 tasksource `filtered-full` 的卡片行数与 datasets-server 不符。

## 2. 首批数据集核对表

### 2.1 主流数据集

split 规模经 datasets-server `info` 与卡片元数据交叉核对，一致。

| 数据集 | id @ revision | 许可证（YAML = API） | 配置与规模 | 字段 / 标签 → 判断类型 | 已知问题 |
|---|---|---|---|---|---|
| BoolQ | `google/boolq` @35b264d0 | cc-by-sa-3.0 | train 9,427 / validation 3,270 | question、passage、answer(bool) → noul | 无带标签 test；calib、dev 须从 train 切；衍生数据继承 share-alike |
| MNLI | `nyu-mll/multi_nli` @da70db2a | cc-by-3.0、cc-by-sa-3.0、mit、other；正文：主体按 OANC，FICTION 另有 CC BY-SA / BY | train 392,702 / validation_matched 9,815 / validation_mismatched 9,832 | premise、hypothesis、genre、label(3) → choice(3) | 无 test（`nyu-mll/glue`:mnli 另有 test_matched 9,796 / test_mismatched 9,847，标签不公开，未核）；许可证混合；`genre` 可作 split_key |
| Banking77 | `PolyAI/banking77` @18469841（2026-10-28） | cc-by-4.0 | train 10,003 / test 3,080 | text、label(77) → choice(77) | 无 validation；10-28 才由加载脚本改为 parquet（DI 文档仍写脚本形式），须钉 revision |
| CLINC150 | `clinc/clinc_oos` @155b9c71 | cc-by-3.0（正文“待补”） | small 7,600 / imbalanced 10,625 / plus 15,250 train；三者 validation 3,100、test 5,500 | text、intent(151，含 oos，索引 42) → choice(150) + oos | 三配置共用 validation / test，manifest 须写明配置；oos 映射为 allow_null 待定（推荐：映射） |
| AG News | `fancyzhx/ag_news` @eb185aad | YAML unknown；正文：供学术界研究及“任何非商业活动” | train 120,000 / test 7,600 | text、label(4) → choice(4) | 无 validation；仅非商业 |
| SST-5 | `SetFit/sst5` @e51bdcd8 | 未声明（卡片无 YAML 头） | train 8,544 / validation 1,101 / test 2,210 | text、label(0–4)、label_text → score(5) | 卡片仅数行：句子级，划分取自第三方仓库 AcademiaSinicaNLPLab/sentiment_dataset；上游 SST 条款未核 |
| NFCorpus | `BeIR/nfcorpus` @b5026a0e + `BeIR/nfcorpus-qrels` @a451b3b2 | 两库 cc-by-sa-4.0（BEIR 统一标注）；项目页（搜索摘要转述，author-reported）：学术用途免费，其他用途见 NutritionFacts.org 条款 | corpus 3,633；queries 3,237；qrels train 110,575 / validation 11,385 / test 12,334 行 | `_id`、title、text；qrels：query-id、corpus-id、score(int) → ranking(候选 = 文档，分级) | qrels 在另一库（tsv）；卡片正文把 nfcorpus 写成“Argument Retrieval”（复制错误）；test 323 个查询（BEIR 卡片表）；两处许可证冲突 |
| MS MARCO | `microsoft/ms_marco` @a47ee7aa | YAML 无；上游 Terms：仅限非商业研究 | v1.1 train 82,326 / validation 10,047 / test 9,650；v2.1 808,731 / 101,093 / 101,092 | query、passages{is_selected, passage_text, url}、answers → ranking(候选 = 段落，二元) | QA 版：每题 ≤10 段，多数恰 1 个正例（v1.1 train 前 46 行：40 行 1 个、4 行 2 个、2 行 0 个）；v1.1 test 前 47 行带标签（v2.1 未抽样） |

### 2.2 Jev 类

| 数据集 | id @ revision | 许可证 | 配置与规模（复核） | 已知问题 |
|---|---|---|---|---|
| tasksource-jev | `tasksource/tasksource-jev-typed-decisions` @8df2e5c9（2026-10-24） | YAML other；行级 license、license_use（best effort）。754 源：commercial 440 / non-commercial 55 / unspecified 259 | default：train 999,832 / validation 15,000 / test 15,000（与既有记录一致；与 full 共用 validation / test）；full 2,550,012；filtered-full 1,304,043（卡片写 1,305,478）；extraction（span）与 vision（图像）不属 v0.1 的判断行 | 前 100 行全是硬标签（前 1,000 行按源轮转，不代表整体）；36 源（full 训练行 250,000，9.8%）按投票 / 均值聚合，是软标签；test 含变体与打包题，按样本评测须 `variant == direct` |
| Open-Jev | `ZefanCai/Open-Jev` @c67699e1（2026-09-20） | cc0-1.0，仅覆盖作者生成内容；customer-control 的简短题面取自 TypeSafe 文档，许可证未核实 | 12 配置。默认 release-v2-redistributable：train 79,116 / calibration 4,672 / validation 3,723 / test 10,356 / ood 15,701，共 113,568（与既有记录一致）；kind choice 25,923 / noul 70,474 / score 17,171。browser-drone-expansion（160,797）包含前者，配置不可相加 | test、ood 标签公开；metadata_json 含特权标签，不得入模；Wikispeedia 行（2,253）因许可证剔除；多为受控合成任务 |
| LocalLLaMA typed-decisions | `LocalLLaMA/typed-decisions` @c9e8c416（2026-11-03） | apache-2.0 | 4 个工作流各 train 300 / test 100；all：train 1,200 / test 400（与既有记录一致）；每 case 5 题，test 共 2,000 个决策 | gold 是 3 次教师采样均值，教师未公开（约 4B 级），测的是与教师的一致度（author-reported）；自一致率 0.735 不是上限（在 train 上拟合的模型已达 0.826）；README 随榜单频繁改动，revision 易变 |

行格式见 §4.1。除判断字段外：tasksource 每行带 `example_id`、`source`、`source_index`、`variant`、`split`、`license`、`license_use`；Open-Jev 带 `split`、`source`、`record_json`（原始整行）、`original_line_number`；LocalLLaMA 带 `workflow`、`factors`、`label_agreement`、`n_questions` 与每题平铺列（`<题名>__label` 等）。

## 3. 通用多标签基准候选与取舍

标准：许可证干净、能钉 revision 且数据在库内（非加载脚本）、标签 ≤100、约 5k–200k 条、有公认基线。

| 候选 | id @ revision | 许可证 | 标签 | 规模 train / val / test | 基线（author-reported） | 取舍 |
|---|---|---|---|---|---|---|
| GoEmotions `simplified` | `google-research-datasets/go_emotions` @add49224 | apache-2.0（YAML = API = 正文） | 28 | 43,410 / 5,426 / 5,427；`raw` 211,225 行（逐评审者） | BERT-base macro-F1 0.46（论文）；后续论文 0.49 | **推荐** |
| LexGLUE `eurlex`（另有 `unfair_tos`） | `coastalcph/lex_glue` @c23fdff1 | cc-by-4.0（正文“待补”） | 100（unfair_tos：8，可空） | 55,000 / 5,000 / 5,000（unfair_tos：5,532 / 2,275 / 1,607） | README 榜单 µ-F1 / m-F1：eurlex：TF-IDF+SVM 63.4 / 47.9，BERT 71.4 / 57.2；unfair_tos：BERT 95.6 / 81.3 | 次选；unfair_tos 作小型补充 |
| Civil Comments | `google/civil_comments` @f2970eb3 | cc0-1.0 | 7 个 float 属性 | 1,804,874 / 97,320 / 97,320 | 无统一多标签基线（unverified） | 备选：软标签，须下采样 |
| MultiEURLEX | `coastalcph/multi_eurlex` @d0318b7f | cc-by-4.0 | 21（level 1；其余级别未核） | en 55,000 / 5,000 / 5,000 | – | 不选：仅加载脚本；与 LexGLUE eurlex 同文档（卡片引同一论文，split 规模相同） |
| SemEval-2018 E-c；Reuters-21578；Jigsaw | `SemEvalWorkshop/sem_eval_2018_task_1` @9bb6e925；`ucirvine/reuters21578` @28838256；`google/jigsaw_toxicity_pred` @9fca5f50 | unknown；other；cc0-1.0（卡片） | 11；ModApte；6 | en 6,838 / 886 / 3,259；9,603 / – / 3,299；159,571 / – / 63,978 | – | 不选：许可证未知或条款不明；均靠加载脚本，Jigsaw 库内没有数据文件（据惯例需从 Kaggle 另取，未读脚本） |

要点（形状来自 first-rows 抽样，不是全量统计）：

- GoEmotions（前 100 行）：多为单标签（1 个标签 82 行、2 个 16 行、3 个 2 行），主指标报 macro-F1 与逐标签校准，不能只报 micro。`raw` 有逐评审者的 28 个 0/1 列，可算软标签；`raw` 含 Reddit 用户名，不进训练文本。
- LexGLUE `eurlex`：类名是数字 id，描述取 nlpaueb/multi-eurlex 的 `eurovoc_descriptors.json`（8.7 MB，7,390 概念、23 种语言；100 个 id 的英文描述全覆盖）。前 22 行平均 5,619 字符（最长 38,439）、约 4.5 个标签，需定截断策略。`unfair_tos` 前 100 行有 88 行无标签。
- 工具选择多标签：搜索只找到 qualifire/tool-selection-quality-benchmark（cc-by-nc-4.0）和 velmann/toolselect-bench（MIT，仅 50 题）；ToolRet 条款各异（DI 文档）。均不收。

## 4. Jev 类型化判断格式对齐与映射草案

### 4.1 字段对齐表

| 项 | 官方 OpenAPI 0.2.0 | tasksource-jev（default） | Open-Jev | LocalLLaMA |
|---|---|---|---|---|
| 粒度 | 一次请求 = state + 多题 | 一行一题；按 (group_id, state) 重组请求 | 一行一题；按 group_id 重组 | 一行一 case：state + 5 题 |
| model | 必填 | 无 | 无 | 无（README 称 state + questions 即请求体） |
| state | 串 / 对象 / 数组（必填） | `state`：纯文本串 | `state_json`：JSON 编码，解码得串或对象 | `state`：JSON 串，解码得对象 |
| 题名 | 自定键，不发给模型 | `question_id`（组内唯一，如 decision） | `metadata_json.question_id`，亦是 id 末段 | `questions` 的键 |
| type | noul / choice / score / box（box 只有 type、instructions） | `kind`：choice / score / noul | `kind`：同左 | `questions[q].type`：同官方 |
| instructions | 可选（串 / 对象 / 数组） | `question` | `question`（常含判定规则） | `instructions` |
| choice 选项 | `criteria` 映射 名 → 描述（必填，1–255 项，描述可省略） | `options`：仅名称，每行随机置换 | `options`：“名: 描述”字符串列表 | `criteria` 映射，同官方 |
| score 等级 | `criteria` 数组（必填，1–10 项，位置即分值，从 0 起） | `options`：等级描述，低 → 高 | `options`：同左 | `criteria` 数组 |
| noul | `criteria` 可选 {true, false} | `options` 为空 | `options` = [no, yes] | `criteria` {false, true} |
| 目标 | （响应）choice / score：probabilities + confidence；noul：概率 | `target`：与 options 对齐的分布；noul 为 [p_true] | `target`：与 options 对齐；noul 为 [p_no, p_yes] | `gold[q]`：官方响应形态（label、probabilities、score 或 noul）+ 平铺列 |
| 硬 / 软 | – | 多为 one-hot；聚合源为软 | 多为 one-hot；有 0.5 / 0.5 并列（前 45 行 3 行） | 全软（教师 3 次均值；160 / 160 个头） |

### 4.2 到设计稿 §3.1 内部契约的映射草案（字段名待定）

| 内部字段 | tasksource-jev | Open-Jev | LocalLLaMA |
|---|---|---|---|
| `id` / `task_id` | `<库>:<group_id>:<question_id>` / `source` | `<库>:<id>` / `source` | `<库>:<case id>:<题名>` / `workflow` |
| `input.query` | `question` | `question` | `questions[q].instructions` |
| `input.context.state` | `state` | `json.loads(state_json)` | `json.loads(state)` |
| `question.type` | `kind` | `kind` | `questions[q].type` |
| `question.options[{id,text}]`（choice） | `options` → `{id: o, text: null}` | 以首个“: ”拆“名: 描述”；名称须唯一，否则整串作名称并记 `parse_ok` | `criteria` 的键与值 |
| `question.levels[{id,text}]`（score，序号从 0） | `options` | `options` | `criteria` 数组 |
| `question.criteria`（noul，可选） | – | – | `criteria` |
| `group` = `{kind: request, id, split_key}` | `group_id` + sha8(`state`) | `group_id` | case `id` |
| `supervision.labels` | argmax(`target`)；noul：`target[0]` ≥ 0.5 | 同左；noul：`target[1]` ≥ 0.5 | `gold[q].label` |
| `supervision.soft`（仅非 one-hot） | `target` | `target`（noul 取 `target[1]`） | `gold[q].probabilities` |
| `provenance` | `source_index`、`example_id`、`variant`、`license`、`license_use`；`label_form` 依 `sources.yaml` 的 `aggregation` | `original_line_number`、`metadata.provenance`（license 文本、target_basis） | `label_form: soft_teacher_mean3`；`teacher: {id: undisclosed, license: unknown}` |

- `label_form` 取值：hard、soft_votes、soft_rating、soft_rule、soft_teacher_mean3；`provenance` 另记 `source{hf, revision, config, split}` 与 `converter@版本`。
- 选项描述放 `question.options[].text`，`candidates` 留给逐候选类型（multi_label、ranking）；备选是 choice 也走 `candidates`，待定（推荐：前者）。
- multi_label、ranking 没有官方对应。同组 noul 扇出（如 Open-Jev 的 mailroom、ir-control）由配方级转换器按 group 合成，并写 `lossy: true`；GoEmotions、LexGLUE 直接走 `candidates`。

### 4.3 转换器必须处理的差异

1. choice 的 `criteria` 必须是对象。tasksource 的 AGENTS.md 示例把 `options` 列表直接作 choice 的 `criteria`，按 OpenAPI 不合法（未对真实 API 实测）；应转成 `{选项: null}`，并检查名称唯一、数量 ≤255。
2. noul 方向不一致：TS [p_true]；OJ [p_no, p_yes]；LL `gold.noul`。
3. score ≤10 级、从 0 起；TS 同一有序任务会在 choice 与 score 间切换（按行哈希），`kind` 属于请求的一部分；TS 的 choice 选项每行置换，不能用位置当标签。
4. 软标签含教师分布时，设计稿 §3.1 要求记录教师身份与许可证，并拒收 Jev / TypeSafe API 输出。LL 教师未公开，无法证明不是 Jev，默认只用于评测与校准评估。
5. 三份数据集都不含 `model`；HTTP 服务的处理待定（推荐：接受任意 `model` 字符串，响应回填实际服务名）。图像（`assets`、box）v0.1 不读，跳过 TS 的 vision 配置。
6. 不得入模：OJ 的 target、metadata_json，LL 的 gold、factors、label_agreement，TS 的 source、variant、license 等。

## 5. 同源与去重

### 5.1 tasksource 与 DI 公开套件

读 `sources.yaml`（754 源，480 个 Hub 库）。各源 `rows` 的训练行合计恰为 full 的 2,550,012，所以行数对应 full 配置，default 的分源构成未公布。DI 取自提交 25c6ec7b（2026-10-20，仍是当前 head）的 README 与 `docs/suite.md`：表 44 项，计分 37 项。

| DI 项 | tasksource 源（Hub 库 @revision） | full 行数 train / val / test |
|---|---|---|
| BANKING77；CLINC150+OOS | banking77（legacy-datasets/banking77 @f5412156，上游 PolyAI）；clinc_oos/plus（clinc/clinc_oos @155b9c71） | 5,558 / 33 / 33；5,633 / 34 / 33 |
| ANLI*（与 DI 同一提交 8e4813d8） | anli/a1、a2、a3（facebook/anli） | 4,261、4,218、4,186 / 26、23、24 / 26、21、25 |
| WinoGrande；HellaSwag（均与 DI 同一提交） | winogrande/winogrande_xl（allenai/winogrande @01e74176）；hellaswag（Rowan/hellaswag @218ec52e） | 10,680 / 42 / 44；12,091 / 53 / 55 |
| ContractNLI* | contract-nli/contractnli_a/seg、contractnli_b/full | 2,104 + 1,955 / 14 + 9 / 12 + 12 |
| Amazon ESCI；NLI4CT | esci（tasksource/esci @8113b17a）；nli4ct_semeval2024（AshtonIsNotHere/… @efabeed1） | 2,413 / 12 / 12；2,308 / 19 / 19 |
| RAGTruth；HoVer | ragtruth/grounding（wandb/RAGTruth-processed @eb4f4b9d）；hover（Dzeniks/hover @8cacc890） | 3,296 / 0 / 19；3,319 / 19 / 20 |
| SGD（不计分） | schema-guided/intent（google-research-datasets/schema_guided_dstc8） | 5,798 / 33 / 33 |
| Humicroedit†；iSarcasmEval†；ARC-Easy / Challenge†（不计分） | humicroedit/subtask-1、2；isarcasm/intended；ai2_arc/* | 1,033 + 1,143；3,171；2,600 + 1,302（train） |

\* non-commercial 源。† 仅按名称与任务匹配，上游身份未做哈希核对。DI 评测上游的 test / validation 分区（如 BANKING77 的整份 test 3,080）；按 tasksource 的 AGENTS.md，`split` 是来源自带分区，train 行取自上游训练分区，风险是“域内训练”，不是逐行泄漏。这一点对打包题、程序生成题、没有原生分区的源未验证；tasksource 自己也声明未排除基准污染，与 DI 私有集是否对齐为 unverified。DI 其余 29 项在 tasksource 里没有对应源（MMLU、BIG-bench 被明确排除）。

### 5.2 tasksource 与首批、多标签候选

| 注册集 | tasksource 源（full 行数 train / val / test） | 风险 |
|---|---|---|
| BoolQ | super_glue/boolq 765 / 2 / 3；boolq_passage 788 / 4 / 3；boolq-natural-perturbations 3,255 / 19 / 19 | 同上游训练域；natural perturbations 卡片只写“BoolQ questions with semantic alteration”，是否含 validation 的问题或段落待哈希核对 |
| MNLI | glue/mnli 612 / 3 / 3；**chaos-mnli-ambiguity/votes 1,439 / 64 / 67**；MoritzLaurer/dataset_train_nli 15,917 / 99 / 96；multilingual-NLI-26lang 4,299 / 25 / 26 | ChaosNLI-MNLI 是 MNLI-matched 开发集的 1,599 对重标（卡片与 ChaosNLI README，doc-verified），与 validation_matched 直接重叠，必须剔除；后两个合集的组成未核 |
| SST-5 | sst5（SetFit/sst5，同库同 revision）3,980 / 23 / 24；glue/sst2 602 / 2 / 3 | SST-2 与 SST-5 同出一语料，按句子哈希 |
| NFCorpus、MS MARCO | 无 | RAGTruth 的 QA 上下文含 MS MARCO 段落（DI 文档，author-reported），段落级可能重叠 |
| GoEmotions、LexGLUE、其他 | go_emotions/simplified 3,157 / 17 / 17（同库同 revision）；lex_glue 的 unfair_tos 2,110 / 13 / 13、ledgar、scotus、case_hold，无 eurlex；civil_comments/*_share 共 11,536 行；jigsaw_toxicity 3,299；toxic_conversations 3,291 | 同上游训练分区；eurlex 无重叠；三个毒性源之间是否共享评论待核 |

Banking77、CLINC150 见 §5.1，另有 `ag_news` 4,310 / 24 / 24，均为上游训练分区。其他同源关系：

- Open-Jev 的 `workflow-controls-v1/*` 与 LocalLLaMA 的四个工作流同名。Open-Jev manifest 称其仿 TypeSafe 公开 evals 页，`official_examples_used: 0`，标签是确定性策略规则、全为 noul；LocalLLaMA 的题与标签另行生成（教师分布）。这是任务族重叠，不是逐行重叠；报告 LocalLLaMA 的“未见工作流”成绩时应排除 Open-Jev 的 `workflow-controls-v1/*`。
- tasksource 的 `sources.yaml` 没有 LocalLLaMA 或 Open-Jev 的条目。

### 5.3 去重规则草案

1. **身份**：每行记 `lineage.upstream = (库, revision, 上游 split, 行号)`。来源有则取：TS 的 `source` + `source_index` + `example_id`；OJ 的 `source` + `original_line_number`；LL 的 `id`；否则用内容哈希。
2. **内容哈希**：对 `state`（成对任务含第二段）做 NFKC、小写、折叠空白后取 sha256，记 `input_hash`。去重只比 `input_hash`，同一输入的不同题面算同一样本。
3. **评测集优先**：训练前对每个 test、dev、calib、eval_only 集合建哈希表。训练池中 `input_hash` 命中的行，连同同 `group_id` / `example_id` 的全部变体，一律剔除。
4. **上游 split**：训练池里 (库, 上游 split) 命中评测集上游 split 的整源整 split 剔除（如 `multi_nli:validation_matched`、`banking77:test`）。TS 以 `source_index` 前缀（重新划分前的 split）判断；打包题与程序生成题该字段为空，改按 `input_hash`。
5. **同源族**：每个数据集标 `family`，已知互含关系写进 `lineage`（ChaosNLI → MNLI dev；RAGTruth → MS MARCO；eurlex ≈ MultiEURLEX-en）。同族数据集一边进训练池、一边做 eval_only 时，必须有剔除记录。
6. **分组划分**：同一请求的所有题、TS 与 OJ 同 `group_id` 的变体、LL 同 case 必须落在同一 split（`group.split_key`）。
7. **默认值**：选了 `eval_only` 套件（DI 公开套件）时，§5.1 的 tasksource 同源源默认剔除，配方可显式关闭；无论开关，报告都标“域内 / 域外”。这是设计稿 §3.1 中“待定”项的推荐。
8. **留痕**：每次运行输出 `dedup_report`（规则、命中数、剔除数、哈希算法版本）。

## 6. 注册表 YAML 草稿

字段名待定。`hf` 为“库 id@revision”，revision 是 2026-11-04 的 `main` 提交。未写 `calib`、`dev` 的条目：各从 train 按内容哈希切约 5%（比例待定）；`X#h50` 表示取 X 的一半；`presets` 是 YAML 锚点。注：tasksource 训练池默认 `license_use == commercial`，按样本评测加 `variant == direct`；LocalLLaMA 的软标签只用于评测与校准评估；Open-Jev 与 LocalLLaMA 的任务族重叠指 `workflow-controls-v1/*`。manifest 路径与 SHA256SUMS 在落盘后生成，此处不填。`terms` 取值：attribution、share_alike、non_commercial、research_only、license_unspecified、mixed_per_row、third_party_text_unverified、teacher_unknown、pii_possible、eval_only_no_train。

```yaml
presets:
  with_val: &with_val {train: train, calib: "validation#h50", dev: "validation#h50", test: test}
  train_test: &train_test {train: train, test: test}
datasets:
- {name: boolq, hf: "google/boolq@35b264d03638db9f4ce671b711558bf7ff0f80d5", judgment: noul, converter: bool_noul_v1,
   license: cc-by-sa-3.0, terms: [share_alike], splits: {train: train, test: validation}, lineage: {family: boolq}}
- {name: multi_nli, hf: "nyu-mll/multi_nli@da70db2af9d09693783c3320c4249840212ee221", judgment: choice, converter: pair_choice_v1,
   license: "cc-by-3.0, cc-by-sa-3.0, mit, other", terms: [mixed_per_row, share_alike],
   splits: {train: train, test: [validation_matched, validation_mismatched]},
   lineage: {family: nli-mnli, overlaps: ["ts:chaos-mnli-ambiguity/votes = validation_matched，须剔除"]}}
- {name: banking77, hf: "PolyAI/banking77@1846984189f2bc0ac51c6fc3a600fa9e1714e0c6", judgment: choice, converter: cls_choice_v1,
   license: cc-by-4.0, terms: [attribution], splits: *train_test, lineage: {family: intent-banking77, in_di: BANKING77}}
- {name: clinc150_plus, hf: "clinc/clinc_oos@155b9c710419136e17307b80d0a13e68cd46b4ec", config: plus, judgment: choice,   # oos -> allow_null 待定
   converter: cls_choice_v1, license: cc-by-3.0, terms: [attribution],
   splits: *with_val, lineage: {family: intent-clinc, in_di: CLINC150+OOS}}
- {name: ag_news, hf: "fancyzhx/ag_news@eb185aade064a813bc0b7f42de02595523103ca4", judgment: choice, converter: cls_choice_v1,
   license: unknown, terms: [non_commercial, license_unspecified], splits: *train_test, lineage: {family: ag-news}}
- {name: sst5, hf: "SetFit/sst5@e51bdcd8cd3a30da231967c1a249ba59361279a3", judgment: score, converter: ordinal_score_v1,
   license: unspecified, terms: [license_unspecified, research_only],
   splits: *with_val, lineage: {family: sst}}
- {name: nfcorpus, hf: "BeIR/nfcorpus@b5026a0e96e8a7ac4f95f482a596389289d46269", qrels: "BeIR/nfcorpus-qrels@a451b3b26d3ae1358f259c1a3a4dd61fcea35a65",
   judgment: ranking, converter: beir_ranking_v1, license: "cc-by-sa-4.0 (Hub 标签); 上游只明示学术用途免费", terms: [research_only, share_alike],
   group_key: query-id, splits: {train: qrels.train, calib: "qrels.validation#h50", dev: "qrels.validation#h50", test: qrels.test},
   lineage: {family: beir-nfcorpus}}
- {name: ms_marco_v1_1, hf: "microsoft/ms_marco@a47ee7aae8d7d466ba15f9f0bfac3b3681087b3a", config: v1.1, judgment: ranking,
   converter: msmarco_listwise_v1, license: "unspecified (Hub); 上游 Terms: 仅限非商业研究", terms: [non_commercial],
   splits: *with_val, lineage: {family: msmarco, appears_in: [ragtruth/grounding]}}
- {name: go_emotions, hf: "google-research-datasets/go_emotions@add492243ff905527e67aeb8b80c082af02207c3", config: simplified,
   judgment: multi_label, converter: multilabel_cands_v1, license: apache-2.0, terms: [pii_possible],
   splits: *with_val, lineage: {family: emotion-goemotions}}
- {name: lex_glue_eurlex, hf: "coastalcph/lex_glue@c23fdff1a6bf74e0e1a71cb86f1e781d37da888c", config: eurlex,
   judgment: multi_label, converter: multilabel_cands_v1, license: cc-by-4.0, terms: [attribution],
   splits: *with_val, lineage: {family: eurlex, same_docs_as: ["multi_eurlex:en"]}}
- {name: tasksource_jev, hf: "tasksource/tasksource-jev-typed-decisions@8df2e5c9a33f19a2c2b8febf291c3acb0a91a8a8", config: default,
   judgment: [noul, choice, score], converter: jev_flat_rows_v1, license: other, terms: [mixed_per_row],
   splits: *with_val, lineage: {family: tasksource, upstream: sources.yaml, di_overlap: true}}
- {name: open_jev_release_v2, hf: "ZefanCai/Open-Jev@c67699e13d0ae25e35b77165a4b6b079bedc8aba", config: release-v2-redistributable,
   judgment: [noul, choice, score], converter: open_jev_rows_v1, license: cc0-1.0, terms: [third_party_text_unverified],
   splits: {train: train, calib: calibration, dev: validation, test: [test, ood]},
   lineage: {family: open-jev, task_family_overlap: [typed_decisions_ll]}}
- {name: typed_decisions_ll, hf: "LocalLLaMA/typed-decisions@c9e8c4167bf9b47edfcba3b9580e69bddd8d9e5a", config: all,
   judgment: [noul, choice, score], converter: jev_request_v1, license: apache-2.0, terms: [teacher_unknown],
   splits: *train_test, lineage: {family: typed-decisions, task_family_overlap: [open_jev_release_v2]}}
- {name: decision_index_public, git: "apolinario/decision-index@25c6ec7b6b5f44a216c63c3a700de3a9c6d6699d", build: "suite rebuild，本地重建，不分发",
   judgment: [choice, noul], converter: di_rows_v1, license: "各上游自带（ANLI CC BY-NC 4.0；RAGTruth 的 MS MARCO 上下文非商业、Yelp 上下文不可再分发）；代码 MIT",
   terms: [eval_only_no_train, non_commercial, mixed_per_row], splits: {eval_only: all},
   lineage: {family: di-public, shares_upstream_with: [tasksource_jev]}}
```

## 7. 未核实项与风险

未核实（括号内为所用或应取的 URL）：

1. NFCorpus 上游条款只读到搜索摘要对项目页的转述（https://www.cl.uni-heidelberg.de/statnlpgroup/nfcorpus/ 未取）；SST 上游条款未取（https://nlp.stanford.edu/sentiment/）。
2. GoEmotions 基线 0.46 / 0.49 经搜索结果转述，论文表格未读（https://arxiv.org/abs/2005.00547）；LexGLUE 基线读自 README 榜单（https://github.com/coastalcph/lex-glue）。
3. tasksource `default` 的分源构成、`source_index` 前缀与上游 split 的对应、打包题的拆分：要读数据才能核，未做。
4. Humicroedit、iSarcasmEval、ARC 与 DI 只是名称匹配；Dzeniks/hover 的切分与 DI 所用 HoVer dev 的关系未核（https://huggingface.co/datasets/Dzeniks/hover）。
5. BoolQ natural perturbations 是否含 BoolQ 开发集内容（https://huggingface.co/datasets/tasksource/boolq-natural-perturbations）；MoritzLaurer 两个 NLI 合集的组成（https://huggingface.co/datasets/MoritzLaurer/dataset_train_nli）。
6. MS MARCO v2.1 的标签覆盖，以及 v1.1 test 的全量标签覆盖：只看了前 46 / 47 行。NFCorpus qrels 的分值范围、CLINC 的 OOS 占比：未核。
7. 官方 API 对 choice 传数组 `criteria` 的实际行为：只依据 OpenAPI，未实测。

风险：

1. revision 易变：LocalLLaMA 随榜单频繁改 README；Banking77 刚在 10-28 改了存储格式。注册表须存 revision 与每文件 SHA256。
2. 许可证混杂：tasksource 有 41.6%（314 / 754）的源没有确认可商用，`license_use` 是 best-effort 分类，不是法律意见；Open-Jev 的 CC0 不覆盖第三方题面。
3. Open-Jev 的 test、ood 标签公开，tasksource 与 LocalLLaMA 的训练池可能同域；报告须标注使用情况。
