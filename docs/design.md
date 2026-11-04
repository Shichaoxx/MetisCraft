# MetisCraft 设计草案（2026-11-04，待用户拍板）

> **已定（2026-11-04）**，以下四条优先于正文；正文等卡 1–3 拍板后统一改写。
> 1. **定位第 ① 部分改为"支持矩阵"**：不单独发布 Awesome 清单。MetisCraft 内收录约 10–20 个当前较好的 Jev 类项目，每行标出能否用 MetisCraft 训练、复现、部署，首批做 3–5 个；清单草稿（AW）只作素材。取代 §1.1 表中的 ① 行、§3.5 和卡 5。
> 2. **部署**：v0.1 交付 Predictor，外加一层沿用 `/v1/systemone` 请求格式的 HTTP 服务，保留按类型的校准；vLLM、SGLang、llama.cpp 的引擎导出放 v0.2，随卡 3 的基座选择。取代决策点 14 的待定状态。
> 3. **通用优先**：MetisCraft 面向通用的类型化判断，不以某个业务任务为中心。不设单一业务旗舰，Schema Linking 只是场景之一，取代卡 4 的推荐；v0.1 验收改为每种判断类型各用一个公开基准，具体基准待拍板（建议见卡 4）。正文里把 Schema Linking 当旗舰或验收用例的地方（§0 第 3 条、§2、卡 2–4、§6）待改写。
> 4. **数据注册表和主流数据集支持是 v0.1 必需项**：MetisCraft 用开源数据集训练和验收。`datasets/registry.yaml`（§3.4）收录主流通用数据集和 Jev 类数据集，每条写明来源与固定 revision、判断类型映射、转换器、许可证与条款标记、split 角色、上游来源（用于去重防泄漏）。首批名单、是否原生读取 Jev 类型化判断格式待拍板，建议见 §3.4 末尾。
>
> 路径约定：`docs/` 下的文件在仓库内；其余 `drafts/`、`partial/`、`research/`、`landscape/` 等路径指维护者本机的调研目录，不随仓库发布。

> **05:20 外部核查**：6 条承重说法已经对抗核查。其中 ranking 不再是差异点，ExSL 证据变弱，"层级多标签没人做"被削弱，卡 1–4 的把握或表述都要随之调整。见文末"外部核查"一节，以及 `drafts/design-verify-frameworks.md`、`design-verify-evidence.md`。正文尚未逐处改写，以文末一节为准。
> 状态：草稿，§0–§7 已写完；已按 `drafts/design-critic.md`（2026-11-04）修订，不同意或改了做法的条目见文末“评审记录”。所有"推荐"只是建议，不替用户做决定；最需要拍板的是 §5 卡 1（构建路线）、卡 3（首发基座族）和决策点 6（v0.1 是否必须有可对照的读出对）。
> 清单草稿缩写：AW = `drafts/awesome/README.md`（本机草稿；已定不单独发布清单，只作支持矩阵的素材）。字段名、包路径、YAML 键名均为示意，未实现、未测试。
> 来源缩写：DP = `research/decision-points.md`；VL = `partial/verify-landscape.md`；VE = `partial/verify-external.md`；RQ = [`docs/research/build-routes.md`](research/build-routes.md)；EN = [`docs/research/scenarios.md`](research/scenarios.md)；SL = [`docs/research/schema-linking.md`](research/schema-linking.md)；SOTA = [`docs/research/decision-model-sota.md`](research/decision-model-sota.md)；S / K = `research/metis-v0.1.0-salvage.md` / `-critique.md`；R = `research/metiscraft-research.md`（v1，部分已推翻）。

## 0. 与昨天相比变了什么

1. **竞品从"没有"变成"已经有"**：ms-swift `--task_type decision`（10-13 合并，7 种读出头、4 种损失、独立校准回调）和 Unsloth（AGPL）已内置决策训练；transformers #50054、TRL #8040 正在标准化，未合并（VL §2.1、§3）。"训练工具稀缺"不再成立，头号风险从"可能发生"变成"已经发生"。
2. **差异化收窄到三块**：读出与基座解耦并在同协议下对照；校准、评测、可追溯作为一等公民；业务场景需要的 multi_label / ranking 类型语义（VL §4；RQ §0 第 1、6 条：ms-swift 与 HF 都把类型写死为 noul / choice / score）。
3. **v0.1 读出范围要重谈**：昨天推荐的"方案甲（答案 token + 标量头）"与 ms-swift `generic_decision` 高度重叠；Schema Linking 的主线读出改为 ③-list（ExSL 式，普通因果注意力），tree 推迟（SL §5.3、§5.8）。
4. **基座和导出被引擎卡住**：引擎可读的带头布局全部绑定 Qwen3.5 混合架构族；Qwen3 / Gemma 只能走答案 token，引擎上拿不到校准；llama.cpp 不支持答案 token 路线（VE §2 第 1–3 条）。v0.1 只承诺"自家 Predictor"；"合并 CausalLM"导出只对 ① 答案 token 读出有意义，③ 的标量头在 lm_head 之外，产物是"骨干 + 独立头文件"，只能由 Predictor 加载。
5. **Awesome 清单已有同类**：AnotiaWang/awesome-decision-models（681★）、andyrewlee/awesome-system-one（42★）已存在（DP 二-F），名字、位置和是否自建要重新决定。

**定位自查**：本稿仍把 MetisCraft 定义为"英文 Awesome Jev-like 清单 + 面向业务场景的 Jev 类决策模型训练工具包"；Schema Linking 是旗舰场景和验收用例，不是项目本身；未漂移成"某个模型"或"Schema Linking 项目"。

## 1. 定位与理念

### 1.1 两部分

| 部分 | 是什么 | 对谁有用 | 不是什么 |
|---|---|---|---|
| ① Awesome Jev-like 清单（英文） | 收录 Jev 类决策模型、训练工具、读出方法、数据与基准、服务引擎、论文；每条带证据级别、固定链接、许可证与条款标记 | 想了解"这个领域现在有什么、哪些数字可信"的人 | 不是星标堆砌；不收无法核实的"击败 Jev"数字而不加标注 |
| ② 训练工具包 | 给定一个业务场景（数据 + 判断类型 + 候选结构），用配方把任意受支持的基座训练成"一次前向给出类型化判断 + 校准概率"的模型，并产出可追溯的评测报告和可加载产物 | 有私有业务判断、需要本地部署、需要可信概率的工程师 | 不是又一个通用 SFT 框架；不自写训练循环、分布式、量化；不是某个具体模型 |

两者的关系：清单是工具包的"地图和证据库"（读出分类、复现目标、竞品对照都来自它）；工具包的同协议对照结果反过来作为清单里少有的受控证据。

### 1.2 理念："训练有素的直觉"

| 理念 | 落到设计上的约束 |
|---|---|
| 快 | 一次前向出全部候选的判断（③-list 共享前缀，SL §5.3）；不靠生成长推理 |
| 知道自己有多准 | 校准是一等公民：每个产物带校准器和其拟合数据身份；评测固定 ECE 口径并同时报 NLL / Brier（SOTA §4.3） |
| 判断有类型 | 类型（noul / choice / score / box / multi_label / ranking）决定输出语义、损失、指标和校准方式，而不是只有一个 logits 张量 |
| 能从反馈中学 | 只预留与 TRL `reward_funcs`、verl `compute_score` 同形的奖励签名，v0.1 对这条理念只承诺一个签名形状测试（§6 验收"反馈接口"行；是否接受见决策点 19）；预测分布与行动策略分开；只观察到所选动作时不补全他项标签（DP 一-13；K#1）。不把 RL 当卖点（SOTA §1 第 4 条） |

工程原则：

- **部件可换**：读出、判断类型、损失、校准器、基座是相互独立的轴，用注册表组合；"同一数据、同一基座、换读出"是一条配置差异，而不是换一个仓库。
- **一事一处**：类型语义只在类型模块定义，指标只在评测模块定义，配方只引用名字。
- **有据可查**：每次运行记录配置、git 提交（含 dirty diff）、环境、数据集版本、权重版本（宪法 §4.1）；产物带 `SHA256SUMS` 与许可证 NOTICE（借鉴 systemone-builder，VL §2.1）。
- **借力成熟库**：训练循环用 HF `Trainer`/Accelerate，适配器用 PEFT，数据用 `datasets`，指标对拍 sklearn / ir_measures；不自建模板 DSL。
- **看得见**：每步计算都能画成张量图，图与代码对齐（§3.6）。

## 2. 竞品格局与真实空白

| 对象 | 读出 | 判断类型 | 读出与基座解耦 | 校准 | 外部评测 / 同协议对照 | 导出 | 许可证 | 证据 |
|---|---|---|---|---|---|---|---|---|
| ms-swift `--task_type decision` | 7 种 `ScoringHead` 子类 + `generic_decision` | noul / choice / score（写死） | 否：每个 loader 固定 `head_cls`；只有 `generic_decision`（verbalizer）可换基座 | 是，两段温度拟合 + OOF | 仅训练期指标；无对照 | 原布局；推理只支持 transformers 后端 | Apache-2.0 | VL §2.2；RQ §1.3（source-verified，`df0bc0f`） |
| Unsloth `FastDecisionModel` | Clef / Laya / `llm` | 未核实（未读源码） | 部分（`llm` 模式） | 是，LBFGS 按类型温度 | 训练期 ECE | GGUF / MLX / llama.cpp | **AGPL-3.0-only**（决策代码逐文件 SPDX） | RQ §0 第 5 条；VL §2.2 |
| transformers #50054 + TRL #8040 | Jev 槽位头；Kev/Von/Clef 后续 PR | noul / choice / score（`Question.questions_types`） | Auto 注册可接新头；损失在 `forward` 内 | 未见 | 未见 | transformers 原生 + `/v1/systemone`（已合并） | Apache-2.0 | RQ §2.2–§2.5（未合并、实验性） |
| 社区（lev、Luce、JevAny、systemone-builder 等） | 各 1–2 种 | 多为三原语 | 少数可换 HF id，但读出不是实验轴 | lev / Luce / OneJev 有 | lev 有 `levbench`；Kev 协议最完整 | 各自服务 | 逐个看 | VL §2.2 |
| 业务化产品（nstranges/system1、DomainSpec、luce.yaml） | — | — | — | — | — | — | — | VL §2.3（README，author-reported） |

**结论**：

1. "训练一个决策模型"已被主流框架覆盖，MetisCraft 不应再造训练器（VL §4）。
2. 真实空白只有三块（VL §4 第 1 条；RQ §0 第 1、6 条）：
   - **类型空白**：主流框架（ms-swift、HF #50054 / TRL #8040）把类型写死为 noul / choice / score（source-verified；RQ §2.4 只 grep 了这三处的 PR diff）；社区有零散的 ranking / subset 实现（OJ `ir-control-v1` 的 pointwise / pairwise / listwise、open-jev 的 Subset 原语、bekko `reranker` 任务头 `02806a8` `modules.py` L343–L350、DI 的 ToolRet / BRIGHT / ESCI，VL §2.1）；**带层级的业务多标签**（如 Schema Linking）目前没人做（VL 结论）。业务场景恰好需要（EN §4.3 表："逐候选标量头 + multi_label"列了场景 1、2、4、5、10、15，共 6 个；ranking 列了 2、12、17，另有场景 5 的召回后排序，计 3–4 个）。
   - **对照空白**：没有一个把"读出 × 基座 × 目标"作为独立轴、在同协议下做对照的工具；也没有对 encoder 强基线的公开同协议数字（SOTA §5 复现目标 1）。
   - **业务配方 + 证据空白**：没有"Awesome 清单 + 业务场景配方 + 逐候选校准评测 + 可追溯产物"的组合；Schema Linking 没有 Jev 格式数据与逐列校准报告（SL §1 第 4、5 条）。
3. 诚实边界：有标签时 encoder 常胜出（CLINC：BGE-small + LR 0.948 > Jev 0.913，author-reported，EN §0 第 4 条），自训练的理由只能是私有化、校准、多标签 / 排序、菜单可变，不能写成"更准"。

## 3. 与路线无关的部分

本节不管选哪条路线都要写，且写法相同；路线只决定它们"挂在谁的接口上"。

### 3.1 数据契约

**原则**：旧 Metis JSONL（`{id, task_id, input:{query, context?}, candidates:[{id,text}], supervision:{kind:"candidate_labels", labels}}`）继续合法，新能力只加**可选**字段（SL §5.1）；旧 `v2-design.md`/`v2-upgrade.md` 的思路带入，旧包名与导入路径兼容承诺不带（K#1、K#2）。

**一行一题**（一个"问题实例"），字段：

| 字段 | 必填 | 含义 |
|---|---|---|
| `id`, `task_id` | 是 | 全局唯一行 id；任务名 |
| `input.query`, `input.context` | query 必填 | 问题文本；上下文（证据、方言、对话历史等自由 dict） |
| `question` | 否（缺省由配方给） | 判断类型规格，见 §3.2：`{type, options?, levels?, allow_null?, min_k?, max_k?}` |
| `candidates[]` | 逐候选类型必填 | `{id, text, level?, parent?, meta?}` |
| `group` | 否 | `{kind, id, split_key}`；`split_key` 让划分器强制按组（如按库）划分 |
| `structure` | 否 | 组级上下文（如表、外键），按 `group.id` 去重存放，行内可只放引用 |
| `supervision` | 训练 / 评测行必填 | `{kind, labels, default?, soft?, roles?}`；`kind ∈ {noul, choice, score, multi_label, ranking, box}`；`soft` 存软标签（如教师分布） |
| `provenance` | 否（转换器产出时必填） | 来源 revision、gold 推导方式与工具版本、`parse_ok` 等；存 `supervision.soft` 时必填教师模型身份及其许可证，转换器拒绝来源是 Jev / TypeSafe API 输出的软标签（TypeSafe 条款） |

**库 → 表 → 列的层级候选**（Schema Linking；SL §5.1、§5.2）：

- 层级用 `level ∈ {database, table, column}` + `parent` 表达，候选 id 带前缀（`t:schools`、`c:schools.City`），保证全局唯一。
- `meta` 放列类型、注释、主外键、示例值、值匹配；序列化模板由配方决定（参考 M-Schema），不写进契约。
- 外键放 `structure.foreign_keys`，供 ⑤ 集合头或层级一致性项使用。
- 一致性规则由类型模块检查：`label(col)=1 ⇒ label(parent table)=1`；违反的行由转换器报告，不静默修正。
- 选库是一个独立的 `choice`（或 pointer）子问题，不和列混在一个多标签集合里（SL §5.3 ④ 行）。
- 小库全量候选；大库允许"两段式"：召回器产出的候选子集写进 `provenance.recall`，第一阶段 SRR 单独报告（SL §5.2）。

**数据集 manifest**：对接宪法的版本目录（`dataset.yaml` + `SHA256SUMS`）；逐 split 校验哈希；manifest 声明 task 规格和 **split 角色**（train / calib / dev / test / eval_only），防泄漏按角色判断并比较 id 与内容哈希，不再只看 split 名是否等于 "test"（S 表"防泄漏与划分"、M2；S:293）。数据身份哈希只覆盖内容，不混入适配器名等元数据（S H6）。

**转换器许可与泄漏**（DP 二-E；EN §6.1）：转换器输出 `license` 字段；Spider / BIRD 派生数据沿用 CC-BY-SA；非商用源（ToxicChat 等）可以用于研究对照，但默认不进"可商用"产物；按 DI 公开套件的源做去重或留出是否默认开启，待定（决策卡 §5）。

### 3.2 判断类型语义

类型决定：输出结构、概率语义、损失族、校准粒度、指标、导出能否兼容官方原语。官方原语与本项目扩展**分开说明**，扩展类型不冒称 Jev 兼容（K#1：旧 `v2-design.md:34-44`，"Jev 的 Score 不能解释为候选列表排序"）。

| 类型 | 官方？ | 输入约束 | 输出 | 概率语义 | 默认损失 | 校准粒度 | 主指标 |
|---|---|---|---|---|---|---|---|
| `noul` | 官方 | 一条陈述 / 准则 | `p_yes ∈ [0,1]` | 伯努利 | BCE（可软标签） | 每类型一温度或 Platt | acc / AUC / NLL / Brier / ECE |
| `choice` | 官方 | 1–255 个选项（官方口径，引擎限制放导出层，VE §3）；`allow_null` 可选 | 选项上的分布（含 null 列时为 k+1 维） | 互斥单选，和为 1 | soft-CE + 小权重 Brier（SOTA §5） | 每类型一温度；可按选项数分桶（借鉴 lev，VL §2.1） | acc / macro-F1（类集合预声明）/ NLL / 多类 Brier / ECE |
| `score` | 官方 | 1–10 个有序等级 | 等级上的分布 + 期望分 | 有序单选 | CE + 序数项（相邻软目标或累积 logit / CORAL，待定，DP 一-1） | 每类型一温度 | NLL / RPS / MAE（期望分）/ ECE；**DI 不测 score**，不能靠 DI 验收 |
| `box` | 官方（Jev 1.14，图像） | 图像 + 目标描述 | 四边坐标（图像宽高的比例） | 非离散分布；官方以 confidence 表达 | — | — | IoU；**v0.1 不做**（只 GUI 场景需要，EN §4.3） |
| `multi_label` | 扩展 | N 个候选（可带层级） | 每候选独立 `p_i` | 每候选伯努利，**不和为 1**；另报"整题全中"概率 | BCE（`pos_weight` 或 focal）+ 可选 pairwise / listwise 排序项 + 可选层级一致性 | 按 `level` 分组温度；可按组大小分桶 | Recall@k、SRR@k / SRR@τ、F6、列级 ROC/PR AUC、逐候选 ECE / Brier |
| `ranking` | 扩展 | N 个候选 | 每候选分数 + 由分数导出的排序；可选 Plackett–Luce 分布 | pointwise 时为相关概率；listwise 时为"排第一"的分布 | pointwise BCE + ListNet；Plackett–Luce 视预算（EN §0 第 6 条） | pointwise 部分按温度校准 | nDCG@k / MRR（以 ir_measures 为准） |

**不变量**（K#2：旧 `v2-upgrade.md:38、55-67`）：

1. 逐候选打分时，已有候选的 raw score 不随无关候选的增删而变（③-list 会违反，见 §7 风险 3；需显式标注为"顺序敏感布局"并测量）。
2. 温度缩放不改 argmax；不能把 NLL 改善说成 accuracy 改善。
3. `calibrate` 生成派生产物，不改父产物；拟合只用 calib split。
4. 预测分布与行动策略（阈值、弃权、top-k 截断）分开存放；策略参数不进模型身份。
5. Predictor 输出统一字段：`raw_scores`、`probabilities`、`probability_semantics`、`calibration`、`decision`、`abstained`、`uncertainty`（K#1 指出旧报告漏了 `uncertainty`）。

**导出到官方语义的映射**：`multi_label` 可降级为"每候选一个 `noul`"，`ranking` 可降级为"每候选一个 `score`"，这样能在 ms-swift / HF 接口上跑通，但丢失集合交互，导出时在产物里写明 `lossy_export: true`（RQ §1.3）。

### 3.3 校准 / 评测 / 可追溯协议

**校准**（SOTA §4.3；K#2 旧 `v2-upgrade.md:62-67`；S:314）：

- 默认：训练后按类型（multi_label 再按 `level`）各拟一个温度；可选 Platt、isotonic、按选项数 / 组大小分桶。
- 只在 calib split 拟合；Schema Linking 的 calib 和 test 必须是**没见过的库**（`group.split_key`）。拒绝在 held-out 集上拟合（借鉴 lev，VL §2.1）。
- 校准器作为派生产物保存：记录拟合数据身份、基座 / adapter 身份、方法与参数。
- 引擎导出时最多写一个全局温度（VE §2 第 3 条），按类型温度只在自家 Predictor 生效；报告里写明这一点。

**评测口径固定**（SOTA §4.3"Same Answer, Different Confidence"；K#20）：

- ECE（口径为本稿建议，待用户确认；各家 10 / 15 bin、等宽 / 等质量不一，SOTA §6）：置信度 = 预测类别的概率（多类）/ 正类概率（二元与逐候选）；15 个等宽分桶；同时报 NLL、Brier，并输出可靠性图。口径作为常量写在一处，报告头部打印。
- 多类 Brier 用全向量定义；macro-F1 的类集合预先声明；单类情况下 balanced accuracy 的处理写明。
- 排序指标以 ir_measures / pytrec_eval 为准，分类与校准以 sklearn 为准；自写版本只用于教学并对拍（DP 一-8）。
- 每个配方必须带无信息基线（多数类 / 先验）和至少一个强基线（encoder 或 BM25 / cross-encoder）；单选打不过无信息基线时暂停加结构（K#2 旧 `v2-upgrade.md:98`）。
- 外部基准定位（DP 一-10）：LL 只作格式烟测与 KL/Brier 对外可比；DI 写"0.3 公开套件（= 0.4 公开部分），自评"，只跑子集；TS 与 DI 至少 11 个基准同源，有泄漏风险。

**评测门禁**（S:334；旧 `prepare_boolq_readout_protocol.py`）：打开 test / 封存 split 之前必须存在覆盖全部 arms × seeds 的实验计划和选择锁；`eval` 默认 split 不是 test。是默认开启还是 opt-in 的"严格模式"，见决策卡。

**可追溯**（宪法 §4.1；S:325；VL §2.1 systemone-builder）：

- 每次运行目录：完整解析后的配置（及其 sha）、git 提交号 + dirty diff、环境（conda 名、关键包版本、CUDA）、数据 manifest 哈希、基座 `SOURCE.json` 与 revision、随机种子。
- 产物 `artifact.json`：判断类型规格、概率语义、校准器、配置哈希、数据哈希、基座身份、git 提交；文件清单 + SHA-256；拒绝符号链接与路径穿越（S 表"产物封存与校验"）。
- 产物附 `NOTICE`：基座权重许可证、每个训练源许可证；非商用源参与训练时自动标为非商用。
- 报告里的每个数字链接到产生它的运行目录。

### 3.4 场景配方格式

类比 LlamaFactory 的 `data/dataset_info.json`（名字 → 文件 / HF 地址 + 列映射）和 `examples/<阶段>/<模型>_<方法>.yaml`（一个可直接跑的完整配置）。MetisCraft 对应两层（Luce 的 `luce.yaml` 分段与 systemone-builder 的 `DomainSpec` 作参照，VL §2.1）：

1. **`datasets/registry.yaml`**（对应 `dataset_info`）：数据名 → manifest 路径 + 期望 sha256（或 `name@version`）+ 转换器名 + 许可证 + split 角色。名字绑定哈希（S 表"本地数据集名册"）。
2. **`recipes/<场景>/<基座族>_<读出>.yaml`**（对应 `examples`）：一个场景配方 = 下列分段，可 `a.b=c` 覆盖：

```yaml
# recipes/schema_linking/qwen3_candidate_list.yaml（示意，字段名未定）
scenario: schema_linking
question: {type: multi_label, levels: [table, column], hierarchy: enforce}
data:     {train: bird@<ver>:train, calib: bird@<ver>:calib_unseen_db, test: bird@<ver>:dev,
           converter: sql_gold_sqlglot, serializer: m_schema_v1}
model:    {base: models/Qwen3-1.7B/<rev>, peft: {kind: lora, r: 16, targets: all_linear}}
readout:  {kind: candidate_scalar, layout: list, shuffle_train: true, permute_infer: 1}
objective: {terms: [{bce: {pos_weight: auto}}, {listwise: {weight: 0.2}}, {hierarchy: {weight: 0.1}}]}
calibration: {method: temperature, group_by: level, split: calib}
eval:     {metrics: [recall@5, recall@10, srr@10, pr_auc, f6, ece, brier], baselines: [bm25, bge_reranker]}
export:   {targets: [predictor]}
trace:    {strict: true}
```

约束：配方只引用注册名，不内嵌代码；默认超参不写死在代码里，新配方先过 sanity 阶梯（过拟合检查 → 学习率扫描 → 学习曲线，DP 一-12）；每个配方附一份 `README` 说明许可证、预期数字的来源和证据级别。首批配方顺序见 EN §5.2（意图识别作冒烟 → Schema Linking → 检索重排 → 内容审核）。

**首批数据集（建议，2026-11-04，待拍板）**：

| 用途 | 数据集 | 判断类型 | 要点（核查见本机 VL） |
|---|---|---|---|
| Jev 类训练 | `ZefanCai/Open-Jev` | 是非、选择、独立多标签、等级 | `release-v2-redistributable` 共 113,568 条，多为合成，英中土三语；CC0 只覆盖作者生成的内容，部分题面取自 TypeSafe 文档 |
| Jev 类训练 | `tasksource/tasksource-jev-typed-decisions` | 多种 | 默认配置 train 999,832 条；标签来自源数据标注或确定性变换，大多是硬标签；许可证按源数据逐行记录；与 Decision Index 至少 11 个基准同源，评测前要按上游去重 |
| Jev 类冒烟与校准评测 | `LocalLLaMA/typed-decisions` | 多种 | 4 个工作流，train 1,200 / test 400 个 case；Apache-2.0；`state` + `questions` 正是 `/v1/systemone` 请求体 |
| 是非 | BoolQ | noul | 旧版已有转换与协议 |
| 选择 | Banking77、CLINC150、AG News | choice | 有 encoder 强基线可对照 |
| 等级 | SST-5 | score | |
| 排序 | NFCorpus、MS MARCO | ranking | NFCorpus 旧版已有闭环 |
| 多标签 | Open-Jev 的独立标签题；通用基准待补 | multi_label | 调研还没覆盖通用多标签基准 |
| 只评测 | Decision Index 公开套件、JevBench | — | DI 只测 choice / noul，公开套件约 11 万个请求，官方一键运行要 96 GB 卡、72 小时；JevBench 输入封存，不能完整复现 |

另建议契约原生读取 Jev 类型化判断格式（`state` + `questions` + 目标），与 v0.1 HTTP 服务的请求格式一致；三份 Jev 类数据集的字段要逐个核对。来源是 Jev / TypeSafe 输出的数据集（如 jimothy、Jevstiller）按 §3.1 一律不收。

### 3.5 Awesome 清单：以现有草稿为准，只列差异与待定项

> **已定（2026-11-04）**：不单独发布清单，改为支持矩阵（见文首）。本节保留，供整理素材时参考。

清单正文以 AW（`drafts/awesome/README.md`，约 82 KB）及同目录 `CONTRIBUTING.md` 为准，本节不再给一份并行的结构规格（一事一处）。以哪份为准、下列差异怎么处理，由用户定（决策点 27）。

AW 现状（据 design-critic §1.3 的核对，本次未重读全文）：结构为 Official APIs → Open-Weight Models（按读出分组）→ Training Code → Datasets → Benchmarks → Serving & Formats → Papers → Business Scenarios → Tutorials & Community → Related Lists；证据用徽标（📄 / 🗣️）表示；手写 Markdown，没有 `entries.yaml`。

| 待定项 | AW 现状 | 本稿建议（不替用户定） |
|---|---|---|
| 源格式 | 手写 README | 未推荐。`entries.yaml` + 生成脚本便于校验固定链接与证据级别，但一人维护多一层工具；也可以只写一个校验现有 README 的脚本 |
| 是否加 Readout taxonomy 节 | 无独立节，读出只作为模型分组 | 加：每类读出链到张量图和代表实现，是清单与工具包的连接点 |
| 是否加 Chinese ecosystem 节 | 无独立节，中文条目散在各节（Chinese-Jev、minimind-jev、ChineseText2SQL 等） | 可不单列，加一个语言标记即可 |
| 是否加 Excluded / flagged 节 | 无 | 随决策点 20（违反条款项目排除还是收录并标注） |
| 证据表示 | 两种徽标 | 映射到四级（source-verified / doc-verified / author-reported / unverified），映射方式待定 |
| 固定链接 | 部分未固定（如 `minimind-jev`） | 首发前逐条补 commit / HF sha / arXiv 版本号 |
| 条款标记 | 未逐条核对 | 首发前逐条核 AGPL / non-commercial / ToS-violation |

**维护**：

- 数字只收"有出处的"，并写明口径；"击败 Jev"类说法默认降级（SOTA §2.9）。
- 剩余缺口：coverage-map §2.1（00:45 版）已部分过时，Schema Linking 文献已由 SL 补上，中文生态在 AW 已有条目。仍缺：ModelScope 系统搜索、国内大厂 API、两份同类 awesome 列表的非训练小节。首发前补。
- 违反条款的项目（jimothy、Jevstiller）：排除还是收录并标注，待定（决策点 20）。

### 3.6 张量可视化约定

已有基础：[`docs/readouts/`](readouts/) 的 7 类读出图 + 总览 + 记号表，`readout-style.tex` 统一配色与宏，用 Tectonic 编译。

约定：

- **记号一处定义**：`readout-0-notation` 的符号表（`B, T, d, V, K, L, M, t*` 等）就是代码里的形状注释用语；代码注释写 `# [B, T, d]`，图上标同样的轴名。
- **图码对齐**：每张图对应一个函数或模块；图的 `.tex` 头注释写源文件路径与提交号，代码 docstring 反向引用图名；CI 用一个小脚本检查双向引用存在（不检查语义）。
- **形状可执行**：每个读出模块带一个 tiny 配置下的形状测试，测试里的形状与图中轴名一致；图的物理边长现状为 `readout-style.tex` 第 2–3 行的取值（T=12、d=5、V=12、K=L=3、B=2、d_h=4）。**这是现状，不是约定**：HANDOFF 第 74 行写明张量图风格待用户定，见决策点 22。
- **注意力可见性**用 `\Mask` 的 5 种模式表示（树前缀 / 双向 / pointer / 普通因果 / 独立块），与代码里的 mask 构造函数一一对应。
- `metiscraft viz` 在核心闭环之后实现（DP 一-16）。未定：是否另出 SVG 给网页；默认语言用英文还是中文。

## 4. 三条互斥的构建路线

三条路线共享 §3 的全部内容（契约、类型、协议、配方、清单、图），区别只在"训练与读出的核心挂在谁身上"。下面每条都要能端到端跑 Schema Linking：BIRD → 转换器 → 训练 → 按库校准 → 评测报告 → 可加载产物。Unsloth 插件路线建议排除（理由：决策代码 AGPL-3.0-only，逐文件 SPDX，与 Apache-2.0 冲突；且 `DECISION_HEADS = ("clef",)` 无注册机制，RQ §0 第 5 条、§3），只作对照；是否确认见决策点 23。

### 路线 A：上游优先（建在 transformers `decision` + TRL `DecisionTrainer` 上）

- **Schema Linking 怎么跑通**：自己写 `CandidateListForDecision(DecisionMixin, PreTrainedModel)` + 自己的 config 类（`DecisionConfig` 是 `@strict`，不能加字段，RQ §2.3），用 `AutoModelForDecision.register(...)` 注册；损失写在 `forward(labels=...)` 里；训练用 TRL `DecisionTrainer`；multi_label 在上游 `Question` 体系里没有，要么新开 PR 扩类型，要么在本地子类里自定义。
- **包结构（与其他路线不同的部分）**：

```
metiscraft/
  hf/            # XxxConfig、XxxForDecision（每种读出一个），register() 在 import 时调用
  hf/questions/  # MultiLabelQuestion、RankingQuestion（本地扩展，或上游 PR 的镜像）
  trainers/      # DecisionTrainer 子类，只补 multi_label/ranking 指标
```

- **依赖与许可证**：transformers（某个包含 #50054 的版本或 PR 分支）、TRL（#8040 合并后）、PEFT；全部 Apache-2.0。
- **成本**：中到大，且**依赖他人时间表**；#50054 已获核心维护者 Approve（11-02），但 mergeable_state 为 dirty、3 项 CI 失败，并要求注明实验性（RQ §0 第 3 条）；现在就开工意味着基于未合并分支，每次 rebase 都要追。
- **风险**：接口自声明实验性；三种 kind 写死，新类型要说服维护者；损失在模型内部，"同一头换损失"的对照不方便；一人维护拿不到 review 带宽；贡献后核心归 HF，工具包本身的定位变薄（RQ §4 路线 A）。
- **HF PR 合并与否**：
  - 合并：路线 A 从"不可行"变为"可行但受限"；对齐成本下降，用户安装体验最好（`pip` 原生）。
  - 不合并或大改：路线 A 直接失效，已写代码大部分作废。
  - 无论合并与否，multi_label / ranking 都不在上游，需要新 PR（会产生公开发帖，属红线 2，需用户批准）。

### 路线 B：做 ms-swift 插件（TRL 版等同路线 A，不单列）

- **Schema Linking 怎么跑通**：`--external_plugins metiscraft_swift_plugin.py` 注册 `ScoringHead` 子类（逐候选标量头）、`ScoringModelLoader` 子类、模板、`loss_map`（BCE + listwise）、`callbacks_map`（按 level 校准、Schema Linking 指标）；multi_label 只能**编码成"每列一个 noul 问题"**放在同一 prompt 里（head 的 `total_q` 每问题一行），因为 `QUESTION_TYPES` 和 `task_type` 都是写死的 `Literal`，不 fork 加不了（RQ §0 第 2 条、§1.3）。
- **包结构（不同部分）**：

```
metiscraft/
  swift_plugin/  # 单文件垫片（绝对导入）+ heads.py / losses.py / callbacks.py / templates.py
  translate/     # MetisCraft 配方 → swift sft 参数；产物 scoring_head.json → artifact.json
```

- **依赖与许可证**：`ms-swift>=4.6.0`（Apache-2.0，插件可保持 Apache-2.0）；随之引入 ms-swift 的全部依赖树。
- **成本**：最小（"一个下午出第一个头"，RQ §4 路线 B），但类型扩展做不了。
- **风险**：`ScoringHead` / `ScoringOutput` / `decision_meta` 10-13 才合并、无稳定承诺；用户要手动传插件文件路径，没有 pip 即生效；绑定 ms-swift 参数体系与 Qwen3.5 主导的模型矩阵；**"每列一个 noul"没有集合交互，⑤ 做不到**；差异化（类型语义）在这条路线上被削弱成"变通"。
- **HF PR 合并与否**：对路线 B 本身影响小；但若 HF 标准化成功，ms-swift 很可能跟进对齐，插件需要再迁一次。

### 路线 C：自建可组合核心（RQ 的推荐；本稿不替用户选）

- **Schema Linking 怎么跑通**：自有 `Readout` / `QuestionType` / `Objective` / `Calibrator` 注册表；`candidate_scalar` 读出 + `list` 布局 + `multi_label` 类型 + BCE / listwise / hierarchy 损失；训练循环用 HF `Trainer` 子类（只重写 `compute_loss` / `create_optimizer`，DP 一-7）；Predictor 与产物自有。另写两个**薄适配层**：导出为 HF `AutoModelForDecision` 兼容格式（`decision.json` 字段命名对齐 #50054）、导出为 ms-swift 可加载的 `scoring_head` 布局。
- **包结构（不同部分）**：

```
metiscraft/
  core/        # registry.py；questions/（6 类型）；readouts/（①②③，接口按 ④⑤ 留）；objectives/；calibrators/
  train/       # Trainer 子类 + collator（按问题数归一损失）
  adapters/    # hf_decision.py（导出 + 可选加载）、swift.py（导出 scoring_head）
```

- **依赖与许可证**：transformers、PEFT、Accelerate、datasets、sklearn、ir_measures；全部 Apache-2.0 / BSD / MIT。ms-swift 与 HF decision 只作为**可选 extras**，不进核心依赖。社区读出实现只读参考，逐个核许可证。
- **成本**：中到大；一人维护核心加两个适配层。
- **风险**：重复造轮子的诱惑（训练循环、分布式、量化都必须借）；上游一改适配层就要跟；"为什么不用 ms-swift"必须靠"类型语义 + 同协议对照 + 业务配方 + 清单"回答；4×3090 能验证的基座上限未测（此前写的"约 9B"无来源，已删；QLoRA 可能更大），v0.1 只承诺 0.6B–4B。
- **HF PR 合并与否**：
  - 合并：适配层从"字段对齐"升级为"真正可被 `AutoModelForDecision` 加载"；可以把某个读出头作为单点贡献送上游（路线 A 的子集）。
  - 不合并：不受影响；适配层只保留字段命名对齐。

### 对照（只列有区别的维度）

| 维度 | A 上游优先 | B ms-swift 插件 | C 自建核心 |
|---|---|---|---|
| multi_label / ranking | 要改上游或本地自定义 | 只能拆成多个 noul / score（有损） | 原生 |
| ⑤ 集合交互读出 | 可（自写模型类） | 不可（每问题独立） | 可（v0.1 可选插件） |
| 同协议读出对照 | 损失在模型内，不便 | 头绑定 loader，部分可 | 原生（设计目标） |
| 现在能否开工 | 否（等 #50054） | 能 | 能 |
| 外部时间表依赖 | 高 | 中（ms-swift 约定会变） | 低（只依赖稳定的 Trainer / PEFT） |
| 开发量 | 大且依赖他人 | 小 | 中到大 |
| 安装体验 | `pip`（合并后） | `pip` + `--external_plugins 路径` | `pip` |
| 许可证 | Apache-2.0 | Apache-2.0 | Apache-2.0 |
| 主要证据 | RQ §2、§4 A | RQ §1、§4 B | RQ §4 C；VL §4 |

## 5. 决策卡

按"今天是否卡住骨架代码"排序。"把握"指推荐本身的把握，不是对结果的预测。已定不再问：名字 MetisCraft、版本 0.1.0.dev0、Apache-2.0、定位、张量图用 Tectonic。

### 卡 1　构建路线（卡住：是，决定包结构和全部依赖）

- **选项**：A 上游优先；B ms-swift 插件；C 自建可组合核心 + 薄适配层（§4）。
- **推荐**：C，并把 B 的"导出到 ms-swift"和 A 的"对齐 HF 字段命名"作为适配层；A 只在 #50054 合并后做单点贡献。
- **把握**：中高。类型空白是硬事实（两边都写死三种 kind）；但 C 的"用户为什么选我们"仍是假设，没有用户验证。
- **证据**：RQ §0 第 1–6 条、§4 对照表；VL §4 第 1、3 条。
- **能否反悔**：部分可。§3 的契约、类型、协议、配方与路线无关，反悔成本主要是 `core/` 与 `train/`；越晚反悔越贵。
- **什么会改变结论**：ms-swift 或 HF 接受 multi_label / ranking 作为新 kind（则 B 或 A 的差异化损失变小，C 的必要性下降）；#50054 合并且损失变为可注册（A 吸引力上升）；有真实用户明确只用 ms-swift。

### 卡 2　v0.1 判断类型（卡住：是，决定 `core/questions` 与契约 `supervision.kind`）

- **选项**：
  - a. noul / choice / score（与竞品相同）；
  - b. a + multi_label；
  - c. a + multi_label + ranking（pointwise + ListNet）；
  - d. 全部含 box。
- **推荐**：c；score 的序数损失（相邻软目标 / CORAL / 累积 logit）先只做一种；box 不做。
- **把握**：中高。multi_label 是 Schema Linking 刚需且是唯一清晰的差异点；ranking 由检索重排（第二个配方）驱动，可以推到 v0.1 末尾。
- **证据**：EN §0 第 3 条、§4.3；RQ §2.4（上游无 multi_label / ranking / box）；DP 一-1、二-C。
- **能否反悔**：加类型容易（注册表）；删类型几乎无成本（预发布）。契约字段应一次留够。
- **什么会改变结论**：时间预算不够（先砍 ranking）；用户实际业务需要图像定位（加 box）。

### 卡 3　首发基座族（卡住：是，决定读出实现顺序、注意力实现与导出目标）

- **选项**：
  - a. **Qwen3（0.6B–4B）走答案 token / 逐候选标量头**：标准因果注意力、生态成熟；引擎上只能走答案 token，**引擎服务拿不到校准**，llama.cpp 不支持；可靠路径是自家 Predictor。
  - b. **Qwen3.5 小模型（d3-edge / lite 尺寸）走引擎可读布局（pplx / d3 255 码 readout）**：三家引擎都认的布局，d3 的模型卡自报"小 Qwen3.5 + 255 码 readout"可行（author-reported，VE §2 第 8 条）；但混合注意力、pplx 要求 SDPA、引擎只施加单一温度、提示词被引擎绑定。
  - c. 两者都做，但只把一个作为验收基座。
- **推荐**：倾向 a 作为 v0.1 验收基座（Schema Linking 的 ③-list 只需普通因果注意力，Qwen3 生态成熟）。**注意：以 ③ 作 v0.1 主线与 SOTA §5 的默认相反**。SOTA §5 的默认配方是"选项字母读出 + soft-CE + 小权重 Brier"，pointer / 标量头、encoder 打分、RL 都默认关闭，直到同骨干同数据自跑出消融。支持 ③-list 的证据只有 SL §5.3 和 ExSL（doc-verified；DeepSeek-Coder-6.7B 全参数微调；代码未找到）。所以选 ③ 是在没有自家消融的情况下押注，§6 验收的"可组合性"行用来尽早在冒烟任务上拿到 ① vs ③ 的消融。第二基座族只用于冒烟配方上的可组合性测试（Gemma 3 等标准注意力族，或 Qwen3.5）；b 的引擎布局只在用户确实需要引擎服务时进 v0.1。**这是本稿最需要用户拍板的一张**。
- **把握**：低（原为中低，因与 SOTA 默认相反而下调）。a 的代价是"引擎可服务"这一卖点后移；b 的代价是首发就要处理混合架构和提示词绑定。
- **证据**：VE §2 第 1–3、8 条，§3"C.5 第 3 条要重写"；SOTA §5 默认配方（反向证据）；SL §5.3 ③-list 行；ExSL（doc-verified）。
- **能否反悔**：读出与基座解耦是本项目的设计目标，换基座应只是换配置；但导出层（pplx 布局）是额外工作，不是可逆的"开关"。
- **什么会改变结论**：用户业务要求必须用 vLLM / SGLang 服务；Qwen3.5 在 4×3090 上 LoRA 显存或 SDPA 兼容性实测有问题；引擎支持按类型温度。

### 卡 4　旗舰场景（卡住：部分，决定第一个转换器和验收用例）

> **已定（2026-11-04）**：不设单一业务旗舰，见文首第 3 条。v0.1 验收建议（待拍板）：是非用 BoolQ，选择用 Banking77（CLINC150 作对照），等级打分用 SST-5，排序用 NFCorpus（旧版已有闭环）；多标签的公开基准调研还没覆盖，待补。以下保留当时的分析。

- **选项**：a. Schema Linking 作旗舰 + 意图识别作冒烟；b. 检索重排作旗舰（数据最丰富）；c. 意图识别作旗舰（最简单）。
- **推荐**：a。Schema Linking 是用户最初场景，且正好落在官方原语和竞品之外；意图识别只作为管线冒烟和同协议对照（诚实结果可能是 encoder 更好）。
- **把握**：高（锚点业务 + 差异化一致）；工作量是主要不确定性。
- **证据**：EN §5.1、§5.2；SL §1 第 1、4、5 条。
- **能否反悔**：可以；配方层面互不依赖。
- **什么会改变结论**：用户补充的实际业务约束（SL §4.1：库规模、方言、延迟、语言）与 BIRD 差太远；CC-BY-SA 再分发问题无法接受。

### 卡 5　Awesome 清单的名字和位置（卡住：否，但首发前必须定）

> **已定（2026-11-04）**：不单独做清单，改为支持矩阵，见文首。以下保留当时的分析。

- **选项**：
  - 位置：a. 本仓库 `awesome/` 子目录；b. 单独仓库。
  - 名字（单独仓库时）：如 `awesome-jev-like`、`awesome-decision-models` 类（与现有重名风险）、`awesome-typed-decisions`；具体名字由用户定。
  - 本卡只讨论位置和名字，前提是自建（定位第 ① 部分）。"不自建、只向 AnotiaWang/awesome-decision-models（681★）或 andyrewlee/awesome-system-one（42★）贡献"等于删掉定位第 ① 部分，是修改已定定位，不作为本卡选项；用户若想走这条路需单独确认。
- **推荐**：倾向 b，差异点是证据分级、固定链接、读出分类、核实过的数字、链接到场景配方；另外可以把勘误以 PR 回馈现有列表（不替代自建；公开发帖，红线 2）。
- **把握**：中。自建的价值取决于能否持续维护；一人维护两仓库有成本。
- **证据**：DP 二-F；coverage-map §1 (i)、§2.1 第 3 条。
- **能否反悔**：位置可迁（`git subtree`）；名字发布后改代价较高。
- **什么会改变结论**：现有列表维护者接受证据分级等结构性改动；用户希望品牌集中在 MetisCraft 一个仓库。

### 其余决策点（去重后，按"卡住骨架"程度排序）

| # | 决策点 | 选项 | 推荐（不替用户定） | 卡住骨架？ | 证据 |
|---|---|---|---|---|---|
| 6 | v0.1 读出范围：**是否必须有一个可对照的读出对** | 甲：①③ 各服务不同类型（① 管 noul/choice，③ 管 multi_label），不构成读出对照；乙：在同一类型、同一数据、同一基座上至少一对读出可只改配置互换（如 choice 上 ① vs ③，③ 逐候选标量头加 softmax 即可服务 choice，推断）；丙：乙 + ②（视卡 3） | 倾向乙（VL §4 建议把"读出 × 基座解耦 + 同协议对照"放进 v0.1 核心；否则头号差异点无法验证）；接口按 ④⑤ 留，⑤ 作 ③-list 上的可选插件；由用户定 | 是 | DP 一-2；VL §4；SL §5.3；design-critic §1.1 |
| 7 | 布局 | pairs / ③-list / tree | ③-list 主推，pairs 作正确性参照，tree 推迟；顺序敏感度 >0.05 则提前做 tree | 是 | SL §5.8；DP 一-5 |
| 8 | 与 HF 决策接口的关系 | 对齐字段 / 自定义后写适配 | 对齐字段命名（随卡 1） | 是（随卡 1） | DP 二-A；RQ §2.2 |
| 9 | 配置系统 | dataclass + OmegaConf / pydantic | 未推荐；dataclass 与 LlamaFactory / TRL 同族，pydantic 校验更严；JSON 不需 PyYAML | 是 | DP 一-8 |
| 10 | 训练后端 | `Trainer` 子类 / Accelerate 自写 | `Trainer` 子类；补 `remove_unused_columns=False`、按问题数归一损失、GAS 等价测试、多卡自动 DataParallel 的坑 | 是 | DP 一-7；K#14、K#15 |
| 11 | 头与 PEFT | PEFT 外单独存 / `modules_to_save` | PEFT 外、strict 加载；头 fp32、两组学习率；梯度裁剪按角色配置 | 是 | DP 一-4；K#14 |
| 12 | noul / choice 默认读出初始化 | verbalizer / MLP + 先验 bias | verbalizer（保留零样本、本身是基线） | 否 | DP 一-3；K#6 |
| 13 | 协议层位置 | 默认开启 / opt-in 严格模式 | 未推荐（S 偏向默认开启） | 否 | DP 一-11 |
| 14 | 导出 | Predictor / + HF CausalLM / + 引擎布局 / + SeqCls | **已定（11-04）**：v0.1 做 Predictor + HTTP 服务，引擎导出放 v0.2，见文首。原推荐：v0.1 只承诺自家 Predictor；"合并 CausalLM"只对 ① 有意义（③ 的产物是骨干 + 独立头文件，只能由 Predictor 加载）；引擎布局随卡 3 | 否 | DP 一-9；VE §3 |
| 15 | Schema Linking 工作量计入 v0.1 | 计入 / 推到 v0.2 | 计入（转换器本身是成果） | 否 | DP 二-D；SL §5.1 |
| 16 | 数据许可与泄漏策略 | 默认过滤非商用 + 按 DI 源去重 / 只标注 | 未推荐；至少默认输出许可证与泄漏报告 | 否 | DP 二-E；VL §3 第 2–3 条 |
| 17 | 外部评测与旧基准重跑 | DI 子集 / Banking77 协议 / 重跑 NFCorpus、BoolQ | Banking77 协议作第二个真实任务候选；重跑需 GPU，属红线 3 | 否 | DP 一-10；K#2 |
| 18 | 默认超参 | 写死 / 配方显式 + sanity 扫描 | 配方显式 + 扫描 | 否 | DP 一-12 |
| 19 | RL / 反馈接口 | 只留奖励签名 / 预留 Policy / Experience | 只留签名。**v0.1 对"能从反馈中学"这条理念只做到一个签名形状测试**（能按 TRL `reward_funcs` 的形状被调用），用户是否接受 | 否 | DP 一-13；SOTA §1 第 4 条 |
| 20 | 违反条款项目的收录 | 排除 / 收录并标注 | 未推荐 | 否 | DP 二-G |
| 21 | 旧内容处置 | 并入 `Shichaoxx/Archive` / 只靠 tag | 未推荐；私有忽略项放 `.git/info/exclude` | 否 | DP 一-15 |
| 22 | 张量图风格、SVG 与默认语言 | 风格（现 `readout-style.tex` 取值是否沿用）；只 PDF/PNG / 另出 SVG；英 / 中 | 未推荐（HANDOFF 第 74 行：风格待用户定） | 否 | DP 一-16 |
| 23 | Unsloth 只读不拷、不做适配 | 确认 / 否 | 建议确认（AGPL）；并规定不对照 Unsloth 决策代码写同功能实现（读后重写仍有衍生作品风险），只参考 Apache-2.0 的 ms-swift `ScoringHead` | 否 | RQ §5 第 3 条；VL §4 |
| 24 | 是否向 ms-swift / HF 提议新 kind | 提 / 不提 | 需用户批准（公开发帖，红线 2） | 否 | RQ §5 第 4 条 |
| 25 | DuSQL / Chase 是否纳入 | 纳入 / 先核许可证 | 先核许可证 | 否 | SL §1 第 8 条 |
| 26 | 实际业务约束 | 用户补充 | — | 否（影响卡 3、卡 4） | SL §4.1 |
| 27 | 清单以哪份为准、怎么补差异 | 以 AW 为准 / 按 §3.5 旧规格重做 | 以 AW 为准，只处理 §3.5 表里的差异 | 否（首发前定） | design-critic §1.3 |
| 28 | 卡 1 推荐 C 的两个适配层（`hf_decision.py`、`swift.py`） | 进 v0.1 / 推到 v0.2 | 推到 v0.2；v0.1 只做字段命名对齐 | 否（随卡 1） | design-critic §2.5 |

## 6. v0.1 范围与验收标准

范围以卡 1–4 和决策点 6 的推荐为假设写出；用户改了哪张卡，这里跟着改。

**在范围内**：§3 全部契约与协议；类型 noul / choice / score / multi_label（ranking 视卡 2）；读出 ①③，且至少一对可对照（决策点 6 乙；②视卡 3）；冒烟配方上 2 个基座族（第二基座只用于可组合性测试）；③-list 布局 + pairs 参照；`Trainer` 子类训练（单卡与 4 卡 DDP）；温度校准；评测报告；自家 Predictor（①的合并 CausalLM 导出可选）；配方：意图识别（冒烟）、Schema Linking（旗舰）；Awesome 清单首版（以 AW 为准）；7 类读出张量图入文档；奖励签名形状测试。

**不在范围内**：box、pointer、tree 布局（除非顺序敏感度触发）、RL、引擎布局导出（除非卡 3 选 b）、`hf_decision.py` / `swift.py` 适配层（决策点 28，推到 v0.2）、自写训练循环、量化、4B 以上基座。

**里程碑与估计**（一人全职人日，全部为推断，未有历史数据校准）：

| 里程碑 | 内容 | 估计人日 |
|---|---|---|
| M0 骨架 | 契约与 manifest、6 类型注册表、配置系统、tiny 随机初始化模型的 CI、形状测试 | 8–12 |
| M1 冒烟闭环 | ①③ 读出、`Trainer` 子类（单卡 + DDP，全局归一）、温度校准、评测报告、Predictor、产物封存、可追溯、Banking77 转换器、encoder + LR 基线、可组合性矩阵 | 15–20 |
| M2 Schema Linking | sqlglot 推 gold 的 BIRD / Spider 转换器、层级 multi_label 与 5 类损失、③-list + pairs、BM25 / bge-reranker 基线、顺序敏感度、按库校准 | 15–25 |
| M3 清单首发 | AW 补固定链接、证据级别与条款标记；张量图入文档、双向引用检查 | 5–8 |
| 合计 | | 约 43–65 |

**砍件顺序**（时间不够时从上往下砍）：ranking → 第二基座（砍掉后"可组合性"降为"1 基座 × 2 读出"，README 写明跨基座未验证）→ 适配层（已默认推迟）→ 评测门禁严格模式。

**验收标准**（全部为可自动运行的测试或报告；涉及真实训练的项需用户批准 GPU 使用，宪法红线 3）：

| 类别 | 标准 | 依据 |
|---|---|---|
| 等价性 | (1) 二元答案 token 读出 = verbalizer 线性头（两行 lm_head 之差），logits 逐位一致（容差写明 dtype）；(2) ③-list 中共享前缀部分的 token 与 pairs 布局逐位相同（K#10 的 adapt 口径，只保证正文 token）；(3) 导出产物在新进程加载后 logits 回读与训练进程一致；(4) 逐候选打分时，删一个无关候选不改变 pairs 布局下其他候选的 raw score（③-list 下改为**测量并报告**顺序敏感度） | S 第 16 行；DP 一-9；K#2 `v2-upgrade.md:38`；SL §5.8 |
| 可组合性 | 冒烟配方（意图识别，0.6B 级）上，至少 2 个基座族 × 至少 2 种读出（① 与 ③，或 ① 与 ②），**只改配置、不改代码**全部跑通，并同表输出同协议对照（含校准列）；CI 里用 tiny 随机初始化模型跑同样的组合矩阵，不占 GPU | VL §4；design-critic §1.1 |
| GAS 等价 | tiny 模型上，`batch=b×GAS=k` 与 `batch=k·b×GAS=1` 的损失与梯度在容差内一致；每问题候选数不同时仍一致；1 卡与 2 卡 DDP 同理。**归一口径**：损失按全局问题数归一，问题数跨 rank all-reduce（类似 `num_items_in_batch`），不用 HF `Trainer` 默认的按 rank 平均；容差按 dtype 写死（如 fp32 1e-6、bf16 1e-2，示意） | DP 一-7（B.4 补充）；口径为推断，未在源码核实 |
| 反馈接口 | 奖励签名形状测试：给定一批 Predictor 输出，奖励函数能按 TRL `reward_funcs` 的签名形状被调用并返回逐样本标量 | §1.2；决策点 19 |
| 校准 | 报告默认输出 NLL、Brier、ECE（口径常量打印在报告头）、可靠性图；温度缩放前后 argmax 不变的测试；校准器只在 calib split 拟合的测试；Schema Linking 的 calib / test 与 train 库不相交的测试 | §3.3；SOTA §4.3 |
| Schema Linking 端到端 | 一条命令从 BIRD（及 Spider）原始文件 → 转换器（sqlglot 推 gold，`parse_ok` 统计）→ 训练 → 按 level 校准 → 报告（表 / 列 Recall@5/10/20、SRR@k、SRR@τ、目标 SRR 下保留列数、PR/ROC AUC、F6、逐列 ECE / Brier、"整题漏召回"校准）；与 BM25 和 cross-encoder 基线同表；顺序敏感度报告；产物可被 Predictor 加载。**不设绝对数字门槛，但设相对且预先登记的门槛**：列级 Recall@10 不低于 BM25 基线；校准后 ECE 不高于校准前；顺序敏感度（不同排列下的标准差）报告值不超过 0.05，超过即触发 tree（SL §5.8 阈值）。不过线时报告照常发布（负结果也是证据），但该配方不标为推荐配方。ExSL 的 BIRD-dev 列级 PR AUC 93.77（doc-verified；DeepSeek-Coder-6.7B 全参数）只作参照，不同基座规模不可直接比 | SL §3.4、§5.5、§5.6、§5.8；SL §1 第 2 条 |
| 冒烟与对照 | 意图识别（Banking77 协议 80/10/10）走通全流程，并输出与 encoder 基线的同协议对照表（含校准列） | EN §5.2；SOTA §5 复现目标 1；K#2 |
| 可追溯 | 任一报告数字可追到运行目录；运行目录含配置、git 提交 + diff、环境、数据与基座身份；产物带 SHA256SUMS 与 NOTICE；NOTICE 自动带上基座许可证，配方 README 必填基座许可证（如 Qwen3-1.7B 的许可证需在配方里写明） | 宪法 §4.1；§3.3 |
| 清单 | 首版每条都有固定链接、许可证、证据级别；校验脚本通过（源格式见决策点 27） | §3.5 |

## 7. 风险

| # | 风险 | 影响 | 缓解 | 证据 |
|---|---|---|---|---|
| 1 | **主流框架补上类型空白**（ms-swift / HF 加入 multi_label / ranking） | 差异化只剩对照与业务配方 | 尽快发布 Schema Linking 配方与对照报告；适配层保证能跟随；清单作为独立价值 | VL §4；RQ §2.5 |
| 2 | **"为什么不用 ms-swift"答不上来** | 没有用户 | README 首屏给出同协议对照表与 Schema Linking 校准报告；不宣称"更准" | EN §0 第 4 条 |
| 3 | **③-list 顺序敏感** | 违反"候选独立"不变量，校准变差 | 训练打乱、推理多排列平均；超过阈值提前做 tree | SL §5.3、§5.8 |
| 4 | **引擎服务时校准失效** | 用户以为概率可信，实际不可信 | 导出产物写明"引擎只施加单一温度 / 不施加温度"；Predictor 为一等产物 | VE §2 第 2–3 条 |
| 5 | **许可证与条款**：Spider / BIRD 为 CC-BY-SA；多个审核数据非商用；Unsloth 决策代码 AGPL；用 Jev 输出训练违反 TypeSafe 条款 | 产物或代码被污染 | 转换器输出许可证字段并进 NOTICE；Unsloth 只读不拷；不以 Jev 输出作训练标签 | EN §6.1；RQ §0 第 5 条；DP 二-G |
| 6 | **数据泄漏**：TS 与 DI 至少 11 个基准同源 | 外部数字虚高 | 按源去重或留出；报告标注"训练源留出"与"零样本"之别 | VL §3 第 3 条；SOTA §6 |
| 7 | **上游接口漂移**：ms-swift 约定、#50054 实验性 | 适配层频繁失效 | 适配层只做导出、放 extras、钉版本 | RQ §1.3、§2.5 |
| 8 | **一人维护 + 范围过大**（核心、两适配层、四个配方、清单） | 拖期、半成品 | v0.1 只承诺卡 2–4 的最小集；按 §6 砍件顺序砍（ranking → 第二基座 → 适配层 → 严格门禁）；适配层默认推到 v0.2 | §6 里程碑表（人日为推断） |
| 9 | **小模型能力上限**：≤5B 与 Jev 差距明显 | 结果不好看 | 目标定为"同协议对照 + 校准"，不追 Jev 分数 | SOTA §1 第 2 条 |
| 10 | **全判肯定等旧问题原因未明** | 新核心复现旧缺陷 | 默认不用"零输出权重 + LoRA B=0"；sanity 阶梯先过 | DP 一-3；K#6 |
| 11 | **调研缺口**：ModelScope 系统搜索、国内大厂 API、两份同类 awesome 列表的非训练小节未覆盖（Schema Linking 文献已由 SL 补上，中文生态在 AW 已有条目） | 清单与竞品判断有盲区 | 首发前补一轮；"不存在"类结论保持两渠道 | coverage-map §2.1（00:45 版，已部分过时）；SL；AW |
| 12 | **算力**：能验证的基座上限未测；GPU 共享 | 大基座结论无法自证 | 只承诺 0.6B–4B；真实实验先申请（红线 3） | 无来源（原"约 9B"已删） |
| 13 | **软标签与参考实现的条款污染**：教师分布来自 Jev / TypeSafe API；对照 AGPL 代码重写 | 产物违反 TypeSafe 条款；代码被认定为 AGPL 衍生 | `provenance` 必填教师身份与许可证，转换器拒收 Jev 来源软标签；不对照 Unsloth 决策代码写同功能实现 | design-critic §2.8 |

## 评审记录（2026-11-04，对照 `drafts/design-critic.md`）

已落实（只列位置）：1.1 → 决策点 6 改写、§6 范围与"可组合性"验收行、卡 3；1.2 → 卡 5；1.3 → §3.5 重写、决策点 27；1.4 → §4 引言；1.5 → §3.6、决策点 22；1.6 → §1.2、决策点 19、验收"反馈接口"行；2.1 → 卡 3（推荐理由与把握）；2.2 → §2 结论 2、§2 表 Unsloth 类型格；2.3 → CLINC / d3 / ExSL 标级别，"约 9B"删除（§4 C、风险 12）；2.4 → 验收 SL 行、GAS 行；2.5 → §6 里程碑与砍件顺序、决策点 28、风险 8；2.6 → §0 第 4 条、决策点 14、§6 范围；2.7 → §4 路线 A 成本；2.8 → §3.1 `provenance`、决策点 23、验收"可追溯"行、风险 13；2.9 → §2 结论 2；2.10 → §3.5 维护、风险 11。

做法与批评不完全一致的地方：

1. **1.1 读出对照不一定要 ②**：批评建议"① 与 ③，或 ① 与 ②"。本稿补充：③ 逐候选标量头加 softmax 就能服务 choice，所以在冒烟配方的 choice 上做 ① vs ③ 即可构成读出对照，不依赖卡 3 是否选 b（推断，未实测）。
2. **1.1 第二基座族**：批评举例 Qwen3.5 或 Gemma。本稿倾向优先标准注意力族（如 Gemma 3），理由是 Qwen3.5 混合注意力与 pplx 要求 SDPA（VE §2），会把冒烟测试变成架构适配；但留给卡 3，不定死。
3. **2.4 相对门槛不过线的后果**：批评只给了门槛。本稿规定不过线时报告照常发布、该配方不标为推荐，而不是阻塞整个 v0.1 发布；理由是负结果对"同协议对照"本身也是证据，且一人维护不宜让单配方卡住发布。
4. **2.5 砍件顺序与 1.1 的冲突**：批评同时要求"可组合性至少 2 个基座族"和"砍件顺序第二位是第二基座"。本稿保留两者，并写明砍掉第二基座时可组合性降为"1 基座 × 2 读出"、README 注明跨基座未验证。
5. **2.5 人日数字**：§6 的人日是本稿的推断，没有历史数据，只用于排序和砍件，不作承诺。
6. **2.9、1.3 的事实**：EN §4.3 的计数和 AW 的结构直接采纳批评的核对结果，本次修订因额度限制未重读原文。

<!-- complete -->

## 外部核查（2026-11-04 05:20，正文未逐处改写，以本节为准）

两个反驳者对本稿依赖的 6 条说法做了对抗核查，全文见 `drafts/design-verify-frameworks.md` 和 `drafts/design-verify-evidence.md`。

| # | 说法 | 判定 | 要点 | 影响 |
|---|---|---|---|---|
| 1 | ms-swift 与 HF/TRL 都把类型写死为 noul/choice/score，不支持 multi_label 和 ranking | corrected | HF #50054 的 `Question.type` 是 Literal 三选一，TRL #8040 也只有三种。ms-swift 的 decision 任务内确实写死，写死处比 RQ 列的多。但 ms-swift 框架本身已有 ranking（`task_type` reranker / generative_reranker，pointwise / listwise reranker 损失）和 multi_label（`problem_type` multi_label_classification，模板转多热） | 卡 2：ranking 已不是差异点，建议先做 b（三种 + multi_label），ranking 列为第一个砍的。卡 4：把 ms-swift 的 reranker 加进基线 |
| 2 | ms-swift 不 fork 能接读出头、损失、回调，但新类型必须 fork | weakened | 前半句经源码确认。"必须 fork"说过头了：多热 gold 加自定义头、BCE 和自定义校准回调，就能在 choice 外壳里做 multi_label；ranking 用 reranker；也可以打猴子补丁。只有把新类型做成一等名称、并让自带的校准和指标按它的语义处理时，才需要 fork。以上都未实测 | 卡 1：仍推荐 C，但把握从中高降为中；对照表里"类型空白是硬事实"要改写 |
| 3 | #50054 已获 approve，但有冲突和 CI 失败；#50203、#50246、TRL #8040 是 draft | confirmed + 两处更正 | Lysandre 于 11-02 approve，mergeable_state 为 dirty。质量 CI 失败来自 10-29 那次运行，26 个测试作业全部 skipped，测试其实从没跑过。TRL #8040 现为 blocked | 路线 A 比本稿写的更不乐观 |
| 4 | ExSL：BIRD-dev 列级 PR AUC 93.77 对 69.18 | corrected | 93.77 是 ExSLc（粗粒度），ExSLf 是 93.67。69.18 的生成式基线用采样计数当置信度，论文自称 crude；按 F6 比，只是 91.04 对 96.21。BCE 和 DeepSeek-Coder-6.7B 属实，没写是全参数还是 LoRA。官方仓库 IBM/extractive-schema-linking 只有 LICENSE 和 README，是空壳 | 卡 3：把握保持低，① 对 ③ 的消融必须做。§6 的 SL 验收要注明 ExSLc 和空仓库 |
| 5 | CLINC：BGE-small + LR 0.948 > Jev 0.913 | corrected | 来源是 arXiv 2610.00346（第三方）。测试集是 600 条 in-scope 加 200 条 OOS，150 个选项。DeBERTa-v3-large 微调 0.953；零样本的 Gemma-4-31B 用选项似然也有 0.940，都高于 Jev 1.13。差异显著性没读到 | §2 结论 3、风险表、卡 4：对照臂加入 DeBERTa-v3-large，复现 600+200 的抽样协议 |
| 6 | "带层级的业务多标签没人做" | weakened | Jev 格式、带多标签 gold 和逐候选校准的 Schema Linking 训练数据或工具，确实没找到。但有反例：jev4pg（Apache-2.0，约 260★）用 Jev 选字段，并在 BIRD Challenging 100 上评测过；jevsql 先选表、再逐个选列，是两级单选；TypeSafe 有层级分类 cookbook（未打开核实）；ModelScope 上有 Snowflake/SQL-Schema-Retrieval | §2 结论 2 删掉"层级"。卡 2 的差异点改写为"多标签 + 逐候选校准"。卡 4 加托管 Jev 字段选择的对照臂，把握从高降为中高 |

<!-- complete -->
