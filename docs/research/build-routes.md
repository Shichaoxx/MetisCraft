# 构建路线的关键问题（上游优先 / 框架插件 / 自建核心）

> 设计前调研材料（2026-11-04）。设计结论以 [`docs/design.md`](../design.md) 为准；文中引用的 coverage-map、`notes/`、`partial/`、`landscape/` 等是维护者本机的调研材料，不随仓库发布。

> 范围：只读调研，未执行第三方代码。2026-11-04。ms-swift `df0bc0f` 与 Unsloth `8d2d7b6` 为浅克隆后阅读；PR 内容来自上轮抓取的原始数据（本机调研目录 `landscape/hf-std-raw/`） + 本轮 `.diff` 与 GitHub GET。
> 证据级别：source-verified / doc-verified / author-reported / unverified。

## 0. 结论（先行）

**推荐：自建可组合核心（路线 3），外加两个薄适配层：ms-swift 插件、与 HF `AutoModelForDecision` 兼容的导出。不押上游。**

1. **两个框架的判断类型都写死为 noul / choice / score 三种**。ms-swift：`QUESTION_TYPES`（`swift/model/decision_head.py` L40）；HF：`Question.questions_types`（#50054 `decision/template_utils.py`）。ranking、multi_label、box 在两边都没有，也没人在做。Schema Linking 要的多标签子集和排序，只能编码成多个 noul/score 问题（无集合交互），要真读出就得自己建。
2. **ms-swift 可以不 fork 注册读出头、模型、模板、损失、回调**（`--external_plugins` 导入 .py 文件，`loss_map` / `callbacks_map` 是普通 dict），但**不能不 fork 加判断类型和 `task_type`**，且没有 pip entry-points，用户要多传一个文件路径。API 是约定，不是公开稳定接口，10-13 才合并。
3. **HF 的决策接口还没合并，并且自己声明是实验性的**：#50054 有 Lysandre 的 Approve（11-02），但 mergeable_state 为 dirty，head 上 3 项 CI 失败（质量检查），Lysandre 的评审要求文档里注明"此 API 属实验性，随时可能变"。后续 Jeva #50203、CLM #50246、TRL #8040 都是 draft，依赖链是 #50054 → #50203 → #50246，TRL #8040 依赖 #50203。**最早也要等 #50054 合并后才能谈贡献**。
4. **损失在 HF 里是模型 forward 的一部分**（传 `labels` 返回 `loss`），不是可注册的钩子；TRL `DecisionTrainer` 复用 `Trainer.compute_loss`，所以第三方想换损失要写自己的 `XxxForDecision` 类。读出头可以通过 `AutoModelForDecision.register(...)`（transformers 的标准 Auto 注册）或 `trust_remote_code` 的 `auto_map` 接入，但 `DecisionConfig` 是 `@strict` 数据类，加字段要走自己的 config 类。
5. **Unsloth 的决策代码是 AGPL-3.0-only**（逐文件 SPDX 头），虽然 README 说"核心 Apache、Studio AGPL"。MetisCraft 是 Apache-2.0，不能拷贝或链接它的决策代码；`DECISION_HEADS = ("clef",)` 也没有注册机制。**排除 Unsloth 插件路线**，只当对照。
6. **定位随证据修正**：JevAny 等项目已提供统一工具包，当前核对见[支持矩阵](../support-matrix.md)。MetisCraft 继续验证业务配方、跨读出受控对照与可追溯部署的价值；这些组合尚未证实为独有空白，业务示例也不限定于 Schema Linking。

待定事项见 §5。

## 1. ms-swift：第三方 pip 包能否不 fork 注册读出头、判断类型、损失、回调

读源码：ms-swift `df0bc0fba64dcacd2e9250df4e924ed9e32e290f`（2026-11-03，main 的 HEAD；下文行号都对这个提交；全部 source-verified，只读，没有执行）。
固定链接前缀：`https://github.com/modelscope/ms-swift/blob/df0bc0fba64dcacd2e9250df4e924ed9e32e290f/`

**小结**：读出头、模型、模板、损失、回调，不 fork 就能注册；判断类型（kind）和新的 `task_type` 不行。没有 pip entry-points 机制，只有 `--external_plugins <文件路径>`，所以 pip 包要提供一个"垫片 .py 文件"，用户把它的路径传进去。

### 1.1 加载机制（唯一入口）

| 项 | 内容 | 位置 |
|---|---|---|
| 参数 | `external_plugins: List[str]`（`.py` 文件路径，不是模块名）；旧名 `custom_register_path` 并入同一列表 | `swift/arguments/base_args/base_args.py` L128–L129、L182–L205 |
| 实现 | `import_external_file(file_path)`：`sys.path.insert(0, 文件所在目录)`，用 `spec_from_file_location` 以合成模块名执行；导入副作用即注册 | `swift/utils/utils.py` L408–L432 |
| 多进程 | 记录已导入文件，dataloader worker 在 spawn/forkserver 下重放 | `swift/utils/utils.py` L435–L486；`swift/dataset/packing.py` L232 |
| 检查点自带插件 | `trust_remote_plugins` 默认 False，开启才导入 checkpoint 的 args.json 里声明的插件 | `base_args.py` L92、L190–L199 |
| 没有 | 全仓 grep `entry_points` / `importlib.metadata` 只有 setup.py 的 console_scripts（L161–L163）和版本检查，没有第三方发现机制 | setup.py L161 |

含义：
- 垫片文件里要用**绝对导入**（`import metiscraft.swift_ext`），因为文件以合成模块名加载，相对导入会失败。
- 用户命令里要多一个 `--external_plugins $(python -c "import metiscraft.swift_plugin as m; print(m.__file__)")`，不能"pip install 即生效"。
- ms-swift 自带的 jev.sh 也是这样用的：`examples/train/decision/jev.sh` L33 `--external_plugins examples/train/decision/dataset.py`。

### 1.2 逐项：能不能不 fork

| 要注册的东西 | 能否 | API 与位置 | 备注 |
|---|---|---|---|
| 新模型 + 新读出头 | 能 | `register_model(ModelMeta(...), exist_ok=False)`（`swift/model/register.py` L31–L42）。子类化 `ScoringModelLoader`，设 `head_cls`（`swift/model/models/decision.py` L27–L43），范例 `JevModelLoader` L101–L115、`KevModelLoader` L297–L344 | `ModelMeta(..., task_type='decision', ...)`，模板用 `TemplateType.*` |
| 读出头类 | 能 | 子类化 `ScoringHead`（`swift/model/decision_head.py` L140–L330）。必须实现 `forward(hidden_states, lm_logits, meta) -> ScoringOutput`、`from_pretrained(model_dir, model, config)`；可选 `extra_state_for_save`（L215）、类属性 `needs_lm_logits` / `needs_output_embeddings` / `keep_fp32` / `encoder_backbone`（L158–L171）；保存格式 `scoring_head.safetensors` + `scoring_head.json`（L218–L219） | 头是基类里的抽象，不用改 ms-swift 就能新增第 8 种读出 |
| 模板（含数据编码） | 能 | `register_template(TemplateMeta, exist_ok=False)`（`swift/template/register.py` L16–L20）；`DecisionTemplate` 在 `swift/template/templates/decision.py` L30–L127，`QwenTemplateMeta(...)` 注册 L134 | 模板的 `_encode` 必须产出 `decision_meta` 与 `target_probs` 标签，形状契约在 `decision_head.py` 文件头 |
| 数据预处理器 | 能 | `swift/dataset/preprocessor/decision.py`；范例通过 `examples/train/decision/dataset.py` 外挂 | 有注册表，未逐行核对 |
| 损失 | 能 | `loss_map` 是普通 dict（`swift/loss/mapping.py` L7–L22）；插件 `from swift.loss import loss_map; loss_map['my_loss'] = MyLoss`；类继承 `BaseLoss`，签名 `__init__(args, trainer)`、`__call__(outputs, labels, *, num_items_in_batch, loss_scale, **kw)`（`swift/loss/base.py` L23–L36）；选择 `--loss_type my_loss` | `loss_type` 是 `Optional[str]`，没有 choices 检查（`swift/trainers/arguments.py` L171，只在 help 里列）；使用点 `swift/trainers/mixin.py` L1147–L1148 |
| 回调 | 能 | `callbacks_map` 是普通 dict（`swift/callbacks/mapping.py` L10–L19）；插件往里加；选择 `--callbacks my_cb`；类签名 `__init__(args, trainer)`；使用点 `swift/trainers/mixin.py` L158–L162 | `--callbacks` 是 `List[str]`，无 choices（`swift/trainers/arguments.py` L177）。决策任务会自动追加 `temperature_calibration`（`swift/arguments/sft_args.py` L249–L252） |
| 评测指标 | 能 | `eval_metrics_map`（`mixin.py` L1141–L1144 使用） | 未核对注册位置 |
| 新的判断类型（box、ranking、multi_label） | **不能不改** | `QUESTION_TYPES = {'noul': 0, 'choice': 1, 'score': 2}`、`KIND_NAMES`（`decision_head.py` L40–L41）是模块常量；模板里 `_normalize_kind`、预处理器里的 kind 分支、各头的 `DEFAULT_RANGES`（L340、L533）、校准回调的按 kind 温度都按这三种写死 | 技术上可 monkeypatch 这个 dict，但等于绑死内部实现；box 在预处理器里有 `region.box`（L609）是多模态数据，不是 box 判断类型 |
| 新的 task_type | **不能不改** | `task_type: Literal[... 'decision']`（`swift/arguments/base_args/model_args.py` L72），经 `HfArgumentParser` 变成 choices（`swift/utils/utils.py` L178–L193）；`TrainerFactory.TRAINER_MAPPING` / `TRAINING_ARGS_MAPPING` 是类属性 dict，值是"模块.类"字符串（`swift/trainers/trainer_factory.py` L13–L40） | 绕法：复用 `task_type='decision'`，然后插件里改 `TRAINER_MAPPING['decision']` 指向自己的 Trainer 子类（猴子补丁，脆弱） |
| 自定义 Trainer | 间接 | 同上，只能替换 dict 里的字符串 | `ScoringTrainer`（`swift/trainers/decision_trainer.py` L12–L30）很薄，只是在 adapter 训练下重新打开 `scoring_head` 的梯度 |

### 1.3 判断：

- **读出头**：覆盖得最好。ms-swift 的 7 种头都是 `ScoringHead` 子类，第三方加第 8 种与自带的同级，不碰上游。
- **判断类型**：这是真正的缺口。三种 kind 是写死的，ranking / multi_label / 逐候选成行 / box 都不在。MetisCraft 要覆盖 Schema Linking（多标签子集、排序），在 ms-swift 里只能走"把多标签拆成逐列 noul"这种变通，或者向上游提 PR。
- **稳定性风险**：`ScoringHead`、`ScoringOutput`、`decision_meta` 的契约在 10-13 才合并，三周内会继续变；没有 semver 保证的公开 API，只有"约定"。`pip install ms-swift>=4.6.0` 是文档要求（`docs/source_en/BestPractices/Decision.md`）。
- **许可证**：ms-swift 根目录 `LICENSE` 是 Apache License 2.0（在该提交上核对），插件可以保持 Apache-2.0。
- **与 Schema Linking 的关系（推断，未实验）**：多标签子集可以编成"每列一个 noul 问题"，排序可编成"每候选一个 score 问题"，同一个 prompt 里放多个问题（head 的 `total_q` 就是每问题一行）。代价是列与列之间没有集合交互（读出⑤做不到），所以只算变通。

## 2. transformers #50054 系列与 TRL #8040 / #7784

### 2.1 PR 一览与评审信号

数据时间：GitHub GET，2026-11-04；"head"为当时 PR head sha。原始数据在本机调研目录 `landscape/hf-std-raw/*.json`，评审与评论为临时抓取，未保留。

| PR | 内容 | 状态（11-04） | 作者 | 评审信号 | 证据 |
|---|---|---|---|---|---|
| transformers #49322 | `transformers serve` 加 `/v1/systemone` 端点，`decision_config` 参数 | **已合并**（10-12） | SunMarc（成员） | 已合并 | source-verified（raw json） |
| transformers #49566 `bb3f575` | 最初的大 PR：Jev/Kev/Von/Clef + 跨仓库加载；76 提交，71 文件，+3874/−411 | open，标签 New model；**已被拆分取代** | ArthurZucker（collaborator） | Cyrilvallez 评论 8 条，认为太大、跨仓库递归加载是"从未允许过的大改动，可能影响很多仓库"，要求拆分；Arthur 10-28 回复"Jev/Kev/Von/Clef 现在是 XxxForDecision（#50054、#50056、#50057、#50058）" | source-verified |
| transformers #50054 `f4a96a9` | 决策核心 + Jev：`transformers/decision/`、`AutoModelForDecision`、`pipeline("decision")`、serve 统一路径；46 提交，36 文件，+2042/−384 | open，非 draft，**mergeable_state=dirty**；10-28 创建，11-02 更新 | ArthurZucker | LysandreJik 10-28 评论（要求兼容现有仓库、要一个把骨干和头包起来的整体 `DecisionModel`、支持 batch）；**11-02 APPROVED**，剩两件：最大的 JEV-35B 与 JEV-VL-27B 没支持（vLLM 布局，头在 lm_head 行里）、serve 与 pipeline 加载出的架构不一致；并要求文档标注实验性；无其他 reviewer；CI：3 失败（`pr-ci / PR CI status`、`Quality gate`、`Check code quality`）、12 成功、docs-check 成功 | source-verified |
| transformers #50056 / #50057 / #50058 | Kev / Von / Clef，逐层堆在 #50054 之上 | open，非 draft，10-29 后无更新 | Arthur | 无评审 | source-verified |
| transformers #50203 `b997f5c` | Jeva：监督损失 + 共享决策输入；+1711 | **draft**，10-31；base 为 `jev-1-core` 分支（堆叠） | kashif（contributor） | 无评审 | source-verified |
| transformers #50246 `077eb2d` | CLM 决策模型 + 对比损失；+2818 | **draft**，11-02；堆叠在 #50203 上 | kashif | 无评审 | source-verified |
| transformers #49747 `f98ec18` | `DecisionConfig.max_options`，头自报能打分的选项数 | open，base 为 `add-jev-models`；CI 等维护者批准（首次贡献者） | apolinario（无关联身份） | 请求 Arthur 评审，无回复 | source-verified |
| TRL #8040 `8971588` | 实验性 `DecisionTrainer`（`trl/experimental/decision/`）+ Jeva BANKING77 示例；+2111 | **draft**，10-30 创建 | kashif（collaborator） | 无评审，1 条运行记录（AMD 890M 1000 步） | source-verified |
| TRL #7784 `e6c7f20` | `examples/sft_decision_model`：SFTTrainer 子类，答案 token 读出，LoRA 训 Qwen3.5-4B | open，非 draft | rycerzes（contributor） | qgallouedec **APPROVED**（10-20）；sergiopaniego（member）10-27 要求先对齐 transformers 的 `DecisionTemplate`再合并；作者 10-28 同意并确认 40000 条里 39999 条 token 一致 | source-verified |

路线图信号：没有找到官方 roadmap 文档；唯一的路线表述是 Lysandre 的评审（"批处理是关键功能"、"应有统一 DecisionModel"）和 Arthur 的"后续 PR 逐个模型加转换"（49566 评论，10-15）。未验证：是否有 HF 内部排期。

### 2.2 拟定类名与接口（#50054 `f4a96a9`，均 source-verified）

- 包 `src/transformers/decision/`：`DecisionConfig`（`configuration_utils.py`，`@strict @dataclass`，字段 `labels`、`temperature`、`chat_template`、`noul_options`、`sanitize_control_tokens`、`adapter`；存为 `decision.json` 与 `decision_template.jinja`）；`DecisionMixin`（`modeling_utils.py`，提供 `from_pretrained`、`checkpoint_config`、`decision_config`、`decide(input_ids, decision_questions, ...)`）；`Question` / `NoulQuestion` / `ChoiceQuestion` / `ScoreQuestion`、`DecisionTemplate`（`template_utils.py`）。
- 模型：`JevForDecision(DecisionMixin, JevPreTrainedModel)`（`models/jev/modeling_jev.py`）。`forward(input_ids, marks, num_options, question_type_ids)` 返回每问题的选项分数；读出是"取每个 `<|mark|>` 前一个 token 的隐状态，过 `nn.Linear(hidden, num_verbalizers)`，加 softcap，按问题类型的偏移取行"，即读出②槽位头。
- Auto：`MODEL_FOR_DECISION_MAPPING_NAMES = {"jev": "JevForDecision"}`、`AutoModelForDecision`（`models/auto/modeling_auto.py`）；Lysandre 评审要求"头不该用 AutoModel 加载，应单独映射"，所以这一条是他们的主动设计。
- 管道：`pipeline("decision", model=..., decision_config=...)`，类 `DecisionPipeline`（`pipelines/decision.py`）。
- 按 PR 描述后续头：Kev（pointer）、Von、Clef（集合交互）在 #50056–#50058；Jeva（#50203）、CLM（#50246）；TRL 侧 `trl.experimental.decision.DecisionTrainer` / `DecisionConfig`，数据整理器 `DataCollatorForJeva`、`DataCollatorForClm`。

### 2.3 第三方如何注册新读出
- **Auto 类注册**：`AutoModelForDecision` 在 diff 中是 `_BaseAutoModelClass` 子类、映射为 `_LazyAutoMapping`。`_BaseAutoModelClass.register(config_class, model_class, exist_ok=False)` 是 transformers 的老接口（在本地的 transformers 4.40.2 里读到：`models/auto/auto_factory.py` L572，doc 级推断可延伸到 PR，未对 PR 分支运行）。所以第三方 pip 包可以：自带 `XxxConfig`、`XxxForDecision(DecisionMixin, PreTrainedModel)`，导入时调用 `AutoModelForDecision.register(XxxConfig, XxxForDecision)`。
- **trust_remote_code**：`auto_map` 里写 `"AutoModelForDecision": "modeling_x.XxxForDecision"` 即可，机制在 `auto_factory.py` L415–L423（4.40.2，同上推断）；`pipeline(..., trust_remote_code=)` 参数在 #50054 的 `pipelines/__init__.py` 里仍在。社区 `vllm-sr/d3*` 已有 `custom_pipelines: {"decision": ...}`，会抢先解析为它们自己的远程代码（rycerzes 在 #49566 的评论）。
- **子类与损失钩子**：损失在 `forward(labels=...)` 内部（`JevaDecisionOutput.loss`，#50203），没有损失注册表。TRL `DecisionTrainer.compute_loss` 直接调 `Trainer.compute_loss`，并要求模型返回 `loss` 和候选 `logits`；指标（`candidate_loss`、`brier`、`entropy`、`null_*`）写死在 trainer 里。可传自己的 `data_collator`、`callbacks`、`peft_config`。
- **约束**：`DecisionConfig` 是 `@strict`，未知字段会报错（rycerzes 实测 `format_version` 触发 TypeError）；第三方设置要放自己的 config 类。`extra_config_files` 与静态方法 `decision_config(head_configs, text_config)` 是"头与骨干分仓库"的钩子，不用就可以略过。

### 2.4 覆盖 multi_label / ranking / 逐候选成行？

| 能力 | 覆盖 | 证据 |
|---|---|---|
| multi_label（子集） | **没有** | 在 #50054 / #49566 / #50203 / #50246 / TRL #8040 的 diff 里 grep `multi_label`、`ranking`、`subset` 均零命中（只有 `allow_null`、`box` 各一次非功能出现）；`Question.questions_types` 为 noul / choice / score 三种 |
| ranking | **没有**（只有 score 的序数档位） | 同上 |
| box | 没有 | 同上 |
| 可空选择（弃权） | 有（Jeva / CLM / TRL） | `allow_null` 在 #50203 出现 24 次，#50246 27 次，TRL 33 次；目标 `target_probs` 长度为 k+1，首列是 null 概率（`normalize_target`） |
| 逐候选成行 | **部分**：CLM（#50246，draft）把每个选项单独编码（`option_input_ids`、`option_chunk_size`，分片与上下文并行） | TRL `compute_loss` 里的处理；#50054 的 Jev 头仍是同一序列里多 `<|mark|>` 槽位 |
| 一次前向多问题 | 有 | `decide` 接收问题列表，用 marks 读出（#50054） |

### 2.5 合并前景（判断，非事实）

- #50054：有 core maintainer 批准，且是 HF 员工自己的堆叠计划，**合并概率较高，但时间不定**（几周量级；这一条是推断）。拖累点：与 main 有冲突（dirty）、CI 质量项失败、两个最大模型未支持、serve 与 pipeline 不一致、Cyrilvallez 对跨仓库加载的顾虑是否已消除没有证据。
- Kev / Von / Clef（#50056–#50058）排在 #50054 后面；Jeva / CLM / TRL #8040 是 draft，依赖链更长，**TRL 的 DecisionTrainer 最晚**，且放在 `experimental` 命名空间，意味着接口随时会改。
- #7784（SFT 示例）已批准，但被要求先对齐 #50054 的 `DecisionTemplate`，所以它也卡在 #50054 上。
- 对我们：即使合并，**判断类型三种的限制不变**；想要 multi_label / ranking 必须在上游新开 PR，且对象是尚未稳定的 `Question` 体系。

## 3. Unsloth：扩展点与 AGPL 范围

读源码：unslothai/unsloth `8d2d7b644a54372576a8e7a022854dca4b8d8c95`（2026-11-03，HEAD）。全部 source-verified。

**扩展点：几乎没有。**
- 入口 `FastDecisionModel`（`unsloth/models/decision.py` L1550）：`from_pretrained(...)`、`get_peft_model`（L1653）、`for_inference`（L1741）、`for_training`（L1746）；训练器 `DecisionTrainer(Trainer)`（L1211），是普通 `Trainer` 子类，可以再子类化。
- 读出头：`DECISION_HEADS = ("clef",)`（`unsloth/models/decision_from_lm.py` L21，L169 校验），只有 Clef 一种头能接到任意 LM 上；另有 Laya 编码器格式（`rl_agent_config.json`，decision.py L1549）。**没有注册表、没有 entry-points**，加头要改源码或猴子补丁常量元组。
- 损失：`_decision_loss(logits, target, mask, ordinal, label_smoothing, brier_weight, ordinal_weight)`（decision.py L171）是私有函数，三个权重开关，不可替换。
- 校准：`_fit_temperature`（`_decision_common.py` L391，LBFGS、`strong_wolfe`）私有。
- 导出：`decision_gguf.py`（`save_pretrained_gguf` L760）、`decision_mlx.py`、`save_pretrained_merged`（decision.py L1476）。
- 判断类型同样是 noul / choice / score 三种（`_decision_common.py` 的 `_label` / `_target_for`）。

**AGPL 范围（要点：README 的说法与源码头不一致）**
- 根 `LICENSE` 是 Apache-2.0，`pyproject.toml` L16 `license = "Apache-2.0"`；`COPYING` 与 `studio/LICENSE.AGPL-3.0` 是 AGPL-3.0 全文。
- README L468–469："双许可，核心 Unsloth 是 Apache-2.0，Studio UI 等可选组件是 AGPL-3.0"。
- **但逐文件 SPDX 头显示**：所有决策文件都是 `SPDX-License-Identifier: AGPL-3.0-only`：`unsloth/models/decision.py`、`_decision_common.py`、`_decision_fast.py`、`decision_from_lm.py`、`decision_datasets.py`、`decision_gguf.py`、`decision_mlx.py`，以及 `scripts/train_decision_from_lm.py`。`unsloth/` 下 203 个 .py 里 105 个带 AGPL 头（`models/` 下 63 / 90）；`loader.py`、`trainer.py`、`save.py` 等是 Apache 头。
- 例外：`unsloth/_vendor/clef/joint_schema_model.py` 是 Cloudflare 的 Apache-2.0 原文（README 说逐字节一致，附上游 LICENSE 与 sha256 清单）；`studio/backend/vendor/laya/` 附 `LICENSE.laya`（Apache-2.0）。
- 结论：决策训练栈按 AGPL-3.0-only 处理。Apache-2.0 项目可以并入 AGPL 项目，反过来不行；MetisCraft 若 import 或拷贝这些文件，合成物要按 AGPL 发布。**不拷、不 import、不写成 Unsloth 的插件**；Unsloth 的做法（Clef 头接任意 LM、Brier 与序数损失、按类型 LBFGS 拟温度、GGUF/MLX 导出）只作为对照读，自己重写。这是源码头级判断，不是法律意见；Unsloth 内部文档不一致，如要精确口径需问其维护者（未做）。

## 4. 三条路线：要写什么、能复用什么、风险

### 路线 A：上游优先（贡献到 transformers / TRL）
- **MetisCraft 要写**：上游风格的 `XxxForDecision` + config + 转换脚本；若要 Schema Linking 的类型，要新开 PR 扩 `Question` 体系（multi_label / ranking），并在 TRL 的 `DecisionTrainer` 指标里加对应项；自己仍要保留工具包之外的调研清单。
- **可复用**：`DecisionMixin.decide`、`DecisionTemplate`、`pipeline("decision")`、serve `/v1/systemone`（#49322 已合并）、`Trainer`/PEFT/FSDP/上下文并行、TRL 的 `DecisionTrainer`。
- **风险**：(1) 接口未合并且声明实验性，每次 rebase 都要追（#50054 现在就是 dirty）；(2) 三种 kind 写死、`DecisionConfig` 是 `@strict`，新类型要说服 3 个以上维护者，周期不可控（推断）；(3) 一人维护，拿不到 review 带宽；(4) 损失在模型内部，调试"同一个头换不同损失"这类对比实验不方便；(5) 贡献后属于 HF，失去工具包定位。
- 适合：只做"把某个头做进 HF 生态"的单点贡献，在 #50054 合并之后。

### 路线 B：做 ms-swift（或 TRL）的插件
- **MetisCraft 要写**：`metiscraft_swift_plugin.py` 垫片（绝对导入）；`ScoringHead` 子类、`ScoringModelLoader` 子类、模板、`loss_map` / `callbacks_map` 条目、预处理器；数据与配置转换（把 MetisCraft 的业务场景配置翻译成 `swift sft` 参数）。
- **可复用**：7 种现成读出头、`ScoringTrainer`、4 个损失、`temperature_calibration` 回调、LoRA/DeepSpeed/Megatron/vLLM 推理与部署、模型/数据集注册。
- **风险**：(1) 只有三种 kind 且 `task_type` 是 `Literal`，Schema Linking 的多标签与排序只能变通；(2) `ScoringHead`/`ScoringOutput`/`decision_meta` 契约 10-13 才合并，没有稳定性承诺；(3) 用户要手动传 `--external_plugins <路径>`，没有 `pip install` 即生效；(4) 绑定 ms-swift 的参数体系和 Qwen3.5 主导的模型矩阵；(5) 若用 TRL 而非 ms-swift，则 TRL 主干无决策代码（#8040 是 draft），等于路线 A。
- 优点：Apache-2.0 对 Apache-2.0，无许可问题；开发量最小，一个下午就能出第一个头；中文社区正好是 ms-swift。
- 适合：作为分发渠道之一，不适合作为核心。

### 路线 C：自建可组合核心（推荐）
- **MetisCraft 要写**：(1) 读出、判断类型、损失、校准、评测的独立接口（至少覆盖本文 7 类读出里 Schema Linking 需要的子集，含 multi_label、ranking）；(2) 业务场景配置（Luce 的 YAML 分段、systemone-builder 的 `DomainSpec` 可借鉴，见 partial/verify-landscape.md）；(3) 到 ms-swift 与 HF `AutoModelForDecision` 的适配层，和 `decision.json` / `scoring_head.safetensors` 导出；(4) 跨读出对照评测；(5) Awesome 清单。
- **可复用**：HF `Trainer` / PEFT / Accelerate / `datasets`；ms-swift 的头和损失可按"适配层"调用，不拷代码（Apache-2.0 允许，仍保留署名）；HF `Question` / `DecisionConfig` 的字段命名，便于将来合并；社区仓库（JevForge、Kev、lev、Luce 等）的读出实现只读参考，许可证逐个看。
- **风险**：(1) 一人维护两套适配层，上游一改就要跟；(2) 容易重复造轮子，要克制，训练循环用 `Trainer`，不自写；(3) 在 ms-swift、HF 已有的情况下，用户为什么还选 MetisCraft，需要靠"多读出对比 + 业务场景配置 + 清单"站住；(4) 4×3090 24GB 下只能验证到约 9B 以内的 LoRA（本地算力限制，推断）。

### 对照表

| 维度 | A 上游优先 | B ms-swift 插件 | C 自建核心 |
|---|---|---|---|
| 新读出头 | 是，但需合并 | 是，不 fork | 是 |
| 新判断类型 | 要改上游 | 要 fork 或变通 | 是 |
| 自定义损失 | 写模型类 | `loss_map` | 是 |
| 自定义回调 | `Trainer` 回调 | `callbacks_map` | 是 |
| 稳定性 | 未合并、实验性 | 约定、无承诺 | 自己负责 |
| 许可证 | Apache-2.0 | Apache-2.0 | Apache-2.0 |
| 用户安装体验 | `pip` 原生 | `--external_plugins 文件` | `pip` |
| 开发量 | 大且依赖他人 | 小 | 中到大 |

## 5. 待定事项

1. 是否采纳"路线 C 为主，B 作为第一个适配层，A 等 #50054 合并后再单点贡献"？
2. Schema Linking 的多标签与排序：接受先用"逐列 noul / 逐候选 score"的编码在 ms-swift 与 HF 上跑通，还是一开始就自建集合交互读出？
3. Unsloth：是否确认"只读不拷"，不做 Unsloth 适配？
4. 是否在后面某一轮去问 ms-swift 与 HF 维护者，要不要把 ranking / multi_label 作为新 kind（会产生公开发帖，待定）。

## 6. 未验证与局限

- 没有对 PR 分支或 ms-swift 实际运行代码，所有 API 结论来自阅读源码；`AutoModelForDecision.register` 与 `auto_map` 的行为是用本机 transformers 4.40.2 的通用 Auto 机制推断的，未对 #50054 实测。
- PR 的"合并前景"是判断，没有 HF 内部排期信息。
- ms-swift 的预处理器注册表、`eval_metrics_map` 注册位置未逐行核对。
- GitHub 共用 token，本轮 GET 约 40 次，未被限流；PR 的 diff 另走 `github.com/...diff`（无 token）。

<!-- complete -->
