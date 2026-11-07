# MetisCraft 设计（v0.1，开发版）

2026-11-05 按已定决策改写。本文是 v0.1 的实现规格：写代码以本文为准，实现进度见 [`status.md`](status.md)。决策的来龙去脉见 §13 和 [`research/`](research/)；改写前的设计草案（含三条构建路线与决策卡的完整分析）保留在提交 `1f866c7` 的同名文件里。

## 1. 定位与理念

MetisCraft 是训练 Jev 类（System One）决策模型的工具包，做这个领域的 LlamaFactory / Unsloth。目标有两层：一是支持通用的 judge；二是，也是更重要的，让特定的垂直业务场景也能做成自己的 Jev——给定场景的数据和判断类型，用配方训练、校准、评测并部署。另有一份支持矩阵，收录当前较好的 Jev 类项目，标出能否用 MetisCraft 训练、复现、部署。

模型一次前向给出类型化判断和校准过的概率。理念是"训练有素的直觉"：

| 理念 | 落到设计上的约束 |
|---|---|
| 快 | 一次前向出全部选项或候选的判断，不靠生成长推理 |
| 知道自己有多准 | 校准是一等公民：每个产物带校准器及其拟合数据身份；评测固定 ECE 口径，同时报 NLL、Brier |
| 判断有类型 | 类型决定输出语义、损失、指标和校准方式（§5） |
| 能从反馈中学 | v0.1 只预留与 TRL `reward_funcs` 同形的奖励签名，并用一个形状测试守住 |

工程原则：部件可换（读出、类型、损失、校准器、基座是独立的轴，用注册表组合）；一事一处；有据可查（每个数字能追到一次运行）；借力成熟库（模型与调度器用 transformers，适配器用 PEFT，下载用 `huggingface_hub`）；看得见（每个读出都有张量图，图与代码对齐，见 [`readouts/`](readouts/)）。

## 2. v0.1 范围

| 项 | v0.1 | 推后 |
|---|---|---|
| 判断类型 | `noul`、`choice`、`score`、`multi_label` | `ranking`（v0.2），`box`（不做） |
| 读出 | ① 答案 token；③ 逐候选标量头（list 布局为主，pairs 布局作参照） | ② 槽位头、④ pointer、⑤ 集合交互、tree 布局 |
| 读出对照 | 选择题上 ① 与 ③ 只改配置即可互换，同协议出对照表 | — |
| 基座 | Qwen3 0.6B–4B，首发 0.6B；CI 用微型随机初始化模型，覆盖 2 种架构 | 4B 以上、第二个真实基座族（可选） |
| 微调 | LoRA（默认）与全参 | 量化、QLoRA |
| 训练 | 自写的最小训练循环（AdamW、线性调度、bf16），单卡与多卡 DDP | RL |
| 校准 | 按（任务，类型）的温度缩放 | Platt、isotonic、分桶 |
| 数据 | 注册表、Jev 格式原生读取、首批转换器、去重 | 排序类转换器（NFCorpus、MS MARCO） |
| 部署 | Predictor + 兼容 `/v1/systemone` 请求格式的 HTTP 服务 | vLLM、SGLang、llama.cpp 导出；ms-swift / HF 适配层 |
| 文档 | 设计、状态、调研、读出张量图、支持矩阵首版、配方说明 | `metiscraft viz` |

## 3. 架构

```
src/metiscraft/
  cli.py            # metiscraft 命令入口
  config.py         # 配置模型（pydantic）、YAML 读取、a.b=c 覆盖、路径解析
  registry.py       # 通用组件注册表
  judgments/        # 判断类型：noul / choice / score / multi_label 的语义、损失、指标挂钩
  data/
    contract.py     # 内部记录契约（§4.3）
    jev.py          # Jev 类型化判断格式的读写（§4.4）
    store.py        # 版本化数据目录（§4.1）
    registry.yaml   # 数据注册表（§4.2）
    fetch.py        # 按固定 revision 下载
    external.py     # 外部构建：按提交运行上游的构建脚本，核对公开哈希（§4.2）
    converters/     # 每个数据集一个转换器
    dedup.py        # 去重与泄漏检查（§4.6）
  modeling/
    backbone.py     # 加载基座、分词器、PEFT
    prompts.py      # 把记录渲染成输入序列
    readouts/       # answer_token.py（①）、candidate_scalar.py（③）
  train/            # collator、训练循环
  calibrate.py      # 温度缩放
  evaluate/         # 指标、报告
  trace.py          # 运行目录、产物封存、NOTICE
  serve/            # predictor.py、server.py（FastAPI）
recipes/            # 配方 YAML
tests/              # pytest；微型随机初始化模型，不占 GPU
```

数据流：`data prepare`（下载 → 转换 → 去重 → 写版本目录）→ `train`（训练 → 产物）→ `calibrate`（在 calib 上拟温度 → 派生产物）→ `eval`（指标与报告）→ `predict` / `serve`。每一步写一个运行目录（§9）。

依赖：torch、transformers、peft、accelerate、huggingface_hub、safetensors、pyyaml、pydantic、numpy；服务用 fastapi、uvicorn（extras `serve`）；测试用 pytest、scikit-learn（对拍指标）。全部是 Apache-2.0、BSD 或 MIT 许可。

## 4. 数据

### 4.1 版本化数据目录

数据根目录由配置 `paths.data_root` 或环境变量 `METISCRAFT_DATA_ROOT` 给出。每个数据集一个版本目录：

```
<data_root>/<数据集>/<版本>/
  dataset.yaml      # 来源（hf_id、revision）、许可证与条款、判断类型、split 角色、转换器版本、状态
  raw/              # 下载的原样文件，只读
  derived/          # 转换后的 MetisCraft JSONL；只能评测的真值放 derived/eval_only/
  manifests/        # algorithm_input.json（训练可读）、eval_only.json（只能评测）
  SHA256SUMS        # 版本目录内全部文件的哈希
```

版本名默认用上游 revision 的前 8 位（如 `rev-35b264d0`）。`dataset.yaml` 缺失、或 `SHA256SUMS` 校验不通过的版本不能用于正式运行。下载缓存放在 `<data_root>/_downloads/metiscraft/`。

### 4.2 数据注册表

`src/metiscraft/data/registry.yaml` 每个数据集一条：名字、`hf`（库 id@revision）、配置名、判断类型、转换器名、许可证、条款标记、split 角色（train / calib / dev / test / eval_only）、上游 lineage。条目与核对依据见 [`research/datasets.md`](research/datasets.md) §6。

首批（2026-11-05 定）：

| 档位 | 数据集 |
|---|---|
| 必收 | BoolQ（noul）、Banking77（choice）、CLINC150 `plus`（choice）、SST-5（score）、GoEmotions `simplified`（multi_label）、Open-Jev `release-v2-redistributable`、LocalLLaMA typed-decisions、tasksource-jev `default`；NFCorpus 只注册，转换器随 ranking 进 v0.2 |
| 可选 | MNLI、AG News、MS MARCO v1.1、LexGLUE `eurlex`、Civil Comments |
| 只评测 | Decision Index 公开套件（本地重建，不分发） |

条款受限的 AG News、MS MARCO、NFCorpus、SST-5 带条款标记；用它们训练的产物在 NOTICE 里标为不可商用。来源是 Jev / TypeSafe API 输出的数据一律不收。

**外部构建**。有的项目不发布数据本身，只发布确定性的构建脚本和产物哈希。这类条目写 `git`（`<owner>/<repo>@<完整提交号>`）和 `params.build`：

- `inputs`：构建脚本要读的 Hugging Face 数据集和分词器，分别映射到已注册的数据集（同一 revision）和权重目录。不同版本的 transformers 可能把同一个分词器解析成不同的类，切分结果略有差别；这时分词器写成 `{weights, class}`，注明上游锁定的版本用哪个类加载；
- `steps`：按顺序原样运行的脚本；
- `verify`：项目在该提交里公开的哈希清单。默认必须一致；项目记录了但没有声明可复现的哈希标 `required: false`，照样逐个对比并记下结果，但不作为接收条件；
- `files`：要拷进 `raw/` 的产物。

`data prepare` 先按提交号检出并核对，再把代码复制到暂存目录里运行，不改上游的一行代码。运行时关闭 Hub，由 MetisCraft 的包装层回答脚本的 `load_dataset` 和 `AutoTokenizer.from_pretrained`：只给声明过的输入，revision 不符或未声明的一律拒绝。必须一致的公开哈希全部一致才接收。构建记录（步骤、输入版本、包版本、核对结果）写进 `dataset.yaml` 的 `source.build`，逐文件核对表和构建日志写进 `manifests/`。第一个例子是 Tev1，见 [`reproductions/tev1.md`](reproductions/tev1.md)。

### 4.3 内部记录契约

一行一题（JSONL），字段：

| 字段 | 必填 | 含义 |
|---|---|---|
| `id`、`task_id` | 是 | 全局唯一行 id；任务名 |
| `input.query` | 是 | 问题或指令文本 |
| `input.context` | 否 | 上下文，如 `state`、段落、对话 |
| `question` | 是 | `{type, options?, levels?, allow_null?}`；`options` / `levels` 是 `[{id, text}]` |
| `candidates` | multi_label 必填 | `[{id, text}]` |
| `group` | 否 | `{id, split_key}`；同组的行必须落在同一 split |
| `supervision` | 训练与评测行必填 | `{kind, labels, soft?}`；`kind` 与 `question.type` 一致 |
| `provenance` | 转换器产出时必填 | 上游库、revision、split、行号、转换器名与版本、`input_hash`、`label_form`；存软标签时必填教师身份与许可证 |

`labels` 的形式：noul 为 `true` / `false`；choice 为选项 id（`allow_null` 时可为 `null`）；score 为等级序号（从 0 起）；multi_label 为正例候选 id 的列表。`soft` 是与选项、等级或候选对齐的概率。

### 4.4 Jev 类型化判断格式

原生读取 Jev 格式，与 HTTP 服务的请求格式一致（OpenAPI 0.2.0）：请求是 `model` + `state` + `questions:{题名:{type, instructions, criteria}}`。choice 的 `criteria` 是"选项名 → 描述"的映射；score 的 `criteria` 是从 0 起的等级描述数组；noul 的 `criteria` 可选。一条请求拆成多行记录：`group.id` 取请求 id，`input.context.state` 取 `state`，`input.query` 取 `instructions`。三份 Jev 类数据集的字段差异与映射见 [`research/datasets.md`](research/datasets.md) §4；读取时不要求 `model` 字段。

### 4.5 转换器

每个转换器把上游 split 转成 §4.3 的记录，并写 `provenance`。v0.1 实现：`bool_noul`（BoolQ）、`cls_choice`（Banking77、CLINC150、可选的 AG News）、`pair_choice`（可选的 MNLI）、`ordinal_score`（SST-5）、`multilabel_cands`（GoEmotions）、`jev_request`（LocalLLaMA）、`open_jev_rows`（Open-Jev）、`jev_flat_rows`（tasksource）。没有官方验证集的数据集，calib 与 dev 从 train 按内容哈希切出（各约 5%），切分规则写进 `dataset.yaml`。复现用的 `tev1` 读 Tev1 构建出的记录，保留其选项顺序（见 [`reproductions/tev1.md`](reproductions/tev1.md)）。

### 4.6 去重与泄漏

- 每行记 `input_hash`：对输入文本做 NFKC、小写、折叠空白后取 sha256。
- 训练前，对所有 calib、dev、test、eval_only 集合建哈希表；训练池中命中的行，连同同组的全部行一起剔除。
- 已知的上游重叠写在注册表的 lineage 里（如 tasksource 的 ChaosNLI 与 MNLI validation_matched），训练时整源整 split 剔除。
- 每次训练输出 `dedup_report.json`：规则、命中数、剔除数、哈希算法版本。

## 5. 判断类型

| 类型 | 输出 | 概率语义 | 默认损失 | 校准 | 主指标 |
|---|---|---|---|---|---|
| `noul` | `p_true` | 伯努利 | BCE（可用软标签） | 温度 | accuracy、NLL、Brier、ECE |
| `choice` | 选项上的分布（`allow_null` 时多一列） | 互斥单选，和为 1 | 交叉熵（可加小权重 Brier） | 温度 | accuracy、macro-F1、NLL、Brier、ECE |
| `score` | 等级上的分布与期望分 | 有序单选 | 交叉熵；可选相邻软目标（序数） | 温度 | accuracy、期望分 MAE、NLL、Brier、ECE |
| `multi_label` | 每候选独立的 `p_i` | 每候选伯努利，不和为 1 | BCE，`pos_weight` 可自动 | 温度（全体候选共用） | micro / macro-F1@0.5、NLL、Brier、逐候选 ECE |

不变量：

1. 温度缩放不改 argmax；不把 NLL 改善说成 accuracy 改善。
2. 校准生成派生产物，不改父产物；拟合只用 calib split。
3. 预测分布与行动策略（阈值、弃权）分开存放，策略参数不进模型身份。
4. pairs 布局下，删掉一个无关候选不改变其他候选的原始分数；list 布局会违反这一条，要测量并报告顺序敏感度。

## 6. 读出

记号与张量图见 [`readouts/README.md`](readouts/README.md)：`B` 批大小，`T` 序列长度，`d` 隐藏维，`V` 词表，`K` 选项或候选数。

**① 答案 token**：在答案位置取 `lm_head` 的 logits `z: [B, V]`，只保留各选项码的 token：`z[:, ell] -> [B, K]`。
- noul 的选项码是 ` yes` / ` no` 两个 token，`p_true = sigmoid(z_yes - z_no)`；这等价于一个线性头（两行 `lm_head` 之差），测试要验证逐位一致。
- choice 与 score 的选项码由码本给出：按固定顺序（`A`–`Z`、`a`–`z`、两位大写字母……）挑出在当前分词器里恰好编码为单个 token 的字符串，前 K 个分给选项。提示里写"码: 选项描述"，所以 Banking77 的 77 个选项也能用 ①。score 默认用从 0 起的数字等级。
- 提示有两种写法（`readout.prompt.style`）。默认 `plain` 是"码: 选项描述"的文本布局，题面过长时只截断状态。`chat_json` 把状态、问题和带字母的选项写成一个 JSON 载荷，放进分词器自带的对话模板（关闭思考，可加系统提示）；等级也用字母，答案是助手回合开头的那个字母。这是 Tev1 训练用的格式，整段一次渲染、不截断，只接受 choice 与 score。
- 参数只有 LoRA（或全参），不加新层，保留零样本能力；零样本的 ① 本身就是基线。

**③ 逐候选标量头**：每个候选取一个表示向量 `h_k: [B, K, d]`，经头 `s_k = w^T MLP(h_k) + b` 得到原始分数 `s: [B, K]`。
- choice、score：对 `s` 做 softmax；multi_label：逐候选 sigmoid；noul：只有一个候选（陈述本身），sigmoid。
- **list 布局**（默认）：共享前缀（指令 + 上下文）后依次接各候选，每个候选以分隔符结尾，`h_k` 取该分隔符位置的隐藏状态；一次前向得到全部候选，普通因果注意力。训练时打乱候选顺序，推理可选多次排列取平均。
- **pairs 布局**（参照）：每个候选与前缀单独组成一条序列，候选之间互不可见；用于正确性测试和顺序敏感度对照。
- 头单独保存，不并入 PEFT；头用 fp32、单独的学习率。

**读出对照**（决策点 6 乙）：同一配方只改 `readout.kind`，就能在 choice 数据集上训练和评测 ① 与 ③，报告同表输出含校准列的对照。

## 7. 训练

- 训练循环是一个约 200 行的最小实现（`train/loop.py`）：AdamW，LoRA 与头两组学习率，transformers 的线性或 cosine 调度（`train.scheduler`），bf16 autocast，梯度裁剪。不套 HF `Trainer`：它对梯度累积与多卡的损失缩放在不同版本间改过多次，而下一条是验收硬指标。
- 损失按全局问题数归一：一个优化步内所有微批、所有卡的问题数先 all-reduce，每个微批回传 `损失总和 × 卡数 / 全局问题数`，DDP 求平均后即为全局均值；用微型模型测试梯度累积（含大小不均的微批）下的等价性。
- 答案 token 读出默认在选项码之间做 softmax。`objective.kind: completion` 改按监督微调的方式训练：把正确选项的码接在提示后面，在最后两个位置分别对整个词表算码和 EOS 的交叉熵，两项取平均作为这道题的损失，再按上一条归一。每题恰好两个答案 token，所以它等于 SFT 报告的逐 token 均值。校准、评测和部署照旧在提示末尾只读选项码的 logits；接在后面的码只影响训练，读出位置看不到它。
- 多卡用 `torchrun` 启动 DDP。
- 多标签可以用 `objective.pos_weight` 给正样本加权（`auto` 取训练集的负正比）。加权把每个正样本的赔率乘上 w，学到的 logit 整体高出 ln w，所以实际用到的 w 写进产物，推理时先减去 ln w，再做温度校准；否则概率会系统性偏高，温度缩放也修不回来。
- 默认超参不写死在代码里，全部在配方中显式给出；新配方先过 sanity 阶梯：小子集过拟合 → 学习率扫描 → 学习曲线。
- 随机种子、数据顺序、LoRA 初始化都由配方里的种子决定，并写进运行目录。

## 8. 校准与评测

- **校准**：按（任务，类型）各拟一个温度，在 calib split 上用 LBFGS 最小化 NLL；校准器存为派生产物，记录拟合数据身份。
- **ECE 口径**（常量，写在一处，报告头部打印）：置信度取预测类别的概率（多类）或正类概率（二元与逐候选）；15 个等宽分桶。同时报 NLL、Brier，并输出可靠性图数据。
- **基线**：每个配方都报无信息基线（多数类 / 先验）；选择、是非、等级任务另报一个 encoder 强基线（句向量 + 逻辑回归，句向量模型在配方里指定），可在时间不够时砍掉。
- **报告**：Markdown，一个配方一份；表头写清数据版本、基座、读出、校准前后。每个数字链接到产生它的运行目录。
- **评测门禁**：`eval` 默认用 dev；用 test 时必须指定已封存的产物，test 的结果不回流到选择。

## 9. 可追溯与产物

- **运行目录**（`paths.runs_root/<配方>/<时间戳>-<短哈希>/`）：解析后的完整配置及其 sha256、git 提交号与未提交改动的 diff、环境（conda 环境名、关键包版本、CUDA）、数据集版本与 manifest 哈希、基座 `SOURCE.json` 与 revision、随机种子、日志。
- **产物**（`artifact/`）：`artifact.json`（判断类型规格、读出、概率语义、选项码本、配置哈希、数据哈希、基座身份、git 提交）、LoRA 权重、头权重（safetensors）、分词器信息、校准器（若有）、`SHA256SUMS`、`NOTICE`（基座许可证、每个训练源的许可证与条款；有非商用源参与训练时整体标为不可商用）。
- 加载产物时先校验 `SHA256SUMS`，拒绝符号链接与路径穿越。

## 10. 推理与部署

- **Predictor**：`Predictor.load(artifact)`，输入内部记录或 Jev 请求，输出统一字段：`raw_scores`、`probabilities`、`probability_semantics`、`calibration`、`decision`、`abstained`、`uncertainty`。
- **HTTP 服务**：`metiscraft serve <artifact>` 启动 FastAPI，`POST /v1/systemone` 接受官方请求格式，按题返回概率与决定；`model` 字段接受任意字符串，响应回填实际服务的产物名。按类型的温度在服务里生效。
- v0.1 只承诺自家 Predictor 与服务；导出到推理引擎放 v0.2。

## 11. 配方

一个配方是一个 YAML 文件，分段如下，命令行可用 `a.b=c` 覆盖：

```yaml
# recipes/banking77/qwen3-0.6b_candidate-scalar.yaml
task: banking77
data:      {dataset: banking77, version: rev-18469841}
model:     {base: qwen3/0.6b, finetune: {kind: lora, r: 16, alpha: 32, targets: all-linear}}
readout:   {kind: candidate_scalar, layout: list, head: {hidden: 256}}
objective: {kind: cross_entropy}    # 答案 token 读出还可用 completion
train:     {epochs: 3, batch_size: 8, grad_accum: 2, lr: 2.0e-4, head_lr: 1.0e-3, scheduler: linear, seed: 42, bf16: true}
calibrate: {method: temperature, split: calib}
eval:      {splits: [dev], baselines: [prior]}
```

配方只引用注册名，不内嵌代码；每个配方目录附 `README.md`，写明许可证、预期数字的来源和证据级别。v0.1 配方：BoolQ、Banking77（① 与 ③ 各一份）、CLINC150、SST-5、GoEmotions，以及一份"垂直场景做成 Jev"的示例（场景待定）。

## 12. 验收标准

全部为可自动运行的测试或报告。真实训练使用单卡、每次不超过 2 小时。

| 类别 | 标准 |
|---|---|
| 单元测试 | `pytest` 全部通过，CPU 上用微型随机初始化模型完成，不下载权重 |
| 等价性 | (1) noul 的 ① 读出与"两行 `lm_head` 之差"的线性头逐位一致；(2) list 与 pairs 布局下，共享前缀部分的 token 与隐藏状态一致；(3) 产物在新进程加载后，logits 与训练进程一致 |
| 可组合性 | 微型模型上，2 种架构 × 2 种读出只改配置就能跑通训练、校准、评测 |
| 梯度累积 | 微型模型上，`batch=b, GAS=k` 与 `batch=k·b, GAS=1` 的损失与梯度在容差内一致 |
| 校准 | 温度缩放前后 argmax 不变；校准器只在 calib split 拟合 |
| 数据 | 首批转换器在样例上通过；`SHA256SUMS` 校验；去重报告产出 |
| 反馈接口 | 奖励函数能按 TRL `reward_funcs` 的签名形状调用，返回逐样本标量 |
| 服务 | HTTP 服务对官方格式的请求返回合法响应，概率与 Predictor 一致 |
| 验收训练 | Qwen3-0.6B 上，BoolQ、Banking77（CLINC150 作对照）、SST-5、GoEmotions 各跑通一次，报告含校准前后与基线；Banking77 上出 ① 与 ③ 的对照。不设绝对分数门槛，负结果照常报告 |
| 可追溯 | 报告里每个数字都能追到运行目录；产物带 `SHA256SUMS` 与 `NOTICE` |

## 13. 决策记录

| 决策 | 结论 | 日期 | 依据 |
|---|---|---|---|
| 定位 | 通用 judge + 让垂直场景做成自己的 Jev；外加支持矩阵，取代单独的 Awesome 清单 | 11-04 | [`research/scenarios.md`](research/scenarios.md) |
| 构建路线 | 自建可组合核心；ms-swift / HF 适配层推到 v0.2，v0.1 只对齐 HF 字段命名 | 11-05 | [`research/build-routes.md`](research/build-routes.md) |
| v0.1 判断类型 | noul / choice / score + multi_label；ranking 推到 v0.2（核查发现主流框架已有 ranking，不再是差异点） | 11-05 | 同上 |
| 首发基座 | Qwen3 0.6B–4B | 11-05 | [`research/decision-model-sota.md`](research/decision-model-sota.md) |
| 读出对照 | 选择题上 ① 与 ③ 只改配置互换，v0.1 必做 | 11-05 | 同上（主流默认是答案 token 读出，标量头需要自家消融） |
| 部署 | Predictor + HTTP 服务；引擎导出推到 v0.2 | 11-04 | 引擎只施加单一温度，按类型校准只能在自家服务里保证 |
| 数据 | 注册表与主流数据集是必需项；原生读取 Jev 格式；首批名单与验收基准见 §4.2、§12 | 11-04、11-05 | [`research/datasets.md`](research/datasets.md) |
| 配置系统 | pydantic + YAML + 点号覆盖 | 11-05 | pydantic 随 fastapi 已在依赖里，校验严格 |
| 训练循环 | 自写最小循环，不用 HF `Trainer` | 11-05 | 全局问题数归一是验收硬指标，`Trainer` 的损失缩放随版本变化，难以保证 |
| 首发基座 | Qwen3-0.6B-Base | 11-05 | 没有对话模板与思考模式的干扰，适合训练判断；旧版也用它 |
| ① 多选项 | 从分词器里挑单 token 的选项码 | 11-05 | 让 ① 在几十个选项时也能用，读出对照才能在 Banking77 上做 |
| 序数损失 | 只做相邻软目标一种 | 11-05 | 先保证主线 |
| Unsloth | 只读不拷，不做适配（决策代码是 AGPL-3.0） | 11-04 | [`research/build-routes.md`](research/build-routes.md) |
| 支持矩阵首版 | 收 13 个项目；首批 Tev1、jebadiah、Hmm、JevForge 与 typed-decisions（读出都在 ① / ③ 内，有公开数字可对照） | 11-05 | [`support-matrix.md`](support-matrix.md) |
| 垂直场景示例 | 银行客服分流：Banking77 经映射表改写成"转哪个队列"与"是否紧急"两个业务问题 | 11-05 | `banking77_support`（注册表） |
| 多标签的正样本加权 | 产物记下实际用到的 `pos_weight`，推理时先减去 ln w 再校准。GoEmotions 验收发现不扣时 NLL、Brier 比先验还差，扣除后 ECE 降到约 0.002 | 11-05 | GoEmotions 验收 |
| 复现公开项目的数据 | 上游构建脚本原样运行，输入由注册表供给，公开哈希全部一致才接收；不重写上游的生成逻辑 | 11-06 | [`reproductions/tev1.md`](reproductions/tev1.md) |
| 上游锁定的库版本与本机不同 | 不为每个项目另建锁版本环境；产物只在输入层受版本影响时，在输入上注明上游的加载方式（如分词器类），仍按公开哈希验收。Tev1 的 21 个哈希里有 3 个只因分词器类不同而差两条计数 | 11-06 | [`reproductions/tev1.md`](reproductions/tev1.md) |

## 14. 待定

- 张量图的风格问题，见 [`readouts/README.md`](readouts/README.md) 末尾。
- 口号与吉祥物。

## 15. 风险

| 风险 | 缓解 |
|---|---|
| 主流框架补上类型空白 | 差异化落在同协议对照、校准与垂直场景配方，而不是类型本身 |
| 有标签时 encoder 往往更准 | 每个配方都报 encoder 基线；自训练的理由写成私有化、校准、多标签、选项集可变，不宣称"更准" |
| list 布局顺序敏感 | 训练打乱顺序，推理可多排列平均，报告顺序敏感度 |
| 许可证与条款 | 注册表带条款标记，NOTICE 自动汇总；不用 Jev 输出作训练标签 |
| 数据泄漏 | §4.6 的去重规则与去重报告 |
| 一人维护、范围过大 | 砍件顺序：encoder 基线 → 第二个真实基座族 → 可选数据集的转换器 |
