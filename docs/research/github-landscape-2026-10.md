# Metis 与相邻 GitHub 项目：定位、差距与开发顺序

核查日期：2026-10-21。Metis 源码基线为 `edb65dd`。本次覆盖 15 个公开仓库，补充此前偏重 Jev 类模型的[调研](jev-task-survey.md)，按任务和工程职责比较。仓库、完整 commit、核查文件、许可证元数据见[检索清单](github-landscape-2026-10.json)。

## 判断

**专用模型后训练仍有使用场景，但“Qwen + LoRA + 独立评分头”本身不足以构成 Metis 的差异。** 相邻工具已经覆盖排序、少样本分类、动态标签、概率校准、模型路由和反馈训练。Metis 目前适合作为早期工具包与研究平台；尚无证据把它定位成质量更高、推理更快或更通用的替代品。

这不是根据项目名称或热度作出的判断。下列仓库在核查版本仍提供非生成评分、分类或决策的实现与使用入口，说明这类职责没有因通用 Agent 的发展而消失。它们的存在也不证明 Metis 已有用户需求；需求仍须由具体任务和成本测量验证。

建议将下一阶段目标收敛为：**围绕候选打分，提供可复现、可校准、可导出的小模型任务适配流程。** 这是产品建议，尚未替换现有 v1 架构，也不是已实现能力。是否采用 encoder、保留 decoder 或引入候选交互，应由同协议实验决定。

## Metis 当前可复用的部分

| 层次 | 已有依据 | 对定位的含义 |
|---|---|---|
| 输入与模型 | query/context/candidates、稳定候选 ID、pairs/dense tree、两种 Qwen3 适配器 | 可作为共同契约；尚不是任意任务或任意 backbone 插件体系 |
| 训练与产物 | full/LoRA、排序 dev 选优、best/final、完整性校验、重载与 Predictor | 是可继续建设的工程基础；单选、多选尚未有同等完整闭环 |
| 排序质量 | NFCorpus test：ScoreHead 0.290978，原始 reranker 0.354783，指标为 nDCG@10 | 当前配方低于强基线；两者训练历史和读出不同，不是头部单因素消融 |
| 分类研究 | BoolQ 第二轮九组均全肯定，balanced accuracy 50%；tiny32 可拟合到训练 accuracy 100% | 尚未建立有效泛化基线；能拟合小训练集不等于能改善真实任务 |
| 效率 | dense tree 的小规模分数、梯度与置换预检 | 不能据此宣称稀疏加速、低延迟或成本优势 |
| 教学与协议 | 张量图、真实 cookbook、失败结果与数据边界说明 | 对学习和复现有价值；还需证明工具包能减少新任务接入成本 |

依据：[当前状态](../project/status.md)、[NFCorpus 协议](../../cookbooks/score-head-plan.md)、[BoolQ 读出研究](../../cookbooks/boolq-readout-selection.md)、[tiny32 诊断](../../cookbooks/boolq-training-diagnostic.md)；源码中的 `training.py` 明确限制质量选优与候选采样为 ranking，`tasks/decisions.py` 已有三种任务决策分支。本次没有重跑模型评测。

## 按类型收集的仓库

下表能力以固定提交的 README 为主；进一步阅读的训练源码列于下一节。没有复现外部指标，不按不同数据和硬件下的自报数字排名。

| 类型 | 仓库与固定版本 | 主要职责 | 对 Metis 的参考价值 |
|---|---|---|---|
| 排序与成对评分 | [Sentence Transformers · b83a7f5](https://github.com/huggingface/sentence-transformers/blob/b83a7f57c677f09c64596e5b80b0a53c7fd78035/README.md) | embeddings、CrossEncoder、排序模型训练与推理 | 当前排序路径的直接对照；优先复用其基线、损失和 evaluator 经验 |
| 检索与 reranker 微调 | [FlagEmbedding · fd1a2bd](https://github.com/FlagOpen/FlagEmbedding/blob/fd1a2bdf69488ffebe0327999d4400d8c8058a0b/README.md) | BGE 检索、reranker、微调与教师分数蒸馏 | 参考正负例格式、强基线和蒸馏配方 |
| 少样本固定标签分类 | [SetFit · d70dad8](https://github.com/huggingface/setfit/blob/d70dad84de757d30f03465d120e0d3fad2c725b0/README.md) | Sentence Transformer + 分类头、少样本训练 | 判断是否需要 decoder；固定标签分类不必先转成逐候选打分 |
| 动态标签分类 | [GLiClass · 6a00f80](https://github.com/Knowledgator/GLiClass/blob/6a00f8067648f25aa0f4f477d167c5f391cb4165/README.md) | 输入时提供标签，单选、多选、层级标签、微调 | 与候选决策接口直接相邻；校准和任务适配值得优先参考 |
| Schema 条件抽取与分类 | [GLiNER2 · ca4f682](https://github.com/fastino-ai/GLiNER2/blob/ca4f68235b50fe4c3ecf3876851d748fe4b15c6c/README.md) | encoder 上的实体、分类、记录、关系抽取，提供训练入口 | 是更广的结构化输出路线；span 抽取需要新输出契约，不能用候选 ID 接口直接代替 |
| 双模型质量与成本路由 | [RouteLLM · 0b64fda](https://github.com/lm-sys/RouteLLM/blob/0b64fdafe049e596a3f5657c219329f24af24198/README.md) | strong/weak 模型选择、阈值与路由评测 | 提供有业务目标的决策任务；路由阈值与概率校准应分别理解 |
| 多方法路由训练框架 | [LLMRouter · c65ee01](https://github.com/ulab-uiuc/LLMRouter/blob/c65ee0122bc9e84f71d571fcdec4a948481b6e88/README.md) | 多种路由算法、训练/推理 CLI、数据生成与评测 | 参考算法可替换性与任务结果评价；简单 MLP 也应是基线 |
| 推理路由系统 | [vLLM Semantic Router · 6c2f023](https://github.com/vllm-project/semantic-router/blob/6c2f023b40d22c0600e5000d6b2df53def23e944/README.md) | 根据策略选择或组合模型、后端与计算资源，接入 Agent harness | 是 Metis 模型可能服务的下游系统；当前没有现成兼容性验证 |
| 合成数据与教师反馈 | [Distilabel · 313fac8](https://github.com/argilla-io/distilabel/blob/313fac85b1a2472dd88db1a31c2b754599f46476/README.md) | 数据生成、标注与反馈流水线 | 可以作为训练数据上游；不是可直接替换 Metis 的模型训练器 |
| LM 程序与任务优化 | [DSPy · 661b2f1](https://github.com/stanfordnlp/dspy/blob/661b2f1a25218f3b44aa125384d11d5508316ecb/README.md) | 模块化 LM 程序，优化提示与权重 | 参考按最终任务指标优化和模块边界；候选 scorer 可以是程序中的一个模块 |
| 通用后训练与奖励模型 | [TRL · 149ff2e](https://github.com/huggingface/trl/blob/149ff2ea7d95a1cd4413421eba96d92a1c8f1424/README.md) | SFT、偏好优化、RL 和各类 Trainer | 参考训练生命周期；偏好奖励与任务正确概率不是同一语义 |
| 分布式 RLHF | [OpenRLHF · dc2a7ad](https://github.com/OpenRLHF/OpenRLHF/blob/dc2a7ad326f619fc5a47737ea47d1920ce2c0b53/README.md) | Ray/vLLM 等训练执行、reward model、Agent RL | 若将来需要规模化可评估复用；当前不足以成为引入重型基础设施的理由 |
| 通用配置化微调 | [LlamaFactory · c683e59](https://github.com/hiyouga/LlamaFactory/blob/c683e59edd8f893abc5e1403184a1d0644817d2a/README.md) | 多模型与训练方法适配、CLI/Web UI | 参考用户入口与配方组织；Metis 暂不宜以广度竞争 |
| 候选决策后训练 | [JevForge · 039badb](https://github.com/zwliJay/jev-forge/blob/039badbafc242b4f8105a59e27f789320efe6e9c/README.md) | 共享前缀候选 scorer、数据、训练、校准、服务与 RLCD | 与 Metis 的目标直接重叠；应比较候选语义、校准和数据协议，而非只比较 head 名称 |
| 任务专用决策流程 | [Luce · bda3d21](https://github.com/scienthoon/luce/blob/bda3d21cdb537ae99662fdba275adc450e90f455/README.md) | init → synth → train → eval → serve，LoRA + decision head | 参考从任务定义到模型上线的使用路径，以及域外校准失败的报告方式 |

## 源码核查带来的具体启示

- **排序训练已有可复用方案。** Sentence Transformers 的 [CrossEncoder 训练说明](https://github.com/huggingface/sentence-transformers/blob/b83a7f57c677f09c64596e5b80b0a53c7fd78035/docs/cross_encoder/training_overview.md)区分标量排序和多类别输出，也包含 causal-LM scorer 的边界；FlagEmbedding 的[微调说明](https://github.com/FlagOpen/FlagEmbedding/blob/fd1a2bdf69488ffebe0327999d4400d8c8058a0b/examples/finetune/reranker/README.md)支持教师分数。通用打分训练不宜作为 Metis 的唯一卖点。
- **轻量基线需要进入实验。** SetFit 有独立 [DistillationTrainer](https://github.com/huggingface/setfit/blob/d70dad84de757d30f03465d120e0d3fad2c725b0/src/setfit/trainer_distillation.py)；LLMRouter 的 [MLPTrainer](https://github.com/ulab-uiuc/LLMRouter/blob/c65ee0122bc9e84f71d571fcdec4a948481b6e88/llmrouter/models/mlprouter/trainer.py)直接使用 CE。任务有成本约束时，应比较简单模型与 decoder，而不是默认更大的模型更合适。
- **校准是独立工程模块。** GLiClass 的[校准源码](https://github.com/Knowledgator/GLiClass/blob/6a00f8067648f25aa0f4f477d167c5f391cb4165/gliclass/calibration.py)区分 sigmoid 与单选 softmax；Luce 的[训练源码](https://github.com/scienthoon/luce/blob/bda3d21cdb537ae99662fdba275adc450e90f455/luce/train.py)包含温度拟合。Metis 可以先实现更简单的温度缩放、版本化保存和重载，无需直接移植复杂校准头。校准本身不能修复没有判别能力的模型。
- **评分头没有因 RL 流行而失效。** TRL 仍提供 [RewardTrainer](https://github.com/huggingface/trl/blob/149ff2ea7d95a1cd4413421eba96d92a1c8f1424/trl/trainer/reward_trainer.py)。但奖励排序、独立二分类、互斥动作概率需要不同损失与语义；不能因为都输出一个数就共用全部评测。
- **是否采用 RL 需要看训练过程。** JevForge 的 [rlcd.py](https://github.com/zwliJay/jev-forge/blob/039badbafc242b4f8105a59e27f789320efe6e9c/jevforge/rlcd.py)包含有限候选精确期望与采样 REINFORCE 两条路径，以及 Brier/KL 项。Metis 应先有可靠监督基线，再验证反馈目标的增益；使用 Brier 或 KL 本身不等于 RL。

## 仓库活跃度与使用边界

本次 15 个仓库的 GitHub 元数据均未标记 archived；这不等于均在积极维护。RouteLLM 核查默认分支的最新提交为 2024-08-10，可作为经典基线，依赖与服务适配应另行核验。Distilabel 的默认分支 README 说明原作者已转向其他项目、社区接手，并建议查看 `develop`；本次固定的是 `main`，未审核 `develop`。仓库 `pushed_at` 可以来自其他分支，不能替代核查提交日期。

许可证清单只是 GitHub 对代码的分类。JevForge 返回 `NOASSERTION`，表示未自动识别，不能推断没有许可证或可自由复用；其他仓库的代码许可证也不自动覆盖模型和数据。Metis 当前没有 `LICENSE` 文件，这会增加外部采用的不确定性；选择许可证属于单独的项目决定，本次没有代为添加。

这些资料足以比较职责和开发方向，不能证明某个外部模型在 Metis 任务上更强，也不能支持未经测量的性能宣传。

## 建议开发顺序与停止条件

| 优先级 | 具体工作 | 完成依据 |
|---|---|---|
| P0 | 选一个真实固定任务，先建立有效监督基线；BoolQ 可继续用于诊断，也可选择意图/工具路由作为第二 cookbook | 固定 train/dev/最终保留集，预先确定指标和预算；比较多数类/简单文本模型、encoder 与现有 decoder，报告不确定性。未胜过简单基线时先查数据、截断与学习曲线 |
| P0 | 补齐 single_choice 的 evaluator、dev 选优与生产产物闭环 | 同一训练入口完成训练、选优、导出、新进程重载与 Predictor；研究格式不能充当生产产物 |
| P1 | 明确 logits、单选 softmax、多选 sigmoid、拒答阈值和校准产物 | dev 或独立 calibration 数据拟合参数；冻结后报告 accuracy/macro-F1、NLL/Brier 和拒答覆盖率。数据不足时使用明确的划分或交叉拟合方案 |
| P1 | 固定任务后测端到端成本 | 相同硬件、输入和候选规模，报告冷/热启动、p50/p95、吞吐、显存及质量；没有优势就不以“快”作定位 |
| P2 | 有瓶颈证据后比较蒸馏、候选交互或共享前缀后端 | 每次控制一个主要变量；同时报告质量与成本，保留 pairs 和简单模型对照 |
| 后续 | 环境反馈 RL、通用 backend 插件、多卡与更广任务 | 监督与生产闭环成立，并有明确反馈信号或容量需求后再进入 |

候选独立评分仍有清晰语义：加入无关候选不应改变已有 raw score。联合候选选择允许集合交互，两者应作为不同契约。Tree 的 dense mask 不能直接提供稀疏性能保证；张量说明与正确性预检继续保留，性能开发由测量触发。

若在预设预算内仍无法得到质量或成本收益，可以将 Metis 收敛为严谨的教学与实验工具，或围绕成熟训练库提供薄适配层。应停止没有证据的 head 堆叠与框架扩张，保留已经有用的契约、配方和失败记录。本次仅形成调研与开发建议，未启动训练、变更架构或发布软件。
