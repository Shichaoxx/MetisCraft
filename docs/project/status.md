# Metis 当前状态

## 方法流程图（2026-10-24）

当前先完善[方法主图](../architecture/framework.md)：以三条候选输入、末尾隐藏向量、评分头参数矩阵和分数条说明主路径，浅色背景区分输入、模型与决策。中英文 README 使用该简图；原工程架构图和详细张量图继续保留。示意数值不作为实验结果，算法与 v1 产物契约未改变，v2 功能仍待实现。

训练与推理[张量图](../architecture/score-head.md)已统一为同一套配色、几何与维度约定，公共前向共享源文件。跨行同名引用替代绕页折线，训练反向单列右至左直虚线；补充权重转置、微批汇总、FP32 边界、LoRA 冻结与梯度说明。此项只调整图稿与解释，不改变训练实现或新增实验结果。

## v2 升级准备（2026-10-22）

已核对当前源码并制定[升级实施方案](../development/v2-upgrade.md)，同步调整 [v2 设计](../architecture/v2-design.md)的开发顺序。当前阶段为准备完成、功能实现待开始；v1 模型、CLI 和产物契约仍是现有实现。

| 阶段 | 范围 | 状态 |
|---|---|---|
| 准备 | 当前源码盘点、兼容边界、实现文件与验收、离线基线 | 已完成；基于 `edb65dd`，155 passed、0 failed、0 skipped |
| 第一阶段 | 单选 TaskSpec/evaluator、dev 选优、CLI 评测、生产导出重载 | 待实现 |
| 第二阶段 | 概率语义、温度校准、拒答策略与派生产物 | 待实现 |
| 第三阶段 | Banking77 生产 cookbook、简单模型/encoder/decoder 对照 | 已选建议任务，数据与训练预算待冻结；未启动 |
| 第四阶段 | 按测量结果扩展适配器、蒸馏和执行优化 | 待前序结果决定 |

基线命令为贡献指南中的完整离线 CPU `python -m pytest -q`；本次 155 项用时 21.53 秒。未新增依赖或更改运行环境，未执行真实训练、留出集评测或模型下载，未推送或发布。准备阶段只形成实施方案，以上结果不表示 v2 功能已通过测试。

## 组件与实现状态

Python 包、import 和 CLI 均为 `metis`。

| 组件 | 当前实现 | 验证范围与待完善项 |
|---|---|---|
| 数据与任务 | JSONL schema、manifest/哈希、配置、DatasetRegistry；ranking、multi_label、single_choice | 单选、多选使用候选打分，没有独立分类头；尚无对应真实任务 cookbook |
| 输入与 attention | 公共输入编译器、pairs/dense tree、逻辑位置、eager/SDPA | tree 已验证小规模分数、梯度与换序一致性；没有稀疏加速或真实训练结果 |
| 模型与读出 | Qwen3 Base + MLP ScoreHead、Qwen3 Reranker yes/no、模型工厂 | 当前仅有两个 Qwen3 适配器 |
| 监督训练 | HF Trainer、full/LoRA、独立 head 学习率、混合精度、候选采样 | 候选采样仅支持 ranking；尚无 RL 或多卡训练 |
| 评测与选优 | IR 指标、完整 dev 选优、原始模型/训练产物评测及报告比较 | 完整指标评测与 dev 选优仅支持 ranking |
| 实验与产物 | 日志、checkpoint 恢复、best/final 导出、完整性校验 | LoRA 产物依赖记录的基座，恢复 checkpoint 依赖其历史选优目录 |
| 推理与调用 | Predictor、候选分数与 ID、TopK/单选/多选、HTTP、workflow 示例 | HTTP 为开发服务；workflow 已验证证据上下文组装，尚未评估最终答案 |
| 教学与 cookbook | 张量教程、训练/推理张量图、NFCorpus 训练协议、BoolQ 读出研究 | NFCorpus 完成工具包完整闭环；BoolQ 使用独立研究入口，未接入通用任务选优与 Predictor |

已有通用 Registry 容器、DatasetRegistry 和模型工厂；内置任务损失与决策集中在 `tasks/`，与 Trainer 和 Predictor 分离。task/objective/backend 插件协议尚未实现。项目提供核心、CPU 模型和分发三组 CI 配置，验证方法见[贡献指南](../../CONTRIBUTING.md)。

## NFCorpus 首轮训练

Qwen3-0.6B-Base + ScoreHead + LoRA 已完成两轮监督排序训练，共 **648 次参数更新**。本轮使用 pairs/SDPA，有 2,556,417 个可训练参数；head 和 LoRA 均有实际更新。

| 选优阶段 | Dev nDCG@10 |
|---|---:|
| 随机 head 初始值，仅作参照 | 0.090762 |
| Step 324 | 0.242403 |
| Step 648，选定 best | 0.263831 |

完整 dev 为 324 query；选定 best 后完成一次 323-query test、新进程重载和实际产物的 workflow 调用。

| 模型 | Test nDCG@10 |
|---|---:|
| 原始 Qwen3-Reranker-0.6B | 0.354783 |
| Base + ScoreHead + LoRA | 0.290978 |

该配置下，ScoreHead 的 nDCG@10 低于原始 reranker。两者的输入模板、读出和训练历史不同，结果用于完整方法的对照。评测固定 BM25 top50，保留零正例召回 query；弱负例和训练正例注入的处理见 [cookbook](../../cookbooks/score-head-plan.md)。

Best/final 分开导出；重载 150 个候选的最大分数差为 0。Workflow 使用模型排序结果构建证据上下文，尚未评估下游 LLM 答案质量。

## BoolQ 读出结构研究

2026-10-06 在真实 BoolQ 标签上完成冻结 Qwen3-0.6B-Base 表示的四种评分头、三个种子实验。固定 384 train / 96 dev / 192 held-out evaluation，各组 360 次优化更新，按 dev NLL 选模型；12 个选定头的新进程重载分数完全一致。

冻结表示阶段，MLP 平均 accuracy 为 63.02%，SwiGLU 为 63.37%，SwiGLU + 标量残差为 64.76%，训练多数类参考为 65.63%。残差相对 MLP 的探索性差值区间包含零。单种子联合 LoRA 阶段完成三组、共 216 次优化更新，accuracy 分别为 61.46%、59.38%、55.21%；没有支持替换默认读出的稳定收益。该研究使用独立保存格式，完整配置与结果见 [BoolQ 消融](../../cookbooks/boolq-readout-ablation.md)。

2026-10-07 完成固定 MLP 的第二轮读出实验：1,536 train / 192 dev / 384 新预留 evaluation，比较末尾 Relevance 标记、Decision 标记与候选正文注意力池化，每种读出使用三个训练种子。每组 3 epochs、288 次优化器调用，共九组；最后一次调用学习率为零。按各组 dev NLL 选 checkpoint，九组全部冻结后才评测新的 evaluation。

| 读出 | 平均 Dev NLL | Evaluation accuracy | 平衡准确率 | 平均 NLL | 平均 Brier |
|---|---:|---:|---:|---:|---:|
| 训练先验常数参考 | — | 58.33% | 50.00% | 0.6844 | 0.2455 |
| 末尾 Relevance | 0.6799 | 58.33% | 50.00% | 0.6818 | 0.2443 |
| 末尾 Decision | 0.6811 | 58.33% | 50.00% | 0.6824 | 0.2446 |
| 候选注意力池化 | 0.6827 | 58.33% | 50.00% | 0.6833 | 0.2449 |

开发集选出的读出仍为 Relevance。九组在固定 0.5 阈值下均预测正类；Decision 和池化相对 Relevance 的 NLL/Brier 配对区间包含零，没有显示分类收益。相同决策不能证明架构等价，微小概率指标变化也不构成任务能力提升。

全部选定 head/LoRA 均有实际更新，池化 query 也有更新，冻结基座保持不变；九组新进程重载的三条 dev 参考分数差均为零。峰值已分配显存约 2.40 GiB。该实验使用独立保存格式，不是生产 `Predictor` 产物；Decision 是现有词表中的文本标记，不是新增可学习 token 或层内注入。协议、置信区间和输入审计见 [BoolQ 读出位置与池化](../../cookbooks/boolq-readout-selection.md)。

两轮研究不改变 v1 模型、冻结 NFCorpus 配方或原实验产物，也不计为 RL 或完整 BoolQ 基准。开源决策模型常用任务与训练方式见 [Jev 风格任务调研](../research/jev-task-survey.md)。

2026-10-09 完成[小训练集拟合诊断](../../cookbooks/boolq-training-diagnostic.md)：在 32 条无截断、正负平衡的训练记录上，现有 Base + MLP + LoRA 的两档学习率分别在 100 与 60 次更新后达到预定停止条件，训练 accuracy 均为 100%。两臂初始化相同，head/LoRA 均更新、冻结基座不变，独立进程重载参考 logits 差为 0。该结果说明小子集可以被拟合，不是开发集或未见评测收益，也不能将此前全肯定预测单独归因于评分头。

## 待完成

2026-10-21 完成[跨类型 GitHub 调研与定位评估](../research/github-landscape-2026-10.md)，核查 15 个仓库并记录固定版本。建议先建立有效任务基线、补齐单选评测与产物闭环，再按质量和成本证据扩展；该建议未改变 v1 架构，未新增训练或评测结果。

通用 task/objective/backend 注册、单选/多选的完整评测与选优、概率校准、更多 backbone、稀疏 attention、RL、多卡、真实训练布局消融，以及第二个任务的工具包完整闭环。Tree 仅有小规模分数、梯度与候选换序正确性预检，尚未验证训练质量或加速。BoolQ 已有多种子联合 LoRA 研究，仍需建立超越简单基线的任务质量。相关实现与扩展方向见[项目对照](../research/related-projects.md)。
