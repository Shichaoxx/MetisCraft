# 版本变更

## 0.1.0rc1 — 尚未发布

首个公开预览以配方连接数据准备、类型化决策模型训练、概率校准与评测，以及产物服务。候选尚未上传 PyPI；API 与产物格式仍可能变化。

### 新增

- `train`、`eval`、`predict`、`serve` 命令串起准备数据、模型拟合、校准、评测与部署。
- Record 支持是非、选择、有序等级、多标签四种判断。
- 答案 token 和逐候选标量头；支持 LoRA、全参或冻结基座；标量头支持 list 与 pairs。
- 内置数据注册表固定来源，保留许可证／条款、划分用途和版本校验和。
- 温度只在 `calib` 拟合，按任务／类型选择；评测同时报告原始与校准后的 NLL、Brier、ECE 和任务指标。
- 运行记录保存配置、数据／基座身份、可取得的源码版本、环境和逐题预测，以及封存产物、校准文件与许可通知。
- Python Predictor 和 JSONL 批量入口支持四类型；`/v1/systemone` HTTP 支持是非、选择、有序等级。
- BoolQ、Banking77、CLINC150、SST-5、GoEmotions、客服分流和 Tev1 配方；证据边界见下文。
- `metiscraft init` 从安装包导出完整 CPU 教程，生成构造数据与随机微型模型，无需下载权重。
- `metiscraft data import` 通过 manifest 导入本地监督 Record JSONL，无需改注册表；同一输入复用并验证既有版本，输入变化需要新版本。
- 公开文档、Cookbook、原理与张量解析提供中英配对。
- 可编辑双语张量图，SVG／PDF 下载、本地数学渲染和同页语言切换。

### 变更

- 训练在加载模型前验证划分用途，拒绝拟合与评测间的输入／组重叠；本地导入保留显式划分。
- `metiscraft-artifact/2` 绑定本地基座内容，包括权重、tokenizer、配置与来源元数据；即使本地校验清单被重新生成，加载时也核对原始身份。
- HTTP 单任务自动映射；多任务使用显式问题到任务映射，缺失映射时报错。
- Record 暴露校准分组、回退与基座验证状态；health 端点也报告基座验证状态。
- choice／score 软标签须和为一；choice 禁用保留 ID `__null__`，导入封存前检查。
- 自定义数据条款保持未解释状态（`commercial_use: null`），不擅自声明可商用；许可通知保留标记并要求核查原条款。

### 迁移与证据

- 缺少基座身份的开发期 `metiscraft-artifact/1` 需要 `--allow-legacy-artifact` 或 `allow_legacy_artifact=True`，发出警告并保持 `legacy_unverified`；当前文件不能追认历史运行。该选项不转换早期 Metis 原型检查点。
- Qwen3.5-4B 的 Tev1 复现达到预定准确率标准：主集 87.3%、政策迁移 99.33%，原版为 88.0%／100%；逐题一致率 94.6%／99.33%。校准仍可能恶化 NLL，历史训练设置不可得。
- CPU 教程不提供真实模型效果证据。GoEmotions 正式运行早于推理修正；同协议 encoder 对照尚未完成。

详见[能力边界](docs/zh/limitations.md)。早期 Metis 0.1.0 保留在 `metis-v0.1.0` tag，与 MetisCraft 版本线不同。

### 可靠性与文档

- 标量头 DDP 排除未使用、未共享的 LM 输出权重，同时保留共享词嵌入；CPU 回归覆盖 tied weights、梯度检查点，以及带累积的两个优化步。
- completion 答案码计入 `model.max_length`；Jev score 拒绝布尔、非整数与 null 金标，不静默强制转换。
- 双语 Cookbook 覆盖真实判断、本地 CPU 闭环和自有数据，命令来自安装包双语模板。
- epoch 尾批不再跨轮补足梯度累积；配方数值范围在模型加载前检查。
- 新导入显式记录硬标签存在性；软标签与歧义历史数据不生成虚构的硬指标，未定义指标序列化为 null。
- 原始与解析后配置身份分开记录；批量预测保持输入顺序。
- 公开文档和发行内容使用明确清单，研发记录保留在 dev。
