# 兼容范围与支持约定

MetisCraft `0.1.0rc1` 是预览候选版。本页定义已验证范围、公开入口和加载规则，
不表示已经承诺与后续候选版或稳定版 `0.1.0` 兼容。

## 运行环境与依赖范围

| 范围 | 已验证内容 | 边界 |
|---|---|---|
| Python | 3.10 和 3.11 | 元数据中的 `>=3.10` 是安装约束；更高版本尚未验证。 |
| CPU 流程 | Linux、POSIX shell、CPU PyTorch、源码及安装后的 wheel | Windows、macOS 及其他操作系统组合不在发行测试矩阵内。 |
| CPU 分布式训练 | 双 rank Gloo 回归测试 | 不代表可以恢复任意进程、主机或网络故障。 |
| GPU 训练 | 各配方记录的 Qwen3、Qwen3.5 运行 | 模型与内核支持取决于基座、Transformers 和 PyTorch 构建；CPU 测试不覆盖全部 CUDA 路径。 |
| 推理 | Transformers、`Predictor`、Python／JSONL 和本地 HTTP | 尚未实现 vLLM、SGLang、llama.cpp 导出。 |

[安装包验收工作流](https://github.com/Shichaoxx/MetisCraft/actions/workflows/release.yml)
只构建一份 wheel，在两个 Python 版本中验证，包含源码目录外的完整流程。
手动运行会跳过发布。

以下依赖组合通过了
[2026-11-10 验收](https://github.com/Shichaoxx/MetisCraft/actions/runs/42762311014)。
这是具体环境的证据，不表示已验证从声明下界到这些版本之间的全部组合：

| Python | PyTorch | Transformers | PEFT | Accelerate |
|---|---|---|---|---|
| 3.10 | `2.14.1+cpu` | `5.20.2` | `0.21.2` | `1.15.0` |
| 3.11 | `2.15.0+cpu` | `5.22.2` | `0.21.2` | `1.15.0` |

安装依赖以 `pyproject.toml` 为准。具体模型家族可能要求高于工具包最低要求的
Transformers 版本。复现结果时保存运行的 `env.json` 和准确的发行包身份；
只满足依赖下界不等于使用同一环境。安装 MetisCraft 本身不会下载权重或数据集。

## 公开接口

| 接口 | 本候选版约定 |
|---|---|
| CLI | 可用选项以 `metiscraft --help` 及子命令帮助为准。配方采用校验后的 schema，未知字段会被拒绝。 |
| Python | 文档入口为 `Record`、配方加载、`run_recipe`、`evaluate_artifact` 与 `Predictor.load`；内部辅助函数不属于扩展 API。 |
| Record／JSONL | 在产物读出支持的前提下提供 `noul`、`choice`、`score`、`multi_label`；保留任务与候选 ID。 |
| `/v1/systemone` | `noul`、`choice`、`score` 使用文档说明的请求和响应字段；多标签使用 Record／JSONL。 |

本候选版 HTTP `confidence` 是选项／等级的最大概率，不等于 TypeSafe 归一化后的
confidence 统计量。Record 的 `score` 决策取众数等级下标，HTTP `score` 返回
零起始等级下标的期望。本次更新保留这些语义。详见[接口约定](interfaces.md)和
[输出语义](tensor-analysis/training-calibration.md#概率分数与-confidence)。

## 产物与校准兼容性

- 新训练写出 `metiscraft-artifact/2`。加载时核对所记录基座的权重、tokenizer、
  配置与来源身份；即使重新生成本地校验清单，内容变化仍会校验失败。
- 保持 `calibration/` 与 `artifact/` 相邻，或显式指定校准路径。校准文件必须
  对应该产物；保留原有任务 ID，或配置明确的 HTTP 任务映射。
- 封存基座可以迁移：用 `--base-path` 指向新位置的相同字节。产物不等于已经完整
  复制全部基座模型。
- 缺少基座身份的历史 `metiscraft-artifact/1` 必须显式使用
  `--allow-legacy-artifact`，加载后仍标记 `legacy_unverified`。未知产物格式被拒绝；
  此开关不能绕过无效的身份记录。
- 训练产物没有受支持的优化器／检查点续训协议。加载产物用于预测不等于恢复训练。

命令见[加载、任务路由与迁移](serving.md)。

## 分布式生命周期

`run_recipe` 创建的通信组会在初始化、训练、产物保存、校准或评测抛出异常时释放。
清理不会新增 barrier；若清理本身在处理另一异常时失败，会记录清理错误并保留原始
异常。正常运行结束后的清理失败会报告给调用方。

调用方预先创建的通信组在成功或失败后均继续由调用方持有；调用方负责协调各 rank
并释放该组。该所有权约定不会修复此前以不同依赖导入顺序初始化的通信组。

双 rank CPU 测试注入建模失败、反向传播后的训练失败和评测失败，并在异常处理完毕后
核对两个 rank 的退出、通信组及 Gloo 线程释放。已经阻塞在集合通信中的 rank 仍可能
依赖后端超时或进程启动器处理；不提供自动重试、弹性恢复或 GPU 容错保证。

## 候选版之间的变化

`0.1.0rc1` 仍处于预览阶段，后续候选版可能调整 API、配方字段或产物格式；变化及迁移
说明记录在[版本变更](../../CHANGELOG_zh.md)。每次运行保存包版本、源码版本、配置、环境和封存
基座。不能假定未来的包会静默迁移旧产物；不支持的格式应明确报错。

[English](../en/compatibility.md)
