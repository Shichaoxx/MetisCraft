# MetisCraft 开发手册

MetisCraft 用一份配方组织决策模型的数据、训练、校准与评测。已有模型的运行目录时，可以直接从下方的预测与服务开始。

## 安装与首次运行

按[安装说明](../installation.md)安装软件。要在 CPU 上跑通一次完整流程，请用 Cookbook 的[首个模型教程](../cookbook/first-model.md)，从准备示例数据走到加载训练产物。

## 数据、训练与校准

1. [准备数据](../data.md)。用 Record 描述输入、任务和标签，分别指定训练、校准和评测数据。
2. [运行训练与评测](../training.md)。设置基座与读出方式，用校准集拟合温度，在评测集上读指标。需要查某个配置项时，见[配方字段](../../cookbook/recipe-reference.md)。

## 预测与服务

要核对配方采用的读出、训练保存的文件，以及校准如何绑定到产物，见[模型结构与训练产物](model-and-artifacts.md)。

[加载与部署产物](../serving.md)介绍如何从运行目录加载模型、批量预测 Record，以及启动 HTTP 服务。`/v1/systemone` 支持 `noul`、`choice` 和 `score`。预测 `multi_label` 时使用标量读出的产物，通过 Python 或 Record JSONL 批量入口调用。

## 技术参考

[接口与产物约定](../interfaces.md)列出判断类型、返回字段和 Python 入口。[兼容范围](../compatibility.md)说明环境和产物版本要求，[能力边界](../limitations.md)说明当前版本适用的场景。
