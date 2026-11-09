# 参与贡献

从[核心文档](docs/zh/index.md)和 [Cookbook](docs/zh/cookbook/index.md)开始。涉及公共 API、判断语义或产物格式的较大变更，请先讨论设计。

## 开发环境

从 PR 的目标分支创建功能分支；当前预览版可用的基线是 `release/0.1.0rc1`。在独立 Python 环境中执行：

```bash
python -m pip install -e '.[dev,serve,reproduce]'
python scripts/check_quality.py
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 python -m pytest
```

质量检查包含 Ruff 格式与规则检查，以及配置和 Record 接口的基础 Pyright 检查。维护者钩子使用 Python 3.11，可通过 `pre-commit install` 安装；CI 只检查格式，不自动改写文件。

测试使用随机微型模型和构造数据，在 CPU 上运行，无需下载权重或数据集。外部数据构建器测试需要 `reproduce` 依赖组。修改安装包工作流时运行[快速上手](docs/zh/quickstart.md)，修改模板或注册表时核对打包资源。

## 实现约定

- 判断语义放在 `judgments/`，指标放在 `evaluate/`，影响结果的默认值放在 `config.py`；实验设置写入配方。
- 训练、校准和评测的用途独立；`eval_only` 标签不能用于训练或温度拟合。
- 产物绑定经过校验的基座内容；缺少原始身份的历史产物保持显式未验证状态。
- Record 与 HTTP 使用一致的任务路由，并保留可检查的校准回退信息。
- 为行为变更增加针对性回归；新读出和判断类型需要形状检查，等价实现需要数值比较。

自有数据无需修改注册表。新增可复用的内置数据集，需要在 `src/metiscraft/data/registry.yaml` 声明固定来源版本、转换器、许可证和条款。不要提交由 Jev 或 TypeSafe 接口输出生成的训练标签。

## 文档与审阅

维护 `docs/public-pages.json` 声明的语言配对，英文为默认语言。导出教程的命令以 `src/metiscraft/templates/quickstart/README.md` 为准；命令须与 CLI 一致，工程检查与真实模型效果分别报告。研发计划和进度在 `dev` 维护，与公共代码和文档分开提交；公开文件不含机器特定信息。

提交消息使用英文祈使句。PR 应说明行为变化、相关验证和实质限制；实验结论链接到运行记录。默认分支与发行包面向使用者，研发记录保留在 `dev`。

## 许可证

代码采用 [Apache-2.0](LICENSE)。提交贡献意味着同意按 Apache-2.0 许可；数据集和模型保留各自许可证。
