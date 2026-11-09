# 安装

MetisCraft 要求 Python **3.10 或更高版本**。建议从 Python 3.11 的独立虚拟环境开始。
下列命令使用 POSIX shell。快速上手只需要 CPU 版 PyTorch；真实模型训练需要与所选模型
匹配的 PyTorch 构建和硬件。

## 从源码安装

从主分支安装当前预览版；已发布的发行包见 [GitHub 发行记录](https://github.com/Shichaoxx/MetisCraft/releases)：

```bash
git clone --branch release/0.1.0rc1 https://github.com/Shichaoxx/MetisCraft.git
cd MetisCraft
python3.11 -m venv .venv
. .venv/bin/activate
python -m pip install '.[serve]'
metiscraft --version
```

如果需要特定的 CPU 或 CUDA 版本，先在这个环境中安装对应的 PyTorch，再安装
MetisCraft。工具包使用当前环境里的 PyTorch。

## 从发行候选 wheel 安装

如果已经拿到构建好的 wheel，在当前环境中安装该文件：

```bash
python -m pip install './metiscraft-0.1.0rc1-py3-none-any.whl[serve]'
metiscraft --version
```

这是本地 wheel 安装，不代表已经可以从 PyPI 执行 `pip install metiscraft`。
可用版本见[发行记录](https://github.com/Shichaoxx/MetisCraft/releases)。

## 可选依赖

| 依赖组 | 用途 |
|---|---|
| `serve` | HTTP 服务所需的 FastAPI 和 Uvicorn |
| `reproduce` | 外部数据构建脚本所需的 `datasets` 和 Jinja2，包括 Tev1 |
| `dev` | 测试、代码检查与开发验证 |

在仓库中开发时使用：

```bash
python -m pip install -e '.[dev,serve,reproduce]'
```

安装包本身不会下载模型权重或数据集。[快速上手](quickstart.md)会在安装后自行生成小规模
本地输入。

[English](../en/installation.md)
