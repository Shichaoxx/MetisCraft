# 安装

MetisCraft 要求 Python **3.10 或更高版本**。建议从 Python 3.11 的独立虚拟环境开始。
下列命令使用 POSIX shell。快速上手只需要 CPU 版 PyTorch；真实模型训练需要与所选模型
匹配的 PyTorch 构建和硬件。

## 从发行候选 wheel 安装

`0.1.0rc1` 已在 [GitHub 发布为预发行版](https://github.com/Shichaoxx/MetisCraft/releases/tag/0.1.0rc1)，发行包通过 GitHub 附件提供。

[下载 wheel](https://github.com/Shichaoxx/MetisCraft/releases/download/0.1.0rc1/metiscraft-0.1.0rc1-py3-none-any.whl) · [下载源码发行包](https://github.com/Shichaoxx/MetisCraft/releases/download/0.1.0rc1/metiscraft-0.1.0rc1.tar.gz) · [SHA256SUMS](https://github.com/Shichaoxx/MetisCraft/releases/download/0.1.0rc1/SHA256SUMS)

在独立 Python 环境中安装适用的 PyTorch，然后在下载目录运行以下命令。`serve` 扩展提供 HTTP 服务依赖。

```bash
sha256sum -c SHA256SUMS --ignore-missing
python -m pip install './metiscraft-0.1.0rc1-py3-none-any.whl[serve]'
metiscraft --version
```

## 从源码安装

从固定的 `0.1.0rc1` 标签获取与本次预发行对应的源码。

```bash
git clone --branch 0.1.0rc1 --depth 1 https://github.com/Shichaoxx/MetisCraft.git
cd MetisCraft
python3.11 -m venv .venv
. .venv/bin/activate
python -m pip install '.[serve]'
metiscraft --version
```

如果需要特定的 CPU 或 CUDA 版本，先在这个环境中安装对应的 PyTorch，再安装
MetisCraft。工具包使用当前环境里的 PyTorch。

本预览版已验证的 Python 支持范围为 3.10 和 3.11。安装元数据中的最低版本要求不代表
已验证更高版本 Python 或全部依赖组合。已验证环境、接口边界与产物加载策略见
[兼容范围与支持约定](compatibility.md)。

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
