# 发布到 PyPI

发布流程见 [release.yml](../.github/workflows/release.yml)，发布进度见
[status.md](status.md)。包名是 `metiscraft`，版本由 `pyproject.toml` 与
`src/metiscraft/__init__.py` 一起维护。

## 首次配置

在 GitHub 仓库设置中创建 `pypi` environment，添加发布审核人，并将允许部署的
ref 限制为 `v*` tag。

在 PyPI 的 [Publishing 设置](https://pypi.org/manage/account/publishing/)中登记
pending publisher；已有项目则在项目的 Publishing 页面添加：

| 字段 | 值 |
|---|---|
| PyPI project name | `metiscraft` |
| GitHub owner | `Shichaoxx` |
| Repository | `MetisCraft` |
| Workflow filename | `release.yml` |
| Environment | `pypi` |

工作流使用短期 OIDC 凭据，无需保存 PyPI API token。首次成功发布会创建项目；
pending publisher 不会预留包名。配置依据见
[PyPI Trusted Publishing 文档](https://docs.pypi.org/trusted-publishers/creating-a-project-through-oidc/)。

## 验证与发布

1. 将两个版本字段改为同一待发布版本，提交后检查 CI。
2. 可从 Actions 手动运行 `release` 验证选定 ref：检查版本一致性、构建 sdist 和
   wheel、检查元数据，再在 Python 3.10 与 3.11 上安装同一份 wheel 并跑完整
   CPU 测试。手动运行只留下 `distributions` artifact，不发布。
   GitHub 要求含 `workflow_dispatch` 的工作流已在默认分支，手动入口才会显示。
3. 发布时，在通过检查的提交上创建并推送 `v<版本>` tag，例如版本 `0.1.0` 对应
   `v0.1.0`。tag 必须与包版本完全一致；旧版的 `metis-v0.1.0` 不触发此流程。
4. tag 触发相同的构建与 wheel 测试。全部通过后，在 `pypi` environment 批准
   发布；发布 job 只下载并上传已经测试的产物，不重新构建。

构建和测试 job 只有仓库读取权限，只有发布 job 获得 `id-token: write`。分离方式
遵循 [PyPI 的发布说明](https://docs.pypi.org/trusted-publishers/using-a-publisher/)。
审核人和 tag 限制在 GitHub environment 中配置，YAML 本身不会创建这些保护规则。

发布版本不能覆盖。上传失败时先查看 PyPI 是否已有部分文件，再决定重试或使用
新版本；流程不设置 `skip-existing`，以免把版本冲突误报为成功。
