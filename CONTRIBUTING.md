# 参与开发

MetisCraft 还在 v0.1 开发中，接口随时会变。当前设计见 [`docs/design.md`](docs/design.md)，进度见 [`docs/status.md`](docs/status.md)。

## 开发环境

```bash
pip install -e ".[dev,serve]"
pytest
ruff check .
```

只跑测试时 CPU 版 PyTorch 就够了：测试用微型随机初始化模型，不下载任何权重或数据。

## 约定

- **一事一处**：判断类型的语义只在 `judgments/` 定义，指标只在 `evaluate/` 定义，配方只引用注册名。
- **默认值有出处**：影响结果的默认值只写在 `config.py`；配方里显式给出训练超参。
- **测试**：新读出或新判断类型要带形状测试；与已有实现等价的部分（比如答案 token 读出与线性头）要带逐位一致的测试。
- **数据**：新数据集先进 `src/metiscraft/data/registry.yaml`，写明固定 revision、许可证与条款；不接受用 Jev / TypeSafe API 输出作标签的数据。
- **提交说明**：英文，祈使句开头，说明为什么改。
- **文档**：中文；项目状态只写在 `docs/status.md`；公开文件里不写本机路径。

## 许可证

代码以 Apache-2.0 发布。提交贡献即表示同意以 Apache-2.0 授权你的贡献。
