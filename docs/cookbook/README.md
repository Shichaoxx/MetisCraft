# MetisCraft Cookbook

讲清 MetisCraft 怎么工作、怎么用好。规范性的定义见 [`design.md`](../design.md)，这里讲原理、做法和经验；所有例子与数字都取自真实运行。

| 篇 | 内容 |
|---|---|
| [原理](principles.md) | 一条记录长什么样；四种判断类型怎样从同一个打分矩阵得到概率和损失；① 与 ③ 两种读出；损失为什么按全局题数归一；校准怎样先撤销加权、再按类型除温度；评测口径与可追溯 |
| [把自己的业务做成 Jev](your-own-jev.md) | 用银行客服分流从头走一遍：把业务问题写成题、映射表、准备数据、写配方、训练评测、部署调用 |
| [最佳实践](best-practices.md) | 读出、训练目标、校准诊断、数据划分与可追溯的 17 条做法，附 v0.1 验收的真实数字 |
| [配方字段参考](recipe-reference.md) | 每个配方字段的类型、默认值和含义，由代码生成 |

正在补充：用 MetisCraft 端到端复现公开项目，第一个是 Tev1，见 [`reproductions/tev1.md`](../reproductions/tev1.md)。

## 图

图按 viz Skill 的规范制作：张量图用 TikZ，沿用 [`readouts/readout-style.tex`](../readouts/readout-style.tex) 的配色与宏；流程图和决策图用 vizkit 规格渲染。导出后都逐张目视检查过，记录在 [`checks/`](checks/)。

| 图 | 用在 | 源码 |
|---|---|---|
| [判断类型：打分 → 概率 → 损失](figures/judgment-types.png)（[PDF](figures/judgment-types.pdf)） | 原理 §2 | [`src/judgment-types.tex`](src/judgment-types.tex) |
| [推理与校准](figures/calibration.png)（[PDF](figures/calibration.pdf)） | 原理 §5 | [`src/calibration.tex`](src/calibration.tex) |
| [最佳实践](figures/best-practices.png)（[SVG](figures/best-practices.svg)） | 最佳实践 | [`src/best-practices.json`](src/best-practices.json) |
| [① 答案 token](../readouts/readout-1-lm-token.png)、[③ 逐候选标量头](../readouts/readout-3-candidate-scalar.png) | 原理 §3 | [`readouts/`](../readouts/) |
| [一次运行的全流程](../reports/20261105-v0.1/figures/01-run-pipeline.png)、[客服分流场景](../reports/20261105-v0.1/figures/02-support-scenario.png) | 业务篇 | [`reports/20261105-v0.1/src/`](../reports/20261105-v0.1/src/) |

重新生成：

```bash
cd docs/cookbook/src
tectonic judgment-types.tex && pdftoppm -png -r 300 -singlefile judgment-types.pdf ../figures/judgment-types
tectonic calibration.tex && pdftoppm -png -r 300 -singlefile calibration.pdf ../figures/calibration
python -m vizkit render best-practices.json --target paper --out <输出目录>   # 需要 viz Skill 的 vizkit
python recipe_reference.py                                                  # 重写 ../recipe-reference.md
```
