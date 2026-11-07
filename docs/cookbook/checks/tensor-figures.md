# 张量图检查记录

两张图都用 `../../readouts/readout-style.tex` 的配色与宏，Tectonic 编译，`pdftoppm -r 300` 导出 PNG，导出后逐张目视检查。

## judgment-types（2026-11-05）

- 形状：$S,M,P,Y\in[B,K]$，$\ell\in[B]$，$N$ 与 $\mathcal L$ 为标量；与 `judgments.batch_loss`（返回损失总和与题数）、`train/loop.accumulate`（乘 world 再除以全局题数）一致。
- 支撑：四行的有效列数依次为 1、4、5、6（是非只用第 0 列）；$M$ 是二值掩码，所有有效格同一颜色；$Y$ 只给非零格上色（选择一格，等级的高斯软目标三格，多标签两格）。
- 几何：$S,M,P,Y$ 同为 $4\times 6$，格子大小一致；$\ell$ 为 $4\times1$。
- 版面：连线都有可见线身，无交叉、无压字；底部三行说明框与署名行完整；无缺字（修正过一次"标量"进入数学模式导致的缺字）。

## calibration（2026-11-05）

- 形状：一道题一行，$S,S',p\in[K]$；$w,T$ 为标量。
- 语义：只有多标签且训练时用了 `pos_weight` 才减 $\ln w$（`judgments.prior_shift`，在 `evaluate/infer.predict` 里施加）；$T$ 按"任务|类型"在 calib 上拟合（`calibrate.fit_calibration`）。
- 版面：两行阶段标题左对齐，$T$ 的虚线不穿过文字；底部说明框与署名行完整，无缺字。
