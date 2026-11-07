# SPC-MRI-GARK 两模型局部证据

2026-11-08，开发研究，M4 未准入。见[报告](../../docs/m3-spc.md)。

- `registration.json`、`runs.json`：144 个 SPC、12 个完整 SDIRK、36 个成熟完整 Radau，共 192 条首批记录；失败保留。
- `reference-*.npz` 与候选 NPZ：共同初值／查询／严格参考、预测阶段、完整阶段通量、驱动系数、原始快曲线与实际返回曲线。
- `order.json`：20 条线性局部诊断和 10 条非线性全局阶数记录，保留粗步误差抵消。
- `predictor-trace.json`：三个大模型失败预测器的缩放残差轨迹；诊断不用于计时。
- 相邻 [`m3-spc-solver-tolerance`](../m3-spc-solver-tolerance/README.md)：独立预登记的 174 条内部求解容差补测，不覆盖首批。
- `literature-provenance.json`、`tests.txt`、`summary.json`、`environment.json`、`manifest.json`：文献版本／系数阅读、自检、独立核验、环境和两目录统一指纹。

只读复核会重算参考和预测器：`bash scripts/run-macos.sh -m research.m3_spc_report`。两批都是单次局部乐观计时，不构成正式加速证据。
