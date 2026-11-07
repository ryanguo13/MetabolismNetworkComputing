# MRI3 内层与历史连续输出证据

2026-11-08。详见 [报告](../../docs/m3-inner-output.md)。没有减少完整审计，没有 M4 准入。

- `registration.json`、`runs.json`、`0.npz`–`11.npz`：24 个内层输出候选与 24 个同保护完整对照；每段三次多项式、查询值、独立与收紧参考、原单段 Hermite 对照。
- `order-registration.json`、`order.json/.npz`：五个宏步的精确线性单步误差阶诊断。
- `history-registration.json`、`history.json`、`history-*.npz`：六条实际 MRI3 八步轨迹，三种输出共享真实端点；不足四端点时历史输出不可用，未使用参考状态构造历史。
- `tests.txt`：五项自检；`summary.json`：参考、曲线、极值、连接、物理收支、阶数与历史重建的复核。
- `environment.json`、`manifest.json`：环境、父提交和本轮证据 SHA-256。

默认只读复核：`bash scripts/run-macos.sh -m research.m3_dense_report`。
