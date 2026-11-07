# 实现优化后的影子误差估计

2026-11-07。与第一版数学公式及阈值相同；批量求值、共享 Jacobian、稠密／稀疏 LU 与插值实现不同。见 [评估报告](../../docs/m3-shadow-evaluation.md)。

- `registration.json`：优化源码、旧开发数据和第二批独立预留起点的指纹。
- `development.json/.npz`：与第一版相同的 68 个候选，用于等价性核验，不计作新的独立样本。
- `new-states.json/.npz`：第二批 36 个新候选、完整独立审计及所有失败。
- `summary.json`：两版指标、局部筛选、批量／标量与误差轨迹等价性、历史指纹复核。
- `tests.txt`：3 项刚性／内部曲线／守恒自检；`historical-checks.txt`：旧验收只读检查输出。
- `environment.json`、`manifest.json`：运行环境、父提交、结果和源文件 SHA-256。

大模型通过本次局部研究筛选，小模型未通过成本筛选。尚无在线误差保证、全窗口性能资格或 M4／PE 准入。

只读复核：`bash scripts/run-macos.sh -m research.m3_shadow_report`。
