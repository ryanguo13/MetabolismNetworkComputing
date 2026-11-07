# 同优化机会的 MRI／完整系统局部成本实验

MRI 与完整 Radau 均可使用稠密小系统／MMD_AT_PLUS_A 稀疏排序；CVODE-KLU 为额外强基线。三个区间、两档内层容差、一次预热和三次轮转开发重复，含所有失败。无正式全窗口性能结论。

- `registration.json`：实际使用的冻结协议及源码。`registration-draft-before-ledger-review.json` 是实验前审查草稿，未用于数据生成。
- `runs.json`、`curves.npz`：耗时、工作量、结构及实际误差、候选与两级参考。
- `summary.json`：压力测试与成本数据的独立重算结果。
- `audit-registration.json`、`optimized-audit.json/.npz`：发现公平基线变快后，另行登记并测量优化高精度审计，重审估计器成本分母。
- `tests.txt`：刚性守恒链的精确矩阵指数核验。
- `manifest.json`：本轮数据、代码与报告指纹。

详见 [报告](../../docs/m3-followup-evaluation.md)。只读复核：`bash scripts/run-macos.sh -m research.m3_followup_report`。
