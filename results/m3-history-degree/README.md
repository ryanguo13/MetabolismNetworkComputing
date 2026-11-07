# 历史多项式次数与端点成本证据

2026-11-08。见 [报告](../../docs/m3-history-cost.md)。未达到 M4，原审计与结构要求保留。

- `registration.json`、`runs.json`、`0.npz`–`5.npz`：三／四／五次历史插值的 90 条已知轨迹重分析；共同区间比较，未使用参考端点构造输出。
- `cost-registration.json`、`cost.json`、`cost-*.npz`：移除无用 Hermite 后的端点＋五次历史输出与完整系统同保护对照；96 条记录含 24 次预热。
- `tests.txt`：三项新增自检；`summary.json`：曲线、误差、收支、成本与全部不合格结果。
- `environment.json`、`manifest.json`：环境、父提交与本轮源码／数据 SHA-256。

只读核验：`bash scripts/run-macos.sh -m research.m3_history_degree_report`。
